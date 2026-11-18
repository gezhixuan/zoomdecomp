"""Qwen2.5-VL / Qwen3-VL wrapper: bit-exact seen raster, pixel-budget control, cached-visual-token injection
(including Qwen3-VL DeepStack features), and an in-place cached-ROI forward with preserved mRoPE positions."""
import contextlib, math, re, types
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_processing_utils import SizeDict

FACTOR = 28  # Qwen2.5-VL default (patch 14 x merge 2); Qwen3-VL uses 32. Functions take `factor` explicitly.
MIN_PIXELS_DEFAULT = 56 * 56
CLIP_MEAN_RGB = (123, 117, 104)


# ---------------- image geometry ----------------
def resize_like_processor(ip, img: Image.Image, rh: int, rw: int) -> Image.Image:
    """Bit-exact replica of the processor's resize: call ITS resize on the uint8 tensor (verified max|diff| = 0)."""
    if img.size == (rw, rh):
        return img
    t = TF.pil_to_tensor(img).unsqueeze(0)
    out = ip.resize(image=t, size=SizeDict(height=rh, width=rw), resample=ip.resample)[0]
    return TF.to_pil_image(out)


def seen_dims(w, h, max_pixels, factor=FACTOR, min_pixels=MIN_PIXELS_DEFAULT):
    return smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)


def fit_dims(w, h, pixels, factor=FACTOR):
    """(rh, rw) that consumes ~`pixels` pixels (i.e. pixels / factor^2 tokens), multiples of factor."""
    return smart_resize(h, w, factor=factor, min_pixels=pixels, max_pixels=pixels)


def crop_box(img, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return img.crop((x0, y0, x1, y1))


def mask_outside(img, box, fill):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bg = Image.new("RGB", img.size, fill)
    bg.paste(img.crop((x0, y0, x1, y1)), (x0, y0))
    return bg


def roi_grid(bbox, W, H, rh, rw, factor=FACTOR, pad_frac=0.5, min_tokens=2):
    """Native bbox (+context pad) -> first-pass merged-token grid box (x0, y0, x1, y1), exclusive upper bounds."""
    x0, y0, x1, y1 = bbox
    pad = pad_frac * max(x1 - x0, y1 - y0)
    x0p, y0p = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1p, y1p = min(float(W), x1 + pad), min(float(H), y1 + pad)
    sx, sy = rw / W, rh / H
    GW, GH = rw // factor, rh // factor
    gx0, gy0 = int(math.floor(x0p * sx / factor)), int(math.floor(y0p * sy / factor))
    gx1, gy1 = int(math.ceil(x1p * sx / factor)), int(math.ceil(y1p * sy / factor))
    gx0, gy0 = max(0, min(gx0, GW - 1)), max(0, min(gy0, GH - 1))
    gx1, gy1 = max(gx0 + 1, min(gx1, GW)), max(gy0 + 1, min(gy1, GH))
    while gx1 - gx0 < min(min_tokens, GW):
        if gx0 > 0: gx0 -= 1
        if gx1 - gx0 < min_tokens and gx1 < GW: gx1 += 1
    while gy1 - gy0 < min(min_tokens, GH):
        if gy0 > 0: gy0 -= 1
        if gy1 - gy0 < min_tokens and gy1 < GH: gy1 += 1
    return (gx0, gy0, gx1, gy1), (GW, GH)


def grid_to_seen_box(g, factor=FACTOR):
    return (g[0] * factor, g[1] * factor, g[2] * factor, g[3] * factor)


def seen_to_native_box(box, W, H, rh, rw):
    return (box[0] * W / rw, box[1] * H / rh, box[2] * W / rw, box[3] * H / rh)


def random_grid_box(g, GW, GH, rng, tries=200):
    gw, gh = g[2] - g[0], g[3] - g[1]
    if gw >= GW and gh >= GH:
        return g, 1.0
    best, best_iou = None, 9.0
    for _ in range(tries):
        rx0 = int(rng.integers(0, GW - gw + 1)); ry0 = int(rng.integers(0, GH - gh + 1))
        cand = (rx0, ry0, rx0 + gw, ry0 + gh)
        ix = max(0, min(cand[2], g[2]) - max(cand[0], g[0])); iy = max(0, min(cand[3], g[3]) - max(cand[1], g[1]))
        inter = ix * iy; iou = inter / (2 * gw * gh - inter)
        if iou < best_iou:
            best, best_iou = cand, iou
        if iou == 0:
            break
    return best, best_iou


def grid_token_indices(g, GW):
    gx0, gy0, gx1, gy1 = g
    return torch.tensor([gy * GW + gx for gy in range(gy0, gy1) for gx in range(gx0, gx1)], dtype=torch.long)


# ---------------- visual cache ----------------
class VisualCache:
    """Post-merger visual tokens per image (`main`), plus Qwen3-VL DeepStack features per layer per image (`deep`)."""
    def __init__(self, main, deep=None):
        self.main = main          # list[(n_i, D)]
        self.deep = deep          # None | list over layers of list[(n_i, D_l)]

    def sub(self, img_idx, idx):
        idx = idx.to(self.main[img_idx].device)
        main = [self.main[img_idx][idx]]
        deep = None if self.deep is None else [[layer[img_idx][idx]] for layer in self.deep]
        return VisualCache(main, deep)


class QwenRunner:
    def __init__(self, model_path, attn="sdpa"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.ip = self.processor.image_processor
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval()
        self.device = self.model.device
        cfg = self.model.config
        self.family = cfg.model_type                     # 'qwen2_5_vl' | 'qwen3_vl'
        vc = cfg.vision_config
        self.patch = vc.patch_size
        self.factor = vc.patch_size * vc.spatial_merge_size
        self.min_pixels = int(self.ip.size.shortest_edge or MIN_PIXELS_DEFAULT)
        self.mean_rgb = tuple(int(round(m * 255)) for m in self.ip.image_mean)
        self.image_token_id = cfg.image_token_id
        self.inner = self.model.model
        self.lm = self.inner.language_model
        ids = {L: self.tok.encode(L, add_special_tokens=False) for L in "ABCD"}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.letter_ids = {k: v[0] for k, v in ids.items()}
        self.digit_ids = {str(d): self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}

    # ---------- inputs ----------
    def build_inputs(self, images, prompt):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        f = self.factor
        inputs = self.processor(text=[text], images=images, return_tensors="pt",
                                min_pixels=4 * f * f, max_pixels=1 << 40)
        exp = torch.tensor([[1, im.size[1] // self.patch, im.size[0] // self.patch] for im in images])
        assert torch.equal(inputs["image_grid_thw"], exp), (inputs["image_grid_thw"], exp)
        return inputs.to(self.device)

    def reference_pixel_values(self, img, max_pixels):
        return self.ip(images=[img], return_tensors="pt", min_pixels=self.min_pixels, max_pixels=max_pixels)

    # ---------- visual cache / override ----------
    @torch.inference_mode()
    def visual_tokens(self, inputs) -> VisualCache:
        out = self.inner.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        main = [t.detach() for t in out.pooler_output]
        deep = None
        ds = getattr(out, "deepstack_features", None)
        if ds:
            sizes = [t.shape[0] for t in main]
            deep = [list(torch.split(layer.detach(), sizes)) for layer in ds]
        return VisualCache(main, deep)

    @contextlib.contextmanager
    def override_visual(self, caches):
        """Make get_image_features return cached tokens (one VisualCache per placeholder image)."""
        inner, dev, dt = self.inner, self.device, torch.bfloat16
        main = [c.main[0].to(dev, dt) for c in caches]
        deep = None
        if any(c.deep is not None for c in caches):
            L = len(caches[0].deep)
            deep = [torch.cat([c.deep[l][0].to(dev, dt) for c in caches], 0) for l in range(L)]
        def fake(pixel_values, image_grid_thw=None, **kw):
            return types.SimpleNamespace(pooler_output=list(main), deepstack_features=deep)
        inner.get_image_features = fake
        try:
            yield
        finally:
            del inner.get_image_features

    # ---------- answering ----------
    def _pack(self, text, lp0, qtype, letters, n_in, n_img):
        res = {"text": text, "n_tokens_in": int(n_in), "n_img_tokens": int(n_img)}
        if qtype == "mcq":
            res["letter_lp"] = {L: lp0[self.letter_ids[L]].item() for L in letters}
        else:
            res["digit_lp"] = {d: lp0[i].item() for d, i in self.digit_ids.items()}
        return res

    @torch.inference_mode()
    def answer(self, inputs, qtype, letters, max_new_tokens=8, caches=None):
        ctx = self.override_visual(caches) if caches is not None else contextlib.nullcontext()
        with ctx:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      output_logits=True, return_dict_in_generate=True,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        lp0 = torch.log_softmax(out.logits[0][0].float(), -1)
        return self._pack(text, lp0, qtype, letters, inputs["input_ids"].shape[1],
                          (inputs["input_ids"] == self.image_token_id).sum())

    @torch.inference_mode()
    def answer_inplace(self, inputs_full, cache_full, keep_idx, qtype, letters):
        """Keep only the ROI's cached image tokens AT THEIR ORIGINAL mRoPE POSITIONS (background tokens dropped),
        single forward, first-token logits. `keep_idx`: indices into the image's merged-token order."""
        ids = inputs_full["input_ids"][0]
        img_pos = (ids == self.image_token_id).nonzero().squeeze(1)
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[img_pos] = False
        keep_idx = keep_idx.to(ids.device)
        keep[img_pos[keep_idx]] = True
        pos, _ = self.inner.get_rope_index(inputs_full["input_ids"], inputs_full["mm_token_type_ids"],
                                           image_grid_thw=inputs_full["image_grid_thw"],
                                           attention_mask=inputs_full["attention_mask"])
        kept_ids = ids[keep][None]
        kept_pos = pos[:, :, keep].contiguous()
        emb = self.inner.get_input_embeddings()(kept_ids)
        mask = kept_ids == self.image_token_id
        roi = cache_full.main[0][keep_idx].to(emb.device, emb.dtype)
        emb = emb.masked_scatter(mask[..., None], roi)
        extra = {}
        if cache_full.deep is not None:
            extra = dict(visual_pos_masks=mask,
                         deepstack_visual_embeds=[layer[0][keep_idx].to(emb.device, emb.dtype) for layer in cache_full.deep])
        out = self.lm(inputs_embeds=emb, position_ids=kept_pos, attention_mask=torch.ones_like(kept_ids), **extra)
        logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
        lp0 = torch.log_softmax(logits[0], -1)
        text = self.tok.decode([int(lp0.argmax())], skip_special_tokens=True).strip()
        return self._pack(text, lp0, qtype, letters, kept_ids.shape[1], int(mask.sum()))


# ---------------- scoring ----------------
def _lse(xs):
    return torch.logsumexp(torch.tensor(xs, dtype=torch.float64), 0).item() if xs else float("-inf")


def score(res, qtype, gold, letters):
    if qtype == "mcq":
        lps = res["letter_lp"]
        others = [lps[L] for L in letters if L != gold]
        m = re.search(r"\b([A-D])\b", res["text"].upper().replace("(", " ").replace(")", " ").replace(".", " "))
        pred = m.group(1) if m else None
        argmax = max(lps, key=lps.get)
        return {"pred": pred, "correct": bool(pred == gold), "gold_lp": lps[gold],
                "gold_logodds": lps[gold] - _lse(others), "argmax": argmax, "argmax_correct": bool(argmax == gold)}
    m = re.search(r"-?\d+", res["text"])
    pred = m.group(0) if m else None
    if gold.lstrip("-").isdigit():
        correct = pred is not None and int(pred) == int(gold)
    else:
        correct = res["text"].strip().lower() == gold.strip().lower()
    dl = res["digit_lp"]; g0 = gold[0] if gold and gold[0] in dl else None
    lo = (dl[g0] - _lse([v for d, v in dl.items() if d != g0])) if g0 else None
    return {"pred": pred, "correct": bool(correct), "gold_lp": dl.get(g0), "gold_logodds": lo,
            "argmax": max(dl, key=dl.get), "argmax_correct": bool(g0 is not None and max(dl, key=dl.get) == g0)}
