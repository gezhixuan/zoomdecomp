"""Model runners with one interface for three families:
  Qwen2.5-VL (28 px/token), Qwen3-VL (32 px/token, DeepStack), InternVL3 (448 tiles, 16x16 tokens per tile = 28 px/token).
Each runner exposes: seen_geometry, resize, build_inputs, visual_tokens, override_visual, grid_token_indices,
answer, answer_inplace, zoom_layout, same_density.  Geometry helpers (roi_grid etc.) live in qwen_runner.py."""
import contextlib, math, re, types
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from transformers.image_processing_utils import SizeDict

from zoomdecomp.qwen_runner import VisualCache, score, _lse  # noqa: F401  (score re-exported)

TILE = 448
TOK_PER_TILE = 256
MAX_TILES = 12


def _pil_resize_with(ip, img, rh, rw):
    if img.size == (rw, rh):
        return img
    t = TF.pil_to_tensor(img).unsqueeze(0)
    out = ip.resize(image=t, size=SizeDict(height=rh, width=rw), resample=ip.resample)[0]
    return TF.to_pil_image(out)


class BaseRunner:
    tiled = False

    def __init__(self, model_path, attn="sdpa"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.ip = self.processor.image_processor
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda").eval()
        self.device = self.model.device
        self.cfg = self.model.config
        self.family = self.cfg.model_type
        self.inner = self.model.model
        self.lm = self.inner.language_model
        self.image_token_id = self.cfg.image_token_id
        self.mean_rgb = tuple(int(round(m * 255)) for m in self.ip.image_mean)
        ids = {L: self.tok.encode(L, add_special_tokens=False) for L in "ABCD"}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.letter_ids = {k: v[0] for k, v in ids.items()}
        self.digit_ids = {str(d): self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}
        self._setup()

    def resize(self, img, rh, rw):
        return _pil_resize_with(self.ip, img, rh, rw)

    def first_token_id(self, text):
        return self.tok.encode(text, add_special_tokens=False)[0]

    def first_token_ids(self, text):
        """First-token ids of the answer under casing variants (model may answer 'Blink' for gold 'blink')."""
        variants = {text, text.lower(), text.upper(), text.capitalize(), text.title()}
        return sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in variants if v})

    def _pack(self, text, lp0, qtype, letters, n_in, n_img, gold_first_id=None):
        res = {"text": text, "n_tokens_in": int(n_in), "n_img_tokens": int(n_img)}
        if qtype == "mcq":
            res["letter_lp"] = {L: lp0[self.letter_ids[L]].item() for L in letters}
        elif qtype == "text":
            if gold_first_id is None:
                g = None
            else:
                ids = gold_first_id if isinstance(gold_first_id, (list, tuple)) else [gold_first_id]
                g = torch.logsumexp(lp0[torch.tensor(ids, device=lp0.device)], 0).item()
            res["text_lp"] = {"gold_first": g, "max": lp0.max().item(), "argmax": self.tok.decode([int(lp0.argmax())])}
        else:
            res["digit_lp"] = {d: lp0[i].item() for d, i in self.digit_ids.items()}
        return res

    @torch.inference_mode()
    def answer(self, inputs, qtype, letters, max_new_tokens=8, caches=None, gold_first_id=None):
        ctx = self.override_visual(caches) if caches is not None else contextlib.nullcontext()
        with ctx:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      output_logits=True, return_dict_in_generate=True,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        lp0 = torch.log_softmax(out.logits[0][0].float(), -1)
        return self._pack(text, lp0, qtype, letters, inputs["input_ids"].shape[1],
                          (inputs["input_ids"] == self.image_token_id).sum(), gold_first_id=gold_first_id)

    def _finish_inplace(self, kept_ids, emb, pos, extra, qtype, letters, n_img, gold_first_id=None, max_new_tokens=8):
        """Greedy decoding over the edited sequence, mirroring `answer`: first-step log-probs plus up to max_new_tokens
        tokens. Positions continue from the last kept position on every axis."""
        mask = torch.ones_like(kept_ids)
        out = self.lm(inputs_embeds=emb, position_ids=pos, attention_mask=mask, use_cache=True, **extra)
        lp0 = torch.log_softmax(self.model.lm_head(out.last_hidden_state[:, -1]).float()[0], -1)
        eos = self.model.generation_config.eos_token_id
        eos = {e for e in (list(eos) if isinstance(eos, (list, tuple)) else [eos]) + [self.tok.eos_token_id] if e is not None}
        past, last_pos, gen, nxt = out.past_key_values, pos[..., -1:], [], int(lp0.argmax())
        for step in range(max_new_tokens):
            gen.append(nxt)
            if nxt in eos or step == max_new_tokens - 1:
                break
            ids = torch.tensor([[nxt]], device=kept_ids.device)
            mask = torch.cat([mask, torch.ones_like(ids)], 1)
            out = self.lm(inputs_embeds=self.inner.get_input_embeddings()(ids), position_ids=last_pos + step + 1,
                          attention_mask=mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax())
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        return self._pack(text, lp0, qtype, letters, kept_ids.shape[1], n_img, gold_first_id=gold_first_id)

    def same_density(self, img):
        """Second image for the same-density arms. Qwen: as is (dims already multiples of factor)."""
        return img, {"kind": "image"}


# ======================================================================================
class QwenRunner(BaseRunner):
    def _setup(self):
        vc = self.cfg.vision_config
        self.patch = vc.patch_size
        self.factor = vc.patch_size * vc.spatial_merge_size
        self.min_pixels = int(self.ip.size.shortest_edge or 56 * 56)

    def seen_geometry(self, W, H, B_px):
        rh, rw = smart_resize(H, W, factor=self.factor, min_pixels=self.min_pixels, max_pixels=B_px)
        return rh, rw, {"kind": "image"}

    def first_image(self, img, I_seen):
        return I_seen

    def zoom_layout(self, seen_w, seen_h, K_px):
        rh, rw = smart_resize(seen_h, seen_w, factor=self.factor, min_pixels=K_px, max_pixels=K_px)
        return rh, rw, {"kind": "image"}

    def build_inputs(self, prompt, images, specs):
        imgs = []
        for im, sp in zip(images, specs):
            if sp["kind"] == "tokens":
                imgs.append(Image.new("RGB", (sp["gw"] * self.factor, sp["gh"] * self.factor)))
            else:
                imgs.append(im)
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        f = self.factor
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt", min_pixels=4 * f * f, max_pixels=1 << 40)
        exp = torch.tensor([[1, im.size[1] // self.patch, im.size[0] // self.patch] for im in imgs])
        assert torch.equal(inputs["image_grid_thw"], exp), (inputs["image_grid_thw"], exp)
        inputs = inputs.to(self.device)
        inputs.n_tokens = [int(g[1] * g[2] // 4) for g in exp.tolist()]
        return inputs

    def reference_pixel_values(self, img, B_px):
        return self.ip(images=[img], return_tensors="pt", min_pixels=self.min_pixels, max_pixels=B_px)

    @torch.inference_mode()
    def visual_tokens(self, inputs):
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

    def grid_token_indices(self, g, GW, spec=None):
        gx0, gy0, gx1, gy1 = g
        return torch.tensor([gy * GW + gx for gy in range(gy0, gy1) for gx in range(gx0, gx1)], dtype=torch.long)

    @torch.inference_mode()
    def answer_inplace(self, inputs_full, cache_full, keep_idx, qtype, letters, gold_first_id=None, max_new_tokens=8):
        ids = inputs_full["input_ids"][0]
        img_pos = (ids == self.image_token_id).nonzero().squeeze(1)
        keep = torch.ones_like(ids, dtype=torch.bool); keep[img_pos] = False
        keep_idx = keep_idx.to(ids.device); keep[img_pos[keep_idx]] = True
        pos, _ = self.inner.get_rope_index(inputs_full["input_ids"], inputs_full["mm_token_type_ids"],
                                           image_grid_thw=inputs_full["image_grid_thw"],
                                           attention_mask=inputs_full["attention_mask"])
        kept_ids = ids[keep][None]; kept_pos = pos[:, :, keep].contiguous()
        emb = self.inner.get_input_embeddings()(kept_ids)
        mask = kept_ids == self.image_token_id
        emb = emb.masked_scatter(mask[..., None], cache_full.main[0][keep_idx].to(emb.device, emb.dtype))
        extra = {}
        if cache_full.deep is not None:
            extra = dict(visual_pos_masks=mask,
                         deepstack_visual_embeds=[layer[0][keep_idx].to(emb.device, emb.dtype) for layer in cache_full.deep])
        return self._finish_inplace(kept_ids, emb, kept_pos, extra, qtype, letters, int(mask.sum()), gold_first_id=gold_first_id, max_new_tokens=max_new_tokens)


# ======================================================================================
class InternVLRunner(BaseRunner):
    tiled = True

    def _setup(self):
        self.factor = 28                      # 448 px tile / 16 tokens per side
        self.patch = 14
        self.min_pixels = TILE * TILE
        self.vfl = self.cfg.vision_feature_layer
        self.vfs = getattr(self.cfg, "vision_feature_select_strategy", "default")
        P = self.processor
        self.img_start, self.img_end, self.img_ctx = P.start_image_token, P.end_image_token, P.image_token

    @staticmethod
    def tiles_for_pixels(px):
        return int(max(1, min(MAX_TILES, round(px / (TILE * TILE)))))

    def seen_geometry(self, W, H, B_px):
        n = self.tiles_for_pixels(B_px)
        cols, rows = get_optimal_tiled_canvas((H, W), (TILE, TILE), 1, n)
        return rows * TILE, cols * TILE, {"kind": "stock", "max_patches": n, "cols": cols, "rows": rows, "tiles": cols * rows}

    def first_image(self, img, I_seen):
        return img  # image 1 always goes through the stock tiling of the RAW image (thumbnail is made from the raw image)

    def zoom_layout(self, seen_w, seen_h, K_px):
        n = self.tiles_for_pixels(K_px)
        cols, rows = get_optimal_tiled_canvas((seen_h, seen_w), (TILE, TILE), 1, n)
        return rows * TILE, cols * TILE, {"kind": "tiles", "n": cols * rows}

    def same_density(self, img):
        w, h = img.size
        cols, rows = max(1, math.ceil(w / TILE)), max(1, math.ceil(h / TILE))
        canvas = Image.new("RGB", (cols * TILE, rows * TILE), self.mean_rgb)
        canvas.paste(img, (0, 0))
        return canvas, {"kind": "tiles", "n": cols * rows}

    def _pixels(self, im, sp):
        if sp["kind"] == "stock":
            out = self.ip(images=[im], return_tensors="pt", crop_to_patches=True, min_patches=1, max_patches=sp["max_patches"])
        elif sp["kind"] == "tiles":
            out = self.ip(images=[im], return_tensors="pt", crop_to_patches=True, min_patches=sp["n"], max_patches=sp["n"])
        else:
            raise ValueError(sp)
        return out["pixel_values"]

    def build_inputs(self, prompt, images, specs):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        pvs, n_tokens = [], []
        for im, sp in zip(images, specs):
            if sp["kind"] == "tokens":
                n_tokens.append(int(sp["n"]))
            else:
                pv = self._pixels(im, sp); pvs.append(pv); n_tokens.append(pv.shape[0] * TOK_PER_TILE)
        for _ in images:
            text = text.replace(self.img_ctx, "<ph>", 1)
        for n in n_tokens:
            text = text.replace("<ph>", self.img_start + self.img_ctx * n + self.img_end, 1)
        enc = self.tok(text, return_tensors="pt")
        pixel_values = torch.cat(pvs, 0) if pvs else torch.zeros(1, 3, TILE, TILE)
        inputs = BatchFeature(data={"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                                    "pixel_values": pixel_values.to(torch.bfloat16)}).to(self.device)
        inputs.n_tokens = n_tokens
        return inputs

    @torch.inference_mode()
    def visual_tokens(self, inputs):
        feats = self.inner.get_image_features(inputs["pixel_values"], vision_feature_layer=self.vfl,
                                              vision_feature_select_strategy=self.vfs, return_dict=True).pooler_output
        flat = feats.reshape(-1, feats.shape[-1]).detach()
        return VisualCache(list(torch.split(flat, inputs.n_tokens)), None)

    @contextlib.contextmanager
    def override_visual(self, caches):
        inner, dev, dt = self.inner, self.device, torch.bfloat16
        feats = torch.cat([c.main[0].to(dev, dt) for c in caches], 0)
        def fake(pixel_values, **kw):
            return types.SimpleNamespace(pooler_output=feats)
        inner.get_image_features = fake
        try:
            yield
        finally:
            del inner.get_image_features

    def grid_token_indices(self, g, GW, spec):
        cols = spec["cols"]; gx0, gy0, gx1, gy1 = g
        idx = []
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                tile = (gy // 16) * cols + (gx // 16)
                idx.append(tile * TOK_PER_TILE + (gy % 16) * 16 + (gx % 16))
        return torch.tensor(idx, dtype=torch.long)

    @torch.inference_mode()
    def answer_inplace(self, inputs_full, cache_full, keep_idx, qtype, letters, gold_first_id=None, max_new_tokens=8):
        ids = inputs_full["input_ids"][0]
        img_pos = (ids == self.image_token_id).nonzero().squeeze(1)
        keep = torch.ones_like(ids, dtype=torch.bool); keep[img_pos] = False
        keep_idx = keep_idx.to(ids.device); keep[img_pos[keep_idx]] = True
        pos = torch.arange(ids.shape[0], device=ids.device)[keep][None]
        kept_ids = ids[keep][None]
        emb = self.inner.get_input_embeddings()(kept_ids)
        mask = kept_ids == self.image_token_id
        emb = emb.masked_scatter(mask[..., None], cache_full.main[0][keep_idx].to(emb.device, emb.dtype))
        return self._finish_inplace(kept_ids, emb, pos, {}, qtype, letters, int(mask.sum()), gold_first_id=gold_first_id, max_new_tokens=max_new_tokens)


_PUNCT = re.compile(r"[^\w\s]")
_ARTICLES = {"a", "an", "the"}


def vqa_normalize(t):
    t = _PUNCT.sub("", t.lower().strip())
    return " ".join(w for w in t.split() if w not in _ARTICLES)


def score_any(res, qtype, gold, letters, answers=None):
    """mcq / blank -> qwen_runner.score; text -> VQA soft accuracy (min(#match/3, 1)), correct := soft >= 0.5."""
    if qtype != "text":
        return score(res, qtype, gold, letters)
    pred = vqa_normalize(res["text"])
    pool = [vqa_normalize(a) for a in (answers or [gold])]
    soft = min(sum(1 for a in pool if a == pred) / 3.0, 1.0)
    lp = res["text_lp"]["gold_first"]
    lo = None if lp is None else (lp - math.log1p(-math.exp(min(lp, -1e-6))))
    return {"pred": pred, "correct": bool(soft >= 0.5), "soft_acc": soft, "gold_lp": lp, "gold_logodds": lo,
            "argmax": res["text_lp"]["argmax"], "argmax_correct": bool(vqa_normalize(res["text_lp"]["argmax"]) == vqa_normalize(gold)[: len(vqa_normalize(res["text_lp"]["argmax"]))])}


def make_runner(model_path, attn="sdpa"):
    from transformers import AutoConfig
    mt = AutoConfig.from_pretrained(model_path).model_type
    if mt in ("qwen2_5_vl", "qwen3_vl", "qwen2_vl"):
        return QwenRunner(model_path, attn=attn)
    if mt == "internvl":
        return InternVLRunner(model_path, attn=attn)
    raise ValueError(f"unsupported model_type {mt}")
