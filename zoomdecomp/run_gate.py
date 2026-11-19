"""Main experiment: first-pass pixel-budget sweep, oracle native zoom, and the second-look conditions on the samples the first pass gets wrong.

Budgets are given as Qwen2.5-VL-equivalent token counts B; the pixel budget B*784 is used for every model family
so that the first-pass downsampling factor is matched across models.

Conditions (image 1 = full image at budget B unless noted):
  base               [full_B]
  base_para          [full_B] with a paraphrased instruction (stability of the failure)
  d_native_zoom      [full_B][encode(crop(I_src, roi) -> K)]            oracle; new pixels + new compute
  d_native_zoom_pad1 same with a 2x larger context pad (stability of the rescue to crop geometry)
  c_seen_zoom        [full_B][encode(crop(I_seen, roi) -> K)]           new compute + upsampling, NO new pixels
  b2_seen_same       [full_B][encode(crop(I_seen, roi))]                re-encode same pixels at same density
  b3_native_same     [full_B][encode(crop(I_src, roi) -> seen dims)]    native pixels at the SAME density as b2
  b1_mask_bg         [encode(mask_outside(I_seen, roi))]                HiDe-style context suppression
  a1_cached_roi      [full_B][cached ROI tokens]                        SIEVE-style reinsertion, no encoder call
  a2_cached_only     [cached ROI tokens]  (positions re-rolled)         cached tokens, background dropped
  a3_cached_inplace  cached ROI tokens kept at ORIGINAL positions, background tokens dropped (no encoder call)
  *_rand             same operation on a same-size random region (sham)
"""
import argparse, json, sys, time, traceback, zlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.data import LOADERS, MCQ_SUFFIX, BLANK_SUFFIX
from zoomdecomp.qwen_runner import (QwenRunner, resize_like_processor, seen_dims, fit_dims, crop_box, mask_outside,
                                    roi_grid, grid_to_seen_box, seen_to_native_box, random_grid_box,
                                    grid_token_indices, score)
from zoomdecomp import paths

STAGE2 = ["c_seen_zoom", "b2_seen_same", "b3_native_same", "b1_mask_bg", "a1_cached_roi", "a2_cached_only",
          "a3_cached_inplace", "d_native_zoom_pad1", "base_para"]
SHAM = ["d_native_zoom", "c_seen_zoom", "b2_seen_same", "b3_native_same", "b1_mask_bg", "a1_cached_roi",
        "a2_cached_only", "a3_cached_inplace"]
PARA = {MCQ_SUFFIX: "Reply with only the letter of the correct option.", BLANK_SUFFIX: "Reply with just the number."}
TOK_PX = 28 * 28  # pixel unit behind the budget labels


def crop_pixels(B, native_box, kmin_tokens=256):
    area = (native_box[2] - native_box[0]) * (native_box[3] - native_box[1])
    return int(max(min(kmin_tokens * TOK_PX, B * TOK_PX), min(B * TOK_PX, area)))


def paraphrase(prompt):
    for a, b in PARA.items():
        if prompt.endswith(a):
            return prompt[: -len(a)] + b
    return prompt + " Answer briefly."


class Ctx:
    """Per-sample geometry for one budget."""
    def __init__(self, R, s, img, B):
        self.R, self.s, self.img, self.B = R, s, img, B
        self.W, self.H = img.size
        self.rh, self.rw = seen_dims(self.W, self.H, B * TOK_PX, factor=R.factor, min_pixels=R.min_pixels)
        self.I_seen = resize_like_processor(R.ip, img, self.rh, self.rw)
        self.inputs_full = R.build_inputs([self.I_seen], s.prompt)
        self.cache = R.visual_tokens(self.inputs_full)
        self.g, (self.GW, self.GH) = roi_grid(s.bbox, self.W, self.H, self.rh, self.rw, factor=R.factor)
        self.g_pad1, _ = roi_grid(s.bbox, self.W, self.H, self.rh, self.rw, factor=R.factor, pad_frac=1.0)

    def boxes(self, g):
        sb = grid_to_seen_box(g, self.R.factor)
        nb = seen_to_native_box(sb, self.W, self.H, self.rh, self.rw)
        return sb, nb

    def zoom(self, g, source):
        """Second image for the zoom arms: crop (seen or native source) -> K pixels; same dims for both sources."""
        sb, nb = self.boxes(g)
        K_px = crop_pixels(self.B, nb)
        th, tw = fit_dims(sb[2] - sb[0], sb[3] - sb[1], K_px, factor=self.R.factor)
        src = crop_box(self.img, nb) if source == "native" else crop_box(self.I_seen, sb)
        return resize_like_processor(self.R.ip, src, th, tw), K_px


def run_condition(ctx, cond, g):
    R, s = ctx.R, ctx.s
    kw = dict(qtype=s.qtype, letters=s.letters)
    sb, nb = ctx.boxes(g)
    gw, gh = g[2] - g[0], g[3] - g[1]
    f = R.factor
    if cond == "base":
        return R.answer(ctx.inputs_full, caches=[ctx.cache], **kw), None
    if cond == "base_para":
        return R.answer(R.build_inputs([ctx.I_seen], paraphrase(s.prompt)), caches=[ctx.cache], **kw), None
    if cond in ("d_native_zoom", "d_native_zoom_pad1"):
        z, _ = ctx.zoom(g, "native")
        return R.answer(R.build_inputs([ctx.I_seen, z], s.prompt), **kw), z.size
    if cond == "c_seen_zoom":
        z, _ = ctx.zoom(g, "seen")
        return R.answer(R.build_inputs([ctx.I_seen, z], s.prompt), **kw), z.size
    if cond == "b2_seen_same":
        z = crop_box(ctx.I_seen, sb)
        return R.answer(R.build_inputs([ctx.I_seen, z], s.prompt), **kw), z.size
    if cond == "b3_native_same":
        z = resize_like_processor(R.ip, crop_box(ctx.img, nb), gh * f, gw * f)
        return R.answer(R.build_inputs([ctx.I_seen, z], s.prompt), **kw), z.size
    if cond == "b1_mask_bg":
        m = mask_outside(ctx.I_seen, sb, R.mean_rgb)
        return R.answer(R.build_inputs([m], s.prompt), **kw), m.size
    idx = grid_token_indices(g, ctx.GW)
    if cond == "a1_cached_roi":
        dummy = Image.new("RGB", (gw * f, gh * f))
        return R.answer(R.build_inputs([ctx.I_seen, dummy], s.prompt), caches=[ctx.cache, ctx.cache.sub(0, idx)], **kw), dummy.size
    if cond == "a2_cached_only":
        dummy = Image.new("RGB", (gw * f, gh * f))
        return R.answer(R.build_inputs([dummy], s.prompt), caches=[ctx.cache.sub(0, idx)], **kw), dummy.size
    if cond == "a3_cached_inplace":
        return R.answer_inplace(ctx.inputs_full, ctx.cache, idx, **kw), (gw * f, gh * f)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--bench", nargs="+", default=["vstar", "zoombench"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    ap.add_argument("--out_dir", default=str(paths.RUNS / "gate_qwen25vl7b"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qtype", default=None)
    ap.add_argument("--all_arms", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = QwenRunner(args.model, attn=args.attn)
    model_name = Path(args.model).name
    print(f"[init] {model_name} family={R.family} factor={R.factor} min_pixels={R.min_pixels} on {R.device}", flush=True)

    for bench in args.bench:
        samples = LOADERS[bench]()
        if args.qtype:
            samples = [x for x in samples if x.qtype == args.qtype]
        if args.limit:
            samples = samples[: args.limit]
        print(f"[{bench}] {len(samples)} samples", flush=True)
        for B in args.budgets:
            out_path = out_dir / f"{bench}_B{B}.jsonl"
            done = {}
            if out_path.exists():
                for line in open(out_path):
                    try:
                        r = json.loads(line); done[(r["uid"], r["cond"])] = bool(r["correct"])
                    except Exception:
                        pass
            fout = open(out_path, "a")
            t0 = time.time(); n_rows = 0; stats = {"base_correct": 0, "d_correct": 0, "rescued": 0, "n": 0}
            for i, s in enumerate(samples):
                try:
                    needed = [c for c in ["base", "d_native_zoom"] + STAGE2 + [c + "_rand" for c in SHAM]
                              if (s.uid, c) not in done]
                    if not needed:
                        continue
                    ctx = Ctx(R, s, s.load_image(), B)
                    g, GW, GH = ctx.g, ctx.GW, ctx.GH
                    sb, nb = ctx.boxes(g)
                    K_px = crop_pixels(B, nb)
                    bx0, by0, bx1, by1 = s.bbox
                    cov = ((bx1 - bx0) * ctx.rw / ctx.W) * ((by1 - by0) * ctx.rh / ctx.H) / (R.factor ** 2)
                    rng = np.random.default_rng(zlib.crc32(f"{s.uid}|{B}".encode()))
                    g_rand, iou = random_grid_box(g, GW, GH, rng)
                    common = {"uid": s.uid, "bench": bench, "qtype": s.qtype, "gold": s.answer, "budget": B,
                              "K": K_px // TOK_PX, "model": model_name, "family": R.family, "factor": R.factor,
                              "img_wh": [ctx.W, ctx.H], "seen_wh": [ctx.rw, ctx.rh], "grid_wh": [GW, GH],
                              "roi_grid": list(g), "roi_grid_rand": list(g_rand), "roi_grid_pad1": list(ctx.g_pad1),
                              "rand_iou": iou, "roi_tokens": (g[2] - g[0]) * (g[3] - g[1]),
                              "bbox_native": list(s.bbox), "bbox_seen_tokens": cov,
                              "meta": s.meta if bench == "vstar" else {}}

                    if args.smoke and i == 0:
                        ref = R.reference_pixel_values(ctx.img, B * TOK_PX)
                        same = torch.equal(ref["image_grid_thw"], ctx.inputs_full["image_grid_thw"].cpu())
                        diff = (ref["pixel_values"].float() - ctx.inputs_full["pixel_values"].float().cpu()).abs().max().item() if same else float("nan")
                        print(f"[smoke] I_seen check B={B}: grid_equal={same} max|dpix|={diff:.3e}", flush=True)
                        r_plain = R.answer(ctx.inputs_full, qtype=s.qtype, letters=s.letters)
                        r_over = R.answer(ctx.inputs_full, qtype=s.qtype, letters=s.letters, caches=[ctx.cache])
                        all_idx = torch.arange(GW * GH)
                        r_inpl = R.answer_inplace(ctx.inputs_full, ctx.cache, all_idx, qtype=s.qtype, letters=s.letters)
                        key = "letter_lp" if s.qtype == "mcq" else "digit_lp"
                        d_over = max(abs(r_plain[key][k] - r_over[key][k]) for k in r_plain[key])
                        d_inpl = max(abs(r_plain[key][k] - r_inpl[key][k]) for k in r_plain[key])
                        print(f"[smoke] override check: '{r_plain['text']}' vs '{r_over['text']}' max|dlp|={d_over:.2e} | "
                              f"inplace(all tokens) check: '{r_inpl['text']}' max|dlp|={d_inpl:.2e}", flush=True)

                    results = {}
                    def run(cond, gg, sham=False):
                        name = cond + ("_rand" if sham else "")
                        if (s.uid, name) in done:
                            return None
                        res, size2 = run_condition(ctx, cond, gg)
                        sc = score(res, s.qtype, s.answer, s.letters)
                        row = dict(common, cond=name, **sc, text=res["text"], n_tokens_in=res["n_tokens_in"],
                                   n_img_tokens=res["n_img_tokens"], second_wh=list(size2) if size2 else None,
                                   lp=res.get("letter_lp") or res.get("digit_lp"))
                        fout.write(json.dumps(row) + "\n"); results[name] = sc
                        return sc

                    sc_base = run("base", g); sc_d = run("d_native_zoom", g)
                    stats["n"] += 1
                    if sc_base: stats["base_correct"] += sc_base["correct"]
                    if sc_d: stats["d_correct"] += sc_d["correct"]
                    if sc_base and sc_d and (not sc_base["correct"]) and sc_d["correct"]: stats["rescued"] += 1
                    base_correct = sc_base["correct"] if sc_base is not None else done.get((s.uid, "base"), False)
                    if args.all_arms or not base_correct:
                        for cond in STAGE2:
                            run(cond, ctx.g_pad1 if cond == "d_native_zoom_pad1" else g)
                        for cond in SHAM:
                            run(cond, g_rand, sham=True)
                    n_rows += len(results); fout.flush()
                    del ctx
                    if (i + 1) % 25 == 0 or args.smoke:
                        el = time.time() - t0
                        print(f"[{bench} B={B}] {i+1}/{len(samples)} rows+={n_rows} base_acc={stats['base_correct']}/{stats['n']} "
                              f"d_acc={stats['d_correct']}/{stats['n']} rescued={stats['rescued']} {el/60:.1f} min", flush=True)
                except Exception as e:
                    print(f"[ERROR] {bench} B={B} {s.uid}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
                    torch.cuda.empty_cache()
            fout.close()
            print(f"[{bench} B={B}] done: {stats} rows+={n_rows} in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
