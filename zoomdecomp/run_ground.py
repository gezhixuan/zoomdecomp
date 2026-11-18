"""Self-grounding experiment: does the model know where to look, and does the decomposition survive its own region?
Stage A (all samples of a benchmark/budget): ask the model, on the first-pass image, for the region needed to answer the question;
parse one box; record IoU with the ground-truth region and whether the box covers the GT centre.
Stage B (primary zoom-rescued cohort of the existing run): rerun the arm ladder with the model's own box in place of the GT box
(same padding, snapping, crop budget and random-twin construction as run_gate2), conditions prefixed "g_".
Output: <run_dir>/ground_<bench>_B<B>.jsonl.  Usage: python -m zoomdecomp.run_ground --model PATH --run_dir DIR --bench vstar --budgets 256 1024"""
import argparse, copy, json, re, sys, time, traceback, zlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.data import LOADERS
from zoomdecomp.qwen_runner import random_grid_box
from zoomdecomp.runners import make_runner, score_any
from zoomdecomp.run_gate2 import Ctx, run_condition, crop_pixels, TOK_PX, STAGE2, SHAM
from zoomdecomp.run_mag import primary_cohort

GROUND_PROMPT = ("Which region of the image must be looked at closely to answer the question below? "
                 "Output only one bounding box as [x1, y1, x2, y2].\nQuestion: {q}")
ARMS = ["c_seen_zoom", "b2_seen_same", "b3_native_same", "b1_mask_bg", "a1_cached_roi", "a2_cached_only", "a3_cached_inplace"]
SHAMS = ["d_native_zoom", "c_seen_zoom", "b2_seen_same", "b1_mask_bg", "a1_cached_roi", "a2_cached_only", "a3_cached_inplace"]


def parse_box(text, family, rw, rh):
    text = text.split("||")[-1]
    clean = re.sub(r"\b[xy][12]?\b\s*[:=]?", " ", text)          # drop placeholder names like "x1, y1, x2, y2" / "x1: 450"
    m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?", clean)
    if not m:
        return None
    x1, y1, x2, y2 = [float(v) for v in m.groups()]
    if max(x1, y1, x2, y2) <= 1.0:                                  # fractions of the image
        x1, x2 = x1 * rw, x2 * rw; y1, y2 = y1 * rh, y2 * rh
    elif family != "qwen2_5_vl":  # Qwen3-VL and InternVL3 emit 0-1000 normalised coordinates; Qwen2.5-VL emits pixels of the input image
        x1, x2 = x1 / 1000 * rw, x2 / 1000 * rw; y1, y2 = y1 / 1000 * rh, y2 / 1000 * rh
    x1, x2 = sorted((max(0.0, min(rw, x1)), max(0.0, min(rw, x2)))); y1, y2 = sorted((max(0.0, min(rh, y1)), max(0.0, min(rh, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


PHRASE_PROMPT = ("Read the question below. Do NOT answer it. Instead, name the thing in the image that one would have to look at to answer it, "
                 "as a short noun phrase (for example 'the glove', 'the number on the blue vehicle', 'the sail of the boat'). Reply with the phrase only.\nQuestion: {q}")


@torch.inference_mode()
def _gen_text(R, prompt, max_new_tokens=16):
    """text-only generation (no image), so the model cannot answer the question instead of naming its target"""
    text = R.processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}], tokenize=False, add_generation_prompt=True)
    enc = R.tok(text, return_tensors="pt").to(R.device)
    out = R.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return R.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
INTERNVL_GROUND = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{phrase}</ref>"


@torch.inference_mode()
def _gen(R, prompt, ctx, max_new_tokens=48):
    inputs = R.build_inputs(prompt, [ctx.first], [ctx.spec1])
    out = R.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return R.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def ground(R, ctx, s):
    """Qwen: one question-conditioned grounding prompt. InternVL3: its native referring-grounding format, with the referred phrase
    first produced by the model from the question (its free-form boxes are unreliable)."""
    q = s.prompt.split("\n")[0]
    if R.family == "internvl":
        phrase = _gen_text(R, PHRASE_PROMPT.format(q=q)).strip().strip(".").strip("'\"").split("\n")[0][:80]
        return phrase + " || " + _gen(R, INTERNVL_GROUND.format(phrase=phrase), ctx, 48)
    return _gen(R, GROUND_PROMPT.format(q=q), ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--run_dir", required=True)
    ap.add_argument("--bench", nargs="+", default=["vstar", "zoombench"]); ap.add_argument("--budgets", nargs="+", type=int, default=[256, 1024])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir); R = make_runner(args.model, attn=args.attn); model_name = Path(args.model).name
    print(f"[init] {model_name} family={R.family}", flush=True)
    for bench in args.bench:
        samples = LOADERS[bench]()
        if args.limit: samples = samples[: args.limit]
        for B in args.budgets:
            src = run_dir / f"{bench}_B{B}.jsonl"
            cohort = primary_cohort(src) if src.exists() else {}
            out_path = run_dir / f"ground_{bench}_B{B}.jsonl"; done = set()
            if out_path.exists():
                for line in open(out_path):
                    try: r = json.loads(line); done.add((r["uid"], r["cond"]))
                    except Exception: pass
            fout = open(out_path, "a"); t0 = time.time(); hits = []; n_rows = 0
            for i, s in enumerate(samples):
                in_cohort = s.uid in cohort
                need = [("ground", None)] + ([(f"g_{c}", c) for c in ["d_native_zoom"] + ARMS] + [(f"g_{c}_rand", c) for c in SHAMS] if in_cohort else [])
                if all((s.uid, n) in done for n, _ in need): continue
                try:
                    ctx = Ctx(R, s, s.load_image(), B)
                    text = ground(R, ctx, s); box_seen = parse_box(text, R.family, ctx.rw, ctx.rh)
                    sx, sy = ctx.W / ctx.rw, ctx.H / ctx.rh
                    pred = (box_seen[0] * sx, box_seen[1] * sy, box_seen[2] * sx, box_seen[3] * sy) if box_seen else None
                    gt = tuple(float(v) for v in s.bbox); cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
                    io = iou(pred, gt) if pred else 0.0; cover = bool(pred and pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3])
                    frac = ((pred[2] - pred[0]) * (pred[3] - pred[1]) / (ctx.W * ctx.H)) if pred else None
                    hits.append((io, cover))
                    common = {"uid": s.uid, "bench": bench, "qtype": s.qtype, "gold": s.answer, "budget": B, "model": model_name, "family": R.family,
                              "img_wh": [ctx.W, ctx.H], "seen_wh": [ctx.rw, ctx.rh], "bbox_gt": list(gt), "bbox_pred": list(pred) if pred else None,
                              "iou": io, "covers_gt_center": cover, "pred_area_frac": frac, "in_cohort": in_cohort}
                    if (s.uid, "ground") not in done:
                        fout.write(json.dumps(dict(common, cond="ground", text=text)) + "\n"); n_rows += 1
                    if in_cohort and pred is not None:
                        s2 = copy.copy(s); s2.bbox = pred
                        ctx2 = Ctx(R, s2, ctx.img, B); g = ctx2.g
                        rng = np.random.default_rng(zlib.crc32(f"{s.uid}|{B}|ground".encode())); g_rand, _ = random_grid_box(g, ctx2.GW, ctx2.GH, rng)
                        kw_common = dict(common, roi_grid=list(g), roi_grid_rand=list(g_rand), roi_tokens=(g[2] - g[0]) * (g[3] - g[1]), K=crop_pixels(B, ctx2.boxes(g)[1]) // TOK_PX)
                        for name, cond in need[1:]:
                            if (s.uid, name) in done: continue
                            res, size2 = run_condition(ctx2, cond, g_rand if name.endswith("_rand") else g)
                            sc = score_any(res, s.qtype, s.answer, s.letters, answers=s.meta.get("answers"))
                            fout.write(json.dumps(dict(kw_common, cond=name, **sc, text=res["text"], n_img_tokens=res["n_img_tokens"], second_wh=list(size2) if size2 else None)) + "\n"); n_rows += 1
                        del ctx2
                    fout.flush(); del ctx
                    if args.smoke or (i + 1) % 50 == 0:
                        ios = np.array([h[0] for h in hits]); cov = np.array([h[1] for h in hits])
                        print(f"[{bench} B={B}] {i+1}/{len(samples)} rows+={n_rows} IoU>=0.5: {(ios>=0.5).mean():.2f} covers-centre: {cov.mean():.2f} medIoU {np.median(ios):.2f} {(time.time()-t0)/60:.1f} min" + (f" | '{text[:60]}' -> {pred and [round(v) for v in pred]} gt {[round(v) for v in gt]}" if args.smoke else ""), flush=True)
                except Exception as e:
                    print(f"[ERROR] {bench} B={B} {s.uid}: {type(e).__name__}: {e}", flush=True); traceback.print_exc(); torch.cuda.empty_cache()
            fout.close(); print(f"[{bench} B={B}] done rows+={n_rows} in {(time.time()-t0)/60:.1f} min", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
