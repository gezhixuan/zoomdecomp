"""Gate experiment, multi-family version (Qwen2.5-VL / Qwen3-VL / InternVL3) built on zoomdecomp.runners.
Same conditions and output schema as run_gate.py (see its docstring); budgets are pixel budgets B x 784."""
import argparse, json, sys, time, traceback, zlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.data import LOADERS, MCQ_SUFFIX, BLANK_SUFFIX
from zoomdecomp.qwen_runner import crop_box, mask_outside, roi_grid, grid_to_seen_box, seen_to_native_box, random_grid_box, score
from zoomdecomp.runners import make_runner, score_any

STAGE2 = ["c_seen_zoom", "b2_seen_same", "b3_native_same", "b1_mask_bg", "a1_cached_roi", "a2_cached_only",
          "a3_cached_inplace", "d_native_zoom_pad1", "base_para"]
SHAM = ["d_native_zoom", "c_seen_zoom", "b2_seen_same", "b3_native_same", "b1_mask_bg", "a1_cached_roi",
        "a2_cached_only", "a3_cached_inplace"]
PARA = {MCQ_SUFFIX: "Reply with only the letter of the correct option.", BLANK_SUFFIX: "Reply with just the number."}
TOK_PX = 28 * 28


def crop_pixels(B, native_box, kmin_tokens=256):
    area = (native_box[2] - native_box[0]) * (native_box[3] - native_box[1])
    return int(max(min(kmin_tokens * TOK_PX, B * TOK_PX), min(B * TOK_PX, area)))


def paraphrase(prompt):
    for a, b in PARA.items():
        if prompt.endswith(a):
            return prompt[: -len(a)] + b
    return prompt + " Answer briefly."


class Ctx:
    def __init__(self, R, s, img, B):
        self.R, self.s, self.img, self.B = R, s, img, B
        self.W, self.H = img.size
        self.rh, self.rw, self.spec1 = R.seen_geometry(self.W, self.H, B * TOK_PX)
        self.I_seen = R.resize(img, self.rh, self.rw)
        self.first = R.first_image(img, self.I_seen)
        self.inputs_full = R.build_inputs(s.prompt, [self.first], [self.spec1])
        self.cache = R.visual_tokens(self.inputs_full)
        self.GW, self.GH = self.rw // R.factor, self.rh // R.factor
        self.g, _ = roi_grid(s.bbox, self.W, self.H, self.rh, self.rw, factor=R.factor)
        self.g_pad1, _ = roi_grid(s.bbox, self.W, self.H, self.rh, self.rw, factor=R.factor, pad_frac=1.0)

    def boxes(self, g):
        sb = grid_to_seen_box(g, self.R.factor)
        return sb, seen_to_native_box(sb, self.W, self.H, self.rh, self.rw)

    def zoom(self, g, source):
        sb, nb = self.boxes(g)
        K_px = crop_pixels(self.B, nb)
        th, tw, spec2 = self.R.zoom_layout(sb[2] - sb[0], sb[3] - sb[1], K_px)
        src = crop_box(self.img, nb) if source == "native" else crop_box(self.I_seen, sb)
        return self.R.resize(src, th, tw), spec2


def run_condition(ctx, cond, g):
    R, s = ctx.R, ctx.s
    kw = dict(qtype=s.qtype, letters=s.letters)
    if s.qtype == "text":
        kw["gold_first_id"] = R.first_token_ids(s.answer); kw["max_new_tokens"] = 12
    sb, nb = ctx.boxes(g)
    gw, gh = g[2] - g[0], g[3] - g[1]
    f = R.factor
    if cond == "base":
        return R.answer(ctx.inputs_full, caches=[ctx.cache], **kw), None
    if cond == "base_para":
        return R.answer(R.build_inputs(paraphrase(s.prompt), [ctx.first], [ctx.spec1]), caches=[ctx.cache], **kw), None
    if cond in ("d_native_zoom", "d_native_zoom_pad1", "c_seen_zoom"):
        z, sp = ctx.zoom(g, "seen" if cond == "c_seen_zoom" else "native")
        return R.answer(R.build_inputs(s.prompt, [ctx.first, z], [ctx.spec1, sp]), **kw), z.size
    if cond in ("b2_seen_same", "b3_native_same"):
        z = crop_box(ctx.I_seen, sb) if cond == "b2_seen_same" else R.resize(crop_box(ctx.img, nb), gh * f, gw * f)
        z2, sp = R.same_density(z)
        return R.answer(R.build_inputs(s.prompt, [ctx.first, z2], [ctx.spec1, sp]), **kw), z.size
    if cond == "b1_mask_bg":
        m = mask_outside(ctx.I_seen, sb, R.mean_rgb)
        sp = {"kind": "tiles", "n": ctx.spec1["tiles"]} if R.tiled else {"kind": "image"}
        return R.answer(R.build_inputs(s.prompt, [m], [sp]), **kw), m.size
    idx = R.grid_token_indices(g, ctx.GW, ctx.spec1)
    tok_spec = {"kind": "tokens", "n": int(len(idx)), "gw": gw, "gh": gh}
    if cond == "a1_cached_roi":
        return R.answer(R.build_inputs(s.prompt, [ctx.first, None], [ctx.spec1, tok_spec]),
                        caches=[ctx.cache, ctx.cache.sub(0, idx)], **kw), (gw * f, gh * f)
    if cond == "a2_cached_only":
        return R.answer(R.build_inputs(s.prompt, [None], [tok_spec]), caches=[ctx.cache.sub(0, idx)], **kw), (gw * f, gh * f)
    if cond == "a3_cached_inplace":
        return R.answer_inplace(ctx.inputs_full, ctx.cache, idx, **kw), (gw * f, gh * f)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", nargs="+", default=["vstar", "zoombench"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qtype", default=None)
    ap.add_argument("--sample", type=int, default=0, help="seeded random subsample of this many samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all_arms", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = make_runner(args.model, attn=args.attn)
    model_name = Path(args.model).name
    print(f"[init] {model_name} family={R.family} factor={R.factor} tiled={R.tiled} on {R.device}", flush=True)

    for bench in args.bench:
        samples = LOADERS[bench]()
        if args.qtype:
            samples = [x for x in samples if x.qtype == args.qtype]
        if args.sample and args.sample < len(samples):
            idx = np.random.default_rng(args.seed).permutation(len(samples))[: args.sample]
            samples = [samples[i] for i in sorted(idx)]
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
                    needed = [c for c in ["base", "d_native_zoom"] + STAGE2 + [c + "_rand" for c in SHAM] if (s.uid, c) not in done]
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
                              "tiles": ctx.spec1.get("tiles"), "n_img_tokens_first": ctx.inputs_full.n_tokens[0],
                              "img_wh": [ctx.W, ctx.H], "seen_wh": [ctx.rw, ctx.rh], "grid_wh": [GW, GH],
                              "roi_grid": list(g), "roi_grid_rand": list(g_rand), "roi_grid_pad1": list(ctx.g_pad1),
                              "rand_iou": iou, "roi_tokens": (g[2] - g[0]) * (g[3] - g[1]),
                              "bbox_native": list(s.bbox), "bbox_seen_tokens": cov,
                              "meta": s.meta if bench == "vstar" else {}}

                    if args.smoke and i == 0:
                        if not R.tiled:
                            ref = R.reference_pixel_values(ctx.img, B * TOK_PX)
                            same = torch.equal(ref["image_grid_thw"], ctx.inputs_full["image_grid_thw"].cpu())
                            diff = (ref["pixel_values"].float() - ctx.inputs_full["pixel_values"].float().cpu()).abs().max().item() if same else float("nan")
                            print(f"[smoke] I_seen check B={B}: grid_equal={same} max|dpix|={diff:.3e}", flush=True)
                        else:
                            n_t = ctx.spec1["tiles"]
                            ref_tiles = R.ip(images=[ctx.I_seen], return_tensors="pt", crop_to_patches=True, min_patches=n_t, max_patches=n_t)["pixel_values"]
                            diff = (ref_tiles[:n_t].float() - ctx.inputs_full["pixel_values"][:n_t].float().cpu()).abs().max().item()
                            print(f"[smoke] canvas check B={B}: tiles={n_t} (+thumb={ctx.inputs_full.n_tokens[0]//256 - n_t}) seen={ctx.I_seen.size} max|dtile|={diff:.3e}", flush=True)
                        r_plain = R.answer(ctx.inputs_full, qtype=s.qtype, letters=s.letters)
                        r_over = R.answer(ctx.inputs_full, qtype=s.qtype, letters=s.letters, caches=[ctx.cache])
                        all_idx = R.grid_token_indices((0, 0, GW, GH), GW, ctx.spec1)
                        r_inpl = R.answer_inplace(ctx.inputs_full, ctx.cache, all_idx, qtype=s.qtype, letters=s.letters)
                        key = {"mcq": "letter_lp", "blank": "digit_lp", "text": "text_lp"}[s.qtype]
                        num = [k for k in r_plain[key] if isinstance(r_plain[key][k], (int, float)) and r_plain[key][k] is not None]
                        d_over = max(abs(r_plain[key][k] - r_over[key][k]) for k in num)
                        d_inpl = max(abs(r_plain[key][k] - r_inpl[key][k]) for k in num)
                        print(f"[smoke] override check: '{r_plain['text']}' vs '{r_over['text']}' max|dlp|={d_over:.2e} | "
                              f"inplace(all grid tokens; n={len(all_idx)} of {r_plain['n_img_tokens']} img tokens) '{r_inpl['text']}' max|dlp|={d_inpl:.2e}", flush=True)

                    results = {}
                    def run(cond, gg, sham=False):
                        name = cond + ("_rand" if sham else "")
                        if (s.uid, name) in done:
                            return None
                        res, size2 = run_condition(ctx, cond, gg)
                        sc = score_any(res, s.qtype, s.answer, s.letters, answers=s.meta.get("answers"))
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
                        print(f"[{bench} B={B}] {i+1}/{len(samples)} rows+={n_rows} base_acc={stats['base_correct']}/{stats['n']} "
                              f"d_acc={stats['d_correct']}/{stats['n']} rescued={stats['rescued']} {(time.time()-t0)/60:.1f} min", flush=True)
                except Exception as e:
                    print(f"[ERROR] {bench} B={B} {s.uid}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
                    torch.cuda.empty_cache()
            fout.close()
            print(f"[{bench} B={B}] done: {stats} rows+={n_rows} in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
