"""Magnification sweep ("how strong a prescription?"): on the primary zoom-rescued cohort of an existing run, re-encode the
target region at linear magnifications m of its first-pass size, from seen pixels (c_m), from native pixels at the same output
size (d_m), and from seen pixels of the same-size random region (c_m_rand). m = 1 reproduces b2 / b3; the native crop's own
magnification is sqrt(native px / seen px) of the region, recorded as `native_mag`.
Output: <out_dir>/mag_<bench>_B<B>.jsonl, same row schema as run_gate2 plus `mag`, `native_mag`, `grid2` (token grid of the second look).
Usage: python -m zoomdecomp.run_mag --model PATH --run_dir runs/gate_x --bench vstar --budgets 256 [--mags 1 1.5 2 3 4 6 8] [--limit N]"""
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.data import LOADERS
from zoomdecomp.qwen_runner import crop_box
from zoomdecomp.runners import make_runner, score_any
from zoomdecomp.run_gate2 import Ctx, TOK_PX

MAGS = [1, 1.5, 2, 3, 4, 6, 8]
MAX_TOKENS2 = 1024


def primary_cohort(path):
    """(uid -> roi_grid_rand) for rows that are zoom-rescued under the primary definition, read from an existing run file."""
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows.setdefault(r["uid"], {})[r["cond"]] = r
    keep = {}
    for uid, d in rows.items():
        b, dn, dr = d.get("base"), d.get("d_native_zoom"), d.get("d_native_zoom_rand")
        if b is None or dn is None or dr is None:
            continue
        ok = lambda r: bool(r["correct"]) and (r.get("gold_logodds") or -1e9) > 0
        if (not b["correct"]) and ok(dn) and (not ok(dr)):
            keep[uid] = dn["roi_grid_rand"]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--run_dir", required=True)
    ap.add_argument("--bench", nargs="+", default=["vstar", "zoombench"]); ap.add_argument("--budgets", nargs="+", type=int, default=[256])
    ap.add_argument("--mags", nargs="+", type=float, default=MAGS); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--max_tokens2", type=int, default=MAX_TOKENS2)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    R = make_runner(args.model, attn=args.attn); model_name = Path(args.model).name; f = R.factor
    print(f"[init] {model_name} factor={f} tiled={R.tiled} mags={args.mags}", flush=True)
    for bench in args.bench:
        samples = {s.uid: s for s in LOADERS[bench]()}
        for B in args.budgets:
            src = run_dir / f"{bench}_B{B}.jsonl"
            if not src.exists():
                print(f"[skip] {src} missing", flush=True); continue
            cohort = primary_cohort(src); uids = [u for u in samples if u in cohort]
            if args.limit: uids = uids[: args.limit]
            out_path = run_dir / f"mag_{bench}_B{B}.jsonl"
            done = set()
            if out_path.exists():
                for line in open(out_path):
                    try:
                        r = json.loads(line); done.add((r["uid"], r["cond"]))
                    except Exception:
                        pass
            fout = open(out_path, "a"); t0 = time.time(); n_rows = 0
            print(f"[{bench} B={B}] cohort={len(uids)} already={len(done)} rows", flush=True)
            for i, uid in enumerate(uids):
                s = samples[uid]
                conds = [(f"{c}_m{m:g}", c, m) for m in args.mags for c in ("c", "d", "c_rand")]
                if all((uid, name) in done for name, _, _ in conds):
                    continue
                try:
                    ctx = Ctx(R, s, s.load_image(), B); g = ctx.g; g_rand = tuple(cohort[uid])
                    sb, nb = ctx.boxes(g); gw, gh = g[2] - g[0], g[3] - g[1]
                    native_mag = float(np.sqrt(((nb[2] - nb[0]) * (nb[3] - nb[1])) / max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))))
                    kw = dict(qtype=s.qtype, letters=s.letters)
                    if s.qtype == "text":
                        kw["gold_first_id"] = R.first_token_ids(s.answer); kw["max_new_tokens"] = 12
                    common = {"uid": uid, "bench": bench, "qtype": s.qtype, "gold": s.answer, "budget": B, "model": model_name, "family": R.family, "factor": f,
                              "img_wh": [ctx.W, ctx.H], "seen_wh": [ctx.rw, ctx.rh], "roi_grid": list(g), "roi_grid_rand": list(g_rand), "roi_tokens": gw * gh, "native_mag": native_mag}
                    for name, c, m in conds:
                        if (uid, name) in done:
                            continue
                        gw2, gh2 = max(2, int(round(gw * m))), max(2, int(round(gh * m)))
                        if gw2 * gh2 > args.max_tokens2:
                            continue
                        tw, th = gw2 * f, gh2 * f
                        if c == "c":
                            src_im = crop_box(ctx.I_seen, sb)
                        elif c == "d":
                            src_im = crop_box(ctx.img, nb)
                        else:
                            sbr, _ = ctx.boxes(g_rand); src_im = crop_box(ctx.I_seen, sbr)
                        z = R.resize(src_im, th, tw); z2, sp = R.same_density(z)
                        res = R.answer(R.build_inputs(s.prompt, [ctx.first, z2], [ctx.spec1, sp]), **kw)
                        sc = score_any(res, s.qtype, s.answer, s.letters, answers=s.meta.get("answers"))
                        row = dict(common, cond=name, arm=c, mag=m, grid2=[gw2, gh2], second_wh=[tw, th], **sc, text=res["text"], n_tokens_in=res["n_tokens_in"], n_img_tokens=res["n_img_tokens"])
                        fout.write(json.dumps(row) + "\n"); n_rows += 1
                    fout.flush(); del ctx
                    if (i + 1) % 25 == 0:
                        print(f"[{bench} B={B}] {i+1}/{len(uids)} rows+={n_rows} {(time.time()-t0)/60:.1f} min", flush=True)
                except Exception as e:
                    print(f"[ERROR] {bench} B={B} {uid}: {type(e).__name__}: {e}", flush=True); traceback.print_exc(); torch.cuda.empty_cache()
            fout.close(); print(f"[{bench} B={B}] done rows+={n_rows} in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
