"""Second, independent placement of the random-region sham: how stable is a failure's regime label to where the sham falls?

The gate runs place one random region per (sample, budget), seeded by crc32("uid|budget"). This script draws another
placement, seeded by crc32("uid|budget|sham<seed>"), and reruns only the eight sham conditions on it, for every sample of
an existing run whose first pass is wrong and whose oracle zoom is right (greedy), i.e. a superset of the rescued cohort.
Target conditions are not rerun: they do not depend on the placement. Rows keep the schema of the stored sham rows, with
cond = "<condition>_rand<seed>", the new region in roi_grid_rand and the first one in roi_grid_rand_first.

--check recomputes the FIRST placement instead (the stored roi_grid_rand) and compares with the stored rows, which verifies
that the weights, the preprocessing and this code path reproduce the original run before anything new is trusted.

Usage: python -m zoomdecomp.rerun_sham --model PATH --run_dir DIR --out_dir DIR2 --bench vstar --budgets 256 1024 [--seed 2] [--limit N]
       python -m zoomdecomp.rerun_sham --model PATH --run_dir DIR --out_dir DIR2 --bench vstar --budgets 256 --check --limit 20"""
import argparse, json, sys, time, zlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.data import LOADERS
from zoomdecomp.qwen_runner import random_grid_box
from zoomdecomp.runners import make_runner, score_any
from zoomdecomp.run_gate2 import Ctx, run_condition, SHAM


def load_rows(path):
    rows = {}
    for line in open(path):
        r = json.loads(line); rows.setdefault(r["uid"], {})[r["cond"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--run_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bench", required=True); ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--check", action="store_true", help="recompute the stored first placement and compare; writes <bench>_B<B>.check.jsonl")
    ap.add_argument("--precheck", type=int, default=8, help="before a new placement, recompute the stored one on this many samples and stop if it is not reproduced")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = make_runner(args.model, attn=args.attn)
    print(f"[init] {Path(args.model).name} family={R.family} on {R.device} seed={args.seed} check={args.check}", flush=True)
    samples = {s.uid: s for s in LOADERS[args.bench]()}
    suffix = "_rand" if args.check else f"_rand{args.seed}"
    for B in args.budgets:
        rows = load_rows(Path(args.run_dir) / f"{args.bench}_B{B}.jsonl")
        uids = [u for u, d in rows.items() if "base" in d and not d["base"]["correct"] and d.get("d_native_zoom", {}).get("correct")
                and all(c + "_rand" in d for c in SHAM)]
        if args.limit: uids = uids[: args.limit]
        out_path = out_dir / (f"{args.bench}_B{B}.check.jsonl" if args.check else f"{args.bench}_B{B}.jsonl")
        done = set()
        if out_path.exists():
            done = {(r["uid"], r["cond"]) for line in open(out_path) for r in [json.loads(line)]}
        fout = open(out_path, "a"); t0 = time.time(); n = same_region = 0; agree = total = 0; dlo = []
        print(f"[{args.bench} B={B}] {len(uids)} first-pass failures that the oracle zoom fixes", flush=True)
        if not args.check and args.precheck and uids:
            ok_ = tot_ = 0; dl = []
            for uid in uids[: args.precheck]:
                s = samples[uid]; d = rows[uid]; ctx = Ctx(R, s, s.load_image(), B); g1 = tuple(d[SHAM[0] + "_rand"]["roi_grid_rand"])
                for c in SHAM:
                    res, _ = run_condition(ctx, c, g1); sc = score_any(res, s.qtype, s.answer, s.letters, answers=s.meta.get("answers")); r0 = d[c + "_rand"]
                    tot_ += 1; ok_ += int(bool(sc["correct"]) == bool(r0["correct"]))
                    if sc.get("gold_logodds") is not None and r0.get("gold_logodds") is not None: dl.append(abs(sc["gold_logodds"] - r0["gold_logodds"]))
                del ctx
            print(f"[precheck B={B}] stored first placement reproduced on {ok_}/{tot_} sham rows; |gold log-odds difference| median {np.median(dl) if dl else float('nan'):.2e}, max {max(dl) if dl else float('nan'):.2e}", flush=True)
            if ok_ < 0.9 * tot_:
                print("[precheck] FAILED: the original run is not reproduced, nothing written", flush=True); sys.exit(3)
        for i, uid in enumerate(uids):
            if all((uid, c + suffix) in done for c in SHAM):
                continue
            s = samples[uid]; d = rows[uid]
            try:
                ctx = Ctx(R, s, s.load_image(), B)
                g = tuple(d["base"]["roi_grid"]); g1 = tuple(d[SHAM[0] + "_rand"]["roi_grid_rand"])
                if g != tuple(ctx.g):
                    print(f"[warn] {uid} B={B}: stored target grid {g} != recomputed {tuple(ctx.g)}; using stored", flush=True)
                if args.check:
                    g2, iou = g1, d[SHAM[0] + "_rand"].get("rand_iou")
                else:
                    rng = np.random.default_rng(zlib.crc32(f"{uid}|{B}|sham{args.seed}".encode()))
                    g2, iou = random_grid_box(g, ctx.GW, ctx.GH, rng); g2 = tuple(g2); same_region += int(g2 == g1)
                for c in SHAM:
                    name = c + suffix
                    if (uid, name) in done:
                        continue
                    r0 = d[c + "_rand"]
                    res, size2 = run_condition(ctx, c, g2)
                    sc = score_any(res, s.qtype, s.answer, s.letters, answers=s.meta.get("answers"))
                    row = dict(r0); row.update(sc, cond=name, text=res["text"], n_tokens_in=res["n_tokens_in"], n_img_tokens=res["n_img_tokens"],
                                               second_wh=list(size2) if size2 else None, lp=res.get("letter_lp") or res.get("digit_lp"),
                                               roi_grid_rand=list(g2), roi_grid_rand_first=list(g1), rand_iou=iou, sham_seed=None if args.check else args.seed)
                    fout.write(json.dumps(row) + "\n"); n += 1
                    if args.check:
                        total += 1; agree += int(bool(sc["correct"]) == bool(r0["correct"]))
                        if sc.get("gold_logodds") is not None and r0.get("gold_logodds") is not None: dlo.append(abs(sc["gold_logodds"] - r0["gold_logodds"]))
                del ctx
            except Exception as e:
                print(f"[ERROR] {uid} B={B}: {type(e).__name__}: {e}", flush=True); torch.cuda.empty_cache()
            if (i + 1) % 25 == 0:
                fout.flush(); print(f"[B={B}] {i+1}/{len(uids)} rows={n} {(time.time()-t0)/60:.1f} min", flush=True)
        fout.close()
        if args.check:
            print(f"[check B={B}] correctness reproduced on {agree}/{total} sham rows; |gold log-odds difference| median {np.median(dlo) if dlo else float('nan'):.2e}, max {max(dlo) if dlo else float('nan'):.2e}", flush=True)
            if total == 0 or agree < 0.9 * total: sys.exit(3)
        else:
            print(f"[B={B}] done rows={n}, second placement identical to the first for {same_region} of {len(uids)} samples, {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
