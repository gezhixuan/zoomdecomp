"""Qualitative panel: one zoom-rescued sample per regime, showing exactly what each arm fed the model. Used for the TextVQA
supplement figure.
CPU only: rebuilds the seen raster / crops with the processor's own resize, reads answers + log-odds from the run JSONL."""
import argparse, json, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from zoomdecomp import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoProcessor
from zoomdecomp.data import LOADERS
from zoomdecomp.qwen_runner import resize_like_processor, seen_dims, crop_box, mask_outside, roi_grid, grid_to_seen_box, seen_to_native_box, fit_dims
from zoomdecomp.figures import load_run, regime_strict, rescued, RAMP, INK, INK2, GRID, REG

TOK_PX = 784


def arms_for(ip, s, img, B, factor=28, min_pixels=3136):
    W, H = img.size
    rh, rw = seen_dims(W, H, B * TOK_PX, factor=factor, min_pixels=min_pixels)
    I_seen = resize_like_processor(ip, img, rh, rw)
    g, (GW, GH) = roi_grid(s.bbox, W, H, rh, rw, factor=factor)
    sb = grid_to_seen_box(g, factor); nb = seen_to_native_box(sb, W, H, rh, rw)
    area = (nb[2] - nb[0]) * (nb[3] - nb[1]); K_px = int(max(min(256 * TOK_PX, B * TOK_PX), min(B * TOK_PX, area)))
    th, tw = fit_dims(sb[2] - sb[0], sb[3] - sb[1], K_px, factor=factor)
    out = {
        "seen (first pass)": (I_seen, sb),
        "b2: seen px, same density": (crop_box(I_seen, sb), None),
        "c: seen px, upsampled": (resize_like_processor(ip, crop_box(I_seen, sb), th, tw), None),
        "d: native px": (resize_like_processor(ip, crop_box(img, nb), th, tw), None),
    }
    return out, (rh, rw), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(paths.RUNS / "gate_qwen25vl7b"))
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--bench", default="vstar"); ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--out", default=str(paths.FIGS))
    ap.add_argument("--regimes", default="1,2,3", help="0-based regime indices to show as rows (0=R1 .. 3=R4)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()
    try: ip = AutoProcessor.from_pretrained(args.model).image_processor
    except Exception:                      # weights not on disk: only .resize/.resample are used and the class defaults (bicubic) match Qwen2.5-VL
        from transformers import Qwen2VLImageProcessor; ip = Qwen2VLImageProcessor()
    wide, lo = load_run(args.run)
    w = wide.loc[args.bench].loc[args.budget]; L = lo.loc[args.bench].loc[args.budget]
    rs = rescued(w).copy(); rs["regime"] = rs.apply(regime_strict, axis=1)
    samples = {s.uid: s for s in LOADERS[args.bench]()}
    rows = {r["uid"]: r for f in glob.glob(f"{args.run}/{args.bench}_B{args.budget}.jsonl") for line in open(f) for r in [json.loads(line)] if r["cond"] == "base"}
    picks = []
    for reg in [int(x) for x in args.regimes.split(",")]:
        cand = rs[rs["regime"] == reg]
        if len(cand) == 0: continue
        # most convincing: largest d log-odds margin over the arm just below it
        key = {1: "b2_seen_same", 2: "c_seen_zoom", 3: "d_native_zoom"}[reg]
        score = L.loc[cand.index][key] - L.loc[cand.index]["base"]
        if reg == 3: score = L.loc[cand.index]["d_native_zoom"] - L.loc[cand.index]["c_seen_zoom"]
        picks.append((reg, score.idxmax()))
    cols = ["seen (first pass)", "b2: seen px, same density", "c: seen px, upsampled", "d: native px"]
    short = {"seen (first pass)": "First pass (seen raster)", "b2: seen px, same density": "b2  seen px, same density", "c: seen px, upsampled": "c  seen px, upsampled", "d: native px": "d  native px"}
    conds = ["base", "b2_seen_same", "c_seen_zoom", "d_native_zoom"]
    # ------------------------------------------------------------- F1b: exemplar grid (rows = regimes, cols = what the model saw)
    prepared = []
    for reg, uid in picks:
        s = samples[uid]; img = s.load_image(); arms, (rh, rw), g = arms_for(ip, s, img, args.budget)
        ratio = max(arms[c][0].size[1] / arms[c][0].size[0] for c in cols)
        prepared.append((reg, uid, s, img, arms, rh, rw, max(0.7, ratio)))
    row_in = [1.5 * r + 0.75 for (*_, r) in prepared]
    fig, axes = plt.subplots(len(picks), 4, figsize=(6.9, sum(row_in)), gridspec_kw=dict(height_ratios=row_in))
    axes = np.atleast_2d(axes); qlines = []
    MIN = "\u2212"
    for i, (reg, uid, s, img, arms, rh, rw, _) in enumerate(prepared):
        meta = rows[uid]
        q = s.prompt.split("\n")[0]
        for j, c in enumerate(cols):
            ax = axes[i, j]; im, box = arms[c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            if box is not None:
                ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="#e34948", lw=1.0))
            lo_v = L.loc[uid][conds[j]]; ok = bool(w.loc[uid][conds[j]])
            ax.set_title(f"{short[c]}\n{im.size[0]}\u00d7{im.size[1]} px", fontsize=6.0, color=INK2, pad=2)
            ax.text(0.5, -0.06, f"Gold log-odds {lo_v:+.1f}".replace("-", MIN) + f"  {'correct' if ok else 'wrong'}", transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                    color=INK, fontweight="bold" if ok else "normal")
        axes[i, 0].set_ylabel(f"{REG[reg]}", fontsize=6.2)   # no internal sample ids in the figure
        qlines.append(f"Q: {q}   gold: {s.answer}   (first pass {img.size[0]}\u00d7{img.size[1]} \u2192 {rw}\u00d7{rh}, target covers {meta['bbox_seen_tokens']:.1f} tokens)")
    fig.tight_layout(h_pad=3.2)
    for i, ql in enumerate(qlines):
        pos = axes[i, 0].get_position(); top = max(axes[i, j].get_position().y1 for j in range(4))
        fig.text(pos.x0 - 0.04, top + 0.42 / fig.get_size_inches()[1], ql, fontsize=6.2, color=INK2, va="bottom", ha="left")
    fig.savefig(Path(args.out) / f"F1b_examples_{args.bench}_B{args.budget}{args.tag}.png", dpi=170, bbox_inches="tight")
    fig.savefig(Path(args.out) / f"F1b_examples_{args.bench}_B{args.budget}{args.tag}.pdf", bbox_inches="tight"); plt.close(fig)
    print("picks:", picks)


if __name__ == "__main__":
    main()
