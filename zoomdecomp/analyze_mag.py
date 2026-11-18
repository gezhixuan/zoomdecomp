"""Magnification sweep analysis + figure ("how strong a prescription does the model need?").
Reads runs/<run>/mag_<bench>_B<B>.jsonl written by run_mag.py. For each model x benchmark: share of the zoom-rescued cohort fixed
(robust criterion) by seen pixels re-tokenized at magnification m (c_m), by native pixels at the same output size (d_m), and by seen
pixels of the random region (c_m_rand); median gold log-odds; the magnification at which c reaches 90% of its plateau.
Curves (figure, per-m table columns) use a fixed sample: rescued failures with every condition at every m <= TOP (= regions small
enough to render at TOP x within the 4,096-token cap), so that all magnifications are compared on the same samples. The "any m"
columns use every rescued failure and every magnification run for it (1-32x, as far as the region fits).
Usage: python -m zoomdecomp.analyze_mag [--out outputs/figures]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
from zoomdecomp import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from zoomdecomp.figures import RUNS, RAMP, GREY, INK, INK2, GRID, BENCH_LABEL, W_TEXT, save, xlabel_under

pd.set_option("display.width", 220)
C_SEEN, C_NATIVE, C_SHAM = RAMP[2], RAMP[3], "#8d8c87"


def load_mag():
    rows = []
    for m, p in RUNS.items():
        for f in glob.glob(str(Path(p) / "mag_*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r["model_short"] = m; rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ok"] = df["correct"].astype(bool) & (df["gold_logodds"].fillna(-1e9) > 0)
    return df


TOP = 16


def fixed_sample(df, top=TOP):
    """Rows with mag <= top for the samples that have c, d and c_rand at every such magnification."""
    sub = df[df.mag <= top]; need = 3 * sub.mag.nunique()
    cnt = sub.groupby(["model_short", "bench", "uid"]).cond.nunique().rename("k")
    keep = cnt[cnt == need].index
    return sub.set_index(["model_short", "bench", "uid"]).loc[lambda x: x.index.isin(keep)].reset_index()


def paired(a, b, B=4000, seed=0):
    """Mean of b - a in points with a 95% sample-bootstrap interval."""
    d = b.astype(int) - a.astype(int); rng = np.random.default_rng(seed); n = len(d)
    bs = d[rng.integers(0, n, (B, n))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def summarise(df):
    g = df.groupby(["model_short", "bench", "arm", "mag"]).agg(n=("uid", "size"), fixed=("ok", "mean"), lo_med=("gold_logodds", "median")).reset_index()
    return g


def prescription(g, model, bench, level=0.9):
    """Smallest magnification at which the seen-pixel curve reaches `level` of its maximum."""
    c = g[(g.model_short == model) & (g.bench == bench) & (g.arm == "c")].sort_values("mag")
    if c.empty: return np.nan, np.nan
    top = c.fixed.max(); hit = c[c.fixed >= level * top]
    return float(hit.mag.iloc[0]), float(top)


def fig_prescription(df, g, out):
    models = [m for m in RUNS if m in set(df.model_short)]; benches = [b for b in ["vstar", "zoombench", "textvqa"] if b in set(df.bench)]
    fig, axes = plt.subplots(len(benches), len(models), figsize=(W_TEXT, 1.05 * len(benches) + 0.3), sharex=True, sharey=True, squeeze=False)
    empty = None
    for i, b in enumerate(benches):
        for j, m in enumerate(models):
            ax = axes[i, j]; sub = g[(g.model_short == m) & (g.bench == b)]
            if sub.empty:
                ax.axis("off"); empty = ax; continue        # InternVL3 is not run on TextVQA (said in the caption and in the setup)
            sc, sd = sub[sub.arm == "c"].sort_values("mag"), sub[sub.arm == "d"].sort_values("mag")
            if len(sc) and len(sd):
                ax.fill_between(sc.mag, 100 * sc.fixed, 100 * sd.fixed.values, color=C_NATIVE, alpha=0.12, lw=0)
            for arm, col, ls in [("c_rand", C_SHAM, "--"), ("d", C_NATIVE, "-"), ("c", C_SEEN, "-")]:
                s_ = sub[sub.arm == arm].sort_values("mag"); main_ = arm != "c_rand"
                ax.plot(s_.mag, 100 * s_.fixed, ls, color=col, lw=1.7 if main_ else 1.0, marker="o", ms=3.6 if main_ else 2.8,
                        mfc=col if main_ else "white", mec="white" if main_ else col, mew=0.6 if main_ else 0.8, zorder=3 if arm == "c" else 2)
            nm = df[(df.model_short == m) & (df.bench == b)].drop_duplicates("uid")["native_mag"].median()
            ax.axvline(nm, color=INK2, lw=0.7, ls=":", zorder=1)
            pm, top = prescription(g, m, b)
            if not np.isnan(pm): ax.plot([pm, pm], [0, 100 * top * 0.9], ls=(0, (2.5, 1.5)), color=C_SEEN, lw=0.9, zorder=1)
            ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1×", "2×", "4×", "8×", "16×"]); ax.minorticks_off()
            ax.set_ylim(-5, 106); ax.set_yticks([0, 50, 100]); ax.spines["left"].set_bounds(0, 100); ax.grid(axis="y")   # pad so markers at 0/100 are not clipped
            ax.text(0.03, 0.95, f"$n$={int(sc.n.max())}", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=INK2)
            if i == 0: ax.set_title(m, fontsize=7.5)
            if j == 0:
                ax.set_ylabel("Fixed (%)", fontsize=6.8)
                ax.text(-0.34, 0.5, BENCH_LABEL[b], transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
    h = [Line2D([], [], color=C_SEEN, lw=1.7, marker="o", ms=3.6), Line2D([], [], color=C_NATIVE, lw=1.7, marker="o", ms=3.6),
         plt.Rectangle((0, 0), 1, 1, color=C_NATIVE, alpha=0.12, lw=0), Line2D([], [], color=C_SHAM, lw=1.0, ls="--", marker="o", ms=2.8, mfc="white"),
         Line2D([], [], color=C_SEEN, lw=0.9, ls=(0, (2.5, 1.5))), Line2D([], [], color=INK2, lw=0.7, ls=":")]
    labels = ["Seen pixels, re-tokenized", "Native pixels, same size", "Gain from new pixels", "Sham of the seen pixels", "90% of the seen-pixel plateau", "Native crop's own magnification"]
    if empty is not None: empty.legend(h, labels, loc="center", frameon=False, fontsize=6.3, handlelength=2.4, labelspacing=0.5, bbox_to_anchor=(0.5, 0.5))
    else: fig.legend(h, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.3, handlelength=2.4, bbox_to_anchor=(0.5, -0.06))
    for j in range(len(models)):                       # a column whose bottom panel is empty shows its x labels one row up
        if not axes[-1, j].axison and len(benches) > 1: axes[-2, j].tick_params(labelbottom=True)
    fig.tight_layout(h_pad=0.9)
    # axis name as a row header left of the tick values of the bottom-left panel (same placement as "Input downsampling" in Fig. 2)
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); FW, FH = fig.get_size_inches(); ax0 = axes[-1, 0]
    tl0 = Bbox.union([t.get_window_extent() for t in ax0.get_xticklabels() if t.get_text()])
    fig.text(ax0.get_position().x0 - 0.06 / FW, tl0.y1 / fig.dpi / FH, "Magnification\nof the region", ha="right", va="top", fontsize=7, linespacing=1.1)
    save(fig, out, "F14_prescription")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(paths.FIGS)); args = ap.parse_args()
    df_all = load_mag()
    if df_all is None:
        print("no mag files yet"); return
    df = fixed_sample(df_all, TOP); g = summarise(df)
    print(f"fixed sample (every condition at every m <= {TOP}):")
    print(g.pivot_table(index=["model_short", "bench", "arm"], columns="mag", values="fixed").round(2).to_string())
    print(g[g.arm == "c"].groupby(["model_short", "bench"]).n.max().to_string())
    print("\nprescription (smallest m with >= 90% of the seen-pixel plateau), plateau, median native magnification; paired c(16x) - c(8x):")
    for m in RUNS:
        for b in ["vstar", "zoombench", "textvqa"]:
            pm, top = prescription(g, m, b)
            if np.isnan(pm): continue
            sub = df[(df.model_short == m) & (df.bench == b)]; w = sub.pivot_table(index="uid", columns="cond", values="ok", aggfunc="first").astype(bool)
            nm = sub.drop_duplicates("uid")["native_mag"].median(); d = paired(w["c_m8"].values, w[f"c_m{TOP}"].values)
            print(f"  {m:14s} {b:9s} n={len(w)} m90={pm:g}  plateau={100*top:.0f}%  native_mag_median={nm:.1f}  c16-c8 = {d[0]:+.0f} [{d[1]:+.0f}, {d[2]:+.0f}]")
    d32 = fixed_sample(df_all, 32)
    print("\nsmaller fixed sample with every m <= 32: c and d at 8/16/24/32 and paired c(32x) - c(8x)")
    for (m, b), sub in d32.groupby(["model_short", "bench"]):
        w = sub.pivot_table(index="uid", columns="cond", values="ok", aggfunc="first").astype(bool); d = paired(w["c_m8"].values, w["c_m32"].values)
        print(f"  {m:14s} {b:9s} n={len(w):3d}  c: " + " ".join(f"{100*w[f'c_m{x}'].mean():3.0f}" for x in (8, 16, 24, 32)) + "  d: " + " ".join(f"{100*w[f'd_m{x}'].mean():3.0f}" for x in (8, 16, 24, 32)) + f"  c32-c8 = {d[0]:+.0f} [{d[1]:+.0f}, {d[2]:+.0f}]")
    print("\nper sample over all magnifications run (all rescued failures): seen pixels suffice at some m (strict: where the random region does not)")
    TAB = paths.TABLES
    mags = sorted(df.mag.unique()); k = len(mags)
    lines = [r"\begin{tabular}{llrr" + "r" * k + "rrrr}", r"\toprule",
             rf" & & & & \multicolumn{{{k}}}{{c}}{{magnification $m$ of the region (fixed sample)}} & & \multicolumn{{3}}{{c}}{{any $m$ (all rescued failures)}} \\",
             rf"\cmidrule(lr){{5-{4+k}}} \cmidrule(lr){{{6+k}-{8+k}}}",
             r"Model & Data & condition & $n$ & " + " & ".join(f"{m:g}$\\times$" for m in mags) + r" & $m_{90}$ & $n$ & fixed & strict \\", r"\midrule"]
    BL = {"vstar": "V$^*$", "zoombench": "ZoomBench", "textvqa": "TextVQA"}
    for m in RUNS:
        for b in ["vstar", "zoombench", "textvqa"]:
            sub = df_all[(df_all.model_short == m) & (df_all.bench == b)]
            if sub.empty: continue
            w = sub.pivot_table(index="uid", columns="cond", values="ok", aggfunc="first")
            amags = sorted(sub.mag.unique())
            def anyof(cols): return np.any([w[c].fillna(False).astype(bool).values for c in cols], axis=0)
            def strict(xs): return np.any([(w[f"{xs}_m{x:g}"].fillna(False).astype(bool) & ~w[f"c_rand_m{x:g}"].fillna(False).astype(bool)).values for x in amags], axis=0)
            some_c, some_d, some_r = anyof([f"c_m{x:g}" for x in amags]), anyof([f"d_m{x:g}" for x in amags]), anyof([f"c_rand_m{x:g}" for x in amags])
            strict_c, strict_d = strict("c"), strict("d")
            pm, top = prescription(g, m, b); gg = g[(g.model_short == m) & (g.bench == b)]; nfix = int(gg[gg.arm == "c"].n.max())
            print(f"  {m:14s} {b:9s} n={len(w)}  seen-some {100*some_c.mean():.0f}% (strict {100*strict_c.mean():.0f}%)  sham-some {100*some_r.mean():.0f}%  native-some {100*some_d.mean():.0f}% (strict {100*strict_d.mean():.0f}%)  not-from-seen {100*(1-strict_c.mean()):.0f}%  m90={pm:g}")
            for arm, lab in [("c", "seen px, re-tokenized"), ("d", "native px, same size"), ("c_rand", "seen px, random region")]:
                row = gg[gg.arm == arm].set_index("mag").reindex(mags)
                cells = " & ".join(f"{100*v:.0f}" for v in row.fixed.values)
                extra = {"c": f"{pm:g}$\\times$ & {len(w)} & {100*some_c.mean():.0f} & {100*strict_c.mean():.0f}", "d": f"-- & & {100*some_d.mean():.0f} & {100*strict_d.mean():.0f}", "c_rand": f"-- & & {100*some_r.mean():.0f} & --"}[arm]
                first = arm == "c"
                lines.append(f"{m if first else ''} & {BL[b] if first else ''} & {lab} & {nfix if first else ''} & {cells} & {extra} \\\\")
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True); (TAB / "mag_sweep.tex").write_text("\n".join(lines) + "\n"); print("wrote", TAB / "mag_sweep.tex")
    fig_prescription(df, g, Path(args.out)); print("figure written")


if __name__ == "__main__":
    main()
