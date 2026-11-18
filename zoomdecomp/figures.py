"""Data figures of the paper (matplotlib, CVPR two-column sizes) and the loaders every analysis shares:
load_run (per-sample outputs of a run folder), rescued (the zoom-rescued failures) and regime_strict (the regime of a failure).
Usage: python -m zoomdecomp.figures"""
import argparse, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
from zoomdecomp import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from PIL import Image

RUNS = {"Qwen2.5-VL-7B": str(paths.RUNS / "gate_qwen25vl7b"),
        "Qwen3-VL-8B": str(paths.RUNS / "gate_qwen3vl8b"),
        "InternVL3-8B": str(paths.RUNS / "gate_internvl3_8b")}
MODEL_C = {"Qwen2.5-VL-7B": "#2a78d6", "Qwen3-VL-8B": "#eb6834", "InternVL3-8B": "#1baf7a"}   # slots 1-3
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]           # ordinal blue 250/350/450/600
GREY, INK, INK2, GRID = "#b9b8b3", "#0b0b0b", "#52514e", "#e6e5e1"
BUDGET_C = ["#b9b8b3", "#6f6e6a", "#0b0b0b"]                   # ordinal neutral ramp for first-pass budgets (coarse -> fine); never the regime ramp
ARMS = ["a1_cached_roi", "a2_cached_only", "a3_cached_inplace", "b1_mask_bg", "b2_seen_same", "b3_native_same", "c_seen_zoom", "d_native_zoom"]
ARM_LABEL = {"a1_cached_roi": "a1 cached\nreinsert", "a2_cached_only": "a2 cached\nROI only", "a3_cached_inplace": "a3 cached\nin place",
             "b1_mask_bg": "b1 mask\nbackground", "b2_seen_same": "b2 seen px\nsame density", "b3_native_same": "b3 native px\nsame density",
             "c_seen_zoom": "c seen px\nupsampled", "d_native_zoom": "d native\nzoom"}
REG = ["R1 reuse", "R2 isolate", "R3 re-tokenize", "R4 acquire"]   # same words as the text
ARM_CODE = {"a1_cached_roi": "a1", "a2_cached_only": "a2", "a3_cached_inplace": "a3", "b1_mask_bg": "b1", "b2_seen_same": "b2", "b3_native_same": "b3", "c_seen_zoom": "c", "d_native_zoom": "d"}
ARM_KEY = "a1 cached re-insertion · a2 region only, new positions · a3 region only, original positions · b1 background masked · b2 seen crop, same density · b3 native crop, same density · c seen crop, upsampled · d native crop, oracle"
BENCH_LABEL = {"vstar": "V* Bench", "zoombench": "ZoomBench", "textvqa": "TextVQA"}

from matplotlib import font_manager as _fm
_FONT_DIR = paths.FONTS
for _f in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"):
    if (_FONT_DIR / _f).exists():
        _fm.fontManager.addfont(str(_FONT_DIR / _f))
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "font.family": ["Arial", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})   # $n$ etc. in Arial too


W_TEXT = 6.875   # CVPR \textwidth (in). Main-text figures are saved exactly this wide, so LaTeX's \linewidth does not rescale their fonts.


def save(fig, out, stem, dpi=170, width=W_TEXT):
    """Save PDF + PNG with a tight vertical extent but exactly `width` inches wide (content centred)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer())
    if bb.width > width + 0.005:
        print(f"  [save] {stem}: content is {bb.width:.3f} in wide (> {width}); LaTeX will shrink it"); box = bb.padded(0.02)
    else:
        cx = (bb.x0 + bb.x1) / 2; box = Bbox([[cx - width / 2, bb.y0 - 0.02], [cx + width / 2, bb.y1 + 0.02]])
    fig.savefig(Path(out) / f"{stem}.pdf", bbox_inches=box); fig.savefig(Path(out) / f"{stem}.png", dpi=dpi, bbox_inches=box); plt.close(fig)


def xlabel_under(fig, axes, text, fontsize=7, gap=0.05):
    """Shared x-label placed just under the lowest tick labels of `axes` (call after the layout is final), centred on their span."""
    fig.canvas.draw(); vis = [ax for ax in axes if ax.axison]
    y = min(t.get_window_extent().y0 for ax in vis for t in ax.get_xticklabels() if t.get_text()) / fig.dpi
    x = (min(ax.get_position().x0 for ax in axes) + max(ax.get_position().x1 for ax in axes)) / 2
    fig.text(x, (y - gap) / fig.get_figheight(), text, ha="center", va="top", fontsize=fontsize)


def budget_axis(g, tol=0.12):
    """Budgets of one model x benchmark with their median first-pass downsampling, coarse -> fine. A budget within `tol` of a
    finer one is dropped (Qwen B4096 = 1.06x vs B16384 = 1.00x), so 'native' is one point: the finest budget."""
    budgets = sorted(g.index.get_level_values(0).unique()); ds = [float(np.median(g.loc[B]["downsample"])) for B in budgets]
    keep = [i for i in range(len(budgets)) if not any(ds[j] >= ds[i] * (1 - tol) for j in range(i + 1, len(budgets)))]
    return [budgets[i] for i in keep], [ds[i] for i in keep]


def load_run(path):
    rows = []
    for f in sorted(glob.glob(str(Path(path) / "*.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["ok"] = df["correct"].astype(bool) & (df["gold_logodds"].fillna(-1e9) > 0)
    wide = df.pivot_table(index=["bench", "budget", "uid"], columns="cond", values="ok", aggfunc="first")
    lo = df.pivot_table(index=["bench", "budget", "uid"], columns="cond", values="gold_logodds", aggfunc="first")
    first = df.groupby(["bench", "budget", "uid"]).agg(img=("img_wh", "first"), seen=("seen_wh", "first"), cov=("bbox_seen_tokens", "first"))
    first = first.loc[wide.index]
    wide["downsample"] = [(a[0] * a[1]) / (b[0] * b[1]) for a, b in zip(first["img"], first["seen"])]
    wide["cov"] = first["cov"].values
    # greedy correctness of the first pass, independent of the log-odds clause (a failure must be a wrong answer)
    g = df[df["cond"] == "base"].set_index(["bench", "budget", "uid"])["correct"].astype(bool)
    wide["base_greedy"] = g.reindex(wide.index).fillna(False).values
    return wide, lo


def regime_strict(r):
    def ok(a):
        v = r.get(a); s = r.get(a + "_rand")
        return bool(v) if pd.notna(v) else False, (bool(s) if (s is not None and pd.notna(s)) else False)
    def hit(a):
        v, s = ok(a); return v and not s
    if any(hit(a) for a in ("a1_cached_roi", "a2_cached_only", "a3_cached_inplace")): return 0
    if any(hit(a) for a in ("b1_mask_bg", "b2_seen_same")): return 1
    if hit("c_seen_zoom"): return 2
    return 3


def rescued(w, variant="primary"):
    """Zoom-rescued failures.
    primary : first-pass answer wrong (greedy), native zoom correct under the robust criterion, and the same-size random
              native crop NOT correct (target-specific rescue).
    confidence-filtered : the earlier definition (base not robust-correct, d robust-correct); includes confidence-only rescues
              and rescues that a random native crop also produces.
    """
    if variant == "confidence-filtered":
        return w[(w["base"] == False) & (w.get("d_native_zoom") == True)]
    spec = (w["d_native_zoom_rand"] != True) if "d_native_zoom_rand" in w else True
    return w[(w["base"] == False) & (w["base_greedy"] == False) & (w.get("d_native_zoom") == True) & spec]


def is_rescued_row(r):
    """Row-level version of rescued(): answer wrong at first pass, native zoom robust-correct, random native crop not."""
    return (r["base"] == False) and (r.get("base_greedy") == False) and (r.get("d_native_zoom") == True) and (r.get("d_native_zoom_rand") != True)


def ds_cells(w, thr=2.0):
    """Mask of the samples that belong to a (benchmark, budget) cell whose median first-pass downsampling is >= thr.
    This is the one cohort rule of the paper: tables and figures pooled over 'first passes downsampled >= 2x' use it, and so do
    the per-cell figures. A per-sample rule (w["downsample"] >= thr) keeps large images of near-native cells and drops small
    images of the 2.8x and 4x cells, which made the pooled n differ from the per-cell n (377/402/411 against 372/397/424)."""
    return (w["downsample"].groupby(level=[0, 1]).transform("median") >= thr).values


def dsample_label(x, cap=True):
    return f"{x:.0f}\u00d7" if x >= 3 else (("Native" if cap else "native") if x < 1.02 else f"{x:.1f}\u00d7")   # 1.06x (B4096 on Qwen) is "1.1x", not native


# ---------------------------------------------------------------- F2: budget curves


# ---------------------------------------------------------------- F3: arm ladder with shams


# ---------------------------------------------------------------- F4: strict regime partition (stacked, 100%)


# ---------------------------------------------------------------- F5: phase diagram (downsample x coverage -> regime share)


# ---------------------------------------------------------------- F6: graded log-odds
def fig_graded(data, out, bench="vstar"):
    models = [m for m in data if bench in data[m][0].index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    conds = ["base"] + ARMS
    for ax, m in zip(axes, models):
        w, lo = data[m]
        B = sorted(w.loc[bench].index.get_level_values(0).unique())[0]  # lowest budget = strongest downsampling
        rs_idx = rescued(w.loc[bench].loc[B]).index
        L = lo.loc[bench].loc[B].loc[rs_idx]
        vals = [L[c].dropna().values for c in conds]
        parts = ax.violinplot(vals, positions=range(len(conds)), widths=0.8, showmeans=False, showextrema=False)
        for pc in parts["bodies"]: pc.set_facecolor(GREY); pc.set_edgecolor("none"); pc.set_alpha(0.6)
        ax.plot(range(len(conds)), [np.median(v) for v in vals], "o", color=INK, ms=3.5)
        shv = [L[c + "_rand"].dropna().values for c in conds if c + "_rand" in L]
        ax.plot(range(1, 1 + len(shv)), [np.median(v) for v in shv], "o", color=INK, ms=3.5, mfc="white")
        ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(len(conds))); ax.set_xticklabels(["Base"] + [ARM_CODE[a] for a in ARMS])
        ds = float(np.median(w.loc[bench].loc[B]["downsample"])); ax.set_title(f"{m} · {dsample_label(ds)} first pass (n={len(rs_idx)})"); ax.grid(axis="y")
    axes[0].set_ylabel("Gold log-odds (zoom-rescued failures)")
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], marker="o", color=INK, ls="", ms=3.5), Line2D([], [], marker="o", color=INK, mfc="white", ls="", ms=3.5)],
                   ["Median, target region", "Median, sham"], frameon=False, loc="upper left", fontsize=5.8, handlelength=1.0)
    # (no key under the figure: the codes are those of Tab. 1, and the long key widened the figure so that LaTeX shrank it)
    fig.tight_layout(); fig.savefig(out / f"F6_graded_{bench}.pdf", bbox_inches="tight"); fig.savefig(out / f"F6_graded_{bench}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- F7: per-sample flow across budgets (alluvial)
def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55):
    verts = [(x0, y0a), ((x0 + x1) / 2, y0a), ((x0 + x1) / 2, y1a), (x1, y1a), (x1, y1b), ((x0 + x1) / 2, y1b), ((x0 + x1) / 2, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def _boot_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); m = x.mean()
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fig_forest(data, out, bench="vstar"):
    """Dumbbell forest: per arm (rows), rescue rate on zoom-rescued samples, target (filled) vs sham (hollow), 95% bootstrap CI;
    columns = budgets (coarsest two + native), model = marker/color within row offsets."""
    models = [m for m in data if bench in data[m][0].index.get_level_values(0)]
    budgets_all = sorted(set.union(*[set(data[m][0].loc[bench].index.get_level_values(0).unique()) for m in models]))
    budgets = [b for b in budgets_all if b in (256, 1024, 4096)]
    arms = ARMS[:-1]  # d is 1.0 by construction
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.9, 2.6), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, B in zip(axes, budgets):
        dss = []
        for k, m in enumerate(models):
            w = data[m][0].loc[bench]
            if B not in w.index.get_level_values(0): continue
            rs = rescued(w.loc[B]); dss.append(float(np.median(w.loc[B]["downsample"])))
            if len(rs) == 0: continue
            off = (k - (len(models) - 1) / 2) * 0.22
            for i, a in enumerate(arms):
                y = len(arms) - 1 - i + off
                mt, lo_t, hi_t = _boot_ci(rs[a].astype(float)) if a in rs else (np.nan,) * 3
                ms_, lo_s, hi_s = _boot_ci(rs[a + "_rand"].astype(float)) if a + "_rand" in rs else (np.nan,) * 3
                ax.plot([ms_, mt], [y, y], color=MODEL_C[m], lw=1.0, alpha=0.8)
                ax.plot([lo_t, hi_t], [y, y], color=MODEL_C[m], lw=2.2, alpha=0.35, solid_capstyle="butt")
                ax.plot(mt, y, "o", color=MODEL_C[m], ms=4.2, label=m if i == 0 else None)
                ax.plot(ms_, y, "o", color=MODEL_C[m], ms=4.2, mfc="white", mew=1.0)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_CODE[a] for a in arms][::-1])
        ax.set_xlim(-0.03, 1.03); ax.spines["bottom"].set_bounds(0, 1); ax.grid(axis="x"); ax.axvline(0, color=INK2, lw=0.6)   # pad so markers at 0/1 are not clipped
        if dss:
            # models can sit at different downsampling for the same nominal budget (B4096: 1.1x on Qwen, 2.8x on InternVL3): name both
            med_ = float(np.median(dss)); main_ = dsample_label(med_, cap=False)
            other = [f"{dsample_label(d_, cap=False)} for {mm.split('-')[0]}" for d_, mm in zip(dss, [m_ for m_ in models if B in data[m_][0].loc[bench].index.get_level_values(0)]) if abs(np.log(d_ / med_)) > np.log(1.3)]
            ax.set_title(("First pass at native resolution" if main_ == "native" else f"First pass {main_} downsampled") + (f" ({', '.join(other)})" if other else ""), fontsize=7)
        ax.set_xlabel("Zoom-rescued failures fixed (%)"); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
        for yb in (0.5, 3.5): ax.axhline(yb, color=GRID, lw=0.6)             # between the families a | b | c
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, handlelength=0.8, bbox_to_anchor=(0.5, -0.11))
    for ax in axes:
        ax.text(1.0, len(arms) - 0.6, "Cached tokens", fontsize=5.6, color=INK2, ha="right", va="center")
        ax.text(1.0, len(arms) - 3.6, "Same density", fontsize=5.6, color=INK2, ha="right", va="center")
        ax.text(1.0, 0.4, "Upsample", fontsize=5.6, color=INK2, ha="right", va="center")
    fig.suptitle(f"{BENCH_LABEL[bench]}: filled = target region, hollow = sham, band = 95% bootstrap CI", y=1.02, fontsize=7)
    # (no key under the figure: the codes are those of Tab. 1, and the long key widened the figure so that LaTeX shrank it)
    fig.tight_layout(); fig.savefig(out / f"F3v2_forest_{bench}.pdf", bbox_inches="tight"); fig.savefig(out / f"F3v2_forest_{bench}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_ladder(data, out, bench="vstar"):
    """Cumulative rescue ladder: x = cumulative intervention set (reuse, +isolate, +upsample, +native), y = cumulative fraction
    of zoom-rescued samples; per budget line; sham ladder dashed/hollow. Caption must report nesting violations."""
    models = [m for m in data if bench in data[m][0].index.get_level_values(0)]
    steps = [("a1_cached_roi", "a2_cached_only", "a3_cached_inplace"), ("b1_mask_bg", "b2_seen_same"), ("c_seen_zoom",), ("d_native_zoom",)]
    names = ["Cached\ntokens", "+ isolate /\nsame density", "+ upsample\nseen px", "+ native\npixels"]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 2.1), sharey=True); axes = np.atleast_1d(axes)
    viol = {}
    for ax, m in zip(axes, models):
        w = data[m][0].loc[bench]
        ns = []; budgets = [B for B in sorted(w.index.get_level_values(0).unique()) if float(np.median(w.loc[B]["downsample"])) >= 2.0 or B == 4096]
        for k, B in enumerate(budgets):
            rs = rescued(w.loc[B]); ds = float(np.median(w.loc[B]["downsample"]))
            if len(rs) == 0: continue
            hit = np.zeros(len(rs), dtype=bool); sham = np.zeros(len(rs), dtype=bool); ys, yss = [], []
            for arms in steps:
                hit |= np.any([rs[a].fillna(False).astype(bool).values for a in arms if a in rs], axis=0)
                sham |= np.any([rs[a + "_rand"].fillna(False).astype(bool).values for a in arms if a + "_rand" in rs], axis=0) if any(a + "_rand" in rs for a in arms) else sham
                ys.append(hit.mean()); yss.append(sham.mean())
            # nesting violation: lower rung works but a higher (more expensive) rung fails
            h1 = np.any([rs[a].fillna(False).astype(bool).values for a in steps[0]], axis=0); h2 = np.any([rs[a].fillna(False).astype(bool).values for a in steps[1]], axis=0); h3 = rs["c_seen_zoom"].fillna(False).astype(bool).values
            viol[(m, B)] = float(((h1 & ~h3) | (h2 & ~h3)).mean())
            col = BUDGET_C[min(k, 2)]
            ax.plot(range(4), ys, "-o", color=col, ms=4, lw=1.4); ns.append(f"{dsample_label(ds, cap=False)} {len(rs)}")
            ax.plot(range(4), yss, "--o", color=col, ms=4, lw=1.0, mfc="white")
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6); ax.set_ylim(0, 1.02); ax.grid(axis="y"); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if ax is axes[0]: ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.set_title(f"{m}\n" + "$n$ = " + "  /  ".join(ns), fontsize=7)
    axes[0].set_ylabel("Zoom-rescued failures fixed,\ncumulative (%, dashed: shams)")
    hh = [Line2D([], [], color=BUDGET_C[k], marker="o", ms=4, lw=1.4) for k in range(3)] + [Line2D([], [], color=BUDGET_C[1], marker="o", ms=4, lw=1.0, ls="--", mfc="white")]
    fig.legend(hh, ["Coarsest first pass (17\u201318\u00d7)", "4\u00d7", "Finest shown (1.1\u00d7; 2.8\u00d7 for InternVL3)", "Shams"], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=6.5, handlelength=1.6)
    fig.suptitle(f"{BENCH_LABEL[bench]}: cumulative ladder", y=1.02)
    fig.tight_layout(); fig.savefig(out / f"F8_ladder_{bench}.pdf", bbox_inches="tight"); fig.savefig(out / f"F8_ladder_{bench}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    return viol


def fig_sankey2(data, out, bench="vstar"):
    """Two-column flow: outcome at the coarsest first pass -> outcome at the finest; nodes include 'correct at first pass' and 'zoom does not help'."""
    models = list(data)
    cats = ["Correct without zoom"] + REG + ["Not rescued"]
    colors = ["#d9d8d3", RAMP[0], RAMP[1], RAMP[2], RAMP[3], "#8d8c87"]
    fig, axes = plt.subplots(1, len(models), figsize=(W_TEXT, 2.6)); axes = np.atleast_1d(axes)
    def cat_of(r):
        if r["base"] == True: return 0
        if is_rescued_row(r): return 1 + regime_strict(r)
        return 5
    for ax, m in zip(axes, models):
        w = data[m][0]
        if bench not in w.index.get_level_values(0): ax.axis("off"); continue
        w = w.loc[bench]; budgets = sorted(w.index.get_level_values(0).unique()); B0, B1 = budgets[0], budgets[-1]
        uids = sorted(set(w.loc[B0].index) & set(w.loc[B1].index))
        seq = np.array([[cat_of(w.loc[B0].loc[u]), cat_of(w.loc[B1].loc[u])] for u in uids]); n = len(uids)
        gap = 0.02 * n; xw = 0.16; pos = []
        for j in range(2):
            cnt = np.bincount(seq[:, j], minlength=6); y = 0; d = {}
            for c in range(6):
                d[c] = (y, cnt[c]); ax.add_patch(plt.Rectangle((j - xw / 2, y), xw, cnt[c], color=colors[c], ec="none"))
                if cnt[c] > 0.05 * n: ax.text(j, y + cnt[c] / 2, f"{cnt[c]}", ha="center", va="center", fontsize=5.6, color="white" if c in (3, 4, 5) else INK)
                y += cnt[c] + gap
            pos.append(d)
        off0 = {c: pos[0][c][0] for c in range(6)}; off1 = {c: pos[1][c][0] for c in range(6)}
        flows = {}
        for a, b in seq: flows[(a, b)] = flows.get((a, b), 0) + 1
        for (a, b), k in sorted(flows.items()):
            col = "#c9c8c3" if a == b else (colors[a] if a != 0 else colors[b])
            _ribbon(ax, xw / 2, 1 - xw / 2, off0[a], off0[a] + k, off1[b], off1[b] + k, col, alpha=0.5 if a != b else 0.35)
            off0[a] += k; off1[b] += k
        ds = [float(np.median(w.loc[B]["downsample"])) for B in (B0, B1)]
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"{dsample_label(ds[0])} first pass", f"{dsample_label(ds[1])} first pass"]); ax.set_xlim(-0.25, 1.25); ax.set_ylim(0, n + gap * 6)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0); ax.set_title(f"{m} (n={n})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, cats, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=5.8, handlelength=0.9, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, f"F7v2_sankey2_{bench}")


def fig_upset(data, out, model="Qwen2.5-VL-7B", budget=256, top=14):
    w, _ = data[model]
    w = w[(w.index.get_level_values(0) != "textvqa") & (w.index.get_level_values(1) == budget)]
    rs = rescued(w)
    arms = ARMS[:-1]
    M = np.zeros((len(rs), len(arms)), dtype=bool)
    for j, a in enumerate(arms):
        ok = rs[a].fillna(False).astype(bool).values
        sh = rs[a + "_rand"].fillna(False).astype(bool).values if a + "_rand" in rs else np.zeros(len(rs), bool)
        M[:, j] = ok & ~sh
    keys = {}
    for row in M:
        k = tuple(row); keys[k] = keys.get(k, 0) + 1
    items = sorted(keys.items(), key=lambda kv: -kv[1])
    shown = items[:top]; rest = sum(v for _, v in items[top:])
    def implied(k):
        if any(k[j] for j, a in enumerate(arms) if a.startswith("a")): return 0
        if any(k[j] for j, a in enumerate(arms) if a in ("b1_mask_bg", "b2_seen_same")): return 1
        if k[arms.index("c_seen_zoom")]: return 2
        return 3
    fig = plt.figure(figsize=(6.9, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 5.0], height_ratios=[1.6, 1.0], wspace=0.02, hspace=0.08)
    ax_top = fig.add_subplot(gs[0, 1]); ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_top); ax_left = fig.add_subplot(gs[1, 0], sharey=ax_mat)
    xs = np.arange(len(shown) + (1 if rest else 0))
    for x, (k, v) in enumerate(shown):
        ax_top.bar(x, v, color=RAMP[implied(k)], width=0.7); ax_top.text(x, v + 0.5, str(v), ha="center", va="bottom", fontsize=5.6, color=INK2)
    if rest:
        ax_top.bar(len(shown), rest, color=GREY, width=0.7); ax_top.text(len(shown), rest + 0.5, f"{rest}\nother", ha="center", va="bottom", fontsize=5.6, color=INK2)
    ax_top.set_ylabel("Zoom-rescued failures\nwith this exact pattern"); ax_top.tick_params(axis="x", length=0, labelbottom=False); ax_top.grid(axis="y")
    for x, (k, v) in enumerate(shown):
        for j in range(len(arms)):
            ax_mat.plot(x, len(arms) - 1 - j, "o", ms=4.5, color=RAMP[3] if k[j] else "white", mec=RAMP[3] if k[j] else GREY, mew=0.8)
        on = [len(arms) - 1 - j for j in range(len(arms)) if k[j]]
        if len(on) > 1: ax_mat.plot([x, x], [min(on), max(on)], color=RAMP[3], lw=1.0)
    ax_mat.set_yticks(range(len(arms))); ax_mat.set_yticklabels([ARM_CODE[a] for a in arms][::-1]); ax_mat.yaxis.tick_right(); ax_mat.tick_params(axis="y", length=0, pad=2)
    ax_mat.set_xticks([]); ax_mat.set_ylim(-0.6, len(arms) - 0.4)
    for sp in ax_mat.spines.values(): sp.set_visible(False)
    sizes = M.sum(0)
    ax_left.barh([len(arms) - 1 - j for j in range(len(arms))], sizes, color=RAMP[2], height=0.6); ax_left.invert_xaxis(); ax_left.set_xlabel("Set size"); ax_left.tick_params(axis="y", length=0, labelleft=False); ax_left.grid(axis="x")
    for sp in ("top", "right", "left"): ax_left.spines[sp].set_visible(False)
    ds = float(np.median(w["downsample"]))
    fig.suptitle(f"{model} · V* + ZoomBench · {dsample_label(ds, cap=False)} first pass · {len(rs)} zoom-rescued failures; set membership = condition beats its sham; bar colour = implied regime", fontsize=6.5, y=0.99)
    fig.text(0.5, -0.02, "None of the 7 conditions (native pixels required) appears as the all-hollow column", ha="center", fontsize=5.8, color=INK2)
    fig.savefig(out / f"A4_upset_{model.split('-')[0]}_B{budget}.pdf", bbox_inches="tight"); fig.savefig(out / f"A4_upset_{model.split('-')[0]}_B{budget}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_sweep(data, out, bench="vstar"):
    """Thinking-Once-style: per model panel, x = first-pass downsampling (log, coarse -> fine), y = share of zoom-rescued
    failures also fixed by at least one condition of each family; the shams of the family hollow/dashed, shaded gap."""
    fam = {"Cached tokens (a1–a3)": (["a1_cached_roi", "a2_cached_only", "a3_cached_inplace"], RAMP[0]),
           "Isolate, same density (b1–b3)": (["b1_mask_bg", "b2_seen_same", "b3_native_same"], RAMP[1]),
           "Upsample seen pixels (c)": (["c_seen_zoom"], RAMP[2])}
    models = [m for m in data if bench in data[m][0].index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(models), figsize=(W_TEXT, 1.85), sharey=True); axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        w = data[m][0].loc[bench]; budgets, xs = budget_axis(w)
        for name, (arms, col) in fam.items():
            ys, ss = [], []
            for B in budgets:
                rs = rescued(w.loc[B])
                hit = np.any([rs[a].fillna(False).astype(bool).values for a in arms if a in rs], axis=0)
                sh = np.any([rs[a + "_rand"].fillna(False).astype(bool).values for a in arms if a + "_rand" in rs], axis=0)
                ys.append(hit.mean()); ss.append(sh.mean())
            ax.fill_between(xs, ss, ys, color=col, alpha=0.10, lw=0)
            ax.plot(xs, ss, "--o", color=col, ms=3.3, lw=0.9, mfc="white", mew=0.9)
            ax.plot(xs, ys, "-o", color=col, ms=4.2, lw=1.6, mec="white", mew=0.6, label=name)
        ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xlim(24, 0.8); ax.minorticks_off()
        tk = {17: "17×", 4: "4×", 1: "Native"}
        for x_ in xs:                                   # a budget away from the three standard ticks (8.5x on TextVQA, 2.8x on InternVL3) gets its own
            if min(abs(np.log(x_ / t_)) for t_ in tk) > np.log(1.3): tk[float(x_)] = dsample_label(x_)     # same rounding as the other figures and the tables
        ax.set_xticks(list(tk)); ax.set_xticklabels(list(tk.values()))
        ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100"]); ax.grid(axis="y"); ax.set_title(m)
    axes[0].text(23, 1.015, "Native zoom", fontsize=5.8, color=INK2, va="bottom", ha="left")
    if len(axes) % 2: axes[len(axes) // 2].set_xlabel("First-pass downsampling")
    axes[0].set_ylabel("Zoom-rescued failures fixed (%)")
    h = [Line2D([], [], color=col, lw=1.6, marker="o", ms=4.2) for _, col in fam.values()] + [Line2D([], [], color=INK2, ls=(0, (3, 2)), lw=0.9, marker="o", ms=3.3, mfc="white")]
    fig.legend(h, list(fam) + ["Sham (same operation, random region)"], frameon=False, loc="lower center", ncol=4, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05 if len(axes) % 2 else 0.12, 1, 1))
    if len(axes) % 2 == 0: xlabel_under(fig, list(axes), "First-pass downsampling")
    save(fig, out, f"F3v3_sweep_{bench}")


def fig_budget_curves_v3(data, out, thumbs=None):
    """Accuracy vs first-pass downsampling (full image, and with the oracle native zoom appended). Under the V* axis: the
    target region of one V* image exactly as the model saw it at each budget, at true relative pixel size, token grid in white."""
    benches = [b for b in ["vstar", "zoombench", "textvqa"] if any(b in data[m][0].index.get_level_values(0) for m in data)]
    from matplotlib.transforms import Bbox
    fig = plt.figure(figsize=(W_TEXT, 2.85))
    gs = fig.add_gridspec(1, len(benches), left=0.092, right=0.985, top=0.925, bottom=0.37, wspace=0.07)
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(benches))]
    for ax, b in zip(axes, benches):
        for m, (w, _) in data.items():
            if b not in w.index.get_level_values(0): continue
            g = w.loc[b]; budgets, xs = budget_axis(g)
            ax.plot(xs, [g.loc[B]["base"].mean() for B in budgets], "--o", color=MODEL_C[m], lw=1.0, ms=3.6, mfc="white", mew=0.9)
            ax.plot(xs, [g.loc[B]["d_native_zoom"].mean() for B in budgets], "-o", color=MODEL_C[m], lw=1.5, ms=4.0, mec="white", mew=0.5)
        ax.set_xscale("log"); ax.set_xlim(24, 0.8); ax.set_xticks([17, 4, 1]); ax.set_xticklabels(["17×", "4×", "Native"]); ax.minorticks_off()
        ax.set_title(BENCH_LABEL[b]); ax.grid(axis="y"); ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"] if ax is axes[0] else [])
    axes[0].set_ylabel("Robust accuracy (%)")
    fig.canvas.draw(); FW, FH = fig.get_size_inches()
    tl0 = Bbox.union([t.get_window_extent() for t in axes[0].get_xticklabels() if t.get_text()])
    x_hdr = axes[0].get_position().x0 - 0.06 / FW                                               # shared right edge of the two row headers
    fig.text(x_hdr, tl0.y1 / fig.dpi / FH, "First-pass\ndownsampling", ha="right", va="top", linespacing=1.1)   # row header left of the tick values
    # legend under the right two panels: colour = model, line = input
    x_leg = (axes[1].get_position().x0 + axes[-1].get_position().x1) / 2
    hm = [Line2D([], [], color=MODEL_C[m], lw=1.5, marker="o", ms=4) for m in data]
    leg1 = fig.legend(hm, list(data), loc="upper center", bbox_to_anchor=(x_leg, 0.2), ncol=3, frameon=False, columnspacing=1.8, handlelength=2.4)
    hs = [Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), marker="o", ms=3.6, mfc="white"), Line2D([], [], color=INK2, lw=1.5, marker="o", ms=4)]
    fig.legend(hs, ["Full image", "Full image + oracle native zoom"], loc="upper center", bbox_to_anchor=(x_leg, 0.12), ncol=2, frameon=False, columnspacing=1.8, handlelength=2.4)
    # (leg1 stays registered in fig.legends; adding it again as an artist drew the model legend twice and made its text look bold)
    if thumbs:
        ax0 = axes[0]
        tl = Bbox.union([t.get_window_extent() for t in ax0.get_xticklabels() if t.get_text()]); y_top = tl.y0 / fig.dpi - 0.05   # inches
        s_in = 0.56 / max(im.size[1] for _, im in thumbs)                                     # one scale for all: true relative size
        for x_tick, im in thumbs:
            xc = ax0.transData.transform((x_tick, 0))[0] / fig.dpi; w_in, h_in = im.size[0] * s_in, im.size[1] * s_in
            ib = fig.add_axes([(xc - w_in / 2) / FW, (y_top - h_in) / FH, w_in / FW, h_in / FH])
            ib.imshow(im, interpolation="nearest", extent=(0, im.size[0], im.size[1], 0)); ib.set_xticks([]); ib.set_yticks([])
            for k in range(28, im.size[0], 28): ib.axvline(k, color="white", lw=0.35, alpha=0.6)
            for k in range(28, im.size[1], 28): ib.axhline(k, color="white", lw=0.35, alpha=0.6)
            for sp in ib.spines.values(): sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
            fig.text(xc / FW, (y_top - h_in - 0.03) / FH, f"{im.size[0]}×{im.size[1]} px", ha="center", va="top", fontsize=5.6, color=INK2)
        fig.text(x_hdr, (y_top - 0.12) / FH, "Target region,\nas the model\nsaw it", ha="right", va="top", fontsize=5.8, color=INK2, linespacing=1.15)   # same right edge as the axis title above
    save(fig, out, "F2v3_budget_curves")


def budget_thumbs(uid_index=21, model=str(paths.MODELS / "Qwen2.5-VL-7B-Instruct")):
    """Target region of one V* image exactly as Qwen2.5-VL saw it at B = 256 / 1024 / 16384 (processor-exact resize, token-aligned)."""
    from transformers import AutoProcessor
    from zoomdecomp.data import load_vstar
    from zoomdecomp.qwen_runner import resize_like_processor, seen_dims, roi_grid, grid_to_seen_box, crop_box
    try: ip = AutoProcessor.from_pretrained(model).image_processor
    except Exception:                                                                         # weights not on disk: only .resize/.resample are used and the class defaults (bicubic) match Qwen2.5-VL
        from transformers import Qwen2VLImageProcessor; ip = Qwen2VLImageProcessor()
    s = load_vstar()[uid_index]; img = s.load_image(); W, H = img.size; thumbs = []
    for B, x_tick in ((256, 17), (1024, 4), (16384, 1)):
        rh, rw = seen_dims(W, H, B * 784); I_seen = resize_like_processor(ip, img, rh, rw)
        g, _ = roi_grid(s.bbox, W, H, rh, rw, factor=28); sb = grid_to_seen_box(g, 28); thumbs.append((x_tick, crop_box(I_seen, sb)))
    return thumbs


def fig_trajectory_raster(data, out, bench="vstar"):
    """Per-sample barcode: rows = questions wrong at the coarsest first pass, columns = first-pass budgets, colour = outcome class."""
    cats = ["Correct without zoom"] + REG + ["Not rescued"]
    colors = ["#e4e3df", RAMP[0], RAMP[1], RAMP[2], RAMP[3], "#8d8c87"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    def cat_of(r):
        if r["base"] == True: return 0
        if is_rescued_row(r): return 1 + regime_strict(r)
        return 5
    models = [m for m in data if bench in data[m][0].index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 3.0)); axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        w = data[m][0].loc[bench]; budgets = sorted(w.index.get_level_values(0).unique())[:3]
        uids = sorted(set.intersection(*[set(w.loc[B].index) for B in budgets]))
        uids = [u for u in uids if w.loc[budgets[0]].loc[u]["base_greedy"] == False]
        seq = np.array([[cat_of(w.loc[B].loc[u]) for B in budgets] for u in uids])
        order = np.lexsort(tuple(seq[:, j] for j in range(seq.shape[1] - 1, -1, -1)))  # sort by col0, then col1, ...
        seq = seq[order]
        ax.imshow(seq, aspect="auto", cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
        ds = [float(np.median(w.loc[B]["downsample"])) for B in budgets]
        ax.set_xticks(range(len(budgets))); ax.set_xticklabels([dsample_label(d) for d in ds]); ax.set_yticks([])
        ax.set_title(f"{m}\n{len(uids)} questions wrong at {dsample_label(ds[0], cap=False)}", fontsize=7)
        for sp in ax.spines.values(): sp.set_visible(False)
        for x in (0.5, 1.5): ax.axvline(x, color="white", lw=2)
    axes[0].set_ylabel("One row per question, sorted by outcome")
    axes[len(axes)//2].set_xlabel("First-pass downsampling")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, cats, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=5.8, handlelength=0.9, columnspacing=0.9)
    fig.suptitle(f"{BENCH_LABEL[bench]}: what happens to each failure as the first pass gets finer", y=1.0, fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"F10_trajectory_{bench}.pdf", bbox_inches="tight"); fig.savefig(out / f"F10_trajectory_{bench}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _pooled_shares(data, bench, min_models=2):
    """Mean strict regime shares over models at matched budgets (budget label B), with median downsample."""
    acc = {}
    for m, (w, _) in data.items():
        if bench not in w.index.get_level_values(0): continue
        g = w.loc[bench]
        for B in sorted(g.index.get_level_values(0).unique()):
            gg = g.loc[B]; ds = float(np.median(gg["downsample"]))
            if ds < 2.0: continue
            rs = rescued(gg)
            if len(rs) < 10: continue
            cnt = np.bincount(rs.apply(regime_strict, axis=1), minlength=4) / len(rs)
            acc.setdefault(B, []).append((ds, cnt, len(rs)))
    out = []
    for B, v in acc.items():
        if len(v) < min_models: continue
        out.append((float(np.mean([t[0] for t in v])), np.mean([t[1] for t in v], axis=0), int(sum(t[2] for t in v)), len(v)))
    out.sort(key=lambda t: -t[0])
    return out


def fig_regime_area(data, out):
    """Phase-diagram look: per benchmark, stacked shares R1..R4 (strict, mean over models) over the first-pass downsampling axis."""
    benches = ["vstar", "zoombench", "textvqa"]
    fig, axes = plt.subplots(1, 3, figsize=(W_TEXT, 2.15), sharey=True)
    for ax, b in zip(axes, benches):
        pts = _pooled_shares(data, b)
        if len(pts) < 2: ax.axis("off"); continue
        xs = np.array([t[0] for t in pts]); sh = np.array([t[1] for t in pts])  # (k, 4)
        cum = np.cumsum(sh, axis=1); lower = np.zeros(len(xs))
        for r in range(4):
            ax.fill_between(xs, lower, cum[:, r], color=RAMP[r], lw=0)
            ax.plot(xs, cum[:, r], color="white", lw=1.0)
            # label at both ends if the band is thick enough
            for j in (0, len(xs) - 1):
                if sh[j, r] >= 0.09:
                    ax.text(xs[j] / (1.08 if j == 0 else 1 / 1.08), (lower[j] + cum[j, r]) / 2, f"{100*sh[j, r]:.0f}", fontsize=5.8, color="white" if r >= 2 else INK, ha="left" if j == 0 else "right", va="center")
            lower = cum[:, r]
        for x_ in xs: ax.axvline(x_, color="white", lw=0.8, alpha=0.6)
        ax.set_xscale("log"); ax.set_xlim(xs.max() * 1.15, xs.min() / 1.15); ax.set_xticks(xs); ax.set_xticklabels([dsample_label(x) for x in xs]); ax.minorticks_off()
        ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50", "100"]); ax.set_title(BENCH_LABEL[b])
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[1].set_xlabel("First-pass downsampling")
    axes[0].set_ylabel("Share of zoom-rescued\nfailures (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[k]) for k in range(4)]
    fig.legend(handles, REG, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=0.9, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, out, "F9v2_regime_area")


def fig_regime_dumbbell(data, out):
    """Arrow plot: share needing native pixels (R4) from the coarsest to the finest downsampled budget, one arrow per benchmark x model."""
    rows = []
    for b in ["vstar", "zoombench", "textvqa"]:
        for m, (w, _) in data.items():
            if b not in w.index.get_level_values(0): continue
            g = w.loc[b]; pts = []
            for B in sorted(g.index.get_level_values(0).unique()):
                gg = g.loc[B]; ds = float(np.median(gg["downsample"]))
                if ds < 2.0: continue
                rs = rescued(gg)
                if len(rs) < 10: continue
                reg = rs.apply(regime_strict, axis=1); pts.append((ds, (reg == 3).mean(), (reg == 0).mean(), len(rs)))
            if len(pts) < 2: continue
            pts.sort(key=lambda t: -t[0]); rows.append((b, m, pts[0], pts[-1]))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
    y = 0; ylabels = []; last_b = None
    for b, m, p0, p1 in rows:
        if last_b is not None and b != last_b: y -= 0.6
        for ax, idx in zip(axes, (1, 2)):
            ax.annotate("", xy=(p1[idx], y), xytext=(p0[idx], y), arrowprops=dict(arrowstyle="-|>", color=MODEL_C[m], lw=1.6, mutation_scale=8))
            ax.plot(p0[idx], y, "o", color=MODEL_C[m], ms=4.5); ax.plot(p1[idx], y, "o", color=MODEL_C[m], ms=4.5, mfc="white")
        ylabels.append((y, f"{BENCH_LABEL[b]} \u00b7 {m}")); y -= 1; last_b = b
    for ax, t in zip(axes, ["% needing native pixels (R4)", "% fixed by cached tokens alone (R1)"]):
        ax.set_yticks([v for v, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=6)
        ax.set_xlim(0, 0.65); ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(["0", "20", "40", "60"]); ax.grid(axis="x"); ax.set_title(t, fontsize=7)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    fig.text(0.5, -0.03, "\u25cf Coarsest first pass (17\u00d7)   \u2192   \u25cb Finest downsampled first pass (4\u00d7; 2.8\u00d7 for InternVL3)", ha="center", va="top", fontsize=6, color=INK2)
    fig.tight_layout(); fig.savefig(out / "F9v3_regime_dumbbell.pdf", bbox_inches="tight"); fig.savefig(out / "F9v3_regime_dumbbell.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_regime_bubbles(data, out, s_max=700):
    """Dot matrix: panels = benchmark x model; rows = regimes (R4 on top); columns = first-pass budgets (coarse -> fine);
    bubble area = strict share of zoom-rescued failures, number printed inside (or beside when small)."""
    benches = ["vstar", "zoombench", "textvqa"]; models = list(data)
    fig, axes = plt.subplots(len(benches), len(models), figsize=(W_TEXT, 4.3), sharex=False, sharey=True)
    for i, b in enumerate(benches):
        for j, m in enumerate(models):
            ax = axes[i, j]; w = data[m][0]
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(length=0)
            if b not in w.index.get_level_values(0):
                ax.set_xticks([]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.55, 3.55); empty = ax
                ax.text(1.0, 0.35, "InternVL3 cannot run\nbelow one 448-px tile", ha="center", va="center", fontsize=6, color=INK2)
                continue
            g = w.loc[b]; cols = []
            for B in sorted(g.index.get_level_values(0).unique()):
                gg = g.loc[B]; ds = float(np.median(gg["downsample"]))
                if ds < 2.0: continue
                rs = rescued(gg)
                if len(rs) < 10: continue
                cnt = np.bincount(rs.apply(regime_strict, axis=1), minlength=4) / len(rs); cols.append((ds, cnt, len(rs)))
            cols.sort(key=lambda t: -t[0])
            for y in range(4): ax.axhline(y, color=GRID, lw=0.6, zorder=0)
            for x, (ds, cnt, n) in enumerate(cols):
                for k in range(4):
                    y = 3 - k; share = cnt[k]
                    ax.scatter([x], [y], s=s_max * share, color=RAMP[k], edgecolor="white", linewidth=0.6, zorder=3)
                    if share >= 0.12:
                        ax.text(x, y, f"{100*share:.0f}", ha="center", va="center", fontsize=5.6, color="white" if k >= 2 else INK, zorder=4)
                    else:
                        ax.text(x + 0.16 + 0.12 * np.sqrt(share / 0.12), y, f"{100*share:.0f}", ha="left", va="center", fontsize=5.2, color=INK2, zorder=4)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{dsample_label(ds)}\n$n$={n}" for ds, _, n in cols], fontsize=6.2, linespacing=1.25)
            ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.55, 3.55)
            ax.set_yticks(range(4)); ax.set_yticklabels(REG[::-1], fontsize=6.5)
            if i == 0: ax.set_title(m, fontsize=7.5, pad=4)
            if j == 0: ax.text(-0.42, 0.5, BENCH_LABEL[b], transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
    # bubble-size key in the otherwise empty panel (InternVL3 has no TextVQA run)
    for x, v in zip((0.0, 0.9, 2.0), (0.1, 0.25, 0.5)):
        empty.scatter([x], [2.35], s=s_max * v, color=GREY, edgecolor="white", linewidth=0.6)
        empty.text(x, 1.55, f"{100*v:.0f}%", ha="center", va="center", fontsize=6.2, color=INK2)
    empty.text(1.0, 3.3, "Bubble area = share of a\ncolumn's rescued failures", ha="center", va="center", fontsize=6, color=INK2)
    fig.tight_layout(rect=(0.02, 0.0, 1, 1)); xlabel_under(fig, list(axes[-1]), "First-pass downsampling"); save(fig, out, "F11_regime_bubbles")


def fig_coverage_lines(data, out):
    """Regime share versus target size on the first pass: one panel per regime, pooled line with Wilson 95% band, thin per-model lines."""
    from statsmodels.stats.proportion import proportion_confint
    edges = [0, 0.5, 1, 2, 4, 8, 1e9]; lab = ["<0.5", "0.5–1", "1–2", "2–4", "4–8", ">8"]
    per = {}
    for m, (w, _) in data.items():
        w = w[(w.index.get_level_values(0) != "textvqa") & ds_cells(w)]; rs = rescued(w)
        per[m] = pd.DataFrame({"reg": rs.apply(regime_strict, axis=1).values, "bin": pd.cut(rs["cov"], edges, labels=lab).astype(str).values})
    allr = pd.concat(per.values())
    n_bin = allr.groupby("bin").size().reindex(lab)
    fig, axes = plt.subplots(1, 4, figsize=(W_TEXT, 1.95), sharey=True)
    x = np.arange(len(lab))
    for k, ax in enumerate(axes):
        for m, df in per.items():
            ys = [((df[df.bin == l].reg == k).mean() if (df.bin == l).sum() >= 8 else np.nan) for l in lab]
            ax.plot(x, 100 * np.array(ys), "-", color=RAMP[k], lw=0.8, alpha=0.45)
        p = np.array([(allr[allr.bin == l].reg == k).mean() for l in lab]); nn = n_bin.values
        lo, hi = proportion_confint((p * nn).round().astype(int), nn, method="wilson")
        ax.fill_between(x, 100 * lo, 100 * hi, color=RAMP[k], alpha=0.16, lw=0)
        ax.plot(x, 100 * p, "-o", color=RAMP[k], lw=1.8, ms=4.2, mec="white", mew=0.6)
        ax.set_title(REG[k], fontsize=7.5, color=INK); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6); ax.set_ylim(0, 60); ax.grid(axis="y")

    axes[0].set_ylabel("Share of zoom-rescued\nfailures (%)")
    from matplotlib.patches import Patch
    axes[0].legend([(Patch(color=RAMP[0], alpha=0.16, lw=0), Line2D([], [], color=RAMP[0], lw=1.8, marker="o", ms=4.2)), Line2D([], [], color=RAMP[0], lw=0.8, alpha=0.6)],
                   ["Pooled, 95% CI", "Single models"], loc="upper left", frameon=False, fontsize=6, handlelength=1.6, borderaxespad=0.2)
    print("  coverage bins n:", ", ".join(f"{l}: {int(v)}" for l, v in n_bin.items()))
    fig.tight_layout(); xlabel_under(fig, list(axes), "Target size on the first pass (tokens)", fontsize=6.8); save(fig, out, "F12_coverage_lines")


def main():
    """Write every data figure of the paper to outputs/figures/ (file names as in the paper's sources)."""
    out = paths.FIGS
    data = {m: load_run(p) for m, p in RUNS.items()}; data = {m: v for m, v in data.items() if v is not None}
    try:
        thumbs = budget_thumbs()                             # needs the V* images and the Qwen2.5-VL processor
    except Exception as e:
        print("thumbnails skipped:", e); thumbs = None
    fig_budget_curves_v3(data, out, thumbs)
    fig_regime_bubbles(data, out); fig_coverage_lines(data, out); fig_sankey2(data, out, "vstar")
    fig_regime_area(data, out); fig_regime_dumbbell(data, out); fig_graded(data, out, "vstar")
    for b in ("vstar", "zoombench", "textvqa"): fig_sweep(data, out, b)
    for b in ("vstar", "zoombench"): fig_trajectory_raster(data, out, b); fig_forest(data, out, b); fig_ladder(data, out, b)
    fig_upset(data, out, "Qwen2.5-VL-7B", 256)
    from zoomdecomp import analyze_mag as _M
    _df = _M.fixed_sample(_M.load_mag(), _M.TOP); _M.fig_prescription(_df, _M.summarise(_df), out)   # same fixed sample as analyze_mag.main
    print("figures written to", out)


if __name__ == "__main__":
    main()
