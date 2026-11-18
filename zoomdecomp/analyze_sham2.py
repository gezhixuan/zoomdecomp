"""Test-retest of the regime labels under a second, independent placement of the random-region sham (zoomdecomp.rerun_sham).

For every rescued failure of the >= 2x cells: the strict label with the first placement and with the second one, the share of
labels that agree, Cohen's kappa, and the regime shares under each placement. The cohort itself uses the sham of the oracle
zoom (a failure is rescued only if the random native crop does not fix it), so membership is also recomputed with the second
placement and reported.

A second block recomputes the numbers the paper quotes (the R4 share per benchmark and budget, the fix rate of condition c
and of its sham at the coarsest budget) with the second placement, and reports the label agreement on the samples whose two
random regions differ (on a small token grid the two draws can coincide) and the direction of the label changes.

Usage: python -m zoomdecomp.analyze_sham2 [--seed 2] [--table outputs/tables/sham2.tex]"""
import argparse, collections, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued, regime_strict, ds_cells
from zoomdecomp.run_gate2 import SHAM
from zoomdecomp import paths

SHAM2 = {"Qwen2.5-VL-7B": "sham2_q25", "Qwen3-VL-8B": "sham2_q3", "InternVL3-8B": "sham2_iv"}
ROOT = paths.RUNS
GROUPS = (("V*+ZoomBench", "V$^*$+ZoomBench", lambda i: i.get_level_values(0) != "textvqa"), ("TextVQA", "TextVQA", lambda i: i.get_level_values(0) == "textvqa"))


def load_second(d, seed):
    rows = [json.loads(l) for f in sorted(glob.glob(str(d / "*.jsonl"))) if not f.endswith(".check.jsonl") for l in open(f)]
    if not rows: return None, None, None
    df = pd.DataFrame(rows); df = df[df.cond.str.endswith(f"_rand{seed}")]
    df["ok"] = df["correct"].astype(bool) & (df["gold_logodds"].fillna(-1e9) > 0)
    wide = df.pivot_table(index=["bench", "budget", "uid"], columns="cond", values="ok", aggfunc="first")
    same = df.groupby(["bench", "budget", "uid"]).apply(lambda g: g.iloc[0]["roi_grid_rand"] == g.iloc[0]["roi_grid_rand_first"])
    def covered(g):                                        # share of the first-pass token grid that the target region covers
        r = g.iloc[0]; a, wh = r["roi_grid"], r.get("grid_wh")
        return (a[2] - a[0]) * (a[3] - a[1]) / (wh[0] * wh[1]) if isinstance(wh, (list, tuple)) else float("nan")
    cover = df.groupby(["bench", "budget", "uid"]).apply(covered)
    return wide, same, cover


def kappa(a, b, k=4):
    a, b = np.asarray(a), np.asarray(b); po = (a == b).mean()
    pe = sum((a == i).mean() * (b == i).mean() for i in range(k))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def level(ds):
    """Budget level of a cell from its median first-pass downsampling."""
    return "17x" if ds > 12 else "9x" if ds > 6 else "4x" if ds > 3.5 else "2.8x"


def cell(s, b, B):
    return s[(s.index.get_level_values(0) == b) & (s.index.get_level_values(1) == B)]


def kappa_cm(M):
    """Cohen's kappa from a confusion matrix."""
    N = M.sum(); po = np.trace(M) / N; pe = (M.sum(1) * M.sum(0)).sum() / N ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def write_table(out, pooled, path):
    """Supplementary table: per benchmark group, one row per model and one pooled row."""
    num = lambda n: f"{n:,}".replace(",", "{,}")
    sh = lambda s: " / ".join(f"{100 * x:.0f}" for x in s)
    row = lambda r: (f"{r['model']} & {r['tex']} & {num(r['n1'])} & {num(r['n2'])} & {num(r['n_both'])} & {100 * r['agree']:.1f} & {r['kappa']:.2f}"
                     f" & {sh(r['s1'])} & {sh(r['s2'])} \\\\")
    L = [r"\begin{tabular}{@{}llrrrrrll@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Zoom-rescued failures} & Labels & & \multicolumn{2}{c}{\rone\ / \rtwo\ / \rthree\ / \rfour\ (\%)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){8-9}",
         r"Model & Data & first & second & both & unchanged (\%) & $\kappa$ & first placement & second placement \\"]
    for name, _, _ in GROUPS:
        rows = [r for r in out + pooled if r["data"] == name]
        if rows: L += [r"\midrule"] + [row(r) for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(L) + "\n"); print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2); ap.add_argument("--table", default=None); args = ap.parse_args()
    out = []
    r4 = collections.defaultdict(dict)                     # (bench, level) -> model -> (share 1, share 2, n1, n2)
    crate = {}                                             # (model, bench) -> c and its sham at the coarsest budget, both placements
    distinct = collections.defaultdict(lambda: np.zeros(2))  # data -> [agreeing, n] over samples whose two random regions differ
    moves = collections.defaultdict(lambda: np.zeros((4, 4), int))
    cnt = collections.defaultdict(lambda: np.zeros((2, 4), int))  # data -> label counts under each placement, pooled over models
    cells = {}                                             # (model, bench, level) -> label counts under each placement
    covs = collections.defaultdict(lambda: ([], []))       # data -> grid share covered by the target region (identical draws, distinct draws)
    for m, p in RUNS.items():
        w, _ = load_run(p); w2, same, cover = load_second(ROOT / SHAM2[m], args.seed)
        if w2 is None: print(f"{m}: no second-placement rows yet"); continue
        w = w[ds_cells(w)]
        both = w.index.intersection(w2.index)                      # first-pass failures the oracle zoom fixes (greedy), second placement available
        A = w.loc[both].copy(); B = A.copy()
        for c in SHAM: B[c + "_rand"] = w2.loc[both, f"{c}_rand{args.seed}"].values      # same targets, shams of the second placement
        r1, r2 = rescued(A), rescued(B)
        for name, tex, sel in GROUPS:
            a, b = r1[sel(r1.index)], r2[sel(r2.index)]
            if len(a) == 0: continue
            la, lb = a.apply(regime_strict, axis=1), b.apply(regime_strict, axis=1); common = la.index.intersection(lb.index)
            sa, sb = np.bincount(la, minlength=4) / len(la), np.bincount(lb, minlength=4) / len(lb)
            agree = (la.loc[common] == lb.loc[common]).mean(); k = kappa(la.loc[common], lb.loc[common])
            idt = same.reindex(common).fillna(False).astype(bool).values; ident = idt.mean()
            eq = (la.loc[common] == lb.loc[common]).values
            distinct[name] += [eq[~idt].sum(), (~idt).sum()]
            cv = cover.reindex(common).values.astype(float); covs[name][0].extend(cv[idt]); covs[name][1].extend(cv[~idt])
            for x, y in zip(la.loc[common].values, lb.loc[common].values): moves[name][x, y] += 1
            cnt[name] += np.stack([np.bincount(la, minlength=4), np.bincount(lb, minlength=4)])
            out.append(dict(model=m, data=name, tex=tex, n1=len(a), n2=len(b), n_both=len(common), agree=agree, kappa=k, s1=sa, s2=sb, ident=ident))
            print(f"{m:14s} {name:13s} rescued: {len(a)} with placement 1, {len(b)} with placement 2, {len(common)} with both | label agreement {100*agree:.1f}%  kappa {k:.2f}"
                  f" | shares R1..R4: " + " ".join(f"{100*x:.0f}" for x in sa) + "  vs  " + " ".join(f"{100*x:.0f}" for x in sb)
                  + f" | largest share difference {100*np.abs(sa-sb).max():.1f} points | same region drawn twice {100*ident:.0f}%")
            print("     confusion (rows: placement 1, columns: placement 2):"); print("     " + pd.crosstab(la.loc[common], lb.loc[common]).to_string().replace("\n", "\n     "))
        # the numbers the paper quotes, per cell, under each placement
        l1, l2 = r1.apply(regime_strict, axis=1), r2.apply(regime_strict, axis=1)
        for (b, Bg), g in A.groupby(level=[0, 1]):
            lv = level(float(np.median(g["downsample"]))); i1, i2 = cell(l1, b, Bg), cell(l2, b, Bg)
            if len(i1) < 10: continue
            r4[(b, lv)][m] = (100 * (i1 == 3).mean(), 100 * (i2 == 3).mean(), len(i1), len(i2))
            cells[(m, b, lv)] = (np.bincount(i1, minlength=4), np.bincount(i2, minlength=4))
            if lv == "17x":
                s1, s2 = cell(r1, b, Bg), cell(r2, b, Bg); f = lambda s, col: 100 * s[col].fillna(False).astype(bool).mean()
                crate[(m, b)] = (f(s1, "c_seen_zoom"), f(s1, "c_seen_zoom_rand"), len(s1), f(s2, "c_seen_zoom"), f(s2, "c_seen_zoom_rand"), len(s2))
    if not out: return
    T = pd.DataFrame(out)
    for name, g in T.groupby("data"):
        tot = g.n_both.sum(); print(f"\n[{name}] pooled over models: label agreement {100*(g.agree*g.n_both).sum()/tot:.1f}% on {tot} failures rescued under both placements; "
                                    f"cohort overlap {100*tot/ (g.n1.sum()):.1f}% of the placement-1 cohort; largest share difference in any model {100*max(np.abs(a-b).max() for a, b in zip(g.s1, g.s2)):.1f} points")
    print("\nR4 share (%), placement 1 -> placement 2, per benchmark and budget")
    for key in sorted(r4):
        v = r4[key]; m1, m2 = np.mean([x[0] for x in v.values()]), np.mean([x[1] for x in v.values()])
        print(f"  {key[0]:10s} {key[1]:5s} mean of models {m1:.1f} -> {m2:.1f} | " + "  ".join(f"{m.split('-')[0]} {a:.1f}->{b:.1f} (n {n1}->{n2})" for m, (a, b, n1, n2) in v.items()))
    print("\nCondition c at the coarsest budget: fixed %, sham %, n  (placement 1 | placement 2)")
    for (m, b), v in sorted(crate.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print("  %-14s %-10s c %.1f sham %.1f n=%d | c %.1f sham %.1f n=%d" % ((m, b) + v))
    for b in ("vstar", "zoombench", "textvqa"):
        vs = [v for (m, bb), v in crate.items() if bb == b]
        if vs: print("  %-10s c range %.0f-%.0f -> %.0f-%.0f | sham range %.0f-%.0f -> %.0f-%.0f" % (b, min(v[0] for v in vs), max(v[0] for v in vs), min(v[3] for v in vs), max(v[3] for v in vs),
                                                                                             min(v[1] for v in vs), max(v[1] for v in vs), min(v[4] for v in vs), max(v[4] for v in vs)))
    print("\nLabel agreement on the samples whose two random regions differ, and direction of the changed labels")
    for name, (e, n) in distinct.items():
        mv = moves[name]
        print(f"  {name:13s} {100 * e / n:.1f}% of {int(n)} | changed labels: {int(np.tril(mv, -1).sum())} to a cheaper regime, {int(np.triu(mv, 1).sum())} to a more expensive one")
        ci, cd = covs[name]
        print(f"  {'':13s} same region drawn twice: {len(ci)} of {len(ci) + len(cd)} = {100 * len(ci) / (len(ci) + len(cd)):.1f}% | median share of the token grid covered by the target region: "
              f"{100 * np.nanmedian(ci):.0f}% for these, {100 * np.nanmedian(cd):.0f}% for the others (grid size missing for {int(np.isnan(ci).sum() + np.isnan(cd).sum())})")
    print("\nPer cell: label counts R1..R4 (placement 1 | placement 2), largest share difference in points, empty regimes")
    worst = collections.defaultdict(float)
    for (m, b, lv), (c1, c2) in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        d = 100 * np.abs(c1 / c1.sum() - c2 / c2.sum()); worst[b] = max(worst[b], d.max())
        empty = " ".join(f"R{i + 1} empty under placement {k}" for k, c in ((1, c1), (2, c2)) for i in range(4) if c[i] == 0)
        print(f"  {m:14s} {b:10s} {lv:5s} {' '.join(map(str, c1))} (n {c1.sum()}) | {' '.join(map(str, c2))} (n {c2.sum()}) | {d.max():.1f} (R{d.argmax() + 1}) {empty}")
    print("  largest share difference of any cell, per benchmark: " + ", ".join(f"{b} {v:.1f}" for b, v in worst.items()))
    pooled = []
    print("\nPooled over models (the pooled rows of the table)")
    for name, tex, _ in GROUPS:
        if name not in moves: continue
        M, c = moves[name], cnt[name]
        r = dict(model="pooled", data=name, tex=tex, n1=int(c[0].sum()), n2=int(c[1].sum()), n_both=int(M.sum()), agree=np.trace(M) / M.sum(), kappa=kappa_cm(M),
                 s1=c[0] / c[0].sum(), s2=c[1] / c[1].sum()); pooled.append(r)
        print(f"  {name:13s} rescued: {r['n1']} / {r['n2']} / {r['n_both']} | unchanged {int(np.trace(M))} = {100 * r['agree']:.2f}%  kappa {r['kappa']:.3f} | shares R1..R4: "
              + " ".join(f"{100 * x:.1f}" for x in r["s1"]) + "  vs  " + " ".join(f"{100 * x:.1f}" for x in r["s2"])
              + f" | largest share difference {100 * np.abs(r['s1'] - r['s2']).max():.1f} points")
    if args.table: write_table(out, pooled, args.table)


if __name__ == "__main__":
    main()
