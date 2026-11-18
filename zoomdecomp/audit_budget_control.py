"""Is the budget effect an artefact of the crop budget K or of cohort turnover?
Writes tables/budget_common.tex (questions rescued at both 17x and 4x) and tables/budget_k256.tex (K fixed at 256 tokens).
Usage: python -m zoomdecomp.audit_budget_control"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS
from zoomdecomp import paths

TAB = paths.TABLES
BL = {"vstar": "V$^*$", "zoombench": "ZoomBench"}


def main():
    R = pd.read_pickle(str(paths.ANALYSIS / "review_audit/rescued_rows.pkl")); R = R[R.crit == "primary"].copy()
    meta = []
    for m, p in RUNS.items():
        for f in glob.glob(p + "/*.jsonl"):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["cond"] != "d_native_zoom": continue
                g = r["roi_grid"]; meta.append(dict(model=m, bench=r["bench"], budget=r["budget"], uid=r["uid"], K=r["K"], roi_tok=(g[2] - g[0]) * (g[3] - g[1])))
    R = R.merge(pd.DataFrame(meta), on=["model", "bench", "budget", "uid"], how="left")
    R["r4"] = (R.strict == 3).astype(int); R["r1"] = (R.strict == 0).astype(int)
    X = R[R.bench.isin(["vstar", "zoombench"])]
    # ---- common cohort
    rows = []
    for (m, b), g in X.groupby(["model", "bench"]):
        a = g[g.budget == 256].set_index("uid"); c = g[g.budget == 1024].set_index("uid"); common = a.index.intersection(c.index)
        t = pd.crosstab(a.loc[common, "strict"], c.loc[common, "strict"]).reindex(index=range(4), columns=range(4), fill_value=0)
        rows.append(dict(model=m, bench=b, n=len(common), n17=len(a), n4=len(c), r4_17=a.loc[common, "r4"].sum(), r4_4=c.loc[common, "r4"].sum(), r1_17=a.loc[common, "r1"].sum(), r1_4=c.loc[common, "r1"].sum(),
                         down=int(t.loc[3, [0, 1, 2]].sum()), up=int(t.loc[[0, 1, 2], 3].sum()), same=int(np.trace(t.values))))
    T = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r" & & & & \multicolumn{2}{c}{\rfour\ share (\%)} & \multicolumn{2}{c}{\rone\ share (\%)} & \multicolumn{2}{c}{label moves} \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}", r"Model & Data & rescued at 17$\times$ / 4$\times$ & both & at 17$\times$ & at 4$\times$ & at 17$\times$ & at 4$\times$ & \rfour$\to$cheaper & cheaper$\to$\rfour \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['model']} & {BL[r['bench']]} & {r['n17']} / {r['n4']} & {r['n']} & {100 * r['r4_17'] / r['n']:.0f} & {100 * r['r4_4'] / r['n']:.0f} & {100 * r['r1_17'] / r['n']:.0f} & {100 * r['r1_4'] / r['n']:.0f} & {r['down']} & {r['up']} \\\\")
    n = T.n.sum()
    lines += [r"\midrule", f"pooled & & {T.n17.sum()} / {T.n4.sum()} & {n} & {100 * T.r4_17.sum() / n:.0f} & {100 * T.r4_4.sum() / n:.0f} & {100 * T.r1_17.sum() / n:.0f} & {100 * T.r1_4.sum() / n:.0f} & {T.down.sum()} & {T.up.sum()} \\\\", r"\bottomrule", r"\end{tabular}"]
    (TAB / "budget_common.tex").write_text("\n".join(lines) + "\n")
    print(T.to_string(index=False)); print(f"pooled n={n}: R4 {100 * T.r4_17.sum() / n:.0f} -> {100 * T.r4_4.sum() / n:.0f}; moves down {T.down.sum()} up {T.up.sum()}; same {T.same.sum()}")
    # ---- K fixed at 256 tokens
    S = X[X.K == 256]
    g = S.groupby(["bench", "budget"]).agg(n=("uid", "size"), R1=("r1", "mean"), R2=("strict", lambda s: (s == 1).mean()), R3=("strict", lambda s: (s == 2).mean()), R4=("r4", "mean"))
    ds = {256: "17$\\times$", 1024: "4$\\times$", 4096: "2.8$\\times$"}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Data & First pass & $n$ & \rone & \rtwo & \rthree & \rfour \\", r"\midrule"]
    for (b, B), r in g.iterrows():
        lines.append(f"{BL[b]} & {ds[B]} & {int(r.n)} & {100 * r.R1:.0f} & {100 * r.R2:.0f} & {100 * r.R3:.0f} & {100 * r.R4:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "budget_k256.tex").write_text("\n".join(lines) + "\n"); print(g.round(2).to_string())
    print("share of rescues with K=256 by budget:", X.groupby("budget").K.apply(lambda s: round((s == 256).mean(), 2)).to_dict())


if __name__ == "__main__":
    main()
