"""Regime partition by question subgroup: V* direct_attributes vs relative_position; ZoomBench multiple choice vs counting.
Writes outputs/tables/subgroups.tex. Usage: python -m zoomdecomp.audit_subgroups"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued, regime_strict, ds_cells
from zoomdecomp import paths

TAB = paths.TABLES
LAB = {("vstar", "direct_attributes"): ("V$^*$", "attribute"), ("vstar", "relative_position"): ("V$^*$", "spatial"), ("zoombench", "mcq"): ("ZoomBench", "multiple choice"), ("zoombench", "blank"): ("ZoomBench", "counting")}


def main():
    sub = {}
    for m, p in RUNS.items():
        for f in glob.glob(str(Path(p) / "*_B*.jsonl")):
            if "mag_" in f: continue
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                if r["cond"] != "base": continue
                key = r["meta"].get("category") if r["bench"] == "vstar" else r["qtype"]
                sub[(r["bench"], r["uid"])] = key
    rows = []
    for m, p in RUNS.items():
        w, _ = load_run(p); w = w[(w.index.get_level_values(0) != "textvqa") & ds_cells(w)]; rs = rescued(w).copy()
        rs["reg"] = rs.apply(regime_strict, axis=1); rs["sub"] = [sub.get((b, u)) for b, _, u in rs.index]
        for (b, s), g in rs.groupby([rs.index.get_level_values(0), "sub"]):
            cnt = np.bincount(g.reg, minlength=4); rows.append(dict(model=m, bench=b, sub=s, n=len(g), c=(g["c_seen_zoom"] == True).mean(), a=((g["a1_cached_roi"] == True) | (g["a2_cached_only"] == True) | (g["a3_cached_inplace"] == True)).mean(), **{f"R{k+1}": 100 * cnt[k] / cnt.sum() for k in range(4)}))
    T = pd.DataFrame(rows); print(T.round(0).to_string(index=False))
    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule", r"Model & Data & Questions & $n$ & \rone & \rtwo & \rthree & \rfour & c fixes & a1--a3 fix \\", r"\midrule"]
    for m in RUNS:
        for key in [("vstar", "direct_attributes"), ("vstar", "relative_position"), ("zoombench", "mcq"), ("zoombench", "blank")]:
            r = T[(T.model == m) & (T.bench == key[0]) & (T["sub"] == key[1])]
            if r.empty: continue
            r = r.iloc[0]; bl, sl = LAB[key]
            lines.append(f"{m if key == ('vstar', 'direct_attributes') else ''} & {bl} & {sl} & {int(r.n)} & {r.R1:.0f} & {r.R2:.0f} & {r.R3:.0f} & {r.R4:.0f} & {100*r.c:.0f} & {100*r.a:.0f} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    from zoomdecomp.wrap_cond import wrap_text                      # condition codes -> \cond{...} (style set in preamble.tex)
    (TAB / "subgroups.tex").write_text(wrap_text("\n".join(lines) + "\n")); print("wrote", TAB / "subgroups.tex")


if __name__ == "__main__":
    main()
