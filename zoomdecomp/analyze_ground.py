"""Self-grounding analysis. Reads runs/<run>/ground_<bench>_B<B>.jsonl (run_ground.py).
A. Does the model know where to look? hit rates of its own box vs the GT region, by budget, split by first-pass outcome and by the
   GT-based strict regime of the zoom-rescued cohort.
B. Does the decomposition survive the model's own box? strict regime partition of the rescued cohort with GT regions vs own regions,
   plus the share of the cohort that no own-region arm (not even the native crop) fixes.
Writes outputs/tables/grounding.tex. Usage: python -m zoomdecomp.analyze_ground"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued, regime_strict
from zoomdecomp import paths

TAB = paths.TABLES
BL = {"vstar": "V$^*$", "zoombench": "ZoomBench"}
REG = ["R1", "R2", "R3", "R4"]
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)


def load():
    rows = []
    for m, p in RUNS.items():
        for f in glob.glob(str(Path(p) / "ground_*.jsonl")):
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                r["model_short"] = m; rows.append(r)
    return pd.DataFrame(rows) if rows else None


def own_regime(w):
    """strict regime from the g_ columns of one sample (same hierarchy as regime_strict); 4 = nothing fixes (incl. native crop)."""
    def ok(c): v = w.get(f"g_{c}"); return bool(v) if v is not None and v == v else False
    def hit(c): return ok(c) and not ok(c + "_rand")
    if not ok("d_native_zoom"): return 4
    for k, tier in enumerate([("a1_cached_roi", "a2_cached_only", "a3_cached_inplace"), ("b1_mask_bg", "b2_seen_same"), ("c_seen_zoom",)]):
        if any(hit(c) for c in tier): return k
    return 3


def main():
    df = load()
    if df is None: print("no ground files yet"); return
    G = df[df.cond == "ground"].copy()
    G["hit"] = G.iou >= 0.5; G["nobox"] = G.bbox_pred.isna()
    # GT-based cohort + regime, first-pass correctness
    info = {}
    for m, p in RUNS.items():
        w, _ = load_run(p)
        for (b, B), g in w.groupby(level=[0, 1]):
            rs = rescued(g); reg = rs.apply(regime_strict, axis=1)
            for uid in g.index.get_level_values(2):
                info[(m, b, B, uid)] = dict(base_ok=bool(g.loc[(b, B, uid), "base"] == True), rescued=uid in rs.index.get_level_values(2), regime=int(reg.loc[(b, B, uid)]) if uid in reg.index.get_level_values(2) else None)
    G["base_ok"] = [info.get((r.model_short, r.bench, r.budget, r.uid), {}).get("base_ok") for r in G.itertuples()]
    G["regime"] = [info.get((r.model_short, r.bench, r.budget, r.uid), {}).get("regime") for r in G.itertuples()]
    print("=" * 100); print("A. own-box quality (all samples): IoU>=0.5 | covers GT centre | no box / refusal | median IoU | median box area (frac of image)")
    A = G.groupby(["model_short", "bench", "budget"]).agg(n=("uid", "size"), hit=("hit", "mean"), cover=("covers_gt_center", "mean"), nobox=("nobox", "mean"), iou_med=("iou", "median"), area_med=("pred_area_frac", "median")).round(2)
    print(A.to_string())
    print("\n-- by first-pass outcome (base correct vs wrong):"); print(G.groupby(["model_short", "bench", "budget", "base_ok"]).agg(n=("uid", "size"), hit=("hit", "mean"), cover=("covers_gt_center", "mean")).round(2).to_string())
    print("\n-- zoom-rescued cohort, by GT-based strict regime (do near-sighted samples get localised?):")
    Rq = G[G.regime.notna()].copy(); Rq["regime"] = Rq.regime.astype(int)
    print(Rq.groupby(["model_short", "budget", "regime"]).agg(n=("uid", "size"), hit=("hit", "mean"), cover=("covers_gt_center", "mean"), nobox=("nobox", "mean")).round(2).to_string())
    # B. partition with own regions
    print("\n" + "=" * 100); print("B. zoom-rescued cohort: strict partition with GT regions vs the model's own region (5th class: nothing fixes, incl. native crop of own box)")
    arms = df[df.cond.str.startswith("g_")].copy(); arms["ok"] = arms["correct"].astype(bool) & (arms["gold_logodds"].fillna(-1e9) > 0)
    W = arms.pivot_table(index=["model_short", "bench", "budget", "uid"], columns="cond", values="ok", aggfunc="first")
    rows = []
    for (m, b, B), g in W.groupby(level=[0, 1, 2]):
        own = g.apply(own_regime, axis=1); gt = pd.Series([info[(m, b, B, u)]["regime"] for u in g.index.get_level_values(3)], index=g.index)
        cnt_own = np.bincount(own, minlength=5); cnt_gt = np.bincount(gt.astype(int), minlength=4)
        cover = Rq[(Rq.model_short == m) & (Rq.bench == b) & (Rq.budget == B)].covers_gt_center.mean()
        rows.append(dict(model=m, bench=b, budget=B, n=len(g), n_cohort=sum(1 for k, v in info.items() if k[0] == m and k[1] == b and k[2] == B and v["rescued"]), cover=cover,
                         **{f"gt_{r}": 100 * cnt_gt[i] / cnt_gt.sum() for i, r in enumerate(REG)}, **{f"own_{r}": 100 * cnt_own[i] / len(g) for i, r in enumerate(REG)}, own_none=100 * cnt_own[4] / len(g),
                         same_label=100 * (own.values == gt.values).mean()))
    T = pd.DataFrame(rows); print(T.round(0).to_string(index=False))
    lines = [r"\begin{tabular}{llrrrrrrrrrrrrr}", r"\toprule", r" & & & \multicolumn{2}{c}{own box} & \multicolumn{4}{c}{GT region} & \multicolumn{5}{c}{model's own region} & \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(lr){10-14}", r"Model & Data & First pass & covers target & $n$ & \rone & \rtwo & \rthree & \rfour & \rone & \rtwo & \rthree & \rfour & none & same label \\", r"\midrule"]
    dsl = {256: "17$\\times$", 1024: "4$\\times$"}
    for r in T.sort_values(["model", "bench", "budget"]).itertuples():
        lines.append(f"{r.model} & {BL[r.bench]} & {dsl.get(r.budget, r.budget)} & {100*r.cover:.0f}\\% & {r.n} & {r.gt_R1:.0f} & {r.gt_R2:.0f} & {r.gt_R3:.0f} & {r.gt_R4:.0f} & {r.own_R1:.0f} & {r.own_R2:.0f} & {r.own_R3:.0f} & {r.own_R4:.0f} & {r.own_none:.0f} & {r.same_label:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True); (TAB / "grounding.tex").write_text("\n".join(lines) + "\n"); print("wrote", TAB / "grounding.tex")
    A.to_csv(str(paths.ANALYSIS / "grounding_quality.csv")); T.to_csv(str(paths.ANALYSIS / "grounding_partition.csv"), index=False)


if __name__ == "__main__":
    main()
