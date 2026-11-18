"""What the partition buys: a budget-stratified zoom policy evaluated as a what-if on the existing outputs.
On the zoom-rescued cohort (V* + ZoomBench, coarsest budget), compare
  always acquire      : native crop for every failure (100% acquisitions, fixes everything by selection)
  re-tokenize first   : run c (seen pixels, upsampled); stop if the model's answer confidence p_max >= tau, else acquire native pixels
  ladder              : a2 (cached tokens only) -> c -> d with the same confidence stop
Confidence p_max is the softmax over the option/digit log-probabilities at the first answer position (no gold needed).
Writes outputs/tables/policy.tex. Usage: python -m zoomdecomp.analyze_policy"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued
from zoomdecomp import paths

TAB = paths.TABLES
TAUS = [0.5, 0.7, 0.8, 0.9, 0.95]


def pmax(lp):
    if not lp: return np.nan
    v = np.array([x for x in lp.values() if x is not None], dtype=float)
    if v.size == 0: return np.nan
    v = np.exp(v - v.max()); return float(v.max() / v.sum())


def main():
    rows = []
    for m, p in RUNS.items():
        w, _ = load_run(p); w = w[(w.index.get_level_values(0) != "textvqa") & (w.index.get_level_values(1) == 256)]
        rs = rescued(w); keep = set(rs.index)
        rec = {}
        for f in glob.glob(str(Path(p) / "*_B256.jsonl")):
            if "mag_" in f or "ground_" in f: continue
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                k = (r["bench"], r["budget"], r["uid"])
                if k in keep and r["cond"] in ("a2_cached_only", "c_seen_zoom"):
                    rec[(k, r["cond"])] = (bool(r["correct"]) and (r.get("gold_logodds") or -1e9) > 0, pmax(r.get("lp")))
        n = len(keep)
        for tau in TAUS:
            # re-tokenize first
            stop_c = np.array([rec.get((k, "c_seen_zoom"), (False, np.nan))[1] >= tau for k in keep]); ok_c = np.array([rec.get((k, "c_seen_zoom"), (False, np.nan))[0] for k in keep])
            fixed = (stop_c & ok_c) | (~stop_c)          # acquired samples are fixed by selection (d fixes every cohort member)
            rows.append(dict(model=m, policy="re-tokenize first", tau=tau, n=n, acq=100 * (~stop_c).mean(), fixed=100 * fixed.mean(), enc=100 * (1 + (~stop_c).mean())))
            # ladder a2 -> c -> d
            stop_a = np.array([rec.get((k, "a2_cached_only"), (False, np.nan))[1] >= tau for k in keep]); ok_a = np.array([rec.get((k, "a2_cached_only"), (False, np.nan))[0] for k in keep])
            fixed_l = (stop_a & ok_a) | (~stop_a & stop_c & ok_c) | (~stop_a & ~stop_c)
            rows.append(dict(model=m, policy="cached, then re-tokenize, then acquire", tau=tau, n=n, acq=100 * (~stop_a & ~stop_c).mean(), fixed=100 * fixed_l.mean(), enc=100 * ((~stop_a).mean() + (~stop_a & ~stop_c).mean())))
        rows.append(dict(model=m, policy="always acquire", tau=np.nan, n=n, acq=100.0, fixed=100.0, enc=100.0))
        rows.append(dict(model=m, policy="never acquire (c only)", tau=np.nan, n=n, acq=0.0, fixed=100 * np.mean([rec.get((k, "c_seen_zoom"), (False, np.nan))[0] for k in keep]), enc=100.0))
    T = pd.DataFrame(rows); pd.set_option("display.width", 200); print(T.round(1).to_string(index=False))
    def table(taus, ladder, name):
        lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Policy & acquisitions (\%) & encoder calls (\%) & rescues kept (\%) \\", r"\midrule"]
        for m in RUNS:
            sub = T[T.model == m]
            lines.append(f"{m} & always acquire & 100 & 100 & 100 \\\\")
            r = sub[sub.policy == "never acquire (c only)"].iloc[0]; lines.append(f" & re-tokenize only & 0 & 100 & {r.fixed:.0f} \\\\")
            for tau in taus:
                r = sub[(sub.policy == "re-tokenize first") & (sub.tau == tau)].iloc[0]; lines.append(f" & re-tokenize, acquire if $p_{{\\max}}<{tau}$ & {r.acq:.0f} & {r.enc:.0f} & {r.fixed:.0f} \\\\")
            if ladder:
                for tau in taus:
                    r = sub[(sub.policy == "cached, then re-tokenize, then acquire") & (sub.tau == tau)].iloc[0]; lines.append(f" & cached $\\to$ re-tokenize $\\to$ acquire ($p_{{\\max}}<{tau}$) & {r.acq:.0f} & {r.enc:.0f} & {r.fixed:.0f} \\\\")
            lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}"]
        TAB.mkdir(parents=True, exist_ok=True); from zoomdecomp.wrap_cond import wrap_text; (TAB / name).write_text(wrap_text("\n".join(lines) + "\n")); print("wrote", TAB / name)
    def table_main(tau, name):
        """Main-text table, one row per model, sized to fit a column at the table font without scaling. Cells that are constant by
        construction stay in the caption and in policy_full.tex: always acquiring is 100 in every column, re-tokenizing alone
        acquires nothing, and encoder calls are 100 plus the acquisitions."""
        lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Re-tokenize only & \multicolumn{2}{c}{Re-tokenize, then acquire} \\",
                 r"\cmidrule(lr){2-2}\cmidrule(l){3-4}", r"Model & rescues kept & acquisitions & rescues kept \\", r"\midrule"]
        for m in RUNS:
            sub = T[T.model == m]; c = sub[sub.policy == "never acquire (c only)"].iloc[0]; r = sub[(sub.policy == "re-tokenize first") & (sub.tau == tau)].iloc[0]
            assert abs(r.enc - (100 + r.acq)) < 1e-6                                   # the relation the caption states
            lines.append(f"{m} & {c.fixed:.0f} & {r.acq:.0f} & {r.fixed:.0f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        from zoomdecomp.wrap_cond import wrap_text; (TAB / name).write_text(wrap_text("\n".join(lines) + "\n")); print("wrote", TAB / name)
    table_main(0.9, "policy.tex"); table((0.7, 0.8, 0.9, 0.95), True, "policy_full.tex")
    table((0.9,), False, "policy_long.tex")        # the main table with every row, shown in the supplement
    T.to_csv(str(paths.ANALYSIS / "policy_whatif.csv"), index=False)


if __name__ == "__main__":
    main()
