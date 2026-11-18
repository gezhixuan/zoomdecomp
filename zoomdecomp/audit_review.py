"""Audit of the per-sample outputs, and the supplement tables it produces.
Cohort variants
  primary              : first-pass answer wrong (greedy), native zoom robust-correct, random native crop not robust-correct
  confidence-filtered  : earlier definition (base not robust-correct, d robust-correct)
  greedy-only          : base greedy wrong, d greedy correct, random native crop greedy wrong; arms and shams judged by greedy correctness
Sections
  A  cohort composition and the strict partition under each variant          -> tables/cohort_variants.tex
  B  R4 decomposition (all cheaper arms fail vs. cheaper arm succeeds but is non-specific) -> tables/r4_decomp.tex
  C  strict vs lenient partition                                              -> tables/lenient_strict.tex
  D  paired differences with question-clustered bootstrap CIs                  -> tables/paired.tex
  E  budget vs target coverage, logistic regression with bench/model FE        -> tables/coverage_logit.tex
  F  processed second-look sizes and token counts per arm (input invariants)   -> tables/dims.tex
Usage: python -m zoomdecomp.audit_review
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued
from zoomdecomp import paths

CHEAP = ["a1_cached_roi", "a2_cached_only", "a3_cached_inplace", "b1_mask_bg", "b2_seen_same", "c_seen_zoom"]
TIERS = [("a1_cached_roi", "a2_cached_only", "a3_cached_inplace"), ("b1_mask_bg", "b2_seen_same"), ("c_seen_zoom",)]
REG = ["R1", "R2", "R3", "R4"]
BL = {"vstar": "V$^*$", "zoombench": "ZoomBench", "textvqa": "TextVQA", "main": "V$^*$+ZoomBench"}
TAB = paths.TABLES
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)


def load_df(path):
    rows = []
    for f in sorted(glob.glob(str(Path(path) / "*.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    df["greedy"] = df["correct"].astype(bool)
    return df


def wide_greedy(df):
    w = df.pivot_table(index=["bench", "budget", "uid"], columns="cond", values="greedy", aggfunc="first")
    first = df.groupby(["bench", "budget", "uid"]).agg(img=("img_wh", "first"), seen=("seen_wh", "first"), cov=("bbox_seen_tokens", "first")).loc[w.index]
    w["downsample"] = [(a[0] * a[1]) / (b[0] * b[1]) for a, b in zip(first["img"], first["seen"])]
    w["cov"] = first["cov"].values
    return w


def val(r, a):
    v = r.get(a)
    return bool(v) if (v is not None and pd.notna(v)) else False


def regime(r, strict=True):
    for k, tier in enumerate(TIERS):
        for a in tier:
            if val(r, a) and (not strict or not val(r, a + "_rand")):
                return k
    return 3


def rows_for(model, rs, tag):
    out = pd.DataFrame(index=rs.index)
    out["model"] = model; out["crit"] = tag
    out["bench"] = rs.index.get_level_values(0); out["budget"] = rs.index.get_level_values(1); out["uid"] = rs.index.get_level_values(2)
    out["downsample"] = rs["downsample"].values; out["cov"] = rs["cov"].values
    out["strict"] = rs.apply(regime, axis=1).values; out["lenient"] = rs.apply(lambda r: regime(r, strict=False), axis=1).values
    out["any_cheaper_ok"] = rs.apply(lambda r: any(val(r, a) for a in CHEAP), axis=1).values
    out["d_rand_ok"] = rs.apply(lambda r: val(r, "d_native_zoom_rand"), axis=1).values
    out["base_greedy_ok"] = rs["base_greedy"].astype(bool).values if "base_greedy" in rs else False
    for a in ["a1_cached_roi", "a2_cached_only", "a3_cached_inplace", "b1_mask_bg", "b2_seen_same", "b3_native_same", "c_seen_zoom", "d_native_zoom"]:
        out[a] = rs.apply(lambda r: val(r, a), axis=1).values
        out[a + "_rand"] = rs.apply(lambda r: val(r, a + "_rand"), axis=1).values
    return out.reset_index(drop=True)


def share_table(R, by, col="strict"):
    t = R.groupby(by + [col]).size().unstack(col, fill_value=0).reindex(columns=range(4), fill_value=0)
    n = t.sum(axis=1)
    pct = (100 * t.div(n, axis=0)).round(0).astype(int)
    pct.columns = REG; pct.insert(0, "n", n)
    return pct


def dsl(x):
    return f"{x:.0f}$\\times$" if x >= 3 else ("native" if x < 1.02 else f"{x:.1f}$\\times$")


def tex(lines, name):
    from zoomdecomp.wrap_cond import wrap_text                      # condition codes -> \cond{...} (style set in preamble.tex)
    (TAB / name).write_text(wrap_text("\n".join(lines) + "\n")); print("wrote", TAB / name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(paths.ANALYSIS / "review_audit"))
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    R_all, DIMS, EXCL = [], [], []
    for m, p in RUNS.items():
        df = load_df(p); wo, _ = load_run(p); wg = wide_greedy(df)
        # first pass downsampled >= 2x, judged per (bench, budget) cell by the median downsampling of the cell, the same rule
        # as figures.py / make_tables.py (a per-sample rule dropped small images from the 2.8x and 4x cells and made n disagree)
        ds = wo["downsample"].groupby(level=[0, 1]).transform("median") >= 2
        prim = rescued(wo[ds]); conf = rescued(wo[ds], "confidence-filtered")
        R_all.append(rows_for(m, prim, "primary")); R_all.append(rows_for(m, conf, "confidence-filtered"))
        # exclusions from the confidence-filtered cohort
        c_only = conf["base_greedy"] == True; nonspec = (conf["d_native_zoom_rand"] == True)
        for b in ["vstar", "zoombench", "textvqa"]:
            cb = conf.index.get_level_values(0) == b
            if cb.sum():
                EXCL.append(dict(model=m, bench=b, n_conf=int(cb.sum()), conf_only=int((c_only & cb).sum()), nonspec=int((nonspec & cb & ~c_only).sum()), n_primary=int((prim.index.get_level_values(0) == b).sum())))
        gres = wg[ds & (wg["base"] == False) & (wg["d_native_zoom"] == True) & (wg["d_native_zoom_rand"] != True)]
        R_all.append(rows_for(m, gres, "greedy-only"))
        d2 = df.copy(); d2["model"] = m
        d2["sw"] = d2["second_wh"].apply(lambda x: x[0] if isinstance(x, list) else np.nan); d2["sh"] = d2["second_wh"].apply(lambda x: x[1] if isinstance(x, list) else np.nan)
        DIMS.append(d2[["model", "bench", "budget", "uid", "cond", "n_img_tokens_first", "n_img_tokens", "sw", "sh", "n_tokens_in", "roi_tokens", "tiles", "K"]])
    R = pd.concat(R_all, ignore_index=True); D = pd.concat(DIMS, ignore_index=True); EX = pd.DataFrame(EXCL)
    R.to_csv(out / "rescued_rows.csv", index=False)
    log = open(out / "audit.txt", "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")
    Rp = R[R.crit == "primary"].copy(); Rc = R[R.crit == "confidence-filtered"]; Rg = R[R.crit == "greedy-only"]
    main_b = Rp.bench.isin(["vstar", "zoombench"])

    # ---------------------------------------------------------------- A
    P("=" * 100); P("A. COHORT: exclusions from the confidence-filtered cohort, and the strict partition under the three variants (first pass >= 2x)")
    EX["conf_only_pct"] = (100 * EX.conf_only / EX.n_conf).round(1); EX["nonspec_pct"] = (100 * EX.nonspec / EX.n_conf).round(1); P(EX.to_string(index=False))
    def three(bsel_p, bsel_c, bsel_g):
        return pd.concat({"primary": share_table(Rp[bsel_p], ["model"]), "confidence-filtered": share_table(Rc[bsel_c], ["model"]), "greedy-only": share_table(Rg[bsel_g], ["model"])}, axis=1)
    T_main = three(main_b, Rc.bench.isin(["vstar", "zoombench"]), Rg.bench.isin(["vstar", "zoombench"]))
    T_tv = three(Rp.bench == "textvqa", Rc.bench == "textvqa", Rg.bench == "textvqa")
    P("[pooled V*+ZB]"); P(T_main.to_string()); P("[TextVQA]"); P(T_tv.to_string())
    for b in ["vstar", "zoombench", "textvqa"]:
        P(f"[{b}, per budget, pooled models]"); P(pd.concat({"primary": share_table(Rp[Rp.bench == b], ["budget"]), "confidence-filtered": share_table(Rc[Rc.bench == b], ["budget"]), "greedy-only": share_table(Rg[Rg.bench == b], ["budget"])}, axis=1).to_string())
    lines = [r"\begin{tabular}{llrrrrrrrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{5}{c}{primary (ours)} & \multicolumn{5}{c}{confidence-filtered} & \multicolumn{5}{c}{greedy-only} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}",
             r"Model & Data & $n$ & \rone & \rtwo & \rthree & \rfour & $n$ & \rone & \rtwo & \rthree & \rfour & $n$ & \rone & \rtwo & \rthree & \rfour \\", r"\midrule"]
    for key, T in [("main", T_main), ("textvqa", T_tv)]:
        for m in RUNS:
            if m not in T.index: continue
            cells = []
            for v in ["primary", "confidence-filtered", "greedy-only"]:
                r = T[v].loc[m]; cells += [str(int(r["n"]))] + [str(int(r[k])) for k in REG]
            lines.append(f"{m} & {BL[key]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "cohort_variants.tex")
    # exclusion summary table rows (appended to r4 table below)

    # ---------------------------------------------------------------- B
    P("\n" + "=" * 100); P("B. R4 DECOMPOSITION (primary cohort). nonspecific = some cheaper arm (a1-a3,b1,b2,c) correct but its sham also correct")
    r4 = Rp[Rp.strict == 3].copy(); r4["type"] = np.where(r4.any_cheaper_ok, "nonspecific", "all_fail")
    g = r4.groupby(["model", "bench", "type"]).size().unstack(fill_value=0).reindex(columns=["all_fail", "nonspecific"], fill_value=0); g["n_R4"] = g.sum(axis=1)
    g["pct_all_fail"] = (100 * g["all_fail"] / g.n_R4).round(0).astype(int); P(g.to_string())
    gp = r4[r4.bench.isin(["vstar", "zoombench"])].groupby(["model", "type"]).size().unstack(fill_value=0).reindex(columns=["all_fail", "nonspecific"], fill_value=0); gp["n_R4"] = gp.sum(axis=1); gp["pct_all_fail"] = (100 * gp.all_fail / gp.n_R4).round(0).astype(int)
    P("[pooled V*+ZB]"); P(gp.to_string())
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{confidence-filtered cohort} & \multicolumn{4}{c}{primary cohort} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
             r"Model & Data & $n$ & conf.-only & non-specific $d$ & $n$ & $n_{\rfour}$ & all cheaper conditions fail & cheaper condition non-specific \\", r"\midrule"]
    for m in RUNS:
        for b in ["vstar", "zoombench", "textvqa"]:
            e = EX[(EX.model == m) & (EX.bench == b)]
            if e.empty: continue
            e = e.iloc[0]; gg = g.loc[(m, b)] if (m, b) in g.index else None
            lines.append(f"{m} & {BL[b]} & {e.n_conf} & {e.conf_only} ({e.conf_only_pct:.0f}\\%) & {e.nonspec} ({e.nonspec_pct:.0f}\\%) & {e.n_primary} & {int(gg.n_R4)} & {int(gg.all_fail)} ({int(gg.pct_all_fail)}\\%) & {int(gg.nonspecific)} ({100 - int(gg.pct_all_fail)}\\%) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "r4_decomp.tex")

    # ---------------------------------------------------------------- C
    P("\n" + "=" * 100); P("C. STRICT vs LENIENT (primary cohort)")
    S = share_table(Rp, ["model", "bench", "budget"]); Lt = share_table(Rp, ["model", "bench", "budget"], col="lenient"); P(pd.concat({"strict": S, "lenient": Lt}, axis=1).to_string())
    dsmed = Rp.groupby(["model", "bench", "budget"]).downsample.median()
    lines = [r"\begin{tabular}{lllrrrrrrrrr}", r"\toprule", r" & & & & \multicolumn{4}{c}{strict} & \multicolumn{4}{c}{lenient} \\", r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
             r"Model & Data & First pass & $n$ & \rone & \rtwo & \rthree & \rfour & \rone & \rtwo & \rthree & \rfour \\", r"\midrule"]
    last = None
    for idx in S.index:
        m, b, B = idx
        if S.loc[idx, "n"] < 10: continue
        if last is not None and m != last: lines.append(r"\midrule")
        lines.append(f"{m if m != last else ''} & {BL[b]} & {dsl(dsmed.loc[idx])} & {int(S.loc[idx,'n'])} & " + " & ".join(str(int(S.loc[idx, k])) for k in REG) + " & " + " & ".join(str(int(Lt.loc[idx, k])) for k in REG) + r" \\")
        last = m
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "lenient_strict.tex")
    # share of rescued failures whose label the sham requirement decides (strict label != lenient label); quoted in Limitations
    for nm, X in (("V*+ZoomBench", Rp[main_b]), ("TextVQA", Rp[~main_b])):
        P(f"labels changed by the sham requirement, {nm} (>=2x cells): {100 * (X.strict != X.lenient).mean():.1f}% of {len(X)}; per model: "
          + ", ".join(f"{m} {100 * (g.strict != g.lenient).mean():.1f}%" for m, g in X.groupby("model")))

    # ---------------------------------------------------------------- D
    P("\n" + "=" * 100); P(f"D. PAIRED DIFFERENCES, question-clustered bootstrap ({args.boot} reps), pooled V*+ZB >=2x, primary cohort")
    pairs = [("b3_native_same", "b2_seen_same", "b3 $-$ b2"), ("a3_cached_inplace", "a2_cached_only", "a3 $-$ a2"), ("a1_cached_roi", "a1_cached_roi_rand", "a1 $-$ sham"),
             ("a2_cached_only", "a2_cached_only_rand", "a2 $-$ sham"), ("b1_mask_bg", "b1_mask_bg_rand", "b1 $-$ sham"), ("b2_seen_same", "b2_seen_same_rand", "b2 $-$ sham"), ("c_seen_zoom", "c_seen_zoom_rand", "c $-$ sham")]
    rows = []
    for m in RUNS:
        X = Rp[main_b & (Rp.model == m)]
        uids, inv = np.unique(X.uid.values, return_inverse=True); U = len(uids)
        W = rng.multinomial(U, np.full(U, 1 / U), size=args.boot); cnt = np.bincount(inv, minlength=U)
        for a, b, lab in pairs:
            dvec = X[a].astype(float).values - X[b].astype(float).values
            s = np.bincount(inv, weights=dvec, minlength=U); est = dvec.mean(); boot = (W @ s) / (W @ cnt); lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(model=m, pair=lab, n=len(X), rate_a=100 * X[a].mean(), rate_b=100 * X[b].mean(), diff=100 * est, lo=100 * lo, hi=100 * hi, only_a=int((X[a] & ~X[b]).sum()), only_b=int((~X[a] & X[b]).sum())))
    PD = pd.DataFrame(rows); P(PD.round(1).to_string(index=False)); PD.to_csv(out / "paired.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Model & Pair & first (\%) & second (\%) & difference (pp) & 95\% CI & discordant (first only / second only) \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["model"] != last: lines.append(r"\midrule")
        lines.append(f"{r['model'] if r['model'] != last else ''} & {r['pair']} & {r['rate_a']:.1f} & {r['rate_b']:.1f} & {r['diff']:+.1f} & [{r['lo']:+.1f}, {r['hi']:+.1f}] & {r['only_a']} / {r['only_b']} \\\\"); last = r["model"]
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "paired.tex")

    # ---------------------------------------------------------------- E
    P("\n" + "=" * 100); P("E. BUDGET vs COVERAGE (primary cohort): logit of strict R4 / R3 on log2 downsample (ld) + log2 target coverage in first-pass tokens (lc) + bench + model FE; cluster-robust SE by question")
    import statsmodels.formula.api as smf
    lrows = []
    for name, X in [("V$^*$+ZoomBench", Rp[main_b]), ("all three", Rp)]:
        X = X.copy(); X["r4"] = (X.strict == 3).astype(int); X["r3"] = (X.strict == 2).astype(int); X["ld"] = np.log2(X.downsample); X["lc"] = np.log2(X["cov"].clip(lower=0.01))
        for dep, dl in [("r4", r"\rfour\ acquisition"), ("r3", r"\rthree\ re-tokenization")]:
            for form, fl in [(f"{dep} ~ ld + lc + C(bench) + C(model)", "both"), (f"{dep} ~ ld + C(bench) + C(model)", "downsampling only"), (f"{dep} ~ lc + C(bench) + C(model)", "coverage only")]:
                fit = smf.logit(form, data=X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(X.uid)[0]})
                P(f"[{name}] n={len(X)} {form}  pseudoR2={fit.prsquared:.3f}  " + "  ".join(f"{k}={fit.params[k]:+.3f} (se {fit.bse[k]:.3f}, p={fit.pvalues[k]:.3f})" for k in ("ld", "lc") if k in fit.params))
                lrows.append(dict(data=name, dep=dl, spec=fl, n=len(X), ld=(fit.params.get("ld"), fit.bse.get("ld"), fit.pvalues.get("ld")), lc=(fit.params.get("lc"), fit.bse.get("lc"), fit.pvalues.get("lc")), r2=fit.prsquared))
    # each benchmark fitted alone (model fixed effects only): the budget coefficient for R4 does not rest on pooling
    n_pooled = len(lrows)
    for b, bl in [("vstar", "V$^*$ alone"), ("zoombench", "ZoomBench alone"), ("textvqa", "TextVQA alone")]:
        X = Rp[Rp.bench == b].copy(); X["r4"] = (X.strict == 3).astype(int); X["ld"] = np.log2(X.downsample); X["lc"] = np.log2(X["cov"].clip(lower=0.01))
        form = "r4 ~ ld + lc + C(model)"
        fit = smf.logit(form, data=X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(X.uid)[0]})
        P(f"[{bl}] n={len(X)} {form}  pseudoR2={fit.prsquared:.3f}  " + "  ".join(f"{k}={fit.params[k]:+.3f} (se {fit.bse[k]:.3f}, p={fit.pvalues[k]:.3f})" for k in ("ld", "lc")))
        lrows.append(dict(data=bl, dep=r"\rfour\ acquisition", spec="both", n=len(X), ld=(fit.params["ld"], fit.bse["ld"], fit.pvalues["ld"]), lc=(fit.params["lc"], fit.bse["lc"], fit.pvalues["lc"]), r2=fit.prsquared))
    def fmt(t):
        if t[0] is None or pd.isna(t[0]): return "--"
        star = "$^{***}$" if t[2] < 0.001 else ("$^{**}$" if t[2] < 0.01 else ("$^{*}$" if t[2] < 0.05 else ""))
        v = f"{t[0]:+.2f}"; v = "0.00" if v in ("+0.00", "-0.00") else v          # no sign on a coefficient that rounds to zero
        return f"{v} ({t[1]:.2f}){star}"
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"Data & Outcome & Predictors & $n$ & $\log_2$ downsampling & $\log_2$ target coverage & pseudo-$R^2$ \\", r"\midrule"]
    for i, r in enumerate(lrows):
        if i == n_pooled: lines.append(r"\midrule")
        lines.append(f"{r['data']} & {r['dep']} & {r['spec']} & {r['n']} & {fmt(r['ld'])} & {fmt(r['lc'])} & {r['r2']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "coverage_logit.tex")
    P("\n-- stratified: strict R4 share (%) by coverage bin x downsampling, pooled models, V*+ZB")
    X = Rp[main_b].copy(); X["covbin"] = pd.cut(X["cov"], [0, 0.5, 1, 2, 4, 8, 16, 1e9], labels=["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", ">16"]); X["dsbin"] = pd.cut(X.downsample, [2, 3, 6, 100], labels=["2-3x", "3-6x", ">6x"])
    t = X.groupby(["covbin", "dsbin"], observed=True).apply(lambda g: pd.Series({"n": len(g), "R4%": round(100 * (g.strict == 3).mean()), "R3%": round(100 * (g.strict == 2).mean())})); P(t.unstack("dsbin").to_string())
    P("\n-- median target coverage (first-pass tokens) per strict regime, pooled V*+ZB >=2x:"); P(X.groupby("strict")["cov"].median().round(2).to_string())

    # ---------------------------------------------------------------- F
    P("\n" + "=" * 100); P("F. INPUT INVARIANTS (all scored samples)")
    for m in RUNS:
        Dm = D[D.model == m]
        pw = Dm.pivot_table(index=["bench", "budget", "uid"], columns="cond", values=["sw", "sh", "n_img_tokens"], aggfunc="first")
        def mism(c1, c2, k):
            a, b = pw[(k, c1)], pw[(k, c2)]; both = a.notna() & b.notna(); return int((a[both] != b[both]).sum()), int(both.sum())
        P(m, "c vs d: size mismatches", mism("c_seen_zoom", "d_native_zoom", "sw"), mism("c_seen_zoom", "d_native_zoom", "sh"), "tokens", mism("c_seen_zoom", "d_native_zoom", "n_img_tokens"),
          "| b2 vs b3: size", mism("b2_seen_same", "b3_native_same", "sw"), mism("b2_seen_same", "b3_native_same", "sh"), "tokens", mism("b2_seen_same", "b3_native_same", "n_img_tokens"),
          "| a2 vs a3 tokens", mism("a2_cached_only", "a3_cached_inplace", "n_img_tokens"))
    arms = ["base", "a1_cached_roi", "a2_cached_only", "a3_cached_inplace", "b1_mask_bg", "b2_seen_same", "b3_native_same", "c_seen_zoom", "d_native_zoom"]
    AL = {"base": "first pass only", "a1_cached_roi": "a1 cached re-insertion", "a2_cached_only": "a2 region only, new positions", "a3_cached_inplace": "a3 region only, original positions", "b1_mask_bg": "b1 background masked",
          "b2_seen_same": "b2 seen crop, same density", "b3_native_same": "b3 native crop, same density", "c_seen_zoom": "c seen crop, upsampled", "d_native_zoom": "d native crop, oracle"}
    sel = D[(D.bench == "vstar") & (D.budget == 256)]
    g = sel.groupby(["model", "cond"]).agg(sw=("sw", "median"), sh=("sh", "median"), tok=("n_img_tokens", "median"), tot=("n_tokens_in", "median"), roi=("roi_tokens", "median"), first=("n_img_tokens_first", "median"))
    P(g.loc[[(m, a) for m in RUNS for a in arms if (m, a) in g.index]].to_string())
    lines = [r"\begin{tabular}{l" + "rrr" * len(RUNS) + "}", r"\toprule", " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in RUNS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(RUNS))),
             "Condition & " + " & ".join(["2nd look (px) & image tok. & input tok."] * len(RUNS)) + r" \\", r"\midrule"]
    for a in arms:
        cells = []
        for m in RUNS:
            if (m, a) not in g.index: cells += ["--", "--", "--"]; continue
            r = g.loc[(m, a)]
            px = "--" if pd.isna(r.sw) else (f"{int(r.sw)}$\\times${int(r.sh)}" if a not in ("a1_cached_roi", "a2_cached_only", "a3_cached_inplace") else "(cached)")
            if a == "b1_mask_bg": px = "full raster"
            cells += [px, f"{int(r.tok)}", f"{int(r.tot)}"]
        lines.append(f"{AL[a]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]; tex(lines, "dims.tex")
    log.close(); R.to_pickle(out / "rescued_rows.pkl"); print("written", out)


if __name__ == "__main__":
    main()
