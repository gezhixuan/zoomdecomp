"""Print every number quoted in the manuscript text, computed from the current cohort definition (figures.rescued).
Usage: python -m zoomdecomp.paper_numbers > analysis/paper_numbers.txt"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued, regime_strict, dsample_label, ds_cells

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
FAM = {"a": ["a1_cached_roi", "a2_cached_only", "a3_cached_inplace"], "b": ["b1_mask_bg", "b2_seen_same", "b3_native_same"], "c": ["c_seen_zoom"]}
ARMS = ["a1_cached_roi", "a2_cached_only", "a3_cached_inplace", "b1_mask_bg", "b2_seen_same", "b3_native_same", "c_seen_zoom"]


def anyof(rs, arms, sham=False):
    cols = [a + ("_rand" if sham else "") for a in arms if a + ("_rand" if sham else "") in rs]
    return np.any([rs[c].fillna(False).astype(bool).values for c in cols], axis=0) if cols else np.zeros(len(rs), bool)


def main():
    data = {m: load_run(p) for m, p in RUNS.items()}
    # ---------------- stage 1 per cell
    print("=" * 110); print("STAGE 1 per cell: n, base acc, zoom acc, rescued (primary), rescued (confidence-filtered), hurt, strict regime shares")
    rows = []
    for m, (w, _) in data.items():
        for (b, B), g in w.groupby(level=[0, 1]):
            ds = float(np.median(g["downsample"])); rs = rescued(g); rc = rescued(g, "confidence-filtered")
            hurt = int(((g["base"] == True) & (g["d_native_zoom"] == False)).sum())
            reg = np.bincount(rs.apply(regime_strict, axis=1), minlength=4) if (ds >= 2 and len(rs) >= 10) else None
            rows.append(dict(model=m, bench=b, B=B, ds=round(ds, 1), n=len(g), base=round(g["base"].mean(), 3), zoom=round(g["d_native_zoom"].mean(), 3), resc=len(rs), resc_conf=len(rc), hurt=hurt,
                             R1=None if reg is None else round(100 * reg[0] / reg.sum()), R2=None if reg is None else round(100 * reg[1] / reg.sum()), R3=None if reg is None else round(100 * reg[2] / reg.sum()), R4=None if reg is None else round(100 * reg[3] / reg.sum())))
    S = pd.DataFrame(rows); print(S.to_string(index=False))
    print("\n-- totals: rescued (primary) all cells:", S.resc.sum(), "| cells with ds>=2:", S[S.ds >= 2].resc.sum(), "| confidence-filtered all cells:", S.resc_conf.sum())
    for b in ["vstar", "zoombench", "textvqa"]:
        sb = S[S.bench == b]
        print(f"-- {b}: rescued range {sb.resc.min()}-{sb.resc.max()}, hurt range {sb.hurt.min()}-{sb.hurt.max()}, zoom acc range {sb.zoom.min()}-{sb.zoom.max()}, base acc at coarsest {sb[sb.B == sb.B.min()].base.min()}-{sb[sb.B == sb.B.min()].base.max()}, base at finest {sb[sb.B == sb.B.max()].base.min()}-{sb[sb.B == sb.B.max()].base.max()}")
        for B in sorted(sb.B.unique()):
            c = sb[(sb.B == B) & sb.R4.notna()]
            if len(c): print(f"   B={B} (ds {c.ds.min()}-{c.ds.max()}): R1 {c.R1.min()}-{c.R1.max()} | R2 {c.R2.min()}-{c.R2.max()} | R3 {c.R3.min()}-{c.R3.max()} | R4 {c.R4.min()}-{c.R4.max()} | R1-R3 {100 - c.R4.max()}-{100 - c.R4.min()}  (n={c.resc.tolist()})")
    # ---------------- pooled-model regime shares per bench x budget (Fig. regime area numbers)
    print("\n" + "=" * 110); print("POOLED-MODEL strict regime shares per bench x budget (ds>=2 cells)")
    for b in ["vstar", "zoombench", "textvqa"]:
        for B in sorted({B for m in data if b in data[m][0].index.get_level_values(0) for B in data[m][0].loc[b].index.get_level_values(0).unique()}):
            cnt = np.zeros(4, int); dss = []
            for m, (w, _) in data.items():
                if b not in w.index.get_level_values(0) or B not in w.loc[b].index.get_level_values(0): continue
                g = w.loc[b].loc[B]; ds = float(np.median(g["downsample"]))
                if ds < 2: continue
                rs = rescued(g); cnt += np.bincount(rs.apply(regime_strict, axis=1), minlength=4); dss.append(ds)
            if cnt.sum(): print(f"  {b} B={B} ds~{np.median(dss):.1f}: n={cnt.sum()} shares R1-R4 = {np.round(100 * cnt / cnt.sum()).astype(int).tolist()}")
    # ---------------- numbers of Fig. 1: pooled 3 models, V*+ZB, B=256
    cnt = np.zeros(4, int)
    for m, (w, _) in data.items():
        ww = w[(w.index.get_level_values(0) != "textvqa") & (w.index.get_level_values(1) == 256)]; cnt += np.bincount(rescued(ww).apply(regime_strict, axis=1), minlength=4)
    print(f"\nFig. 1: pooled V*+ZB at B=256: n={cnt.sum()} counts={cnt.tolist()} shares={np.round(100 * cnt / cnt.sum()).astype(int).tolist()}")
    # ---------------- family sweep numbers (V*), per model per budget
    print("\n" + "=" * 110); print("FAMILY RATES on rescued samples (any arm of family; sham = any sham of family), per bench/model/budget")
    for b in ["vstar", "zoombench", "textvqa"]:
        for m, (w, _) in data.items():
            if b not in w.index.get_level_values(0): continue
            for B in sorted(w.loc[b].index.get_level_values(0).unique()):
                g = w.loc[b].loc[B]; rs = rescued(g); ds = float(np.median(g["downsample"]))
                if len(rs) < 10: continue
                s = f"  {b:9s} {m:14s} {dsample_label(ds):>7s} n={len(rs):3d} |"
                for f, arms in FAM.items():
                    s += f" {f}: {100 * anyof(rs, arms).mean():4.0f} ({100 * anyof(rs, arms, True).mean():3.0f}) |"
                for a in ["a1_cached_roi", "b2_seen_same", "b3_native_same"]:
                    s += f" {a[:2]}: {100 * rs[a].fillna(False).astype(bool).mean():3.0f} ({100 * rs[a + '_rand'].fillna(False).astype(bool).mean():3.0f}) |"
                print(s)
    # ---------------- pooled arm vs sham (Table 2), ds>=2, V*+ZB
    print("\n" + "=" * 110); print("POOLED ARM vs SHAM (V*+ZB, ds>=2) per model; and TextVQA")
    for sel_name, sel in [("V*+ZB", lambda w: (w.index.get_level_values(0) != "textvqa")), ("TextVQA", lambda w: (w.index.get_level_values(0) == "textvqa"))]:
        for m, (w, _) in data.items():
            ww = w[sel(w) & ds_cells(w)]; rs = rescued(ww)
            if len(rs) == 0: continue
            cells = []
            for a in ARMS:
                t = rs[a].fillna(False).astype(bool).mean(); sh = rs[a + "_rand"].fillna(False).astype(bool).mean(); cells.append(f"{a[:2]} {100 * t:3.0f} ({100 * sh:3.0f}) d={100 * (t - sh):+3.0f}")
            b3only = ((rs["b3_native_same"] == True) & (rs["b2_seen_same"] == False)).mean(); b2only = ((rs["b2_seen_same"] == True) & (rs["b3_native_same"] == False)).mean()
            pad = rs["d_native_zoom_pad1"].fillna(False).astype(bool).mean(); para = (rs["base_para"] == False).mean()
            reg = rs.apply(regime_strict, axis=1); len_ = rs.apply(lambda r: next((k for k, tier in enumerate([("a1_cached_roi", "a2_cached_only", "a3_cached_inplace"), ("b1_mask_bg", "b2_seen_same"), ("c_seen_zoom",)]) if any(r.get(a) == True for a in tier)), 3), axis=1)
            print(f"  [{sel_name}] {m:14s} n={len(rs)} | " + " | ".join(cells))
            print(f"      b3-only {100 * b3only:.1f}%  b2-only {100 * b2only:.1f}% | pad survive {100 * pad:.0f}% | base wrong under paraphrase {100 * para:.0f}% | strict R1-R4 {np.round(100 * np.bincount(reg, minlength=4) / len(rs)).astype(int).tolist()} lenient {np.round(100 * np.bincount(len_, minlength=4) / len(rs)).astype(int).tolist()}")
            for b in ["vstar", "zoombench"]:
                rb = rs[rs.index.get_level_values(0) == b]
                if len(rb): print(f"      {b}: pad survive {100 * rb['d_native_zoom_pad1'].fillna(False).astype(bool).mean():.0f}%  paraphrase-persist {100 * (rb['base_para'] == False).mean():.0f}%  n={len(rb)}")
    # ---------------- hierarchy violations per cell (lenient hits, as in the ladder figure)
    print("\n" + "=" * 110); print("HIERARCHY VIOLATIONS per cell: (a-family or b-family rescues) and c does not  [lenient hits]; strict version in brackets")
    for b in ["vstar", "zoombench", "textvqa"]:
        for m, (w, _) in data.items():
            if b not in w.index.get_level_values(0): continue
            out = []
            for B in sorted(w.loc[b].index.get_level_values(0).unique()):
                g = w.loc[b].loc[B]; rs = rescued(g); ds = float(np.median(g["downsample"]))
                if len(rs) < 10 or ds < 2: continue
                h1, h2, h3 = anyof(rs, FAM["a"]), anyof(rs, ["b1_mask_bg", "b2_seen_same"]), anyof(rs, ["c_seen_zoom"])
                s1, s2, s3 = anyof(rs, FAM["a"], True), anyof(rs, ["b1_mask_bg", "b2_seen_same"], True), anyof(rs, ["c_seen_zoom"], True)
                v = ((h1 & ~h3) | (h2 & ~h3)).mean(); vs = (((h1 & ~s1) & ~(h3 & ~s3)) | ((h2 & ~s2) & ~(h3 & ~s3))).mean()
                out.append(f"{dsample_label(ds)}: {100 * v:.0f}% [{100 * vs:.0f}%] n={len(rs)}")
            if out: print(f"  {b:9s} {m:14s} " + " | ".join(out))
    # ---------------- coverage medians per regime
    print("\n" + "=" * 110); print("TARGET COVERAGE (first-pass tokens) median per strict regime, pooled V*+ZB ds>=2, per model and pooled")
    allr = []
    for m, (w, _) in data.items():
        ww = w[(w.index.get_level_values(0) != "textvqa") & ds_cells(w)]; rs = rescued(ww).copy(); rs["reg"] = rs.apply(regime_strict, axis=1); rs["model"] = m; allr.append(rs[["reg", "cov", "model"]])
        print(f"  {m}: " + ", ".join(f"R{k + 1} {rs[rs.reg == k]['cov'].median():.2f} (n={int((rs.reg == k).sum())})" for k in range(4)))
    A = pd.concat(allr); print("  pooled: " + ", ".join(f"R{k + 1} {A[A.reg == k]['cov'].median():.2f} (n={int((A.reg == k).sum())})" for k in range(4)))
    # ---------------- flows: V* 17x -> native per model
    print("\n" + "=" * 110); print("FLOW V*: of questions zoom-rescued at the coarsest first pass, outcome at the finest first pass")
    for m, (w, _) in data.items():
        wv = w.loc["vstar"]; budgets = sorted(wv.index.get_level_values(0).unique()); B0, B1 = budgets[0], budgets[-1]
        r0 = rescued(wv.loc[B0]); u = r0.index; g1 = wv.loc[B1].reindex(u)
        corr = (g1["base"] == True).mean(); res = g1.apply(lambda r: (r["base"] == False) and (r.get("base_greedy") == False) and (r.get("d_native_zoom") == True) and (r.get("d_native_zoom_rand") != True), axis=1)
        reg1 = g1[res].apply(regime_strict, axis=1) if res.any() else pd.Series(dtype=int)
        print(f"  {m}: n0={len(u)} at {dsample_label(float(np.median(wv.loc[B0]['downsample'])))}; at {dsample_label(float(np.median(wv.loc[B1]['downsample'])))}: correct w/o zoom {100 * corr:.0f}%, still rescued {100 * res.mean():.0f}% (regimes {np.bincount(reg1, minlength=4).tolist()}), other {100 * (1 - corr - res.mean()):.0f}%")


if __name__ == "__main__":
    main()
