"""Generate LaTeX tables for the paper from the run directories (booktabs)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp.figures import RUNS, load_run, rescued, regime_strict, ARMS, ds_cells
from zoomdecomp import paths

OUT = paths.TABLES; OUT.mkdir(parents=True, exist_ok=True)
BL = {"vstar": "V$^*$", "zoombench": "ZoomBench", "textvqa": "TextVQA"}
SHORT = {"Qwen2.5-VL-7B": "Qwen2.5-VL-7B", "Qwen3-VL-8B": "Qwen3-VL-8B", "InternVL3-8B": "InternVL3-8B"}


def dsl(x):
    return f"{x:.0f}$\\times$" if x >= 3 else ("native" if x < 1.02 else f"{x:.1f}$\\times$")


def main():
    data = {m: load_run(p) for m, p in RUNS.items()}; data = {m: v for m, v in data.items() if v is not None}
    # ---------------- Table 1: stage 1 + strict regime shares
    rows = []
    for m, (w, _) in data.items():
        for (b, B), g in w.groupby(level=[0, 1]):
            ds = float(np.median(g["downsample"])); rs = rescued(g)
            resc = len(rs); hurt = int(((g["base"] == True) & (g["d_native_zoom"] == False)).sum())
            if ds >= 2.0 and resc >= 10:
                cnt = np.bincount(rs.apply(regime_strict, axis=1), minlength=4); frac = cnt / cnt.sum()
                reg = " / ".join(f"{100*f:.0f}" for f in frac)
            else:
                reg = "--"
            rows.append((m, b, B, ds, len(g), g["base"].mean(), g["d_native_zoom"].mean(), resc, hurt, reg))
    lines = [r"\begin{tabular}{llrrrrrrl}", r"\toprule",
             r"Model & Benchmark & First pass & $n$ & Base & +Zoom & Rescued & Hurt & R1 / R2 / R3 / R4 (\%) \\", r"\midrule"]
    last = None
    for m, b, B, ds, n, base, zoom, resc, hurt, reg in rows:
        mm = SHORT[m] if (m, b) != last else ""; bb = BL[b] if (m, b) != last else ""
        if last is not None and m != last[0]: lines.append(r"\midrule")
        lines.append(f"{mm} & {bb} & {dsl(ds)} & {n} & {base:.3f} & {zoom:.3f} & {resc} & {hurt} & {reg} \\\\")
        last = (m, b)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "stage1.tex").write_text("\n".join(lines) + "\n")
    # trimmed version for the main text: only first passes downsampled >= 2x (where the regime partition is defined)
    lines = [r"\begin{tabular}{llrrrrrrl}", r"\toprule",
             r"Model & Benchmark & First pass & $n$ & Base & +Zoom & Rescued & Hurt & R1 / R2 / R3 / R4 (\%) \\", r"\midrule"]
    last = None
    for m, b, B, ds, n, base, zoom, resc, hurt, reg in rows:
        if ds < 2.0: continue
        mm = SHORT[m] if (m, b) != last else ""; bb = BL[b] if (m, b) != last else ""
        if last is not None and m != last[0]: lines.append(r"\midrule")
        lines.append(f"{mm} & {bb} & {dsl(ds)} & {n} & {base:.3f} & {zoom:.3f} & {resc} & {hurt} & {reg} \\\\")
        last = (m, b)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "stage1_main.tex").write_text("\n".join(lines) + "\n")

    # ---------------- Table 2: pooled arm vs sham on rescued samples (first pass >= 2x), per model, V* + ZoomBench
    arms = ARMS[:-1]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(arms) + "@{}}", r"\toprule",                       # fits a column at the table font: n is printed and goes in the caption
             "Model & " + " & ".join(["a1", "a2", "a3", "b1", "b2", "b3", "c"]) + r" \\", r"\midrule"]
    for m, (w, _) in data.items():
        ww = w[(w.index.get_level_values(0) != "textvqa") & ds_cells(w)]; rs = rescued(ww)
        cells = []
        for a in arms:
            t = rs[a].mean() if a in rs else np.nan; s = rs[a + "_rand"].mean() if a + "_rand" in rs else np.nan
            cells.append(f"{100*t:.0f} ({100*s:.0f})")
        lines.append(f"{SHORT[m]} & " + " & ".join(cells) + r" \\"); print(f"  arms.tex: {SHORT[m]} n={len(rs)} (quoted in the caption of tab:arms)")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "arms.tex").write_text("\n".join(lines) + "\n")

    # ---------------- Table 3: robustness (2x pad, paraphrase, b3-only), per model pooled >= 2x
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & $n$ & fixed with 2$\times$ padding & failure persists under paraphrase & b3 fixes, b2 does not \\", r"\midrule"]
    for m, (w, _) in data.items():
        ww = w[(w.index.get_level_values(0) != "textvqa") & ds_cells(w)]; rs = rescued(ww)
        pad = rs["d_native_zoom_pad1"].mean() if "d_native_zoom_pad1" in rs else np.nan
        para = (rs["base_para"] == False).mean() if "base_para" in rs else np.nan
        b3o = ((rs["b3_native_same"] == True) & (rs["b2_seen_same"] == False)).mean() if "b3_native_same" in rs else np.nan
        lines.append(f"{SHORT[m]} & {len(rs)} & {100*pad:.0f}\\% & {100*para:.0f}\\% & {100*b3o:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "robustness.tex").write_text("\n".join(lines) + "\n")
    from zoomdecomp.wrap_cond import wrap_dir; wrap_dir(OUT)        # condition codes -> \cond{...} (style set in preamble.tex)
    print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
