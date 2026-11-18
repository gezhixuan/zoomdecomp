"""Where benchmarks, checkpoints, per-sample outputs and generated files live.

Every location can be set with an environment variable. The defaults are folders in the repository root.

    ZOOMDECOMP_DATA    benchmarks (vstar_bench, ZoomBench, TextVQA_GT_bbox)      default: data/
    ZOOMDECOMP_MODELS  model checkpoints                                          default: models/
    ZOOMDECOMP_RUNS    per-sample outputs of every condition (jsonl)              default: runs/
    ZOOMDECOMP_OUT     tables, figures and analysis files written by the scripts  default: outputs/
    ZOOMDECOMP_FONTS   optional folder with Arial TTF files for the figures       default: fonts/
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dir(var, default):
    return Path(os.environ.get(var, REPO / default))


DATA = _dir("ZOOMDECOMP_DATA", "data")
MODELS = _dir("ZOOMDECOMP_MODELS", "models")
RUNS = _dir("ZOOMDECOMP_RUNS", "runs")
OUT = _dir("ZOOMDECOMP_OUT", "outputs")
FONTS = _dir("ZOOMDECOMP_FONTS", "fonts")
TABLES, FIGS, ANALYSIS = OUT / "tables", OUT / "figures", OUT / "analysis"
for _d in (TABLES, FIGS, ANALYSIS):
    _d.mkdir(parents=True, exist_ok=True)
