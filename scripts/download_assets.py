"""Download the three benchmarks and the three checkpoints of the paper from the Hugging Face Hub.

Benchmarks go to ZOOMDECOMP_DATA (default data/), checkpoints to ZOOMDECOMP_MODELS (default models/).
Resumable. Check the license of each benchmark and model on its Hub page before use.
"""
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp import paths

JOBS = [  # (repo id, repo type, local folder name)
    ("craigwu/vstar_bench", "dataset", "vstar_bench"),
    ("inclusionAI/ZoomBench", "dataset", "ZoomBench"),
    ("jrzhang/TextVQA_GT_bbox", "dataset", "TextVQA_GT_bbox"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "model", "Qwen2.5-VL-7B-Instruct"),
    ("Qwen/Qwen3-VL-8B-Instruct", "model", "Qwen3-VL-8B-Instruct"),
    ("OpenGVLab/InternVL3-8B-hf", "model", "InternVL3-8B-hf"),
]

failed = []
for repo, kind, name in JOBS:
    dst = (paths.DATA if kind == "dataset" else paths.MODELS) / name
    t0 = time.time()
    print(f"[get ] {repo} ({kind}) -> {dst}", flush=True)
    try:
        snapshot_download(repo, repo_type=kind, local_dir=str(dst), max_workers=8)
        print(f"[done] {name} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:                                   # keep going, report at the end
        traceback.print_exc()
        failed.append((name, repr(e)))
print("failed:", failed)
sys.exit(1 if failed else 0)
