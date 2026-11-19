"""Download the three benchmarks and the three checkpoints of the paper from the Hugging Face Hub.

Benchmarks go to ZOOMDECOMP_DATA (default data/), checkpoints to ZOOMDECOMP_MODELS (default models/).
Each repository is pinned to the revision the paper used. Resumable. Check the license of each benchmark and model on its
Hub page before use.
"""
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoomdecomp import paths

JOBS = [  # (repo id, repo type, local folder name, revision)
    ("craigwu/vstar_bench", "dataset", "vstar_bench", "d9ae62c903da0c98336e85c5ee89cd863b04b4da"),
    ("inclusionAI/ZoomBench", "dataset", "ZoomBench", "b788097e57d30510c6877824833234a73bf80d25"),
    ("jrzhang/TextVQA_GT_bbox", "dataset", "TextVQA_GT_bbox", "4a0fae0da919af44db46d93ab1a1e128daad2de6"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "model", "Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ("Qwen/Qwen3-VL-8B-Instruct", "model", "Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    ("OpenGVLab/InternVL3-8B-hf", "model", "InternVL3-8B-hf", "259a3b64a14623c0ec91a045cb43f7c5af5fa6af"),
]

failed = []
for repo, kind, name, rev in JOBS:
    dst = (paths.DATA if kind == "dataset" else paths.MODELS) / name
    t0 = time.time()
    print(f"[get ] {repo} ({kind}) @ {rev[:7]} -> {dst}", flush=True)
    try:
        snapshot_download(repo, repo_type=kind, revision=rev, local_dir=str(dst), max_workers=8)
        print(f"[done] {name} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:                                   # keep going, report at the end
        traceback.print_exc()
        failed.append((name, repr(e)))
print("failed:", failed)
sys.exit(1 if failed else 0)
