#!/usr/bin/env bash
# Every model run of the paper, one command per line. Run from the repository root after scripts/download_assets.py.
# Each line needs one GPU and writes one JSON-lines file under runs/, or extends it. The first four blocks are
# independent of each other. The last four read the outputs of the first four. Run the file top to bottom, or hand
# the lines to a job scheduler.
set -euo pipefail

# First pass, oracle zoom, the eight conditions and their shams: Qwen2.5-VL on V* and ZoomBench
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench vstar --budgets 256 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench vstar --budgets 1024 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench vstar --budgets 4096 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench vstar --budgets 16384 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench zoombench --budgets 256 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench zoombench --budgets 1024 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench zoombench --budgets 4096 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate --model models/Qwen2.5-VL-7B-Instruct --bench zoombench --budgets 16384 --out_dir runs/gate_qwen25vl7b

# The same for Qwen3-VL
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench vstar --budgets 256 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench vstar --budgets 1024 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench vstar --budgets 4096 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench vstar --budgets 16384 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench zoombench --budgets 256 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench zoombench --budgets 1024 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench zoombench --budgets 4096 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate --model models/Qwen3-VL-8B-Instruct --bench zoombench --budgets 16384 --out_dir runs/gate_qwen3vl8b

# The same for InternVL3
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench vstar --budgets 256 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench vstar --budgets 1024 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench vstar --budgets 4096 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench zoombench --budgets 256 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench zoombench --budgets 1024 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf --bench zoombench --budgets 4096 --out_dir runs/gate_internvl3_8b

# The same on TextVQA (1,000 sampled questions, both Qwen models)
python -m zoomdecomp.run_gate2 --model models/Qwen2.5-VL-7B-Instruct --bench textvqa --budgets 64 --sample 1000 --seed 0 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate2 --model models/Qwen2.5-VL-7B-Instruct --bench textvqa --budgets 128 --sample 1000 --seed 0 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate2 --model models/Qwen2.5-VL-7B-Instruct --bench textvqa --budgets 256 --sample 1000 --seed 0 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate2 --model models/Qwen2.5-VL-7B-Instruct --bench textvqa --budgets 1024 --sample 1000 --seed 0 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate2 --model models/Qwen3-VL-8B-Instruct --bench textvqa --budgets 64 --sample 1000 --seed 0 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate2 --model models/Qwen3-VL-8B-Instruct --bench textvqa --budgets 128 --sample 1000 --seed 0 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate2 --model models/Qwen3-VL-8B-Instruct --bench textvqa --budgets 256 --sample 1000 --seed 0 --out_dir runs/gate_qwen3vl8b
python -m zoomdecomp.run_gate2 --model models/Qwen3-VL-8B-Instruct --bench textvqa --budgets 1024 --sample 1000 --seed 0 --out_dir runs/gate_qwen3vl8b

# Magnification sweep at the coarsest budget
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench vstar --budgets 256
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench zoombench --budgets 256
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench textvqa --budgets 64
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench vstar --budgets 256
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench zoombench --budgets 256
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench textvqa --budgets 64
python -m zoomdecomp.run_mag --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench vstar --budgets 256
python -m zoomdecomp.run_mag --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench zoombench --budgets 256

# Magnification sweep, larger magnifications
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench vstar --budgets 256 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench zoombench --budgets 256 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench textvqa --budgets 64 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench vstar --budgets 256 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench zoombench --budgets 256 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench textvqa --budgets 64 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench vstar --budgets 256 --mags 12 16 24 32 --max_tokens2 4096
python -m zoomdecomp.run_mag --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench zoombench --budgets 256 --mags 12 16 24 32 --max_tokens2 4096

# The conditions with each model's own predicted region
python -m zoomdecomp.run_ground --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench vstar --budgets 256 1024
python -m zoomdecomp.run_ground --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench zoombench --budgets 256 1024
python -m zoomdecomp.run_ground --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench vstar --budgets 256 1024
python -m zoomdecomp.run_ground --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --bench zoombench --budgets 256 1024
python -m zoomdecomp.run_ground --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench vstar --budgets 256 1024
python -m zoomdecomp.run_ground --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --bench zoombench --budgets 256 1024

# Second, independent placement of every sham
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench vstar --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench vstar --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench zoombench --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench zoombench --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench textvqa --budgets 64 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench textvqa --budgets 128 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench textvqa --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench vstar --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench vstar --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench zoombench --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench zoombench --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench textvqa --budgets 64 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench textvqa --budgets 128 --seed 2
python -m zoomdecomp.rerun_sham --model models/Qwen3-VL-8B-Instruct --run_dir runs/gate_qwen3vl8b --out_dir runs/sham2_q3 --bench textvqa --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench vstar --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench vstar --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench vstar --budgets 4096 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench zoombench --budgets 256 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench zoombench --budgets 1024 --seed 2
python -m zoomdecomp.rerun_sham --model models/InternVL3-8B-hf --run_dir runs/gate_internvl3_8b --out_dir runs/sham2_iv --bench zoombench --budgets 4096 --seed 2
