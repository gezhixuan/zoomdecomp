# What Does a Zoom Actually Add?

Code and per-sample outputs for the paper *What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in
Vision-Language Models* by Zhixuan Ge, Anqi Li, Kexin Geng and Wei Qiu.

A vision-language model that answers a fine-grained question wrongly on the full image often answers it correctly after a
zoom into the relevant region. A zoom changes several things at once: it removes the surrounding context, re-tokenizes pixels
the model has already seen at a higher density, and, if the first pass was downsampled, shows pixels the model never received.
This repository holds the harness that separates these factors failure by failure. It fixes the first pass, runs eight
second-look conditions and their shams on every failure that an oracle zoom rescues, and assigns each failure to the cheapest
condition that fixes it.

## Layout

| Path | Content |
|---|---|
| `zoomdecomp/runners.py`, `qwen_runner.py` | Model wrappers for Qwen2.5-VL, Qwen3-VL and InternVL3: the exact raster the encoder saw, pixel-budget control, cached visual tokens |
| `zoomdecomp/data.py` | Loaders for V*, ZoomBench and TextVQA with target regions |
| `zoomdecomp/run_gate.py`, `run_gate2.py` | Main experiment: first-pass budget sweep, oracle zoom, the eight conditions and their shams |
| `zoomdecomp/run_mag.py` | Magnification sweep |
| `zoomdecomp/run_ground.py` | The same conditions with each model's own predicted region |
| `zoomdecomp/rerun_sham.py` | Second, independent placement of every sham |
| `zoomdecomp/make_tables.py`, `audit_*.py`, `analyze_*.py`, `paper_numbers.py` | Tables and the numbers quoted in the text |
| `zoomdecomp/figures.py`, `fig_examples.py` | Data figures and the per-sample example panels |
| `zoomdecomp/paths.py` | Where data, checkpoints, outputs and generated files live |
| `scripts/download_assets.py` | Downloads the benchmarks and checkpoints from the Hugging Face Hub |
| `scripts/paper_runs.sh` | Every model run of the paper, one command per line |

## Setup

Python 3.11. The experiments were run with torch 2.6.0 (CUDA 12.4) and transformers 5.12.1 on a single GPU per job.

```bash
pip install -r requirements.txt
```

All locations default to folders in the repository root and can be changed with environment variables, see
`zoomdecomp/paths.py`: `ZOOMDECOMP_DATA`, `ZOOMDECOMP_MODELS`, `ZOOMDECOMP_RUNS`, `ZOOMDECOMP_OUT`.

## Reproduce the tables from the released outputs

No GPU, no benchmark and no checkpoint is needed for this part. Download `zoomdecomp_outputs.tar.gz` (18 MB) from the
Releases page of this repository and unpack it into `runs/`:

```bash
mkdir -p runs && tar -xzf zoomdecomp_outputs.tar.gz -C runs
python -m zoomdecomp.make_tables            # Tab. 2, S10, S12
python -m zoomdecomp.audit_review           # Tab. S1 to S6
python -m zoomdecomp.audit_budget_control   # Tab. S7, S8 (run audit_review first)
python -m zoomdecomp.analyze_mag            # Tab. S9, Fig. 5
python -m zoomdecomp.analyze_ground         # Tab. S11
python -m zoomdecomp.analyze_sham2 --table outputs/tables/sham2.tex   # Tab. S13
python -m zoomdecomp.analyze_policy         # Tab. 3, S14, S15
python -m zoomdecomp.audit_subgroups        # Tab. S16
python -m zoomdecomp.paper_numbers          # every number quoted in the text
```

The LaTeX tables are written to `outputs/tables/` and are byte-identical to the tables of the paper.
`python -m zoomdecomp.figures` writes the data figures to `outputs/figures/`. One of them draws image thumbnails if the V*
images and the Qwen2.5-VL processor are on disk. The figures use Arial if a folder with the TTF files is given in
`ZOOMDECOMP_FONTS`, and DejaVu Sans otherwise.
`python -m zoomdecomp.fig_examples --bench textvqa --budget 64` draws the example panels of the supplement and needs the
benchmark images. Figs. 1 and 2 are illustrations and are not generated here.

## Run the experiments

```bash
python scripts/download_assets.py           # three benchmarks, three checkpoints (about 50 GB)

# one model, one benchmark, one first-pass budget (in units of 28x28 pixels)
python -m zoomdecomp.run_gate  --model models/Qwen2.5-VL-7B-Instruct --bench vstar --budgets 256 --out_dir runs/gate_qwen25vl7b
python -m zoomdecomp.run_gate2 --model models/InternVL3-8B-hf        --bench vstar --budgets 256 --out_dir runs/gate_internvl3_8b
python -m zoomdecomp.run_gate2 --model models/Qwen2.5-VL-7B-Instruct --bench textvqa --budgets 64 --sample 1000 --seed 0 --out_dir runs/gate_qwen25vl7b

# follow-up runs on an existing output folder
python -m zoomdecomp.run_mag    --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench vstar --budgets 256
python -m zoomdecomp.run_ground --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --bench vstar --budgets 256 1024
python -m zoomdecomp.rerun_sham --model models/Qwen2.5-VL-7B-Instruct --run_dir runs/gate_qwen25vl7b --out_dir runs/sham2_q25 --bench vstar --budgets 256 --seed 2
```

`scripts/paper_runs.sh` lists every run of the paper with its model, benchmark and budget, one command per line.

## Output format

Each run folder holds one JSON-lines file per benchmark and budget. A line is one condition on one sample.

| Field | Meaning |
|---|---|
| `uid`, `bench`, `qtype`, `gold` | Sample id, benchmark, question type, gold answer |
| `budget`, `K` | First-pass budget and crop budget, in units of 28x28 pixels |
| `img_wh`, `seen_wh`, `grid_wh` | Source size, size of the raster the encoder saw, its token grid |
| `roi_grid`, `roi_grid_rand` | Target region and the random region of the shams, in token-grid cells |
| `cond` | Condition: `base`, `a1_cached_roi`, `a2_cached_only`, `a3_cached_inplace`, `b1_mask_bg`, `b2_seen_same`, `b3_native_same`, `c_seen_zoom`, `d_native_zoom`. The suffix `_rand` marks the sham, `c_m<k>` and `d_m<k>` the magnification sweep |
| `pred`, `text`, `correct` | Greedy answer, raw reply, correctness |
| `gold_logodds`, `lp` | Log-odds of the gold answer and the log-probabilities it is computed from |
| `second_wh`, `n_img_tokens`, `n_tokens_in` | Size of the second look, image tokens and total input tokens |

A condition fixes a sample if `correct` is true and `gold_logodds` is positive.

## Third-party material

The benchmarks (V*, ZoomBench, the TextVQA subset with text regions released with ViCrop) and the models are downloaded
from their own repositories and keep their own licenses. The released outputs contain model replies, gold answers and region
coordinates, and no images.

## License

The code is released under the MIT license, see `LICENSE`.

## Citation

```bibtex
@article{ge2026zoom,
  title   = {What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in Vision-Language Models},
  author  = {Ge, Zhixuan and Li, Anqi and Geng, Kexin and Qiu, Wei},
  year    = {2026}
}
```
