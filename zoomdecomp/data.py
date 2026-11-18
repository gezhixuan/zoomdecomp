"""Benchmark loaders -> unified Sample records (bbox in native pixel xyxy)."""
import io, json, re
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from zoomdecomp import paths

VSTAR = (paths.DATA / "vstar_bench")
ZOOM = (paths.DATA / "ZoomBench/data/test.parquet")
TEXTVQA = (paths.DATA / "TextVQA_GT_bbox/data")
TEXT_SUFFIX = "Answer the question using a single word or phrase."
MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."
BLANK_SUFFIX = "Answer with a single number."


@dataclass
class Sample:
    uid: str
    bench: str
    qtype: str            # "mcq" | "blank"
    prompt: str
    answer: str           # letter for mcq, string for blank
    letters: list
    bbox: tuple           # native xyxy (union of target boxes)
    image_path: str = None
    image_bytes: bytes = None
    meta: dict = field(default_factory=dict)

    def load_image(self):
        im = Image.open(self.image_path) if self.image_path else Image.open(io.BytesIO(self.image_bytes))
        return im.convert("RGB")


def _reformat_vstar_text(text):
    """'(A) rubber' -> 'A. rubber'; replace the trailing instruction with our common suffix."""
    out, letters = [], []
    for line in text.split("\n"):
        if not line.strip() or line.startswith("Answer with the option"):
            continue
        m = re.match(r"^\(([A-D])\)\s*(.*)$", line)
        if m:
            out.append(f"{m.group(1)}. {m.group(2)}"); letters.append(m.group(1))
        else:
            out.append(line)
    return "\n".join(out) + "\n" + MCQ_SUFFIX, letters


def load_vstar():
    samples = []
    for line in open(VSTAR / "test_questions.jsonl"):
        r = json.loads(line)
        img = VSTAR / r["image"]
        ann = json.load(open(str(img)[:-4] + ".json"))
        boxes = ann["bbox"]  # [[x, y, w, h], ...]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        prompt, letters = _reformat_vstar_text(r["text"])
        assert r["label"] in letters, (r, letters)
        samples.append(Sample(uid=f"vstar_{r['question_id']}", bench="vstar", qtype="mcq", prompt=prompt,
                              answer=r["label"], letters=letters, bbox=(x0, y0, x1, y1), image_path=str(img),
                              meta={"category": r["category"], "boxes": boxes, "targets": ann["target_object"]}))
    return samples


def load_zoombench():
    pf = pq.ParquetFile(str(ZOOM)); samples = []
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["id", "query", "response", "bbox", "question_type", "image"]).to_pylist():
            q = r["query"].strip(); qt = r["question_type"]; ans = str(r["response"]).strip()
            if qt == "mcq":
                letters = re.findall(r"(?m)^([A-D])\.\s", q)
                prompt = q + "\n" + MCQ_SUFFIX
                if ans not in letters:
                    continue
            else:
                letters = []; prompt = q + "\n" + BLANK_SUFFIX
            b = [float(v) for v in r["bbox"]]
            img = r["image"]; data = img["bytes"] if isinstance(img, dict) else img
            samples.append(Sample(uid=f"zoom_{r['id']}", bench="zoombench", qtype=qt, prompt=prompt, answer=ans,
                                  letters=letters, bbox=(b[0], b[1], b[2], b[3]), image_bytes=data))
    return samples


def load_textvqa():
    """TextVQA val questions with a single GT text region (ViCrop authors). answer = majority of 10 human answers."""
    import collections, glob
    samples = []
    for f in sorted(glob.glob(str(TEXTVQA / "*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["image", "question", "answer", "dataset_id", "bbox"]).to_pylist():
                answers = [a.strip().lower() for a in r["answer"]]
                gold = collections.Counter(answers).most_common(1)[0][0]
                x, y, w, h = [float(v) for v in r["bbox"]]
                img = r["image"]; data = img["bytes"] if isinstance(img, dict) else img
                samples.append(Sample(uid=f"textvqa_{r['dataset_id']}", bench="textvqa", qtype="text",
                                      prompt=r["question"].strip() + "\n" + TEXT_SUFFIX, answer=gold, letters=[],
                                      bbox=(x, y, x + w, y + h), image_bytes=data, meta={"answers": answers}))
    return samples


LOADERS = {"vstar": load_vstar, "zoombench": load_zoombench, "textvqa": load_textvqa}
