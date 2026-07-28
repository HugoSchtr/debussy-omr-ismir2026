#!/usr/bin/env python3
"""Zero-shot evaluation of Zeus (Mayer et al., OLiMPiC/ICDAR24) checkpoints on a
paired system-level OMR dataset (used for Table 4 and Table 5 of the ISMIR 2026
Debussy Dataset paper).

This is a cleaned, standalone re-implementation of the core evaluation logic in
the original exploratory notebook (`zeus_eval.ipynb`): load a Zeus model + tags,
binarize input images per a configurable method, run inference via the vendored
`zeus.py` (TensorFlow, invoked as a subprocess -- Zeus's own CLI contract),
compute SER via `ser_metric.py`, CER via the same character-level Levenshtein
convention used everywhere else in this codebase (`fine_tuning/eval_functions.py`
-- Zeus's own `ser_metric.py` only reports SER/SERnotuplets), and optionally TEDn
via `tedn_metric.py`'s `TEDn_lmx_xml`. Exploratory/visualization cells from the
notebook (matplotlib previews, interactive binarization comparisons) are
intentionally NOT reproduced here -- this script only computes the final
per-model SER/CER/TEDn numbers.

CRITICAL correction vs. the original submission (see README.md for details):
the original submission applied `binarization_method: adaptive` uniformly to
ALL THREE Zeus checkpoints. Pixel-statistics evidence gathered during the
camera-ready revision showed this is only correct for the plain GrandStaff-LMX
checkpoint; the Camera-GrandStaff-LMX and Synthetic/OLiMPiC checkpoints are
trained on genuinely grayscale/photographic or unbinarized images. This script's
default preprocessing is keyed by checkpoint name (see DEFAULT_BINARIZE_BY_MODEL
below) and applies the corrected policy by default. Pass --binarize explicitly
to override this and reproduce the original (uncorrected) submission numbers.

Usage:
    python eval_zeus.py \\
        --model models/zeus-grandstaff-lmx-1.0-2024-02-12.model \\
        --dataset /path/to/paired_system_level \\
        --output ./results/zeus_grandstaff_lmx \\
        --compute-tedn --tedn-flavor lmx
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# --- olimpic-icdar24 (Mayer et al.) -- required for the LMX linearizer and
# TEDn implementation. Not pip-installable: clone
# https://github.com/ufal/olimpic-icdar24 and set OLIMPIC_ICDAR24_DIR, or place
# it as a sibling ../../olimpic-icdar24 directory next to this repo (default).
OLIMPIC_ICDAR24_DIR = os.environ.get(
    "OLIMPIC_ICDAR24_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "olimpic-icdar24"),
)
if OLIMPIC_ICDAR24_DIR not in sys.path:
    sys.path.insert(0, OLIMPIC_ICDAR24_DIR)

# --- ser_metric.py (vendored, this directory) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ser_metric  # noqa: E402


# =============================================================================
# Corrected per-checkpoint binarization policy (camera-ready revision)
# =============================================================================
# WHY: pixel-statistics evidence (see REVISION_PLAN.md item T1.3 in the paper
# repo) showed the original submission's uniform `binarization_method: adaptive`
# was a mismatch for two of the three checkpoints:
#   - zeus-grandstaff-lmx:        trained on near-binary GrandStaff renders
#                                 (~87% of pixels > brightness 200) -> adaptive
#                                 binarization matches training and is CORRECT.
#   - zeus-camera-grandstaff-lmx: trained on genuinely grayscale/photographic
#                                 "_distorted" Camera-GrandStaff images (sampled
#                                 pixel stats: ~68% > 200, ~25% mid-gray, vs.
#                                 ~87%/~10% for the clean variant) -> forcing
#                                 adaptive binarization at eval time does NOT
#                                 match training. CORRECTED to `none`.
#   - zeus-olimpic:               trained on synthetic/OLiMPiC data without any
#                                 binarization step (confirmed via zeus.py's own
#                                 dataset-description "threshold:" transform
#                                 convention: the official synthetic training
#                                 command in the Zeus README never sets it)
#                                 -> CORRECTED to `none`.
# This is a real, citable methodological fix affecting the Camera-GrandStaff and
# Synthetic Zeus zero-shot rows in Table 5 -- see the paper's methodology
# section and REVISION_PLAN.md for the full writeup. Use --binarize to override
# this and reproduce the original (uncorrected) submission numbers.
DEFAULT_BINARIZE_BY_MODEL = {
    "zeus-grandstaff-lmx-1.0-2024-02-12.model": "adaptive",
    "zeus-camera-grandstaff-lmx-1.0-2024-02-12.model": "none",
    "zeus-olimpic-1.0-2024-02-12.model": "none",
}


def resolve_binarize_method(model_path: str, override: str | None) -> str:
    if override is not None:
        return override
    model_name = Path(model_path).name
    if model_name in DEFAULT_BINARIZE_BY_MODEL:
        return DEFAULT_BINARIZE_BY_MODEL[model_name]
    print(
        f"WARNING: no known corrected default binarization for model '{model_name}'. "
        f"Falling back to 'adaptive' (the original submission's uniform choice) -- "
        f"pass --binarize explicitly if this checkpoint needs different treatment.",
        file=sys.stderr,
    )
    return "adaptive"


# =============================================================================
# Image binarization (matches the original zeus_eval.ipynb protocol exactly)
# =============================================================================

def apply_binarization(img_pil: Image.Image, method: str = "adaptive") -> Image.Image:
    """Apply binarization to a PIL Image. Returns an RGB PIL Image."""
    if method == "none" or method is None:
        return img_pil.convert("RGB")

    img_gray = ImageOps.grayscale(img_pil)
    img_array = np.array(img_gray)

    if method == "otsu":
        hist, bin_edges = np.histogram(img_array, bins=256, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight = hist.astype(float) / hist.sum()
        cumsum_weight = np.cumsum(weight)
        cumsum_mean = np.cumsum(weight * bin_centers)
        global_mean = cumsum_mean[-1]
        denominator = cumsum_weight * (1 - cumsum_weight)
        valid = denominator != 0
        bcv = np.zeros_like(cumsum_weight)
        bcv[valid] = ((global_mean * cumsum_weight[valid] - cumsum_mean[valid]) ** 2) / denominator[valid]
        threshold = bin_centers[np.argmax(bcv)]
        img_binary = (img_array > threshold).astype(np.uint8) * 255

    elif method == "adaptive":
        from scipy.ndimage import uniform_filter

        block_size = 35
        C = 10
        local_mean = uniform_filter(img_array.astype(float), size=block_size, mode="reflect")
        img_binary = (img_array > (local_mean - C)).astype(np.uint8) * 255

    elif method == "fixed":
        img_binary = (img_array > 128).astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return Image.fromarray(img_binary, mode="L").convert("RGB")


# =============================================================================
# Metrics
# =============================================================================
# CER uses the same character-level Levenshtein convention as
# fine_tuning/eval_functions.py's compute_poliphony_metrics -- the canonical
# metric implementation referenced throughout this project. Zeus's own
# ser_metric.py only reports SER/SERnotuplets (token-level), so CER is computed
# here directly.

def _levenshtein(a, b) -> int:
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def compute_cer(gold_lmx_list: list, pred_lmx_list: list) -> float:
    total_edit, total_len = 0, 0
    for gold, pred in zip(gold_lmx_list, pred_lmx_list):
        gold_chars = list(gold.replace(" ", ""))
        pred_chars = list(pred.replace(" ", ""))
        total_edit += _levenshtein(gold_chars, pred_chars)
        total_len += len(gold_chars)
    return 100.0 * total_edit / total_len if total_len else 0.0


def compute_tedn(gold_musicxml_list: list, pred_lmx_list: list, flavor: str = "lmx") -> float:
    """Corpus-level TEDn (%), via olimpic-icdar24's TEDn_lmx_xml."""
    from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml

    total_gold_cost, total_edit_cost, n_errors = 0, 0, 0
    for gold_xml, pred_lmx in zip(gold_musicxml_list, pred_lmx_list):
        try:
            result = TEDn_lmx_xml(predicted_lmx=pred_lmx, gold_musicxml=gold_xml, flavor=flavor)
            total_gold_cost += result.gold_cost
            total_edit_cost += result.edit_cost
        except Exception as e:
            n_errors += 1
            print(f"  TEDn failed for one sample: {e}", file=sys.stderr)
    if n_errors:
        print(f"  TEDn: {n_errors}/{len(gold_musicxml_list)} samples failed", file=sys.stderr)
    return 100.0 * total_edit_cost / total_gold_cost if total_gold_cost else 100.0


# Tuplet-exception filtering, matching ser_metric.py's SERnotuplets convention.
_TUPLET_EXCEPTIONS = {"tuplet:start", "tuplet:stop"}
_TUPLET_EXCEPTION_RE = re.compile(r"^\d+in\d+$")


# =============================================================================
# Dataset preparation
# =============================================================================

def discover_samples(dataset_dir: Path, exclude_documents: list) -> list:
    """Find (musicxml_path, image_path, document, folio) tuples under dataset_dir,
    matching the paired_system_level layout: dataset_dir/document/folio/sample.{musicxml,jpg}."""
    exclude_set = set(exclude_documents or [])
    samples = []
    for musicxml_file in sorted(dataset_dir.rglob("*.musicxml")):
        rel_parts = musicxml_file.relative_to(dataset_dir).parts
        document = rel_parts[0]
        if document in exclude_set:
            continue
        image_file = musicxml_file.with_suffix(".jpg")
        if not image_file.exists():
            continue
        folio = rel_parts[1] if len(rel_parts) > 2 else ""
        samples.append({
            "musicxml_path": musicxml_file,
            "image_path": image_file,
            "document": document,
            "folio": folio,
            "sample_id": musicxml_file.stem,
        })
    return samples


def linearize_gt(musicxml_path: Path, Linearizer) -> tuple[str, str]:
    error_buffer = io.StringIO()
    linearizer = Linearizer(errout=error_buffer, fail_on_unknown_tokens=False)
    tree = ET.parse(musicxml_path)
    root = tree.getroot()
    for part in root.findall(".//part"):
        linearizer.process_part(part)
    return " ".join(linearizer.output_tokens), error_buffer.getvalue()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a Zeus .model directory (e.g. models/zeus-grandstaff-lmx-1.0-2024-02-12.model)")
    parser.add_argument("--dataset", required=True, help="Path to the paired_system_level dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for predictions and results")
    parser.add_argument("--zeus-script-dir", default=str(Path(__file__).resolve().parent),
                        help="Directory containing zeus.py (defaults to this script's directory)")
    parser.add_argument("--binarize", default=None, choices=["none", "adaptive", "otsu", "fixed"],
                        help="Override the corrected per-checkpoint default binarization (see module docstring). "
                             "Pass --binarize adaptive for ALL models to reproduce the original (uncorrected) submission.")
    parser.add_argument("--exclude-documents", nargs="*", default=[], help="Document IDs to exclude")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compute-tedn", action="store_true", help="Also compute TEDn (slow: ~30-120s/sample)")
    parser.add_argument("--tedn-flavor", default="lmx", choices=["lmx", "full"])
    parser.add_argument("--zeus-timeout", type=int, default=1200, help="Timeout (s) for the zeus.py subprocess")
    args = parser.parse_args()

    from app.linearization.Linearizer import Linearizer

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    binarize_method = resolve_binarize_method(args.model, args.binarize)
    print(f"Model: {args.model}")
    print(f"Binarization: {binarize_method}"
          f"{' (explicit override)' if args.binarize else ' (corrected default)'}")

    # --- Discover samples & linearize ground truth ---
    samples = discover_samples(dataset_dir, args.exclude_documents)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Dataset: {len(samples)} samples")

    gt_lmx, gt_musicxml_raw = [], []
    for s in samples:
        lmx, _errors = linearize_gt(s["musicxml_path"], Linearizer)
        s["gt_lmx"] = lmx
        gt_lmx.append(lmx)
        gt_musicxml_raw.append(s["musicxml_path"].read_text(encoding="utf-8"))

    # --- Build the batch pickle Zeus expects (image bytes + gt lmx) ---
    batch_samples = []
    for s in samples:
        img = Image.open(s["image_path"])
        img_processed = apply_binarization(img, method=binarize_method)
        buf = io.BytesIO()
        img_processed.save(buf, format="PNG")
        batch_samples.append({
            "path": s["sample_id"],
            "image": buf.getvalue(),
            "lmx": s["gt_lmx"],
            "musicxml": "",
        })

    batch_pickle_path = output_dir / f"batch_{binarize_method}.pickle"
    with open(batch_pickle_path, "wb") as f:
        pickle.dump(batch_samples, f)

    # --- Run Zeus (subprocess -- zeus.py is TensorFlow, its own CLI contract) ---
    print(f"Running zeus.py --load {args.model} ...")
    test_arg = str(batch_pickle_path.with_suffix(""))
    cmd = ["python3", "zeus.py", "--load", os.path.abspath(args.model), "--exp", str(output_dir.resolve()), "--test", test_arg]
    start = time.time()
    proc = subprocess.run(cmd, cwd=args.zeus_script_dir, capture_output=True, text=True, timeout=args.zeus_timeout)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"zeus.py failed (exit {proc.returncode}) after {elapsed:.1f}s")
    print(f"zeus.py completed in {elapsed:.1f}s ({elapsed / max(len(samples), 1):.2f}s/sample)")

    predicted_file = output_dir / f"{batch_pickle_path.with_suffix('').name}.predicted.lmx"
    with open(predicted_file, "r", encoding="utf-8") as f:
        predictions = [line.rstrip("\r\n") for line in f]
    if len(predictions) != len(samples):
        print(f"WARNING: {len(predictions)} predictions for {len(samples)} samples -- truncating/padding", file=sys.stderr)

    # --- Metrics ---
    ser_metrics = ser_metric.ser_metric(gt_lmx, predictions)
    cer = compute_cer(gt_lmx, predictions)

    result = {
        "model": args.model,
        "binarize": binarize_method,
        "binarize_is_override": args.binarize is not None,
        "n_samples": len(samples),
        "SER": ser_metrics["SER"],
        "SERnotuplets": ser_metrics["SERnotuplets"],
        "CER": cer,
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "excluded_documents": args.exclude_documents,
    }

    if args.compute_tedn:
        print(f"Computing TEDn (flavor={args.tedn_flavor})...")
        result[f"TEDn_{args.tedn_flavor}"] = compute_tedn(gt_musicxml_raw, predictions, flavor=args.tedn_flavor)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS ({Path(args.model).name}, binarize={binarize_method})")
    print("=" * 60)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}%")
    print(f"\nSaved to {results_path}")

    # Cleanup the intermediate pickle (predictions/results.json are kept).
    batch_pickle_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
