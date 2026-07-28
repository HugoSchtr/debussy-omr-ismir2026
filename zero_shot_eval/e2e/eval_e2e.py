#!/usr/bin/env python3
"""Zero-shot evaluation of the FCN/CRNN/CNNT end-to-end OMR baselines
(Ríos-Vila et al.) on a paired system-level dataset (Table 4 / Table 5 of the
ISMIR 2026 Debussy Dataset paper).

This is a cleaned, standalone re-implementation of the core evaluation logic
in the original exploratory notebook (`e2e_pianoform_eval.ipynb`): load a
Lightning checkpoint's raw `nn.Module` (no Lightning/gin dependency needed at
inference time), preprocess an image (binarize -> grayscale -> resize -> rotate
90 degrees, matching the training pipeline in `data.py`), run CTC greedy
decoding, convert the predicted **kern (bekrn) tokens to MusicXML via music21
with a 4-level sanitization fallback (matching `run_smt_evaluation_optimized.py`'s
protocol), linearize both prediction and ground truth to LMX via the Zeus
Linearizer, and compute SER/CER/LER (standard and "filtered" -- unfair tokens
removed, see UNFAIR_TOKENS below) plus optional TEDn. Exploratory/visualization
cells from the notebook (matplotlib plots, per-sample distribution plots,
per-document breakdown tables, prediction-vs-GT image galleries) are
intentionally NOT reproduced here -- only the inference -> metrics pipeline
needed to reproduce the paper's numbers is kept.

Usage:
    python eval_e2e.py \\
        --checkpoint model_weights/CNNT-v5-camera.ckpt --arch CNNT \\
        --dataset /path/to/paired_system_level \\
        --output ./results/CNNT-v5-camera \\
        --binarize adaptive --compute-tedn
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, groupby
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

# olimpic-icdar24 (Mayer et al.) -- required for the LMX linearizer. Not
# pip-installable: clone https://github.com/ufal/olimpic-icdar24 and set
# OLIMPIC_ICDAR24_DIR, or place it as a sibling ../olimpic-icdar24 directory
# next to this repo (default).
OLIMPIC_ICDAR24_DIR = os.environ.get(
    "OLIMPIC_ICDAR24_DIR",
    str(Path(__file__).resolve().parent.parent / "olimpic-icdar24"),
)
if OLIMPIC_ICDAR24_DIR not in sys.path:
    sys.path.insert(0, OLIMPIC_ICDAR24_DIR)


# =============================================================================
# Model architectures (from model/E2E_Score_Unfolding.py in the source repo,
# re-defined here without gin/lightning so this script has no training-only
# dependencies at inference time)
# =============================================================================

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        self.padding = None
        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h // 2, padding_h - padding_h // 2, padding_w // 2, padding_w - padding_w // 2]
                    padding = (0, 0)
        else:
            padding = (0, 0)
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        return self.point_conv(x)


class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)

    def forward(self, inputs):
        import random
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1, 1), kernel=3, activation=nn.ReLU, dropout=0.4):
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=kernel // 2)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel, padding=kernel // 2)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x


class DSCBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.4):
        super().__init__()
        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout / 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm_layer(x)
        x = self.conv3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1, 1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2, 2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2, 2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2, 2), dropout=dropout),
            ConvBlock(in_c=256, out_c=512, stride=(2, 1), dropout=dropout),
        ])
        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=512, out_c=512, stride=(1, 1), dropout=dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1, 1), dropout=dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1, 1), dropout=dropout),
            DSCBlock(in_c=512, out_c=512, stride=(1, 1), dropout=dropout),
        ])

    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim, len_max, device):
        super().__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)
        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start=0):
        if isinstance(start, int):
            return x + self.pe[:, :, start:start + x.size(2)].to(x.device)
        for i in range(x.size(0)):
            x[i] = x[i] + self.pe[0, :, start[i]:start[i] + x.size(2)]
        return x


class RecurrentScoreUnfolding(nn.Module):
    """CRNN decoder: LSTM-based."""
    def __init__(self, out_cats):
        super().__init__()
        self.dec_lstm = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.out_dense = nn.Linear(512, out_cats)

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        x = inputs.reshape(b, c, h * w).permute(0, 2, 1)
        x, _ = self.dec_lstm(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)


class TransformerScoreUnfolding(nn.Module):
    """CNNT decoder: Transformer-based."""
    def __init__(self, out_cats, max_len):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.pos_encoding = PositionalEncoding1D(dim=512, len_max=max_len, device=self.dummy_param.device)
        transf_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.dec_transf = nn.TransformerEncoder(transf_layer, num_layers=1)
        self.out_dense = nn.Linear(512, out_cats)

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        x = inputs.reshape(b, c, h * w)
        x = self.pos_encoding(x)
        x = x.permute(0, 2, 1)
        x = self.dec_transf(x)
        x = self.out_dense(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)


class PageDecoder(nn.Module):
    """FCN decoder: fully convolutional (5x5 conv)."""
    def __init__(self, out_cats):
        super().__init__()
        self.dec_conv = nn.Conv2d(in_channels=512, out_channels=out_cats, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, inputs):
        x = self.dec_conv(inputs)
        x = F.log_softmax(x, dim=1)
        b, c, h, w = x.size()
        x = x.reshape(b, c, h * w)
        x = x.permute(2, 0, 1)
        return x


class E2EScore_FCN(nn.Module):
    def __init__(self, in_channels, out_cats):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = PageDecoder(out_cats=out_cats)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class E2EScore_CRNN(nn.Module):
    def __init__(self, in_channels, out_cats):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = RecurrentScoreUnfolding(out_cats=out_cats)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class E2EScore_CNNT(nn.Module):
    def __init__(self, in_channels, out_cats, max_len):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = TransformerScoreUnfolding(out_cats=out_cats, max_len=max_len)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


ARCH_CLASSES = {"FCN": E2EScore_FCN, "CRNN": E2EScore_CRNN, "CNNT": E2EScore_CNNT}


def load_e2e_model(checkpoint_path: str, arch: str, device: torch.device, cnnt_max_len: int = 10000):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    i2w = hp["i2w"]
    blank_idx = hp["blank_idx"]
    out_cats = blank_idx + 1

    if arch == "CRNN":
        model = E2EScore_CRNN(in_channels=1, out_cats=out_cats)
    elif arch == "CNNT":
        model = E2EScore_CNNT(in_channels=1, out_cats=out_cats, max_len=cnnt_max_len)
    elif arch == "FCN":
        model = E2EScore_FCN(in_channels=1, out_cats=out_cats)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        state_dict[k[len("model."):] if k.startswith("model.") else k] = v
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, i2w, blank_idx


# =============================================================================
# Preprocessing + CTC decoding (matches the training pipeline in data.py:
# grayscale -> optional binarization -> resize to fixed height -> rotate 90 CW)
# =============================================================================

def apply_binarization(img_pil: Image.Image, method: str = "none", block_size: int = 35, C: int = 10) -> Image.Image:
    if method == "none" or method is None:
        return ImageOps.grayscale(img_pil)

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
        if block_size % 2 == 0:
            block_size += 1
        local_mean = uniform_filter(img_array.astype(float), size=block_size, mode="reflect")
        img_binary = (img_array > (local_mean - C)).astype(np.uint8) * 255
    elif method == "fixed":
        img_binary = (img_array > 128).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return Image.fromarray(img_binary, mode="L")


def preprocess_image(image_path: str, binarization_method: str, target_height: int) -> torch.Tensor:
    img_pil = Image.open(image_path).convert("RGB")
    img_gray = apply_binarization(img_pil, method=binarization_method)
    img_np = np.array(img_gray)

    h_orig, w_orig = img_np.shape[:2]
    if h_orig != target_height:
        scale = target_height / h_orig
        new_width = int(np.ceil(w_orig * scale))
        img_np = cv2.resize(img_np, (new_width, target_height), interpolation=cv2.INTER_LINEAR)

    img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    img_tensor = transforms.ToTensor()(img_np)
    return img_tensor.unsqueeze(0)


@torch.inference_mode()
def ctc_greedy_decode(model, img_tensor, i2w, blank_idx, device):
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor).permute(1, 0, 2).contiguous()[0]
    out_best = torch.argmax(pred, dim=1)
    out_best = [k for k, _ in groupby(out_best.tolist())]
    return [i2w[c] for c in out_best if c != blank_idx]


def tokens_to_kern(tokens: list) -> str:
    raw = "".join(tokens)
    return raw.replace("<t>", "\t").replace("<b>", "\n").replace("<s>", " ")


def predict_single(model, image_path, i2w, blank_idx, device, binarization, target_height):
    img_tensor = preprocess_image(image_path, binarization, target_height)
    tokens = ctc_greedy_decode(model, img_tensor, i2w, blank_idx, device)
    return tokens, tokens_to_kern(tokens)


# =============================================================================
# **kern sanitization (3-level fallback, matching run_smt_evaluation_optimized.py)
# =============================================================================

def sanitize_kern_basic(kern_content):
    if not kern_content or not kern_content.strip():
        return None
    lines = kern_content.split("\n")
    sanitized_lines, has_kern_header, has_terminator, current_spine_count = [], False, False, 0
    for i, line in enumerate(lines):
        line = line.rstrip()
        if not line and i == 0:
            continue
        if line.startswith("**ekern") or line.startswith("**kern"):
            line = "\t".join(["**kern" if col.startswith("**e") or col.startswith("**k") else col for col in line.split("\t")])
            has_kern_header = True
            current_spine_count = line.count("**kern")
            sanitized_lines.append(line)
            continue
        if line.startswith("*-"):
            has_terminator = True
            if current_spine_count > 0:
                line = "\t".join(["*-"] * current_spine_count)
            sanitized_lines.append(line)
            continue
        if line.startswith("*"):
            tokens = line.split("\t")
            splits = sum(1 for t in tokens if t == "*^")
            joins = sum(1 for t in tokens if t == "*v")
            current_spine_count += splits - joins
            sanitized_lines.append(line)
            continue
        if line.startswith("!!!") or line.startswith("!!"):
            sanitized_lines.append(line)
            continue
        if line and not line.startswith("!"):
            tokens = line.split("\t")
            if current_spine_count > 0 and len(tokens) > current_spine_count:
                line = "\t".join(tokens[:current_spine_count])
            elif current_spine_count > 0 and len(tokens) < current_spine_count:
                tokens.extend(["."] * (current_spine_count - len(tokens)))
                line = "\t".join(tokens)
            sanitized_lines.append(line)
        elif line:
            sanitized_lines.append(line)
    if not has_kern_header and sanitized_lines:
        first_data = next((l for l in sanitized_lines if not l.startswith("!") and not l.startswith("*") and l.strip()), None)
        if first_data:
            current_spine_count = len(first_data.split("\t")) or 1
            sanitized_lines.insert(0, "\t".join(["**kern"] * current_spine_count))
    if not has_terminator:
        current_spine_count = current_spine_count or 1
        sanitized_lines.append("\t".join(["*-"] * current_spine_count))
    return "\n".join(sanitized_lines) if len(sanitized_lines) >= 3 else None


def sanitize_kern_advanced(kern_content):
    kern_content = sanitize_kern_basic(kern_content)
    if not kern_content:
        return None
    lines = kern_content.split("\n")
    sanitized_lines, base_columns = [], 0
    for line in lines:
        if line.startswith("**kern"):
            base_columns = line.count("**kern")
            sanitized_lines.append(line)
            continue
        if line.startswith("*-") or line.startswith("!"):
            sanitized_lines.append(line)
            continue
        if line.startswith("*"):
            tokens = [t for t in line.split("\t") if t]
            cleaned = ["*" if t in ("*^", "*v", "*x") else t for t in tokens]
            if base_columns > 0:
                cleaned = (cleaned + ["*"] * base_columns)[:base_columns] if len(cleaned) < base_columns else cleaned[:base_columns]
            sanitized_lines.append("\t".join(cleaned))
            continue
        if not line.strip():
            continue
        tokens = [t if t else "." for t in line.split("\t")]
        if base_columns > 0:
            tokens = (tokens + ["."] * base_columns)[:base_columns] if len(tokens) < base_columns else tokens[:base_columns]
        sanitized_lines.append("\t".join(tokens))
    final_lines = []
    for line in sanitized_lines:
        if line.startswith("*-") and base_columns > 0:
            line = "\t".join(["*-"] * base_columns)
        final_lines.append(line)
    return "\n".join(final_lines) if len(final_lines) >= 3 else None


def sanitize_kern_nuclear(kern_content):
    if not kern_content or not kern_content.strip():
        return None
    lines = kern_content.split("\n")
    base_columns = 0
    for line in lines:
        if line.startswith("**kern") or line.startswith("**ekern"):
            base_columns = sum(1 for t in line.split("\t") if t.startswith("**"))
            break
    if base_columns == 0:
        for line in lines:
            if line and not line.startswith("!") and not line.startswith("*"):
                tokens = [t for t in line.split("\t") if t]
                if tokens:
                    base_columns = len(tokens)
                    break
    base_columns = base_columns or 2

    final_lines = ["\t".join(["**kern"] * base_columns)]
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("**") or line.startswith("*-"):
            continue
        if line.startswith("!"):
            tokens = [t if t.startswith("!") else "!" for t in line.split("\t")]
            tokens = (tokens + ["!"] * base_columns)[:base_columns]
            final_lines.append("\t".join(tokens))
            continue
        if line.startswith("*"):
            cleaned = []
            for token in line.split("\t"):
                if token.startswith("*clef") or token.startswith("*k[") or token.startswith("*M") or token.startswith("=") or token == "*":
                    cleaned.append(token)
                else:
                    cleaned.append("*")
            cleaned = (cleaned + ["*"] * base_columns)[:base_columns]
            final_lines.append("\t".join(cleaned))
            continue
        tokens = [t if t else "." for t in line.split("\t")]
        tokens = (tokens + ["."] * base_columns)[:base_columns]
        final_lines.append("\t".join(tokens))
    final_lines.append("\t".join(["*-"] * base_columns))
    return "\n".join(final_lines) if len(final_lines) >= 3 else None


class ConversionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ConversionTimeout("Conversion timed out")


def kern_to_musicxml(kern_content: str, output_path: str, timeout_sec: int = 10) -> bool:
    from music21 import converter as m21_converter
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(timeout_sec)
        score = m21_converter.parse(kern_content, format="humdrum")
        score.write("musicxml", fp=output_path)
        signal.alarm(0)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
    except (Exception, ConversionTimeout):
        signal.alarm(0)
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def convert_kern_to_musicxml_with_fallback(kern_content: str, output_path: str) -> tuple:
    if kern_to_musicxml(kern_content, output_path):
        return True, "direct"
    for level, sanitize_fn in (
        ("sanitize_basic", sanitize_kern_basic),
        ("sanitize_advanced", sanitize_kern_advanced),
        ("sanitize_nuclear", sanitize_kern_nuclear),
    ):
        sanitized = sanitize_fn(kern_content)
        if sanitized and kern_to_musicxml(sanitized, output_path):
            return True, level
    return False, "failed"


def linearize_musicxml_file(musicxml_path: str, Linearizer) -> str | None:
    try:
        tree = ET.parse(musicxml_path)
        part = tree.getroot().find(".//part")
        if part is None:
            return None
        linearizer = Linearizer(errout=io.StringIO(), fail_on_unknown_tokens=False)
        linearizer.process_part(part)
        return " ".join(linearizer.output_tokens)
    except Exception:
        return None


def linearize_gt_musicxml(musicxml_path: str, Linearizer) -> str | None:
    try:
        linearizer = Linearizer(errout=io.StringIO(), fail_on_unknown_tokens=False)
        tree = ET.parse(musicxml_path)
        for part in tree.getroot().findall(".//part"):
            linearizer.process_part(part)
        return " ".join(linearizer.output_tokens)
    except Exception:
        return None


# =============================================================================
# Metrics
# =============================================================================

def levenshtein_tokens(a, b) -> int:
    if isinstance(a, str):
        a = a.split()
    if isinstance(b, str):
        b = b.split()
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    distances = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        prev, distances = distances, [i] + [0] * len_b
        for j in range(1, len_b + 1):
            distances[j] = min(distances[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return distances[-1]


def levenshtein_chars(a: str, b: str) -> int:
    return levenshtein_tokens(list(a), list(b))


_TUPLET_EXCEPTIONS = {"tuplet:start", "tuplet:stop"}
_TUPLET_EXCEPTION_RE = re.compile(r"^\d+in\d+$")

# Tokens that appear in GT MusicXML (linearized by Zeus) but that kern-based
# models structurally cannot produce (the kern->MusicXML converter never
# generates the corresponding MusicXML elements). Removing these from both GT
# and prediction before comparison yields the "filtered" fSER/fCER/fLER
# metrics -- a fairer measure of what the model can actually represent. See
# VOCABULARY_EVALUATION_v2.md in the original evaluation repo.
UNFAIR_TOKENS = {
    "slur:start", "slur:stop",
    "staccato", "tenuto", "accent", "strong-accent", "arpeggiate", "trill-mark", "fermata",
    "grace", "grace:slash",
    "tremolo:1", "tremolo:2", "tremolo:3", "tremolo:4",
    "tremolo:single", "tremolo:start", "tremolo:stop", "tremolo:unmeasured",
}


def filter_unfair_tokens(tokens: list) -> list:
    return [t for t in tokens if t not in UNFAIR_TOKENS]


def compute_tedn_for_pair(pred_lmx: str, gt_musicxml_path: str, flavor: str) -> float | None:
    """Corpus-contribution TEDn for one sample; returns None on failure."""
    from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
    try:
        with open(gt_musicxml_path, "r", encoding="utf-8") as f:
            gold_musicxml = f.read()
        result = TEDn_lmx_xml(predicted_lmx=pred_lmx, gold_musicxml=gold_musicxml, flavor=flavor, errout=io.StringIO())
        return result.edit_cost, result.gold_cost
    except Exception:
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file (e.g. CNNT-v5-camera.ckpt)")
    parser.add_argument("--arch", required=True, choices=["FCN", "CRNN", "CNNT"])
    parser.add_argument("--dataset", required=True, help="Path to the paired_system_level dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for predictions and results")
    parser.add_argument("--binarize", default="adaptive", choices=["none", "adaptive", "otsu", "fixed"],
                        help="Preprocessing binarization method. NOTE: the original evaluation used "
                             "'adaptive' uniformly for ALL models including camera-trained variants -- "
                             "see README.md for why this is likely low-impact (unlike the Zeus case) "
                             "but was not re-verified with full rigor.")
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--cnnt-max-len", type=int, default=10000)
    parser.add_argument("--exclude-documents", nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compute-tedn", action="store_true")
    parser.add_argument("--tedn-flavors", nargs="*", default=["lmx", "full"], choices=["lmx", "full"])
    parser.add_argument("--device", default=None, help="cuda|cpu|mps (default: auto-detect)")
    args = parser.parse_args()

    from app.linearization.Linearizer import Linearizer

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    (output_dir / "predictions_kern").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions_musicxml").mkdir(parents=True, exist_ok=True)

    # --- Dataset discovery ---
    exclude_set = set(args.exclude_documents)
    musicxml_files = [
        f for f in sorted(dataset_dir.rglob("*.musicxml"))
        if f.relative_to(dataset_dir).parts[0] not in exclude_set
    ]
    if args.max_samples:
        musicxml_files = musicxml_files[: args.max_samples]

    dataset = []
    for mxml_file in musicxml_files:
        img_file = mxml_file.with_suffix(".jpg")
        if not img_file.exists():
            continue
        rel = mxml_file.relative_to(dataset_dir)
        dataset.append({
            "sample_id": mxml_file.stem,
            "image_path": str(img_file),
            "gt_musicxml_path": str(mxml_file),
            "document": rel.parts[0],
        })
    print(f"Dataset: {len(dataset)} samples ({len(set(s['document'] for s in dataset))} documents)")

    print("Precomputing GT linearizations...")
    gt_data = {}
    for sample in dataset:
        lmx = linearize_gt_musicxml(sample["gt_musicxml_path"], Linearizer)
        if lmx:
            gt_data[sample["sample_id"]] = lmx.split()
    print(f"GT linearizations: {len(gt_data)}/{len(dataset)}")

    # --- Load model & run predictions ---
    print(f"Loading {args.arch} from {args.checkpoint} ...")
    model, i2w, blank_idx = load_e2e_model(args.checkpoint, args.arch, device, args.cnnt_max_len)

    print(f"Running predictions (binarize={args.binarize})...")
    kern_predictions = {}
    t0 = time.time()
    for i, sample in enumerate(dataset):
        try:
            _, kern_string = predict_single(model, sample["image_path"], i2w, blank_idx, device, args.binarize, args.target_height)
            kern_predictions[sample["sample_id"]] = kern_string
            (output_dir / "predictions_kern" / f"{sample['sample_id']}.krn").write_text(kern_string)
        except Exception as e:
            print(f"  Prediction failed for {sample['sample_id']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dataset)}")
    print(f"Predictions: {len(kern_predictions)}/{len(dataset)} in {time.time() - t0:.1f}s")

    print("Converting kern -> MusicXML -> LMX...")
    pred_lmx = {}
    conversion_stats = Counter()
    for sid, kern_str in kern_predictions.items():
        mxml_path = str(output_dir / "predictions_musicxml" / f"{sid}.musicxml")
        success, method = convert_kern_to_musicxml_with_fallback(kern_str, mxml_path)
        conversion_stats[method] += 1
        if success:
            lmx = linearize_musicxml_file(mxml_path, Linearizer)
            if lmx:
                pred_lmx[sid] = lmx.split()
            else:
                conversion_stats["linearization_failed"] += 1
    conv_rate = 100.0 * len(pred_lmx) / max(len(kern_predictions), 1)
    print(f"Conversion: {dict(conversion_stats)} -> {len(pred_lmx)}/{len(kern_predictions)} ({conv_rate:.1f}%)")

    with open(output_dir / "predicted_lmx.json", "w", encoding="utf-8") as f:
        json.dump({sid: " ".join(toks) for sid, toks in pred_lmx.items()}, f)

    # --- Metrics (global, corpus-level; failed conversions count as 100% error) ---
    acc = Counter()  # total_edit / total_gt for each of: tok, char, line, ftok, fchar, fline

    def _acc(gt_tok, pred_tok):
        gt_f, pred_f = filter_unfair_tokens(gt_tok), filter_unfair_tokens(pred_tok)
        gt_chars, pred_chars = "".join(gt_tok), "".join(pred_tok)
        gt_f_chars, pred_f_chars = "".join(gt_f), "".join(pred_f)
        acc["edit_tok"] += levenshtein_tokens(gt_tok, pred_tok)
        acc["len_tok"] += len(gt_tok)
        acc["edit_char"] += levenshtein_chars(gt_chars, pred_chars)
        acc["len_char"] += len(gt_chars)
        acc["edit_line"] += levenshtein_tokens([" ".join(gt_tok)], [" ".join(pred_tok)])
        acc["len_line"] += 1
        acc["edit_ftok"] += levenshtein_tokens(gt_f, pred_f)
        acc["len_ftok"] += len(gt_f)
        acc["edit_fchar"] += levenshtein_chars(gt_f_chars, pred_f_chars)
        acc["len_fchar"] += len(gt_f_chars)
        acc["edit_fline"] += levenshtein_tokens([" ".join(gt_f)], [" ".join(pred_f)])
        acc["len_fline"] += 1

    n_success, n_conv_failed, n_no_gt = 0, 0, 0
    for sample in dataset:
        sid = sample["sample_id"]
        gt_tok = gt_data.get(sid)
        if gt_tok is None:
            n_no_gt += 1
            continue
        if sid not in pred_lmx:
            _acc(gt_tok, [])  # conversion failed -> counted as a full miss
            n_conv_failed += 1
            continue
        _acc(gt_tok, pred_lmx[sid])
        n_success += 1

    def pct(edit_key, len_key):
        return 100.0 * acc[edit_key] / acc[len_key] if acc[len_key] else 0.0

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "binarize": args.binarize,
        "n_samples": len(dataset),
        "n_success": n_success,
        "n_conversion_failed": n_conv_failed,
        "n_no_gt": n_no_gt,
        "conversion_rate_pct": conv_rate,
        "conversion_stats": dict(conversion_stats),
        "SER": pct("edit_tok", "len_tok"),
        "CER": pct("edit_char", "len_char"),
        "LER": pct("edit_line", "len_line"),
        "fSER": pct("edit_ftok", "len_ftok"),
        "fCER": pct("edit_fchar", "len_fchar"),
        "fLER": pct("edit_fline", "len_fline"),
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "excluded_documents": args.exclude_documents,
    }

    if args.compute_tedn:
        # NOTE: matching the original notebook's protocol (cell 10b), the global
        # (corpus-level) TEDn is computed only over samples with a successful
        # kern->MusicXML->LMX conversion -- unlike SER/CER/LER above, conversion
        # failures do NOT contribute a 100%-error term to this aggregate (the
        # notebook only reflected failures in the separate mean/median-over-samples
        # view, which this script does not reproduce). Report n_conversion_failed
        # alongside TEDn so this scope is clear when comparing to SER/CER/LER.
        print(f"Computing TEDn (flavors={args.tedn_flavors}, over {len(pred_lmx)} convertible samples)...")
        for flavor in args.tedn_flavors:
            total_edit, total_gold = 0, 0
            for sample in dataset:
                sid = sample["sample_id"]
                if sid not in pred_lmx or sid not in gt_data:
                    continue
                out = compute_tedn_for_pair(" ".join(pred_lmx[sid]), sample["gt_musicxml_path"], flavor)
                if out is None:
                    continue
                edit_cost, gold_cost = out
                total_edit += edit_cost
                total_gold += gold_cost
            result[f"TEDn_{flavor}"] = 100.0 * total_edit / total_gold if total_gold else 100.0

    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS ({args.arch}, {Path(args.checkpoint).name}, binarize={args.binarize})")
    print("=" * 60)
    for k in ("SER", "CER", "LER", "fSER", "fCER", "fLER"):
        print(f"  {k}: {result[k]:.2f}%")
    for k, v in result.items():
        if k.startswith("TEDn_"):
            print(f"  {k}: {v:.2f}%")
    print(f"\nSaved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
