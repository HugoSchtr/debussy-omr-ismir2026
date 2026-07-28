#!/usr/bin/env python3
"""
SMT Full-Page Evaluation Script for SLURM Cluster

Evaluation protocol aligned with Zeus evaluation notebook:
- Token-level SER (Levenshtein on token lists, NOT character-level)
- Global SER = total_edit_distance / total_gt_tokens
- Multiple binarization options: none, adaptive, otsu
- Vocabulary analysis & OOV-tuplet filtering
- Document exclusion
- Timestamped output directories
- Dedicated logs folder

Dataset: paired_by_document_layout (full-page level images)
Model: PRAIG/smt-fp-grandstaff

Pipeline:
1. Image → SMT full-page model → **ekern prediction
2. **ekern → MusicXML (music21 + advanced sanitization)
3. MusicXML → Zeus Linearizer → predicted LMX tokens
4. GT MusicXML → Zeus Linearizer → GT LMX tokens
5. Compare with token-level Levenshtein → SER
"""

import sys
import os
import argparse
import io
import json
import re
import yaml
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from music21 import converter


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging with both console and file output."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# IMAGE PREPROCESSING - Multiple binarization methods (matching notebook)
# =============================================================================

def apply_binarization(img_pil: Image.Image, method: str = 'adaptive',
                       block_size: int = 35, C: int = 10) -> Image.Image:
    """
    Apply binarization to a PIL Image using different methods.
    Matches the notebook's binarization protocol exactly.

    Args:
        img_pil: PIL Image object
        method: 'adaptive', 'otsu', 'fixed', or 'none' (no binarization, just RGB)
        block_size: Local neighborhood size for adaptive thresholding
        C: Constant subtracted from local mean

    Returns:
        RGB PIL Image (binarized or original RGB)
    """
    if method == 'none' or method is None:
        return img_pil.convert('RGB')

    # Convert to grayscale
    img_gray = ImageOps.grayscale(img_pil)
    img_array = np.array(img_gray)

    if method == 'otsu':
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

    elif method == 'adaptive':
        if block_size % 2 == 0:
            block_size += 1
        local_mean = uniform_filter(img_array.astype(float), size=block_size, mode='reflect')
        img_binary = (img_array > (local_mean - C)).astype(np.uint8) * 255

    elif method == 'fixed':
        img_binary = (img_array > 128).astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return Image.fromarray(img_binary, mode='L').convert('RGB')


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(base_dir: str, test_mode: Optional[str] = None,
                 test_document: str = None, test_folio: str = None,
                 max_samples: Optional[int] = None,
                 exclude_documents: Optional[List[str]] = None,
                 dataset_type: str = "system") -> List[Dict]:
    """Load dataset from paired_system_level or paired_by_document_layout directory.
    
    Args:
        dataset_type: "system" for paired_system_level (nested), "fullpage" for paired_by_document_layout (flat)
    """
    if dataset_type == "fullpage":
        base_path = Path(base_dir) / "paired_by_document_layout"
    else:
        base_path = Path(base_dir) / "paired_system_level"

    if not base_path.exists():
        raise ValueError(f"Dataset directory not found: {base_path}")

    exclude_set = set(exclude_documents) if exclude_documents else set()
    dataset = []
    documents = sorted([d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')])

    for doc_dir in documents:
        document_id = doc_dir.name

        if document_id in exclude_set:
            continue

        if test_mode == "document" and document_id != test_document:
            continue

        if dataset_type == "fullpage":
            # Full-page: flat structure, files directly in document directory
            # Files named like: {document_id}_{folio}.jpg
            for img_file in sorted(doc_dir.glob("*.jpg")):
                sample_id = img_file.stem
                gt_musicxml = doc_dir / f"{sample_id}.musicxml"

                if gt_musicxml.exists():
                    # Extract folio from filename (e.g., "12148_btv1b10073993b_f2" -> "f2")
                    folio_id = sample_id.split('_')[-1] if '_' in sample_id else "unknown"

                    if test_mode == "folio" and (document_id != test_document or folio_id != test_folio):
                        continue

                    dataset.append({
                        'sample_id': sample_id,
                        'image_path': str(img_file),
                        'gt_musicxml_path': str(gt_musicxml),
                        'document': document_id,
                        'folio': folio_id,
                        'filename': f"{sample_id}.musicxml",
                    })

                    if max_samples and len(dataset) >= max_samples:
                        break
        else:
            # System-level: nested structure with folio directories
            for folio_dir in sorted(doc_dir.iterdir()):
                if not folio_dir.is_dir():
                    continue

                folio_id = folio_dir.name

                if test_mode == "folio" and (document_id != test_document or folio_id != test_folio):
                    continue

                for img_file in sorted(folio_dir.glob("*.jpg")):
                    sample_id = img_file.stem
                    gt_musicxml = folio_dir / f"{sample_id}.musicxml"

                    if gt_musicxml.exists():
                        dataset.append({
                            'sample_id': sample_id,
                            'image_path': str(img_file),
                            'gt_musicxml_path': str(gt_musicxml),
                            'document': document_id,
                            'folio': folio_id,
                            'filename': f"{sample_id}.musicxml",
                        })

                        if max_samples and len(dataset) >= max_samples:
                            break

                if max_samples and len(dataset) >= max_samples:
                    break

        if max_samples and len(dataset) >= max_samples:
            break

    if exclude_set:
        logging.info(f"Excluded {len(exclude_set)} document(s): {', '.join(sorted(exclude_set))}")

    return dataset


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_name: str, device: str):
    """Load SMT model from HuggingFace with PE buffer fix."""
    from smt_model import SMTModelForCausalLM
    from smt_model.modeling_smt import PositionalEncoding2D, PositionalEncoding1D

    logging.info(f"Loading model: {model_name}")
    model = SMTModelForCausalLM.from_pretrained(model_name)

    if device == "cpu":
        model = model.cpu()
    else:
        model = model.to(device)

    model.eval()

    # Reinitialize positional encoding buffers (transformers 5.x + PyTorch 2.10 fix)
    config = model.config
    h_max = config.maxh // model.height_reduction
    w_max = config.maxw // model.width_reduction
    model.pos2D = PositionalEncoding2D(dim=config.d_model, h_max=h_max, w_max=w_max).to(device)
    model.decoder.position_encoding = PositionalEncoding1D(
        dim=config.d_model, len_max=int(config.maxlen)
    ).to(device)
    logging.info("Reinitialized positional encoding buffers (transformers 5.x compatibility fix)")

    logging.info(f"Model loaded on {device}")
    return model, device


# =============================================================================
# FAST PREDICT WITH KV-CACHE
# =============================================================================

@torch.inference_mode()
def fast_predict(model, input_tensor):
    """
    Fast autoregressive prediction with KV-caching.
    Verified identical output to model.predict() with ~2x speedup.
    """
    device = input_tensor.device

    # Encode image (once)
    encoder_output = model.forward_encoder(input_tensor)

    # Pre-compute encoder features (once)
    encoder_output_2D = model.pos2D(encoder_output)
    encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1)
    encoder_features_2D = torch.flatten(encoder_output_2D, start_dim=2, end_dim=3).permute(0, 2, 1)

    # Pre-compute cross-attention K,V for each decoder layer (once)
    num_layers = len(model.decoder.decoder.layers)
    cross_kv_cache = []
    for layer in model.decoder.decoder.layers:
        ca = layer.cross_attn
        k = ca._split_heads(ca.k_proj(encoder_features_2D))
        v = ca._split_heads(ca.v_proj(encoder_features))
        cross_kv_cache.append((k, v))

    # Autoregressive decoding with KV-cache
    bos_id = model.w2i['<bos>']
    eos_name = '<eos>'
    text_sequence = []
    self_kv_cache = [None] * num_layers
    current_token_id = bos_id

    for step in range(model.maxlen - 1):
        new_token = torch.tensor([[current_token_id]], device=device)
        x = model.decoder.embedding(new_token)
        x = model.decoder.position_encoding(x, start=step)

        for layer_idx, dec_layer in enumerate(model.decoder.decoder.layers):
            # Self-attention with KV-cache
            sa = dec_layer.self_attn
            q = sa._split_heads(sa.q_proj(x))
            k = sa._split_heads(sa.k_proj(x))
            v = sa._split_heads(sa.v_proj(x))

            if self_kv_cache[layer_idx] is not None:
                past_k, past_v = self_kv_cache[layer_idx]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            self_kv_cache[layer_idx] = (k, v)

            if sa.has_flash_attn:
                sa_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=sa.scale,
                )
            else:
                w = (q @ k.transpose(-2, -1)) * sa.scale
                sa_out = F.softmax(w, dim=-1) @ v

            sa_out = sa.out_proj(sa._merge_heads(sa_out))
            x = dec_layer.norm_layers[0](x + dec_layer.dropout_layers[0](sa_out))

            # Cross-attention with cached encoder K,V
            ca = dec_layer.cross_attn
            q_ca = ca._split_heads(ca.q_proj(x))
            cached_k, cached_v = cross_kv_cache[layer_idx]

            if ca.has_flash_attn:
                ca_out = F.scaled_dot_product_attention(
                    q_ca, cached_k, cached_v, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=ca.scale,
                )
            else:
                w = (q_ca @ cached_k.transpose(-2, -1)) * ca.scale
                ca_out = F.softmax(w, dim=-1) @ cached_v

            ca_out = ca.out_proj(ca._merge_heads(ca_out))
            x = dec_layer.norm_layers[1](x + dec_layer.dropout_layers[1](ca_out))

            # FFN
            ffn_out = dec_layer.ffn(x)
            x = dec_layer.norm_layers[2](x + dec_layer.dropout_layers[2](ffn_out))

        x_drop = model.decoder.dropout(x)
        logits = model.decoder.vocab_projection(x_drop)
        current_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        token_name = model.i2w[current_token_id]
        if token_name == eos_name:
            break
        text_sequence.append(token_name)

    return text_sequence


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_sample(model, image_path: str, device: str,
                          max_seq_len: int = 8096,
                          binarization_method: str = 'adaptive',
                          binarization_block_size: int = 35,
                          binarization_C: int = 10) -> str:
    """Run SMT prediction on a single image."""
    # Load image as RGB PIL
    pil_img = Image.open(image_path).convert('RGB')

    # Apply binarization (matching notebook protocol)
    pil_img = apply_binarization(
        pil_img, method=binarization_method,
        block_size=binarization_block_size, C=binarization_C,
    )

    # Resize to target_height=256, width rounded to multiple of 16
    original_width, original_height = pil_img.size
    target_height = 256
    scale = target_height / original_height
    new_width = max((int(original_width * scale) // 16) * 16, 16)
    new_height = max((target_height // 16) * 16, 16)

    pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

    # Transform: RGB → Grayscale → Tensor
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        predictions = fast_predict(model, img_tensor)
        kern_pred = "".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')

    return kern_pred


def run_predictions(model, dataset: List[Dict], predictions_dir: Path,
                    device: str, config: Dict, checkpoint_data: Dict) -> Dict[str, str]:
    """Run predictions on all samples with checkpointing."""
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    processed_samples = set(checkpoint_data.get('processed_samples', []))

    logging.info(f"Running predictions on {len(dataset)} samples...")
    logging.info(f"Resuming: {len(processed_samples)} samples already processed")
    logging.info(f"Binarization: {config.get('binarization_method', 'adaptive')}")

    for sample in tqdm(dataset, desc="Predictions"):
        sample_id = sample['sample_id']
        pred_file = predictions_dir / f"{sample_id}_prediction.txt"

        if sample_id in processed_samples and pred_file.exists():
            predictions[sample_id] = str(pred_file)
            continue

        try:
            kern_pred = predict_single_sample(
                model, sample['image_path'], device,
                config.get('max_seq_len', 8096),
                binarization_method=config.get('binarization_method', 'adaptive'),
                binarization_block_size=config.get('binarization_block_size', 35),
                binarization_C=config.get('binarization_C', 10),
            )

            with open(pred_file, 'w') as f:
                f.write(kern_pred)

            predictions[sample_id] = str(pred_file)
            processed_samples.add(sample_id)

            if len(predictions) % config.get('save_checkpoint_every', 50) == 0:
                save_checkpoint(predictions_dir.parent, {'processed_samples': list(processed_samples)})

        except Exception as e:
            logging.error(f"Prediction failed for {sample_id}: {e}")
            continue

    save_checkpoint(predictions_dir.parent, {'processed_samples': list(processed_samples)})

    return predictions


# =============================================================================
# KERN SANITIZATION
# =============================================================================

def sanitize_kern(kern_content: str) -> Optional[str]:
    """Basic **kern sanitization: fix headers and terminators."""
    if not kern_content or not kern_content.strip():
        return None

    lines = kern_content.split('\n')
    sanitized_lines = []
    current_spine_count = 0

    for i, line in enumerate(lines):
        line = line.rstrip()

        if line.startswith('**ekern') or line.startswith('**kern'):
            line = '\t'.join(['**kern' if col.startswith('**') else col
                              for col in line.split('\t')])
            current_spine_count = line.count('**kern')
            sanitized_lines.append(line)
            continue

        if line.startswith('*-'):
            if current_spine_count > 0:
                line = '\t'.join(['*-'] * current_spine_count)
            sanitized_lines.append(line)
            continue

        sanitized_lines.append(line)

    return '\n'.join(sanitized_lines)


def sanitize_kern_advanced(kern_content: str) -> Optional[str]:
    """
    Advanced **kern sanitization: flatten spine manipulations, fix column counts,
    remove embedded interpretation tokens from data columns.
    """
    if not kern_content or not kern_content.strip():
        return None

    lines = kern_content.split('\n')
    sanitized = []
    base_cols = 0

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        if line.startswith('**ekern') or line.startswith('**kern'):
            tokens = line.split('\t')
            base_cols = sum(1 for t in tokens if t.startswith('**'))
            sanitized.append('\t'.join(['**kern'] * base_cols))
            continue

        if line.startswith('*-'):
            sanitized.append('\t'.join(['*-'] * base_cols))
            continue

        if line.startswith('!'):
            tokens = line.split('\t')
            while len(tokens) < base_cols:
                tokens.append('!')
            sanitized.append('\t'.join(tokens[:base_cols]))
            continue

        # Process all lines: check each column for spine ops
        tokens = line.split('\t')
        cleaned_tokens = []
        is_interp_line = line.startswith('*')

        for t in tokens:
            t = t.strip()
            if not t:
                cleaned_tokens.append('*' if is_interp_line else '.')
                continue
            if t in ('*^', '*v', '*x'):
                cleaned_tokens.append('*')
                continue
            # Remove interpretation tokens embedded in data columns
            if not is_interp_line and ' ' in t:
                parts = t.split(' ')
                data_parts = [p for p in parts if not p.startswith('*')]
                cleaned_tokens.append(' '.join(data_parts) if data_parts else '.')
                continue
            cleaned_tokens.append(t)

        while len(cleaned_tokens) < base_cols:
            cleaned_tokens.append('*' if is_interp_line else '.')
        cleaned_tokens = cleaned_tokens[:base_cols]

        sanitized.append('\t'.join(cleaned_tokens))

    if len(sanitized) < 3:
        return None

    return '\n'.join(sanitized)


# =============================================================================
# LINEARIZATION - WITH CACHING
# =============================================================================

def setup_olimpic():
    """Setup Olimpic linearization."""
    olimpic_path = str(Path("./olimpic-icdar24").resolve())

    if not os.path.exists(olimpic_path):
        raise ValueError(f"Olimpic path not found: {olimpic_path}")

    if olimpic_path not in sys.path:
        sys.path.append(olimpic_path)

    from app.linearization.Linearizer import Linearizer
    from app.symbolic.MxlFile import MxlFile
    import xml.etree.ElementTree as ET

    return Linearizer, MxlFile, ET


def linearize_musicxml(musicxml_path: str, Linearizer, MxlFile, ET) -> Optional[str]:
    """Linearize predicted MusicXML using Zeus's linearization."""
    try:
        if musicxml_path.endswith('.mxl'):
            mxl = MxlFile.load_mxl(musicxml_path)
        else:
            with open(musicxml_path, 'r') as f:
                xml_content = f.read()
            mxl = MxlFile(ET.ElementTree(ET.fromstring(xml_content)))

        try:
            part = mxl.get_piano_part()
        except Exception:
            part = mxl.tree.find("part")

        if part is None or part.tag != "part":
            return None

        linearizer = Linearizer()
        linearizer.process_part(part)
        return " ".join(linearizer.output_tokens)

    except Exception:
        return None


def linearize_gt_musicxml(musicxml_path: str, Linearizer, MxlFile, ET) -> Tuple[Optional[str], str]:
    """
    Linearize GT MusicXML directly from file (matching notebook protocol).
    Uses fail_on_unknown_tokens=False and captures error output.
    Extracts only the piano part (matching predicted linearization).
    Returns (linearized_string, error_output).
    """
    try:
        error_buffer = io.StringIO()
        linearizer = Linearizer(errout=error_buffer, fail_on_unknown_tokens=False)

        # Use MxlFile to extract the piano part (same as predicted linearization)
        with open(musicxml_path, 'r') as f:
            xml_content = f.read()
        mxl = MxlFile(ET.ElementTree(ET.fromstring(xml_content)))

        try:
            part = mxl.get_piano_part()
        except Exception:
            # Fallback: look for <part-name>Piano</part-name> (not in olimpic's allowlist)
            tree_root = mxl.tree.getroot()
            part_list = tree_root.find('part-list')
            part = None
            if part_list is not None:
                piano_names = {"Piano", "Grand Piano", "Pianoforte", "Acoustic Grand Piano",
                               "Harpsichord", "Piano (2)"}
                for sp in part_list.findall('score-part'):
                    pname = sp.find('part-name')
                    if pname is not None and pname.text in piano_names:
                        pid = sp.attrib['id']
                        candidates = [p for p in tree_root.findall('part') if p.attrib.get('id') == pid]
                        if candidates:
                            part = candidates[0]
                            break
            # Last resort: first <part>
            if part is None:
                all_parts = tree_root.findall('part')
                if not all_parts:
                    return None, "No <part> element found"
                part = all_parts[0]

        if part is None or part.tag != "part":
            return None, "Could not extract piano part"

        linearizer.process_part(part)

        tokens = linearizer.output_tokens
        error_output = error_buffer.getvalue()

        return " ".join(tokens), error_output

    except Exception as e:
        return None, str(e)


def precompute_gt_linearizations(dataset: List[Dict], cache_dir: Path,
                                 Linearizer, MxlFile, ET) -> Dict[str, Dict]:
    """
    Precompute all GT linearizations with error tracking.
    Returns dict mapping sample_id → {lmx, tokens, n_tokens, ...}.
    """
    cache_file = cache_dir / "gt_linearizations_cache.json"

    if cache_file.exists():
        logging.info("Loading GT linearizations from cache...")
        with open(cache_file, 'r') as f:
            raw_cache = json.load(f)
        # Convert old string format to dict format
        gt_data = {}
        for sid, val in raw_cache.items():
            if isinstance(val, str):
                tokens = val.split()
                gt_data[sid] = {'lmx': val, 'tokens': tokens, 'n_tokens': len(tokens)}
            else:
                gt_data[sid] = val
        if len(gt_data) >= len(dataset) * 0.9:
            logging.info(f"Loaded {len(gt_data)} GT linearizations from cache")
            return gt_data

    logging.info(f"Precomputing GT linearizations for {len(dataset)} samples...")
    gt_data = {}

    for sample in tqdm(dataset, desc="GT Linearization"):
        sample_id = sample['sample_id']
        lmx, error_output = linearize_gt_musicxml(sample['gt_musicxml_path'], Linearizer, MxlFile, ET)
        if lmx:
            tokens = lmx.split()
            gt_data[sample_id] = {
                'lmx': lmx,
                'tokens': tokens,
                'n_tokens': len(tokens),
                'linearization_errors': error_output.count('[ERROR]'),
                'linearization_warnings': error_output.count('[WARNING]'),
            }

    # Save cache (serializable)
    cache_for_save = {sid: d['lmx'] for sid, d in gt_data.items()}
    with open(cache_file, 'w') as f:
        json.dump(cache_for_save, f)

    logging.info(f"Precomputed {len(gt_data)} GT linearizations")
    return gt_data


# =============================================================================
# KERN TO MUSICXML CONVERSION
# =============================================================================

def kern_to_musicxml(kern_content: str, output_path: str) -> bool:
    """Convert **kern to MusicXML using music21 (file-based for reliability)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.krn', delete=False) as tmp:
            tmp.write(kern_content)
            tmp_path = tmp.name
        score = converter.parse(tmp_path)
        os.unlink(tmp_path)
        tmp_path = None
        score.write('musicxml', fp=output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


def convert_predictions_to_musicxml(predictions_dir: Path, output_dir: Path,
                                    use_sanitization: bool = True) -> Tuple[Dict, Dict, Dict]:
    """
    Convert all **kern predictions to MusicXML with multi-strategy pipeline.
    Strategy: direct → advanced sanitization → basic sanitization.
    """
    musicxml_dir = output_dir / "predictions_musicxml"
    musicxml_dir.mkdir(parents=True, exist_ok=True)

    musicxml_files = {}
    conversion_methods = {}
    success_direct = 0
    success_advanced = 0
    success_basic = 0
    failed_count = 0

    prediction_files = list(predictions_dir.glob("*_prediction.txt"))

    logging.info(f"Converting {len(prediction_files)} predictions to MusicXML...")
    logging.info(f"Sanitization: {'ENABLED (basic + advanced)' if use_sanitization else 'DISABLED'}")

    for pred_file in tqdm(prediction_files, desc="Converting"):
        sample_id = pred_file.stem.replace('_prediction', '')

        with open(pred_file, 'r') as f:
            kern_content = f.read()

        musicxml_path = musicxml_dir / f"{sample_id}.musicxml"

        # Strategy 1: Direct
        if kern_to_musicxml(kern_content, str(musicxml_path)):
            musicxml_files[sample_id] = str(musicxml_path)
            conversion_methods[sample_id] = 'music21_direct'
            success_direct += 1
            continue

        if use_sanitization:
            # Strategy 2: Advanced sanitization
            sanitized_adv = sanitize_kern_advanced(kern_content)
            if sanitized_adv and kern_to_musicxml(sanitized_adv, str(musicxml_path)):
                musicxml_files[sample_id] = str(musicxml_path)
                conversion_methods[sample_id] = 'sanitize_advanced'
                success_advanced += 1
                continue

            # Strategy 3: Basic sanitization
            sanitized = sanitize_kern(kern_content)
            if sanitized and kern_to_musicxml(sanitized, str(musicxml_path)):
                musicxml_files[sample_id] = str(musicxml_path)
                conversion_methods[sample_id] = 'sanitize_basic'
                success_basic += 1
                continue

        conversion_methods[sample_id] = 'failed'
        failed_count += 1

    total = len(prediction_files)
    success_count = success_direct + success_advanced + success_basic

    stats = {
        'total': total,
        'successful': success_count,
        'success_direct': success_direct,
        'success_advanced': success_advanced,
        'success_basic': success_basic,
        'failed': failed_count,
        'success_rate': 100 * success_count / total if total > 0 else 0,
    }

    if total > 0:
        logging.info(f"Conversion: {success_count}/{total} ({stats['success_rate']:.1f}%)")
        logging.info(f"   Direct: {success_direct}, Advanced sanitize: {success_advanced}, Basic: {success_basic}")
        logging.info(f"   Failed: {failed_count}/{total}")

    return musicxml_files, conversion_methods, stats


# =============================================================================
# METRICS - Token-level Levenshtein (matching notebook protocol)
# =============================================================================

def levenshtein_tokens(a: list, b: list) -> int:
    """
    Compute Levenshtein distance between two TOKEN lists.

    WARNING: Do NOT use Levenshtein on joined strings - that computes
    CHARACTER distance which is ~5-10x larger! (notebook protocol)
    """
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
        prev_distances, distances = distances, [i] + [0] * len_b
        for j in range(1, len_b + 1):
            distances[j] = min(
                distances[j - 1] + 1,
                prev_distances[j] + 1,
                prev_distances[j - 1] + (a[i - 1] != b[j - 1]),
            )
    return distances[-1]


def levenshtein_chars(a: str, b: str) -> int:
    """Compute character-level Levenshtein distance between two strings."""
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    distances = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        prev_distances, distances = distances, [i] + [0] * len_b
        for j in range(1, len_b + 1):
            distances[j] = min(
                distances[j - 1] + 1,
                prev_distances[j] + 1,
                prev_distances[j - 1] + (a[i - 1] != b[j - 1]),
            )
    return distances[-1]


def get_system_breaks_from_musicxml(musicxml_path: str) -> list:
    """Extract system break positions (measure indices) from a GT MusicXML file.
    Returns list of 0-based measure indices where a new system starts.
    """
    import xml.etree.ElementTree as _ET
    try:
        tree = _ET.parse(musicxml_path)
        root = tree.getroot()
        measures = root.findall('.//measure')
        breaks = []
        for i, m in enumerate(measures):
            for p in m.findall('print'):
                if p.get('new-system') == 'yes':
                    breaks.append(i)
        return breaks
    except Exception:
        return []


def split_into_systems(tokens: list, system_break_measures: list) -> list:
    """Split a token list into system-level chunks using GT system break positions.

    Args:
        tokens: Zeus linearization token list.
        system_break_measures: 0-based measure indices where new systems start
            (from get_system_breaks_from_musicxml).

    Returns:
        List of strings, one per system (joined tokens for that system).
    """
    # Find positions of 'measure' tokens
    measure_positions = [i for i, t in enumerate(tokens) if t == 'measure']
    if not measure_positions:
        return [' '.join(tokens)] if tokens else []

    # Build token-index boundaries for each system
    boundaries = [0] + system_break_measures + [len(measure_positions)]
    # Deduplicate and sort (first measure is always system 1)
    boundaries = sorted(set(boundaries))

    systems = []
    for s in range(len(boundaries) - 1):
        start_m = boundaries[s]
        end_m = boundaries[s + 1]
        if start_m >= len(measure_positions):
            break
        tok_start = measure_positions[start_m]
        tok_end = measure_positions[end_m] if end_m < len(measure_positions) else len(tokens)
        system_str = ' '.join(tokens[tok_start:tok_end])
        if system_str:
            systems.append(system_str)

    return systems if systems else [' '.join(tokens)]





# =============================================================================
# VOCABULARY COMPARISON (Model vs Zeus vs Dataset — token-level)
# =============================================================================

def run_vocabulary_comparison(model, gt_data: Dict[str, Dict], output_dir: Path,
                             vocab_info: Dict = None) -> Dict:
    """
    Build a comprehensive vocabulary comparison between:
      A. Model vocabulary (kern space, from model.w2i)
      B. Zeus LMX vocabulary (ALL_TOKENS from olimpic linearizer)
      C. Evaluation dataset vocabulary (actual tokens in GT LMX sequences)
      D. MusicXML-level OOV features (from vocabulary analysis)

    NOTE: Because Zeus's _emit() drops tokens not in ALL_TOKENS, the GT LMX
    only ever contains ALL_TOKENS members (C is subset of B always).
    The REAL OOV is at the MusicXML level: features like dynamics, wedges,
    pedals, and exotic tuplets that Zeus silently drops during linearization.
    These are captured by run_vocabulary_analysis() and merged here.

    Returns a dict with vocabulary sets, per-sample OOV info from MusicXML
    analysis, and aggregate statistics.
    """
    olimpic_path = str(Path("./olimpic-icdar24").resolve())
    if olimpic_path not in sys.path:
        sys.path.append(olimpic_path)

    # --- A. Model vocab (kern space) ---
    model_kern_vocab = set(model.w2i.keys()) if hasattr(model, 'w2i') else set()
    logging.info(f"Model kern vocab size: {len(model_kern_vocab)} tokens")

    # --- B. Zeus LMX vocab ---
    try:
        from app.linearization.vocabulary import ALL_TOKENS
        zeus_vocab = set(ALL_TOKENS)
    except ImportError as e:
        logging.warning(f"Could not import Zeus vocabulary: {e}")
        zeus_vocab = set()
    logging.info(f"Zeus LMX vocab size: {len(zeus_vocab)} tokens")

    # --- C. Evaluation dataset vocab (from GT LMX sequences) ---
    dataset_token_counts = Counter()
    per_sample_tokens = {}  # sample_id → set of unique tokens
    for sample_id, gt_entry in gt_data.items():
        if isinstance(gt_entry, str):
            tokens = gt_entry.split()
        else:
            tokens = gt_entry.get('tokens', gt_entry.get('lmx', '').split())
        token_set = set(tokens)
        per_sample_tokens[sample_id] = token_set
        for tok in tokens:
            dataset_token_counts[tok] += 1

    dataset_vocab = set(dataset_token_counts.keys())
    logging.info(f"Dataset vocab size: {len(dataset_vocab)} unique token types")
    logging.info(f"Dataset total tokens: {sum(dataset_token_counts.values())}")

    # --- Vocabulary set operations ---
    common_vocab = zeus_vocab & dataset_vocab
    oov_tokens = dataset_vocab - zeus_vocab  # in GT but NOT in Zeus
    unused_vocab = zeus_vocab - dataset_vocab  # in Zeus but never in GT

    logging.info(f"Common vocab (Zeus ∩ Dataset): {len(common_vocab)} tokens")
    logging.info(f"OOV tokens (Dataset \\ Zeus):   {len(oov_tokens)} token types")
    logging.info(f"Unused vocab (Zeus \\ Dataset): {len(unused_vocab)} token types")

    # OOV token frequency breakdown
    oov_token_counts = {tok: dataset_token_counts[tok] for tok in oov_tokens}
    total_oov_instances = sum(oov_token_counts.values())
    total_all_instances = sum(dataset_token_counts.values())
    oov_instance_ratio = total_oov_instances / total_all_instances if total_all_instances > 0 else 0

    logging.info(f"LMX-level OOV instances: {total_oov_instances}/{total_all_instances} ({oov_instance_ratio*100:.2f}%)")
    if total_oov_instances == 0:
        logging.info("   (Expected: Zeus _emit() drops non-ALL_TOKENS before output, so LMX is always subset of ALL_TOKENS)")

    # --- MusicXML-level OOV (the REAL vocabulary gap) ---
    musicxml_oov_by_category = {}
    musicxml_oov_total = 0
    n_samples_musicxml_oov = 0
    per_sample_analysis = {}

    if vocab_info:
        musicxml_oov_by_category = vocab_info.get('oov_by_category', {})
        for cat, counts in musicxml_oov_by_category.items():
            cat_total = sum(counts.values()) if isinstance(counts, (dict, Counter)) else 0
            musicxml_oov_total += cat_total
        n_samples_musicxml_oov = vocab_info.get('n_any_oov', 0)
        for entry in vocab_info.get('analysis', []):
            per_sample_analysis[entry['sample_id']] = entry

    logging.info(f"")
    logging.info(f"MusicXML-level OOV (features dropped by Zeus during linearization):")
    logging.info(f"   Samples affected:   {n_samples_musicxml_oov}/{len(gt_data)}")
    logging.info(f"   Total features dropped: {musicxml_oov_total}")
    for cat in sorted(musicxml_oov_by_category, key=lambda c: -sum(
            musicxml_oov_by_category[c].values()
            if isinstance(musicxml_oov_by_category[c], (dict, Counter)) else [0])):
        counts = musicxml_oov_by_category[cat]
        total = sum(counts.values()) if isinstance(counts, (dict, Counter)) else 0
        logging.info(f"   {cat}: {total} instances")

    # --- Per-sample OOV density (based on MusicXML analysis) ---
    per_sample_oov = []
    for sample_id, gt_entry in gt_data.items():
        if isinstance(gt_entry, str):
            tokens = gt_entry.split()
        else:
            tokens = gt_entry.get('tokens', gt_entry.get('lmx', '').split())
        n_lmx_tokens = len(tokens)
        analysis_entry = per_sample_analysis.get(sample_id, {})
        n_musicxml_oov = analysis_entry.get('oov_count', 0)
        oov_density = n_musicxml_oov / (n_lmx_tokens + n_musicxml_oov) if (n_lmx_tokens + n_musicxml_oov) > 0 else 0
        per_sample_oov.append({
            'sample_id': sample_id,
            'lmx_tokens': n_lmx_tokens,
            'musicxml_oov_count': n_musicxml_oov,
            'oov_density': round(oov_density, 4),
            'has_any_oov': analysis_entry.get('has_any_oov', False),
            'oov_categories': analysis_entry.get('oov_categories', {}),
        })

    # --- Save comparison report ---
    comparison = {
        'model_kern_vocab_size': len(model_kern_vocab),
        'zeus_lmx_vocab_size': len(zeus_vocab),
        'dataset_vocab_size': len(dataset_vocab),
        'common_vocab_size': len(common_vocab),
        'lmx_oov_token_types': len(oov_tokens),
        'lmx_oov_note': 'Always 0: Zeus _emit() drops non-ALL_TOKENS before output',
        'unused_vocab_types': len(unused_vocab),
        'musicxml_oov_total_features': musicxml_oov_total,
        'musicxml_oov_samples_affected': n_samples_musicxml_oov,
        'musicxml_oov_by_category': {
            cat: dict(counts) if isinstance(counts, (dict, Counter)) else {}
            for cat, counts in musicxml_oov_by_category.items()
        },
        'total_token_instances': total_all_instances,
        'model_kern_vocab': sorted(model_kern_vocab),
        'zeus_lmx_vocab': sorted(zeus_vocab),
        'common_vocab': sorted(common_vocab),
        'unused_vocab': sorted(unused_vocab),
        'dataset_token_counts': dict(sorted(dataset_token_counts.items(), key=lambda x: -x[1])),
        'per_sample_oov': per_sample_oov,
    }

    comparison_file = output_dir / "vocabulary_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    logging.info(f"Vocabulary comparison saved to {comparison_file}")

    return {
        'zeus_vocab': zeus_vocab,
        'dataset_vocab': dataset_vocab,
        'common_vocab': common_vocab,
        'oov_tokens': oov_tokens,  # LMX-level (always empty)
        'musicxml_oov_total': musicxml_oov_total,
        'musicxml_oov_samples': n_samples_musicxml_oov,
        'musicxml_oov_by_category': musicxml_oov_by_category,
        'comparison': comparison,
    }


# =============================================================================
# VOCABULARY ANALYSIS (matching notebook protocol)
# =============================================================================

def run_vocabulary_analysis(dataset: List[Dict], output_dir: Path) -> Dict:
    """
    Scan GT MusicXML for OOV elements that the model can't predict.
    Returns analysis dict with per-sample OOV info and sets of affected samples.
    """
    import xml.etree.ElementTree as ET

    # Load Zeus vocabulary
    olimpic_path = str(Path("./olimpic-icdar24").resolve())
    if olimpic_path not in sys.path:
        sys.path.append(olimpic_path)

    try:
        from app.linearization.vocabulary import (
            ALL_TOKENS, TIME_MODIFICATION_TOKENS, CLEF_TOKENS,
            BEATS_TOKENS, BEAT_TYPE_TOKENS,
        )
        model_vocab = set(ALL_TOKENS)
        supported_beats = set()
        for b in BEATS_TOKENS:
            try:
                supported_beats.add(int(b.split(':')[1]))
            except (IndexError, ValueError):
                pass
        supported_beat_types = set()
        for b in BEAT_TYPE_TOKENS:
            try:
                supported_beat_types.add(int(b.split(':')[1]))
            except (IndexError, ValueError):
                pass
    except ImportError as e:
        logging.warning(f"Could not import vocabulary: {e}")
        return {'analysis': [], 'samples_with_oov_tuplets': set(), 'oov_by_category': {}, 'n_clean': len(dataset), 'n_oov_tuplets': 0}

    logging.info(f"Vocabulary analysis: {len(model_vocab)} Zeus tokens")

    vocab_analysis = []
    all_oov_tokens = Counter()
    oov_by_category = defaultdict(Counter)

    for sample in tqdm(dataset, desc="Vocab analysis"):
        xml_file = sample['gt_musicxml_path']
        oov_tokens = []
        oov_categories = defaultdict(list)

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Tuplets
            for time_mod in root.findall(".//time-modification"):
                actual = time_mod.find("actual-notes")
                normal = time_mod.find("normal-notes")
                if actual is not None and normal is not None:
                    tuplet_token = f"{actual.text}in{normal.text}"
                    if tuplet_token not in TIME_MODIFICATION_TOKENS:
                        oov_tokens.append(tuplet_token)
                        oov_categories['tuplet'].append(tuplet_token)

            # Time signatures
            for time_elem in root.findall(".//time"):
                beats = time_elem.find("beats")
                beat_type = time_elem.find("beat-type")
                if beats is not None:
                    try:
                        bv = int(beats.text)
                        if bv not in supported_beats:
                            oov_tokens.append(f"beats:{bv}")
                            oov_categories['time_signature'].append(f"beats:{bv}")
                    except (ValueError, TypeError):
                        pass
                if beat_type is not None:
                    try:
                        btv = int(beat_type.text)
                        if btv not in supported_beat_types:
                            oov_tokens.append(f"beat-type:{btv}")
                            oov_categories['time_signature'].append(f"beat-type:{btv}")
                    except (ValueError, TypeError):
                        pass

            # Clefs
            for clef in root.findall(".//clef"):
                sign = clef.find("sign")
                line = clef.find("line")
                if sign is not None and line is not None:
                    ct = f"clef:{sign.text}{line.text}"
                    if ct not in CLEF_TOKENS:
                        oov_tokens.append(ct)
                        oov_categories['clef'].append(ct)

            # Dynamics (NOT in Zeus vocab at all)
            for dyn in root.findall(".//dynamics"):
                for child in dyn:
                    oov_tokens.append(f"dynamics:{child.tag}")
                    oov_categories['dynamics'].append(child.tag)

            # Wedges
            for wedge in root.findall(".//wedge"):
                oov_tokens.append(f"wedge:{wedge.get('type', 'unknown')}")
                oov_categories['wedge'].append(wedge.get('type', 'unknown'))

            # Pedal
            for pedal in root.findall(".//pedal"):
                oov_tokens.append(f"pedal:{pedal.get('type', 'unknown')}")
                oov_categories['pedal'].append(pedal.get('type', 'unknown'))

        except Exception as e:
            oov_categories['error'] = [str(e)]

        has_oov_tuplets = 'tuplet' in oov_categories
        has_any_oov = len(oov_tokens) > 0
        vocab_analysis.append({
            'sample_id': sample['sample_id'],
            'filename': sample.get('filename', ''),
            'has_oov_tuplets': has_oov_tuplets,
            'has_any_oov': has_any_oov,
            'oov_count': len(oov_tokens),
            'oov_categories': dict(oov_categories),
        })

        for tok in oov_tokens:
            all_oov_tokens[tok] += 1
        for cat, toks in oov_categories.items():
            for tok in toks:
                oov_by_category[cat][tok] += 1

    # Summary
    n_oov_tuplets = sum(1 for v in vocab_analysis if v['has_oov_tuplets'])
    n_any_oov = sum(1 for v in vocab_analysis if v['has_any_oov'])
    n_clean_tuplets = sum(1 for v in vocab_analysis if not v['has_oov_tuplets'])
    n_fully_clean = sum(1 for v in vocab_analysis if not v['has_any_oov'])

    logging.info(f"Vocabulary analysis complete:")
    logging.info(f"   Total samples: {len(vocab_analysis)}")
    logging.info(f"   Fully clean (no OOV at all): {n_fully_clean}")
    logging.info(f"   Clean (no OOV tuplets only): {n_clean_tuplets}")
    logging.info(f"   Has OOV tuplets: {n_oov_tuplets}")
    logging.info(f"   Has any OOV: {n_any_oov}")

    for cat in sorted(oov_by_category, key=lambda c: -sum(oov_by_category[c].values())):
        total = sum(oov_by_category[cat].values())
        logging.info(f"   OOV {cat}: {total} instances")

    # Save analysis
    analysis_file = output_dir / "vocabulary_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'total_samples': len(vocab_analysis),
            'fully_clean_samples': n_fully_clean,
            'clean_no_oov_tuplets': n_clean_tuplets,
            'oov_tuplet_samples': n_oov_tuplets,
            'any_oov_samples': n_any_oov,
            'oov_by_category': {cat: dict(counts) for cat, counts in oov_by_category.items()},
            'per_sample': vocab_analysis,
        }, f, indent=2)

    # Build sets
    samples_with_oov_tuplets = set(v['sample_id'] for v in vocab_analysis if v['has_oov_tuplets'])
    samples_with_any_oov = set(v['sample_id'] for v in vocab_analysis if v['has_any_oov'])

    return {
        'analysis': vocab_analysis,
        'samples_with_oov_tuplets': samples_with_oov_tuplets,
        'samples_with_any_oov': samples_with_any_oov,
        'oov_by_category': dict(oov_by_category),
        'n_fully_clean': n_fully_clean,
        'n_clean_tuplets': n_clean_tuplets,
        'n_oov_tuplets': n_oov_tuplets,
        'n_any_oov': n_any_oov,
    }


# =============================================================================
# UNFAIR TOKENS — LMX tokens that kern-based models structurally cannot produce
# =============================================================================
UNFAIR_TOKENS = frozenset({
    'slur:start', 'slur:stop',
    'staccato', 'tenuto', 'arpeggiate', 'accent', 'strong-accent',
    'trill-mark', 'fermata',
    'tremolo:1', 'tremolo:2', 'tremolo:3', 'tremolo:4',
    'tremolo:single', 'tremolo:start', 'tremolo:stop', 'tremolo:unmeasured',
})


# =============================================================================
# EVALUATION (matching notebook protocol)
# =============================================================================

def evaluate_all(dataset: List[Dict], musicxml_files: Dict[str, str],
                 gt_data: Dict[str, Dict], conversion_methods: Dict[str, str],
                 vocab_info: Dict, vocab_comparison: Dict, output_dir: Path,
                 Linearizer, MxlFile, ET) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate all predictions against GT using token-level SER, CER, and LER.
    Also computes filtered variants (fSER, fCER, fLER) that remove UNFAIR_TOKENS
    from both GT and predicted sequences before comparison.
    """
    results = []
    total_edit = 0
    total_gt_tokens = 0
    # CER accumulators
    total_char_edit = 0
    total_gt_chars = 0
    # LER accumulators (system-level)
    total_system_edit = 0
    total_gt_systems = 0
    # Filtered accumulators
    total_edit_f = 0
    total_gt_tokens_f = 0
    total_char_edit_f = 0
    total_gt_chars_f = 0
    total_system_edit_f = 0
    total_gt_systems_f = 0
    total_unfair_gt = 0
    total_unfair_pred = 0

    logging.info(f"Evaluating {len(musicxml_files)} predictions...")

    for sample in tqdm(dataset, desc="Evaluation"):
        sample_id = sample['sample_id']

        # Need both a converted MusicXML and a GT linearization
        if sample_id not in musicxml_files:
            results.append({
                'sample_id': sample_id,
                'document': sample['document'],
                'folio': sample['folio'],
                'status': 'no_musicxml',
            })
            continue

        # Get GT tokens
        gt_entry = gt_data.get(sample_id)
        if gt_entry is None:
            results.append({
                'sample_id': sample_id,
                'document': sample['document'],
                'folio': sample['folio'],
                'status': 'no_gt_linearization',
            })
            continue

        # Handle both old cache format (string) and new format (dict)
        if isinstance(gt_entry, str):
            gt_lmx = gt_entry
            gt_tokens = gt_lmx.split()
        else:
            gt_lmx = gt_entry['lmx']
            gt_tokens = gt_entry.get('tokens', gt_lmx.split())

        # Linearize predicted MusicXML
        pred_lmx = linearize_musicxml(musicxml_files[sample_id], Linearizer, MxlFile, ET)
        if not pred_lmx:
            results.append({
                'sample_id': sample_id,
                'document': sample['document'],
                'folio': sample['folio'],
                'status': 'linearization_failed',
            })
            continue

        pred_tokens = pred_lmx.split()

        # TOKEN-level Levenshtein (matching notebook protocol — NOT character level!)
        edit_dist = levenshtein_tokens(gt_tokens, pred_tokens)

        # CER: character-level Levenshtein on joined linearization (no spaces)
        gt_chars = ''.join(gt_tokens)
        pred_chars = ''.join(pred_tokens)
        char_edit_dist = levenshtein_chars(gt_chars, pred_chars)

        # LER: system-level Levenshtein (split using GT MusicXML system breaks)
        system_breaks = get_system_breaks_from_musicxml(sample['gt_musicxml_path'])
        gt_systems = split_into_systems(gt_tokens, system_breaks)
        pred_systems = split_into_systems(pred_tokens, system_breaks)
        system_edit_dist = levenshtein_tokens(gt_systems, pred_systems)

        # Filtered metrics: remove UNFAIR_TOKENS from both sides
        gt_tokens_f = [t for t in gt_tokens if t not in UNFAIR_TOKENS]
        pred_tokens_f = [t for t in pred_tokens if t not in UNFAIR_TOKENS]
        edit_dist_f = levenshtein_tokens(gt_tokens_f, pred_tokens_f)
        gt_chars_f = ''.join(gt_tokens_f)
        pred_chars_f = ''.join(pred_tokens_f)
        char_edit_dist_f = levenshtein_chars(gt_chars_f, pred_chars_f)
        gt_systems_f = split_into_systems(gt_tokens_f, system_breaks)
        pred_systems_f = split_into_systems(pred_tokens_f, system_breaks)
        system_edit_dist_f = levenshtein_tokens(gt_systems_f, pred_systems_f)
        n_unfair_gt = len(gt_tokens) - len(gt_tokens_f)
        n_unfair_pred = len(pred_tokens) - len(pred_tokens_f)

        # Accumulate global metrics
        total_edit += edit_dist
        total_gt_tokens += len(gt_tokens)
        total_char_edit += char_edit_dist
        total_gt_chars += len(gt_chars)
        total_system_edit += system_edit_dist
        total_gt_systems += len(gt_systems)
        total_edit_f += edit_dist_f
        total_gt_tokens_f += len(gt_tokens_f)
        total_char_edit_f += char_edit_dist_f
        total_gt_chars_f += len(gt_chars_f)
        total_system_edit_f += system_edit_dist_f
        total_gt_systems_f += len(gt_systems_f)
        total_unfair_gt += n_unfair_gt
        total_unfair_pred += n_unfair_pred

        sample_ser = (edit_dist / len(gt_tokens) * 100) if len(gt_tokens) > 0 else 0
        sample_cer = (char_edit_dist / len(gt_chars) * 100) if len(gt_chars) > 0 else 0
        sample_ler = (system_edit_dist / len(gt_systems) * 100) if len(gt_systems) > 0 else 0
        sample_fser = (edit_dist_f / len(gt_tokens_f) * 100) if len(gt_tokens_f) > 0 else 0
        sample_fcer = (char_edit_dist_f / len(gt_chars_f) * 100) if len(gt_chars_f) > 0 else 0
        sample_fler = (system_edit_dist_f / len(gt_systems_f) * 100) if len(gt_systems_f) > 0 else 0

        results.append({
            'sample_id': sample_id,
            'document': sample['document'],
            'folio': sample['folio'],
            'status': 'success',
            'gt_tokens': len(gt_tokens),
            'pred_tokens': len(pred_tokens),
            'edit_distance': edit_dist,
            'SER': sample_ser,
            'CER': sample_cer,
            'LER': sample_ler,
            'fSER': sample_fser,
            'fCER': sample_fcer,
            'fLER': sample_fler,
            'unfair_gt_tokens': n_unfair_gt,
            'unfair_pred_tokens': n_unfair_pred,
            'conversion_method': conversion_methods.get(sample_id, 'unknown'),
        })

    # Compute global metrics
    success_results = [r for r in results if r.get('status') == 'success']
    n_success = len(success_results)

    global_ser = (total_edit / total_gt_tokens * 100) if total_gt_tokens > 0 else 0
    global_cer = (total_char_edit / total_gt_chars * 100) if total_gt_chars > 0 else 0
    global_ler = (total_system_edit / total_gt_systems * 100) if total_gt_systems > 0 else 0
    global_fser = (total_edit_f / total_gt_tokens_f * 100) if total_gt_tokens_f > 0 else 0
    global_fcer = (total_char_edit_f / total_gt_chars_f * 100) if total_gt_chars_f > 0 else 0
    global_fler = (total_system_edit_f / total_gt_systems_f * 100) if total_gt_systems_f > 0 else 0

    sample_sers = [r['SER'] for r in success_results]
    mean_ser = float(np.mean(sample_sers)) if sample_sers else 0
    std_ser = float(np.std(sample_sers, ddof=1)) if len(sample_sers) > 1 else 0
    median_ser = float(np.median(sample_sers)) if sample_sers else 0

    sample_cers = [r['CER'] for r in success_results]
    mean_cer = float(np.mean(sample_cers)) if sample_cers else 0
    std_cer = float(np.std(sample_cers, ddof=1)) if len(sample_cers) > 1 else 0
    median_cer = float(np.median(sample_cers)) if sample_cers else 0

    sample_lers = [r['LER'] for r in success_results]
    mean_ler = float(np.mean(sample_lers)) if sample_lers else 0
    std_ler = float(np.std(sample_lers, ddof=1)) if len(sample_lers) > 1 else 0
    median_ler = float(np.median(sample_lers)) if sample_lers else 0

    sample_fsers = [r['fSER'] for r in success_results]
    mean_fser = float(np.mean(sample_fsers)) if sample_fsers else 0
    std_fser = float(np.std(sample_fsers, ddof=1)) if len(sample_fsers) > 1 else 0
    median_fser = float(np.median(sample_fsers)) if sample_fsers else 0

    sample_fcers = [r['fCER'] for r in success_results]
    mean_fcer = float(np.mean(sample_fcers)) if sample_fcers else 0
    std_fcer = float(np.std(sample_fcers, ddof=1)) if len(sample_fcers) > 1 else 0
    median_fcer = float(np.median(sample_fcers)) if sample_fcers else 0

    sample_flers = [r['fLER'] for r in success_results]
    mean_fler = float(np.mean(sample_flers)) if sample_flers else 0
    std_fler = float(np.std(sample_flers, ddof=1)) if len(sample_flers) > 1 else 0
    median_fler = float(np.median(sample_flers)) if sample_flers else 0

    pct_unfair_gt = (total_unfair_gt / total_gt_tokens * 100) if total_gt_tokens > 0 else 0

    logging.info(f"{'=' * 60}")
    logging.info(f"METRIC DEFINITIONS")
    logging.info(f"{'=' * 60}")
    logging.info(f"All metrics are computed in LMX token space (Zeus/Olimpic linearization).")
    logging.info(f"Both GT and predictions are linearized through the same Zeus pipeline,")
    logging.info(f"which enforces GT ⊆ ALL_TOKENS and Pred ⊆ ALL_TOKENS.")
    logging.info(f"Lower values = better performance (0% = perfect match).")
    logging.info(f"")
    logging.info(f"SER  = Symbol Error Rate: token-level Levenshtein edit distance")
    logging.info(f"       between GT and predicted LMX sequences, divided by |GT tokens|.")
    logging.info(f"CER  = Character Error Rate: character-level Levenshtein on the")
    logging.info(f"       concatenated token strings (no spaces), divided by |GT chars|.")
    logging.info(f"LER  = Line Error Rate: sequence-level Levenshtein over system LMX")
    logging.info(f"       strings. System breaks from GT MusicXML new-system elements.")
    logging.info(f"       Per-sample (each full-page image), then globally.")
    logging.info(f"")
    logging.info(f"fSER = Filtered SER: same as SER but after removing {len(UNFAIR_TOKENS)} unfair")
    logging.info(f"       token types from both GT and prediction (slurs, articulations,")
    logging.info(f"       grace notes, tremolo — elements kern models cannot produce).")
    logging.info(f"fCER = Filtered CER: same as CER on filtered token sequences.")
    logging.info(f"fLER = Filtered LER: same as LER on filtered system sequences.")
    logging.info(f"")
    logging.info(f"{'=' * 60}")
    logging.info(f"EVALUATION RESULTS")
    logging.info(f"{'=' * 60}")
    logging.info(f"Evaluated: {n_success}/{len(dataset)} samples")
    logging.info(f"Unfair tokens removed: {total_unfair_gt} GT ({pct_unfair_gt:.2f}%), {total_unfair_pred} pred")
    logging.info(f"")
    logging.info(f"GLOBAL SER (total_edit / total_gt_tokens):")
    logging.info(f"   Full dataset:           {global_ser:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE SER:")
    logging.info(f"   Mean +/- Std:           {mean_ser:.2f}% +/- {std_ser:.2f}%")
    logging.info(f"   Median:                 {median_ser:.2f}%")
    logging.info(f"")
    logging.info(f"GLOBAL CER (total_char_edit / total_gt_chars):")
    logging.info(f"   Full dataset:           {global_cer:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE CER:")
    logging.info(f"   Mean +/- Std:           {mean_cer:.2f}% +/- {std_cer:.2f}%")
    logging.info(f"   Median:                 {median_cer:.2f}%")
    logging.info(f"")
    logging.info(f"GLOBAL LER (system-level Levenshtein, SMT-plusplus method):")
    logging.info(f"   Full dataset:           {global_ler:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE LER:")
    logging.info(f"   Mean +/- Std:           {mean_ler:.2f}% +/- {std_ler:.2f}%")
    logging.info(f"   Median:                 {median_ler:.2f}%")
    logging.info(f"")
    logging.info(f"--- FILTERED METRICS (unfair tokens removed) ---")
    logging.info(f"")
    logging.info(f"GLOBAL fSER:")
    logging.info(f"   Full dataset:           {global_fser:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE fSER:")
    logging.info(f"   Mean +/- Std:           {mean_fser:.2f}% +/- {std_fser:.2f}%")
    logging.info(f"   Median:                 {median_fser:.2f}%")
    logging.info(f"")
    logging.info(f"GLOBAL fCER:")
    logging.info(f"   Full dataset:           {global_fcer:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE fCER:")
    logging.info(f"   Mean +/- Std:           {mean_fcer:.2f}% +/- {std_fcer:.2f}%")
    logging.info(f"   Median:                 {median_fcer:.2f}%")
    logging.info(f"")
    logging.info(f"GLOBAL fLER:")
    logging.info(f"   Full dataset:           {global_fler:.2f}% (n={n_success})")
    logging.info(f"")
    logging.info(f"PER-SAMPLE fLER:")
    logging.info(f"   Mean +/- Std:           {mean_fler:.2f}% +/- {std_fler:.2f}%")
    logging.info(f"   Median:                 {median_fler:.2f}%")
    logging.info(f"{'=' * 60}")

    metrics_df = pd.DataFrame(results)
    metrics_csv = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Save comprehensive summary
    summary = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'PRAIG/smt-fp-grandstaff',
        'total_dataset': len(dataset),
        'evaluated': n_success,
        'failed': len(dataset) - n_success,
        'global_SER': round(global_ser, 4),
        'mean_SER': round(mean_ser, 4),
        'std_SER': round(std_ser, 4),
        'median_SER': round(median_ser, 4),
        'total_edit_distance': int(total_edit),
        'total_gt_tokens': int(total_gt_tokens),
        'global_CER': round(global_cer, 4),
        'mean_CER': round(mean_cer, 4),
        'std_CER': round(std_cer, 4),
        'median_CER': round(median_cer, 4),
        'total_char_edit_distance': int(total_char_edit),
        'total_gt_chars': int(total_gt_chars),
        'global_LER': round(global_ler, 4),
        'mean_LER': round(mean_ler, 4),
        'std_LER': round(std_ler, 4),
        'median_LER': round(median_ler, 4),
        'total_system_edit_distance': int(total_system_edit),
        'total_gt_systems': int(total_gt_systems),
        'global_fSER': round(global_fser, 4),
        'mean_fSER': round(mean_fser, 4),
        'std_fSER': round(std_fser, 4),
        'median_fSER': round(median_fser, 4),
        'global_fCER': round(global_fcer, 4),
        'mean_fCER': round(mean_fcer, 4),
        'std_fCER': round(std_fcer, 4),
        'median_fCER': round(median_fcer, 4),
        'global_fLER': round(global_fler, 4),
        'mean_fLER': round(mean_fler, 4),
        'std_fLER': round(std_fler, 4),
        'median_fLER': round(median_fler, 4),
        'unfair_tokens_removed_gt': int(total_unfair_gt),
        'unfair_tokens_removed_pred': int(total_unfair_pred),
        'unfair_tokens_pct_gt': round(pct_unfair_gt, 4),
        'unfair_token_types': sorted(UNFAIR_TOKENS),
        'failed_samples': [
            {'sample_id': r['sample_id'], 'document': r['document'],
             'folio': r['folio'], 'reason': r['status']}
            for r in results
            if r.get('status') not in ('success', None) and 'SER' not in r
        ],
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return metrics_df, summary


# =============================================================================
# TEDn EVALUATION
# =============================================================================

def _compute_tedn_single(args):
    """Worker function for parallel TEDn computation of both full and lmx flavors."""
    sample_id, pred_lmx, gt_musicxml_path = args
    try:
        olimpic_path = str(Path("./olimpic-icdar24").resolve())
        if olimpic_path not in sys.path:
            sys.path.append(olimpic_path)
        from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml

        with open(gt_musicxml_path, 'r') as f:
            gold_musicxml = f.read()

        result_full = TEDn_lmx_xml(
            predicted_lmx=pred_lmx,
            gold_musicxml=gold_musicxml,
            flavor='full',
            errout=io.StringIO(),
        )
        result_lmx = TEDn_lmx_xml(
            predicted_lmx=pred_lmx,
            gold_musicxml=gold_musicxml,
            flavor='lmx',
            errout=io.StringIO(),
        )
        return (
            sample_id,
            result_full.edit_cost, result_full.gold_cost,
            result_lmx.edit_cost, result_lmx.gold_cost,
            None,
        )
    except Exception as e:
        return sample_id, 0, 0, 0, 0, str(e)


def run_tedn_evaluation(dataset: List[Dict], musicxml_files: Dict[str, str],
                        gt_data: Dict[str, Dict], predictions_dir: Path,
                        output_dir: Path, config: Dict,
                        Linearizer, MxlFile, ET) -> Optional[Dict]:
    """
    Run TEDn (Tree Edit Distance, normalized) evaluation.
    Requires predicted LMX and GT MusicXML files.
    WARNING: Computationally expensive (~30-120s per sample).
    """
    import multiprocessing

    workers = config.get('tedn_workers', 4)

    # Build (sample_id, pred_lmx, gt_path) tuples — both flavors computed per worker call
    tasks = []
    for sample in dataset:
        sid = sample['sample_id']
        if sid not in musicxml_files:
            continue

        pred_lmx = linearize_musicxml(musicxml_files[sid], Linearizer, MxlFile, ET)
        if not pred_lmx:
            continue

        tasks.append((sid, pred_lmx, sample['gt_musicxml_path']))

    if not tasks:
        logging.warning("TEDn: no valid samples to evaluate")
        return None

    logging.info(f"TEDn: evaluating {len(tasks)} samples with {workers} workers (both full and lmx)...")

    total_edit_full = 0
    total_gold_full = 0
    total_edit_lmx = 0
    total_gold_lmx = 0
    tedn_results = []
    failed = 0

    with multiprocessing.Pool(workers) as pool:
        for sid, edit_full, gold_full, edit_lmx, gold_lmx, error in tqdm(
            pool.imap_unordered(_compute_tedn_single, tasks),
            total=len(tasks), desc="TEDn"
        ):
            if error:
                logging.warning(f"TEDn failed for {sid}: {error}")
                failed += 1
                continue
            norm_full = edit_full / gold_full if gold_full > 0 else 1.0
            norm_lmx = edit_lmx / gold_lmx if gold_lmx > 0 else 1.0
            total_edit_full += edit_full
            total_gold_full += gold_full
            total_edit_lmx += edit_lmx
            total_gold_lmx += gold_lmx
            tedn_results.append({
                'sample_id': sid,
                'edit_cost_full': edit_full,
                'gold_cost_full': gold_full,
                'TEDn_full': round(norm_full * 100, 4),
                'edit_cost_lmx': edit_lmx,
                'gold_cost_lmx': gold_lmx,
                'TEDn_lmx': round(norm_lmx * 100, 4),
            })

    global_full = total_edit_full / total_gold_full * 100 if total_gold_full > 0 else 0
    global_lmx = total_edit_lmx / total_gold_lmx * 100 if total_gold_lmx > 0 else 0
    tedns_full = [r['TEDn_full'] for r in tedn_results]
    tedns_lmx = [r['TEDn_lmx'] for r in tedn_results]
    mean_full = float(np.mean(tedns_full)) if tedns_full else 0
    median_full = float(np.median(tedns_full)) if tedns_full else 0
    mean_lmx = float(np.mean(tedns_lmx)) if tedns_lmx else 0
    median_lmx = float(np.median(tedns_lmx)) if tedns_lmx else 0

    logging.info(f"{'=' * 60}")
    logging.info(f"TEDn RESULTS")
    logging.info(f"{'=' * 60}")
    logging.info(f"Evaluated: {len(tedn_results)}/{len(tasks)} samples ({failed} failed)")
    logging.info(f"")
    logging.info(f"TEDn-full (all MusicXML content):")
    logging.info(f"   Global TEDn-full: {global_full:.2f}%")
    logging.info(f"   Mean TEDn-full:   {mean_full:.2f}%")
    logging.info(f"   Median TEDn-full: {median_full:.2f}%")
    logging.info(f"")
    logging.info(f"TEDn-lmx (LMX-covered concepts only):")
    logging.info(f"   Global TEDn-lmx:  {global_lmx:.2f}%")
    logging.info(f"   Mean TEDn-lmx:    {mean_lmx:.2f}%")
    logging.info(f"   Median TEDn-lmx:  {median_lmx:.2f}%")
    logging.info(f"{'=' * 60}")

    tedn_df = pd.DataFrame(tedn_results)
    tedn_df.to_csv(output_dir / "tedn_metrics.csv", index=False)

    tedn_summary = {
        'evaluated': len(tedn_results),
        'failed': failed,
        'full': {
            'global_TEDn': round(global_full, 4),
            'mean_TEDn': round(mean_full, 4),
            'median_TEDn': round(median_full, 4),
            'total_edit_cost': int(total_edit_full),
            'total_gold_cost': int(total_gold_full),
        },
        'lmx': {
            'global_TEDn': round(global_lmx, 4),
            'mean_TEDn': round(mean_lmx, 4),
            'median_TEDn': round(median_lmx, 4),
            'total_edit_cost': int(total_edit_lmx),
            'total_gold_cost': int(total_gold_lmx),
        },
    }

    tedn_summary_file = output_dir / "tedn_summary.json"
    with open(tedn_summary_file, 'w') as f:
        json.dump(tedn_summary, f, indent=2)

    return tedn_summary


# =============================================================================
# PER-DOCUMENT BREAKDOWN
# =============================================================================

def per_document_breakdown(metrics_df: pd.DataFrame, output_dir: Path):
    """Compute and save per-document SER breakdown."""
    success = metrics_df[metrics_df['status'] == 'success'].copy()
    if success.empty:
        return

    doc_stats = []
    for doc, group in success.groupby('document'):
        total_edit = group['edit_distance'].sum()
        total_gt = group['gt_tokens'].sum()
        global_ser = (total_edit / total_gt * 100) if total_gt > 0 else 0
        mean_ser = group['SER'].mean()
        std_ser = group['SER'].std(ddof=1) if len(group) > 1 else 0

        doc_stats.append({
            'document': doc,
            'n_systems': len(group),
            'global_SER': round(global_ser, 2),
            'mean_SER': round(mean_ser, 2),
            'std_SER': round(std_ser, 2),
            'total_gt_tokens': int(total_gt),
            'total_edit_distance': int(total_edit),
        })

    doc_df = pd.DataFrame(doc_stats).sort_values('global_SER', ascending=False)
    doc_df.to_csv(output_dir / "per_document_breakdown.csv", index=False)

    logging.info(f"Per-document breakdown saved ({len(doc_stats)} documents)")
    if len(doc_df) > 0:
        logging.info(f"   Best:  {doc_df.iloc[-1]['document']} ({doc_df.iloc[-1]['global_SER']:.2f}% SER)")
        logging.info(f"   Worst: {doc_df.iloc[0]['document']} ({doc_df.iloc[0]['global_SER']:.2f}% SER)")


# =============================================================================
# REPORTING
# =============================================================================

def save_report(summary: Dict, conversion_stats: Dict, vocab_info: Dict,
                output_dir: Path):
    """Save human-readable evaluation report."""
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SMT EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date:  {summary['evaluation_date']}\n")
        f.write(f"Model: {summary['model']}\n\n")

        f.write("METRIC DEFINITIONS\n")
        f.write("-" * 70 + "\n")
        f.write("All metrics computed in LMX token space (Zeus/Olimpic linearization).\n")
        f.write("Both GT and predictions linearized through the same Zeus pipeline,\n")
        f.write("which enforces GT ⊆ ALL_TOKENS — evaluation is vocabulary-fair by construction.\n")
        f.write("Lower values = better performance (0% = perfect match).\n\n")
        f.write("SER  = Symbol Error Rate: token-level Levenshtein edit distance\n")
        f.write("       between GT and predicted LMX sequences, divided by |GT tokens|.\n")
        f.write("CER  = Character Error Rate: character-level Levenshtein on the\n")
        f.write("       concatenated token strings (no spaces), divided by |GT chars|.\n")
        f.write("LER  = Line Error Rate: sequence-level Levenshtein over system LMX\n")
        f.write("       strings. System breaks from GT MusicXML new-system elements.\n")
        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples:  {summary['total_dataset']}\n")
        f.write(f"Evaluated:      {summary['evaluated']}\n")
        f.write(f"Failed:         {summary['failed']}\n\n")

        failed_samples = summary.get('failed_samples', [])
        if failed_samples:
            f.write("FAILED SAMPLES\n")
            f.write("-" * 70 + "\n")
            by_reason: dict = {}
            for s in failed_samples:
                by_reason.setdefault(s['reason'], []).append(s['sample_id'])
            for reason, ids in sorted(by_reason.items()):
                f.write(f"{reason} ({len(ids)}):\n")
                for sid in sorted(ids):
                    f.write(f"   {sid}\n")
            f.write("\n")

        f.write("CONVERSION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Success rate:        {conversion_stats['success_rate']:.1f}%\n")
        f.write(f"   Direct:           {conversion_stats['success_direct']}\n")
        f.write(f"   Adv. sanitize:    {conversion_stats['success_advanced']}\n")
        f.write(f"   Basic sanitize:   {conversion_stats['success_basic']}\n")
        f.write(f"   Failed:           {conversion_stats['failed']}\n\n")

        f.write("SER (Token-level Levenshtein)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Global SER:                  {summary['global_SER']:.2f}%\n")
        f.write(f"Mean +/- Std:                {summary['mean_SER']:.2f}% +/- {summary['std_SER']:.2f}%\n")
        f.write(f"Median:                      {summary['median_SER']:.2f}%\n\n")

        f.write("CER (Character-level Levenshtein)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Global CER:                  {summary['global_CER']:.2f}%\n")
        f.write(f"Mean +/- Std:                {summary['mean_CER']:.2f}% +/- {summary['std_CER']:.2f}%\n")
        f.write(f"Median:                      {summary['median_CER']:.2f}%\n\n")

        f.write("LER (System-level Levenshtein, SMT-plusplus method)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Global LER:                  {summary['global_LER']:.2f}%\n")
        f.write(f"Mean +/- Std:                {summary['mean_LER']:.2f}% +/- {summary['std_LER']:.2f}%\n")
        f.write(f"Median:                      {summary['median_LER']:.2f}%\n\n")

        f.write("VOCABULARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Note: MusicXML features like dynamics, wedges, pedals, and exotic\n")
        f.write(f"tuplets are dropped by Zeus during linearization but do NOT affect\n")
        f.write(f"evaluation fairness — both GT and predictions go through the same pipeline.\n")
        for cat, counts in vocab_info.get('oov_by_category', {}).items():
            total = sum(counts.values()) if isinstance(counts, (dict, Counter)) else 0
            f.write(f"   {cat}: {total} instances\n")

        # TEDn results (if available)
        if 'tedn' in summary:
            f.write("\nTEDn (Tree Edit Distance, normalized)\n")
            f.write("-" * 70 + "\n")
            tedn = summary['tedn']
            f.write(f"Evaluated:        {tedn['evaluated']} ({tedn['failed']} failed)\n\n")
            f.write(f"TEDn-full (all MusicXML content):\n")
            t = tedn['full']
            f.write(f"   Global TEDn-full: {t['global_TEDn']:.2f}%\n")
            f.write(f"   Mean TEDn-full:   {t['mean_TEDn']:.2f}%\n")
            f.write(f"   Median TEDn-full: {t['median_TEDn']:.2f}%\n\n")
            f.write(f"TEDn-lmx (LMX-covered concepts only):\n")
            t = tedn['lmx']
            f.write(f"   Global TEDn-lmx:  {t['global_TEDn']:.2f}%\n")
            f.write(f"   Mean TEDn-lmx:    {t['mean_TEDn']:.2f}%\n")
            f.write(f"   Median TEDn-lmx:  {t['median_TEDn']:.2f}%\n")

        f.write("\n" + "=" * 70 + "\n")

    logging.info(f"Report saved to {report_file}")


# =============================================================================
# CHECKPOINTING
# =============================================================================


def save_checkpoint(output_dir: Path, data: Dict):
    """Save checkpoint."""
    checkpoint_file = output_dir / "checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SMT Evaluation")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output = Path(config.get('output_dir', './results'))
    output_dir = base_output / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dedicated logs directory
    logs_dir = Path(config.get('logs_dir', './logs'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"evaluation_{timestamp}.log"

    logger = setup_logging(config.get('log_level', 'INFO'), log_file)

    logger.info("=" * 70)
    logger.info("SMT EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Config:     {args.config}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Log file:   {log_file}")
    logger.info(f"Timestamp:  {timestamp}")

    # Save config copy to output dir
    config_copy = output_dir / "config_used.yaml"
    with open(config_copy, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_type = config.get('dataset_type', 'system')
    logger.info(f"Dataset type: {dataset_type}")
    dataset = load_dataset(
        config['dataset_base_dir'],
        config.get('test_mode'),
        config.get('test_document'),
        config.get('test_folio'),
        config.get('max_samples'),
        exclude_documents=config.get('exclude_documents'),
        dataset_type=dataset_type,
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Shared predictions directory (reusable across runs, tagged by binarization)
    binarization_method = config.get('binarization_method', 'adaptive')
    predictions_subdir = f"predictions_{binarization_method}"
    predictions_dir = base_output / predictions_subdir
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint from shared predictions
    checkpoint_data = {}
    existing_preds = list(predictions_dir.glob("*_prediction.txt"))
    if existing_preds:
        checkpoint_data = {
            'processed_samples': [f.stem.replace('_prediction', '') for f in existing_preds]
        }
        logger.info(f"Found {len(existing_preds)} existing predictions (reusing)")

    # Check if all predictions already exist (skip model loading if so)
    dataset_ids = {s['sample_id'] for s in dataset}
    existing_ids = set(checkpoint_data.get('processed_samples', []))
    missing_ids = dataset_ids - existing_ids

    if missing_ids:
        logger.info(f"Need predictions for {len(missing_ids)} samples, loading model...")
        model, device = load_model(config['model_name'], config['device'])
        predictions = run_predictions(model, dataset, predictions_dir, device, config, checkpoint_data)
        logger.info(f"Predictions completed: {len(predictions)} samples")
        # Keep model reference for vocabulary comparison (kern vocab extraction)
        _model_ref = model
        del model
    else:
        logger.info("All predictions already exist, skipping model loading.")
        predictions = {s['sample_id']: str(predictions_dir / f"{s['sample_id']}_prediction.txt") for s in dataset}
        _model_ref = None

    # Convert to MusicXML (output in timestamped dir)
    musicxml_files, conversion_methods, conversion_stats = convert_predictions_to_musicxml(
        predictions_dir, output_dir, config.get('use_sanitization', True),
    )

    # Setup linearization
    logger.info("Setting up Olimpic linearization...")
    Linearizer, MxlFile, ET = setup_olimpic()

    # Precompute GT linearizations (cached in base output dir so it persists)
    gt_data = precompute_gt_linearizations(dataset, base_output, Linearizer, MxlFile, ET)

    # Vocabulary analysis
    logger.info("Running vocabulary analysis...")
    vocab_info = run_vocabulary_analysis(dataset, output_dir)

    # Vocabulary comparison (Model vs Zeus vs Dataset — token-level)
    logger.info("Running vocabulary comparison...")
    if _model_ref is None:
        # If model was not loaded (predictions were cached), load it briefly for vocab
        logger.info("Loading model briefly for vocabulary extraction...")
        _model_ref, _ = load_model(config['model_name'], config.get('device', 'cpu'))
    vocab_comparison = run_vocabulary_comparison(_model_ref, gt_data, output_dir, vocab_info)
    del _model_ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate
    metrics_df, summary = evaluate_all(
        dataset, musicxml_files, gt_data, conversion_methods,
        vocab_info, vocab_comparison, output_dir, Linearizer, MxlFile, ET,
    )

    # TEDn evaluation (optional, computationally expensive)
    if config.get('enable_tedn', False):
        logger.info("Running TEDn evaluation (this may take a while)...")
        tedn_results = run_tedn_evaluation(
            dataset, musicxml_files, gt_data, predictions_dir,
            output_dir, config, Linearizer, MxlFile, ET,
        )
        # Merge TEDn into summary
        if tedn_results:
            summary['tedn'] = tedn_results
            summary_file = output_dir / "evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
    else:
        logger.info("TEDn evaluation disabled (set enable_tedn: true in config to enable)")

    # Per-document breakdown
    per_document_breakdown(metrics_df, output_dir)

    # Save report
    save_report(summary, conversion_stats, vocab_info, output_dir)

    # Symlink "latest" to this run for convenience
    latest_link = base_output / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Latest:  {latest_link} -> {output_dir.name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
