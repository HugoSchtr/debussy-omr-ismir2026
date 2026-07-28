"""
Data pipeline for SMT fine-tuning on Debussy handwritten manuscripts.

Handles:
  - MusicXML → kern conversion via music21
  - Image loading/preprocessing (grayscale, resize, binarization)
  - Tokenization using pretrained SMT vocabulary
  - Teacher forcing with configurable error rate
  - Batch collation with padding
"""

import os
import re
import random
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
logger = logging.getLogger(__name__)


# =============================================================================
# MusicXML → kern conversion
# =============================================================================
# Standalone converter (verovio primary, music21 fallback), vendored alongside
# this file as musicxml_to_kern.py -- see that file's docstring for an exact
# round-trip test suite covering 1/2/3/4-staff cases, ties, chords,
# accidentals, and dotted durations.
from musicxml_to_kern import musicxml_to_kern  # noqa: E402


# Standalone tokens to drop (dynamics/hairpins that leak through spine filtering)
_DROP_TOKENS = {'p', 'pp', 'ppp', 'f', 'ff', 'fff', 'mf', 'mp', 'sf', 'sfz', 'fp',
                'fz', 'rfz', 'cresc', 'decresc', 'dim',
                '<', '>', '[', ']'}


def decompose_kern_token(token: str) -> List[str]:
    """Decompose a compound Kern note/rest token into atomic tokens.

    Matches the bekrn (broken Kern) format used by PRAIG/fp-grandstaff.

    Examples:
        '4c#'     → ['4', 'c', '#']
        '4.E'     → ['4', '.', 'E']   (dot = separate atom)
        '16f##JK' → ['16', 'f', '##', 'J', 'K']
        '4r'      → ['4', 'r']
        '.'       → ['.']
        'q8cc'    → ['q', '8', 'cc']
    """
    # Non-note tokens returned as-is
    if (token.startswith('*') or token.startswith('=') or
            token in ('<b>', '<t>', '<s>', '<>', '.', ':', '<pad>', '<bos>', '<eos>') or
            token.startswith('!')):
        return [token]

    # Strip characters not in fp-grandstaff vocab:
    # ornaments (^/tTsm,), articulations (';?), slur marks (()~), stem direction ('`),
    # fermata (;), visibility modifier (y outside ties)
    token = re.sub(r'[()~^/\\;,?\'`tTsmp]', '', token)
    # Strip tuplet ratios (%N) before decomposition — never valid atoms
    token = re.sub(r'%-?\d+', '', token)
    if not token:
        return []

    atoms = []
    i = 0
    n = len(token)

    # 1. Grace prefix: q or qq
    if i < n and token[i] == 'q':
        j = i
        while j < n and token[j] == 'q':
            j += 1
        # Only treat as grace if followed by a digit (duration)
        if j < n and token[j].isdigit():
            atoms.append(token[i:j])
            i = j

    # 2. Duration: sequence of digits
    j = i
    while j < n and token[j].isdigit():
        j += 1
    if j > i:
        atoms.append(token[i:j])
        i = j

    # 3. Duration dots
    while i < n and token[i] == '.':
        atoms.append('.')
        i += 1

    # 4. Pitch: repeated letter (a-g or A-G) or 'r' for rest
    if i < n:
        if token[i] == 'r':
            atoms.append('r')
            i += 1
        elif token[i].isalpha() and token[i].lower() in 'abcdefg':
            j = i
            pitch_char = token[j]
            while j < n and token[j] == pitch_char:
                j += 1
            atoms.append(token[i:j])
            i = j

    # 5. Accidentals with optional editorial suffix (ONE suffix only)
    # fp-grandstaff vocab: #, ##, -, --, n + optional single suffix (X/i/j/x/y/Z)
    while i < n and token[i] in '#-n':
        j = i + 1
        # Double accidentals
        if j < n and token[j] == token[i] and token[i] in '#-':
            j += 1
        # At most ONE editorial/cautionary suffix: X i j x y Z
        if j < n and token[j] in 'XijxyZ':
            j += 1
        atoms.append(token[i:j])
        i = j

    # 6. Beam markers with optional direction
    # fp-grandstaff vocab has K>/K</L>/L< but NOT J>/J< — strip direction for J
    while i < n and token[i] in 'JKkL':
        beam_char = token[i]
        j = i + 1
        if j < n and token[j] in '<>' and beam_char != 'J':
            j += 1
        atoms.append(token[i:j])
        i = j

    # 7. Tie/slur markers with optional direction/modifier
    # fp-grandstaff vocab: [, ], [y — but NOT ]> or ]<
    while i < n and token[i] in '[]_':
        bracket = token[i]
        j = i + 1
        if bracket == '[' and j < n and token[j] == 'y':
            j += 1  # [y is in vocab
        # Skip > < y after ] — not in vocab
        atoms.append(token[i:j])
        i = j

    # 8. Skip any remaining characters (ornaments, editorial marks)
    # Don't emit them — they'd be OOV

    return atoms if atoms else [token]


def _normalize_kern_token(token: str, w2i: dict = None) -> str:
    """Normalize a single kern note/rest token to match SMT vocabulary.
    
    Applies:
      - Convert invisible rests ryy → r (e.g. 4ryy → 4r)
      - Move [ tie-start from prefix to suffix (kern → ekern convention)
      - Strip slur markers ( and ) — not in SMT vocabulary
      - Strip staccato ' — not in SMT vocabulary
      - Strip turn ornament ~ — not in SMT vocabulary
      - Strip editorial marker @ — not in SMT vocabulary
      - Convert JK → Jk (SMT uses lowercase k)
      - Strip cautionary X / editorial i if result is in vocabulary
    """
    # Invisible rests → regular rests
    token = token.replace('ryy', 'r')
    # Move tie-start [ from prefix to suffix
    if token.startswith('[') and len(token) > 1:
        token = token[1:] + '['
    # Strip slur open/close
    token = token.replace('(', '').replace(')', '')
    # Strip staccato (not in SMT vocab at all)
    token = token.replace("'", '')
    # Strip turn ornament (not in SMT vocab)
    token = token.replace('~', '')
    # Strip editorial marker
    token = token.replace('@', '')
    # JK → Jk (partial beam marker)
    token = token.replace('JK', 'Jk')
    # Strip cautionary accidental X, editorial ficta i, and accent >
    # only if the result is a known vocabulary token
    if w2i is not None and token not in w2i:
        # Try stripping X and i first
        stripped = token.replace('X', '').replace('i', '')
        if stripped in w2i:
            token = stripped
        elif stripped != token:
            # Try also stripping > (accent) from already X/i-stripped version
            stripped2 = stripped.replace('>', '')
            if stripped2 in w2i:
                token = stripped2
        else:
            # Try stripping just > (accent)
            stripped3 = token.replace('>', '')
            if stripped3 in w2i:
                token = stripped3
        # Strip tuplet ratios %N (e.g. 40%3EE → 40EE)
        if token not in w2i and '%' in token:
            stripped_pct = re.sub(r'%-?\d+', '', token)
            if stripped_pct in w2i:
                token = stripped_pct
    return token


_MANIP_TOKENS = {'*^', '*v', '*-', '*+'}


def kern_to_token_string(kern_content: str, w2i: dict = None, decompose: bool = False) -> str:
    """Convert raw kern content to the tokenized string format used by SMT.

    The SMT model expects kern encoded as:
      - Lines separated by <b>
      - Tabs separated by <t>
      - Spaces separated by <s>

    Filters out non-kern spines (e.g. **dynam), global/local comments,
    and non-essential interpretation lines. Normalizes kern tokens to
    match the **ekern_1.0 vocabulary.

    Output stays fixed-width: one column per ORIGINAL **kern spine (from the exclusive
    interpretation header), for the whole sample. A *^ spine split therefore does not grow
    the number of output columns — the split voices are folded back into their parent
    spine's single output column, space-joined, reusing the same convention already used
    for chords (multiple simultaneous pitches in one cell). This is what keeps a real
    *^/*v-bearing passage from being silently misread: `origin[i]` tracks, for the CURRENT
    column i (which shifts every time a *^/*v/*-/*+ row is processed), which ORIGINAL kern
    spine it still belongs to, so later rows are attributed to the correct staff even after
    the column layout has changed (found 2026-07-24: the previous version tracked a single
    fixed index list computed once from the header, so any row after a split silently read
    from the wrong columns — confirmed on-corpus to hit ~41% of all rows in 71.7% of files;
    same fix applied to smt-finetune/data.py, the primary copy).
    """
    lines = kern_content.strip().split('\n')

    # Step 1: Identify which columns are **kern spines, and set up the origin map.
    origin = None  # origin[i] = original kern-spine index for current column i, or None
    n_kern = 0
    for line in lines:
        if line.startswith('**'):
            cols = line.split('\t')
            origin = []
            for c in cols:
                if c.startswith('**kern'):
                    origin.append(n_kern)
                    n_kern += 1
                else:
                    origin.append(None)
            break

    # For decomposed mode, keep interpretations relevant to the fp-grandstaff vocab:
    # *clef, *k[], *M, *met(), *staff, *^, *v, *-, *+, bare *
    _KEEP_INTERP_DECOMPOSED = {'*^', '*v', '*-', '*+', '*'}

    # Step 2: Filter and process lines
    data_lines = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        # Skip global comments (!!!) and local comments (!)
        if line.startswith('!'):
            continue

        cols_now = line.split('\t') if origin is not None else None
        is_manip_line = (origin is not None and line.startswith('*') and not line.startswith('**')
                          and any(c in _MANIP_TOKENS for c in cols_now))

        # Skip non-essential interpretation lines
        if line.startswith('*'):
            cols = line.split('\t')
            if origin is not None:
                first_kern = next((cols[i] for i, o in enumerate(origin)
                                    if o is not None and i < len(cols)), cols[0])
            else:
                first_kern = cols[0]

            if decompose and first_kern.startswith('**'):
                continue  # Skip **kern headers in decomposed mode (not in fp vocab)

            # Keep clef, key signature, time signature, meter, staff, spine manipulations
            if first_kern.startswith('*clef') or first_kern.startswith('*k[') or \
               first_kern.startswith('*M') or first_kern.startswith('*met(') or \
               first_kern.startswith('*staff') or \
               first_kern in ('*^', '*v', '*-', '*+') or is_manip_line or \
               first_kern.startswith('**'):
                # Map staff numbering: verovio outputs *staff1/*staff2,
                # fp-grandstaff vocab expects *staff0/*staff+1
                if decompose and first_kern.startswith('*staff'):
                    cols = line.split('\t') if '\t' in line else [line]
                    mapped = []
                    for c in cols:
                        if c == '*staff1':
                            mapped.append('*staff0')
                        elif c == '*staff2':
                            mapped.append('*staff+1')
                        else:
                            mapped.append(c)
                    line = '\t'.join(mapped)
                    cols_now = line.split('\t') if origin is not None else cols_now
            else:
                continue  # Skip

        # Filter to (original) kern columns only, following the current origin map
        if origin is not None:
            cols = cols_now if cols_now is not None else line.split('\t')
            per_slot = [[] for _ in range(n_kern)]
            for i, val in enumerate(cols):
                o = origin[i] if i < len(origin) else None
                if o is not None:
                    per_slot[o].append(val)
            is_barline_line = all(c.startswith('=') for c in cols if c)
            if is_manip_line or line.startswith('*') or is_barline_line:
                # structural row (interpretation, spine manipulation, or barline): one
                # representative token per original spine, these aren't simultaneous
                # content and would just duplicate (e.g. "= =") if space-joined
                out_cols = [vals[0] if vals else '.' for vals in per_slot]
            else:
                # real note/rest data: fold simultaneous split-voice content into its
                # parent spine's cell via space-join (same convention as chords)
                out_cols = [' '.join(v for v in vals if v) if vals else '.' for vals in per_slot]
            line = '\t'.join(out_cols)

            # Advance the origin map past this row's spine manipulations, for all
            # subsequent rows (data and interpretation alike) until the next one.
            if is_manip_line:
                new_origin = []
                i = 0
                while i < len(cols):
                    tok = cols[i]
                    o = origin[i] if i < len(origin) else None
                    if tok == '*^':
                        new_origin.append(o)
                        new_origin.append(o)
                        i += 1
                    elif tok == '*v':
                        run_origin = o
                        while i < len(cols) and cols[i] == '*v':
                            i += 1
                        new_origin.append(run_origin)
                    elif tok == '*-':
                        i += 1  # spine terminates: drop this column going forward
                    elif tok == '*+':
                        new_origin.append(None)  # new spine, not tracked as kern content
                        i += 1
                    else:
                        new_origin.append(o)
                        i += 1
                origin = new_origin

        data_lines.append(line)
    
    # w2i passed to normalization: skip vocab-aware cleanup in decomposed mode
    norm_w2i = None if decompose else w2i

    token_parts = []
    for line in data_lines:
        parts = line.split('\t')
        normalized = []
        for p in parts:
            if p.startswith('**kern'):
                if decompose:
                    continue  # Skip header in decomposed mode
                p = '**ekern_1.0'
            elif p.startswith('='):
                # Strip barline numbers: =1 → =, =12 → =, keep =- and ==
                p = re.sub(r'^(=+)-?$', r'\1', re.sub(r'^(=+)\d+', r'\1', p))
            elif not p.startswith('*'):
                # Handle space-separated chord/dynamic tokens within a column
                if ' ' in p:
                    sub_parts = p.split(' ')
                    # Strip .ZZZ null-voice prefixes in sub-tokens
                    sub_parts = ['.' if (s.startswith('.ZZZ') or s.startswith('.zzz')) else s
                                 for s in sub_parts]
                    sub_parts = [_normalize_kern_token(s, norm_w2i) for s in sub_parts]
                    sub_parts = [s for s in sub_parts if s and s not in _DROP_TOKENS]
                    if decompose:
                        # Decompose each note into atoms, separate notes with <s>
                        decomposed_notes = []
                        for sp in sub_parts:
                            atoms = decompose_kern_token(sp)
                            if atoms:
                                decomposed_notes.append(' '.join(atoms))
                        p = ' <s> '.join(decomposed_notes) if decomposed_notes else '.'
                    else:
                        p = ' '.join(sub_parts) if sub_parts else '.'
                else:
                    # Strip .ZZZ null-voice prefix BEFORE normalization/decomposition
                    if p.startswith('.ZZZ') or p.startswith('.zzz'):
                        p = '.'
                    p = _normalize_kern_token(p, norm_w2i)
                    if p in _DROP_TOKENS:
                        p = '.'
                    elif decompose:
                        atoms = decompose_kern_token(p)
                        p = ' '.join(atoms) if atoms else '.'
            normalized.append(p)
        # Drop standalone dynamics/hairpins that leaked through
        normalized = [p for p in normalized if p not in _DROP_TOKENS]
        if normalized:
            token_parts.append(' <t> '.join(normalized))
    
    return ' <b> '.join(token_parts)


# =============================================================================
# Image preprocessing
# =============================================================================

def adaptive_binarize(image: np.ndarray, block_size: int = 35, C: int = 10) -> np.ndarray:
    """Apply adaptive binarization to a grayscale image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a grayscale image (uint8, H x W).
    
    Augmentations suitable for handwritten music scores:
      - Small rotation (simulates scan skew)
      - Brightness/contrast variation
      - Gaussian noise (simulates paper texture)
      - Erosion/dilation (simulates ink thickness variation)
      - Slight vertical shift
    """
    h, w = img.shape

    # Random small rotation (-2 to +2 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Brightness & contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Erosion or dilation (ink thickness)
    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # Vertical shift (1-3 pixels)
    if random.random() < 0.3:
        shift = random.randint(-3, 3)
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def load_and_preprocess_image(
    image_path: str,
    target_height: int = 256,
    max_width: int = 3056,
    binarize: bool = True,
    block_size: int = 35,
    C: int = 10,
    augment: bool = False,
) -> np.ndarray:
    """Load image, convert to grayscale, optionally binarize, and resize.
    
    Returns a float32 array of shape (1, H, W) normalized to [0, 1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Augment before binarization (on raw grayscale)
    if augment:
        img = augment_image(img)

    if binarize:
        img = adaptive_binarize(img, block_size, C)

    # Resize keeping aspect ratio, height = target_height
    h, w = img.shape
    new_w = int(w * target_height / h)
    
    # Clamp width
    if new_w > max_width:
        new_w = max_width
    
    # Round to multiple of 16 (needed for ConvNeXt encoder with 16x downsampling)
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (target_height // 16) * 16)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] and add channel dimension
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    
    return img


# =============================================================================
# Vocabulary builder
# =============================================================================

def build_kern_vocab_from_samples(
    samples,
    w2i: Dict[str, int] = None,
    decompose: bool = False,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build w2i/i2w by converting all MusicXML samples to kern and collecting tokens.

    Args:
        samples: List of (img_path, mxml_path) tuples.
        w2i: Optional existing vocab for context-aware normalization
             (e.g., stripping cautionary X when the bare form exists).
        decompose: Use decomposed (atomic) kern format.

    Returns:
        (w2i, i2w) with <pad>=0, <bos>=1, <eos>=2, then sorted data tokens.
    """
    all_tokens: set = set()
    for sample in samples:
        mxml_path = sample[-1]
        kern = musicxml_to_kern(mxml_path)
        if kern is None:
            continue
        token_str = kern_to_token_string(kern, w2i=w2i, decompose=decompose)
        all_tokens.update(token_str.split())

    new_w2i: Dict[str, int] = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    new_i2w: Dict[int, str] = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    next_idx = 3
    for token in sorted(all_tokens):
        if token not in new_w2i:
            new_w2i[token] = next_idx
            new_i2w[next_idx] = token
            next_idx += 1

    logger.info(f"Kern vocabulary from dataset: {len(new_w2i)} tokens "
                f"({len(all_tokens)} data + 3 special)")
    return new_w2i, new_i2w


# =============================================================================
# Dataset
# =============================================================================

class DebussyKernDataset(Dataset):
    """Dataset for Debussy manuscripts with MusicXML→kern on-the-fly conversion.
    
    Expects directory structure:
        dataset_dir/
            doc_id/
                page/
                    sample.jpg
                    sample.musicxml
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],  # list of (image_path, musicxml_path)
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        target_height: int = 256,
        max_width: int = 3056,
        max_seq_len: int = 1512,
        binarize: bool = True,
        block_size: int = 35,
        C: int = 10,
        teacher_forcing_error_rate: float = 0.2,
        is_train: bool = True,
        decompose: bool = False,
    ):
        self.w2i = w2i
        self.i2w = i2w
        self.target_height = target_height
        self.max_width = max_width
        self.max_seq_len = max_seq_len
        self.binarize = binarize
        self.block_size = block_size
        self.C = C
        self.teacher_forcing_error_rate = teacher_forcing_error_rate if is_train else 0.0
        self.is_train = is_train
        self.vocab_size = len(w2i)
        self.decompose = decompose
        
        # Pre-convert all MusicXML to kern and tokenize
        self.data = []
        self.skipped = 0
        for img_path, mxml_path in samples:
            kern = musicxml_to_kern(mxml_path)
            if kern is None:
                self.skipped += 1
                continue
            
            token_str = kern_to_token_string(kern, w2i=w2i, decompose=decompose)
            tokens = token_str.split(' ')
            
            # Map tokens to indices, skip unknown tokens
            token_ids = [w2i['<bos>']]
            unknown_count = 0
            for t in tokens:
                if t in w2i:
                    token_ids.append(w2i[t])
                else:
                    unknown_count += 1
            token_ids.append(w2i['<eos>'])
            
            # Truncate if too long
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len - 1] + [w2i['<eos>']]
            
            if unknown_count > 0:
                logger.debug(f"{os.path.basename(img_path)}: {unknown_count} unknown tokens")
            
            self.data.append({
                'image_path': img_path,
                'musicxml_path': mxml_path,
                'token_ids': token_ids,
                'kern_string': token_str,
            })
        
        if self.skipped > 0:
            logger.warning(f"Skipped {self.skipped}/{len(samples)} samples (MusicXML→kern conversion failed)")
        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        image = load_and_preprocess_image(
            item['image_path'],
            target_height=self.target_height,
            max_width=self.max_width,
            binarize=self.binarize,
            block_size=self.block_size,
            C=self.C,
            augment=self.is_train,
        )
        image = torch.from_numpy(image)
        
        # Original token IDs (used for labels — always clean)
        label_ids = item['token_ids']
        
        # Teacher forcing: corrupt decoder input tokens (NOT labels)
        decoder_input_ids = label_ids.copy()
        if self.teacher_forcing_error_rate > 0 and self.is_train:
            for i in range(1, len(decoder_input_ids) - 1):  # skip <bos> and <eos>
                if random.random() < self.teacher_forcing_error_rate:
                    decoder_input_ids[i] = random.randint(0, self.vocab_size - 1)
        
        return {
            'image': image,
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'kern_string': item['kern_string'],
            'musicxml_path': item['musicxml_path'],
        }


def batch_collate(batch):
    """Collate function: pad images (with white=1.0) and tokens (with pad=0) to batch max size.
    
    Returns:
        encoder_input: (B, 1, max_H, max_W) - padded images
        decoder_input: (B, max_L-1) - corrupted tokens without <eos> (teacher forcing)
        labels: (B, max_L-1) - clean tokens without <bos>
    """
    images = [item['image'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    kern_strings = [item['kern_string'] for item in batch]
    musicxml_paths = [item['musicxml_path'] for item in batch]

    # Pad images to max size in batch (pad with 1.0 = white)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded_images = torch.ones(len(images), 1, max_h, max_w, dtype=torch.float32)
    for i, img in enumerate(images):
        padded_images[i, :, :img.shape[1], :img.shape[2]] = img
    
    # Pad token sequences
    max_len = max(len(t) for t in label_ids)
    
    # decoder_input = corrupted tokens without <eos>
    # labels = clean original tokens without <bos>
    decoder_inputs = torch.zeros(len(decoder_input_ids), max_len - 1, dtype=torch.long)
    labels = torch.zeros(len(label_ids), max_len - 1, dtype=torch.long)
    
    for i in range(len(batch)):
        dec_len = len(decoder_input_ids[i]) - 1
        decoder_inputs[i, :dec_len] = decoder_input_ids[i][:-1]  # corrupted, without <eos>
        labels[i, :dec_len] = label_ids[i][1:]  # clean, without <bos>
    
    return {
        'encoder_input': padded_images,
        'decoder_input': decoder_inputs,
        'labels': labels,
        'kern_strings': kern_strings,
        'musicxml_paths': musicxml_paths,
    }


# =============================================================================
# Data loading utilities
# =============================================================================

def discover_samples(
    dataset_dir: str,
    exclude_documents: Optional[List[str]] = None,
    flat: bool = False,
) -> List[Tuple[str, str]]:
    """Discover all (image, musicxml) pairs from the dataset directory.
    
    Args:
        flat: If True, expects flat structure: doc_id/sample.{jpg,musicxml}
              If False, expects nested structure: doc_id/page/sample.{jpg,musicxml}
    """
    dataset_dir = Path(dataset_dir)
    exclude = set(exclude_documents) if exclude_documents else set()
    
    samples = []
    for doc_dir in sorted(dataset_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if doc_id in exclude:
            logger.info(f"Excluding document: {doc_id}")
            continue
        
        if flat:
            # Flat: files directly in doc_dir
            for musicxml_file in sorted(doc_dir.glob('*.musicxml')):
                stem = musicxml_file.stem
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = musicxml_file.with_suffix(ext)
                    if candidate.exists():
                        img_path = str(candidate)
                        break
                if img_path is None:
                    logger.warning(f"No image found for {musicxml_file}")
                    continue
                samples.append((img_path, str(musicxml_file)))
        else:
            # Nested: doc_id/page/files
            for page_dir in sorted(doc_dir.iterdir()):
                if not page_dir.is_dir():
                    continue
                for musicxml_file in sorted(page_dir.glob('*.musicxml')):
                    stem = musicxml_file.stem
                    img_path = None
                    for ext in ['.jpg', '.jpeg', '.png']:
                        candidate = musicxml_file.with_suffix(ext)
                        if candidate.exists():
                            img_path = str(candidate)
                            break
                    if img_path is None:
                        logger.warning(f"No image found for {musicxml_file}")
                        continue
                    samples.append((img_path, str(musicxml_file)))
    
    logger.info(f"Discovered {len(samples)} samples from {dataset_dir}")
    return samples


def _group_by_document(
    samples: List[Tuple[str, str]],
    flat: bool = False,
) -> Dict[str, List[Tuple[str, str]]]:
    """Group samples by document ID extracted from path structure.
    
    Args:
        flat: If True, doc_id is parts[-2] (flat: doc_id/file).
              If False, doc_id is parts[-3] (nested: doc_id/page/file).
    """
    doc_samples: Dict[str, List[Tuple[str, str]]] = {}
    for img_path, mxml_path in samples:
        parts = Path(img_path).parts
        doc_id = parts[-2] if flat else parts[-3]
        doc_samples.setdefault(doc_id, []).append((img_path, mxml_path))
    return doc_samples


def split_by_document(
    samples: List[Tuple[str, str]],
    train_ratio: float = 0.75,
    seed: int = 42,
    flat: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split samples into train/val by document ID (no document bleed).
    
    Document ID is extracted from the path structure: dataset_dir/doc_id/page/file
    """
    doc_samples = _group_by_document(samples, flat=flat)
    doc_ids = sorted(doc_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    n_train = int(len(doc_ids) * train_ratio)
    train_docs = set(doc_ids[:n_train])
    
    train = []
    val = []
    for doc_id in doc_ids:
        if doc_id in train_docs:
            train.extend(doc_samples[doc_id])
        else:
            val.extend(doc_samples[doc_id])
    
    logger.info(f"Split: {len(train)} train ({len(train_docs)} docs), "
                f"{len(val)} val ({len(doc_ids) - len(train_docs)} docs)")
    return train, val


def split_by_document_three_way(
    samples: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    flat: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split samples into train/val/test by document ID (no document bleed).
    
    Args:
        train_ratio: Fraction of documents for training.
        val_ratio: Fraction of documents for validation.
        The remaining (1 - train_ratio - val_ratio) goes to test.
    """
    doc_samples = _group_by_document(samples, flat=flat)
    doc_ids = sorted(doc_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    n_train = int(len(doc_ids) * train_ratio)
    n_val = int(len(doc_ids) * val_ratio)
    
    train_doc_ids = doc_ids[:n_train]
    val_doc_ids = doc_ids[n_train:n_train + n_val]
    test_doc_ids = doc_ids[n_train + n_val:]
    
    train = [s for d in train_doc_ids for s in doc_samples[d]]
    val = [s for d in val_doc_ids for s in doc_samples[d]]
    test = [s for d in test_doc_ids for s in doc_samples[d]]
    
    logger.info(f"3-way split: {len(train)} train ({len(train_doc_ids)} docs), "
                f"{len(val)} val ({len(val_doc_ids)} docs), "
                f"{len(test)} test ({len(test_doc_ids)} docs)")
    return train, val, test


def kfold_by_document(
    samples: List[Tuple[str, str]],
    n_folds: int = 5,
    seed: int = 42,
    val_ratio: float = 0.15,
    flat: bool = False,
) -> List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]]:
    """Generate k-fold cross-validation splits by document (no document bleed).
    
    Returns a list of (train, val, test) tuples, one per fold.
    In each fold the held-out documents form the test set, and a fraction of
    the remaining documents is used for validation.
    
    Args:
        val_ratio: Fraction of remaining (non-test) documents to use for validation.
        flat: If True, doc_id is parts[-2] of path (flat dir structure).
    """
    doc_samples = _group_by_document(samples, flat=flat)
    doc_ids = sorted(doc_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    # Split doc_ids into n_folds roughly equal chunks
    fold_size = len(doc_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(doc_ids)
        folds.append(doc_ids[start:end])
    
    splits = []
    for fold_idx in range(n_folds):
        test_ids = folds[fold_idx]
        remaining_ids = [d for i, chunk in enumerate(folds) if i != fold_idx for d in chunk]
        
        n_val = max(1, int(len(remaining_ids) * val_ratio))
        val_ids = remaining_ids[:n_val]
        train_ids = remaining_ids[n_val:]
        
        train = [s for d in train_ids for s in doc_samples[d]]
        val = [s for d in val_ids for s in doc_samples[d]]
        test = [s for d in test_ids for s in doc_samples[d]]
        
        logger.info(f"Fold {fold_idx + 1}/{n_folds}: "
                    f"{len(train)} train ({len(train_ids)} docs), "
                    f"{len(val)} val ({len(val_ids)} docs), "
                    f"{len(test)} test ({len(test_ids)} docs)")
        splits.append((train, val, test))
    
    return splits


# =============================================================================
# Lightning DataModule
# =============================================================================

class DebussyDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for Debussy fine-tuning."""

    def __init__(
        self,
        dataset_dir: str,
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        batch_size: int = 4,
        num_workers: int = 4,
        target_height: int = 256,
        max_width: int = 3056,
        max_seq_len: int = 1512,
        binarize: bool = True,
        block_size: int = 35,
        C: int = 10,
        teacher_forcing_error_rate: float = 0.2,
        train_ratio: float = 0.75,
        val_ratio: Optional[float] = None,
        exclude_documents: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.w2i = w2i
        self.i2w = i2w
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_height = target_height
        self.max_width = max_width
        self.max_seq_len = max_seq_len
        self.binarize = binarize
        self.block_size = block_size
        self.C = C
        self.teacher_forcing_error_rate = teacher_forcing_error_rate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.exclude_documents = exclude_documents
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        all_samples = discover_samples(self.dataset_dir, self.exclude_documents)
        
        common_args = dict(
            w2i=self.w2i,
            i2w=self.i2w,
            target_height=self.target_height,
            max_width=self.max_width,
            max_seq_len=self.max_seq_len,
            binarize=self.binarize,
            block_size=self.block_size,
            C=self.C,
        )
        
        if self.val_ratio is not None:
            # 3-way split: train / val / test
            train_samples, val_samples, test_samples = split_by_document_three_way(
                all_samples, self.train_ratio, self.val_ratio, self.seed
            )
            self.test_dataset = DebussyKernDataset(
                samples=test_samples,
                teacher_forcing_error_rate=0.0,
                is_train=False,
                **common_args,
            )
        else:
            # 2-way split: train / val (original behavior)
            train_samples, val_samples = split_by_document(
                all_samples, self.train_ratio, self.seed
            )
        
        self.train_dataset = DebussyKernDataset(
            samples=train_samples,
            teacher_forcing_error_rate=self.teacher_forcing_error_rate,
            is_train=True,
            **common_args,
        )
        
        self.val_dataset = DebussyKernDataset(
            samples=val_samples,
            teacher_forcing_error_rate=0.0,
            is_train=False,
            **common_args,
        )
        
        parts = [f"Train: {len(self.train_dataset)}", f"Val: {len(self.val_dataset)}"]
        if self.test_dataset is not None:
            parts.append(f"Test: {len(self.test_dataset)}")
        logger.info(" | ".join(parts))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=batch_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # predict one at a time for autoregressive decoding
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=batch_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        ds = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=batch_collate,
            pin_memory=True,
        )
