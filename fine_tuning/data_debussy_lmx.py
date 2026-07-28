"""
Data pipeline for fine-tuning on Debussy dataset using LMX format.

Converts MusicXML files from the Debussy dataset to linearized MusicXML (LMX)
for fine-tuning models that were trained on LMX vocabulary.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# Import the LMX linearizer from olimpic-icdar24 (https://github.com/ufal/olimpic-icdar24).
# Clone that repo and point OLIMPIC_ICDAR24_DIR at it (env var, defaults to a sibling
# ./olimpic-icdar24 directory next to this repo) -- see fine_tuning/README.md.
OLIMPIC_ICDAR24_DIR = os.environ.get(
    'OLIMPIC_ICDAR24_DIR', os.path.join(os.path.dirname(__file__), '..', 'olimpic-icdar24')
)
if OLIMPIC_ICDAR24_DIR not in sys.path:
    sys.path.insert(0, OLIMPIC_ICDAR24_DIR)
from app.linearization.Linearizer import Linearizer
from app.symbolic.MxlFile import MxlFile

from data import augment_image

logger = logging.getLogger(__name__)


# =============================================================================
# Image preprocessing (same as data_lmx.py)
# =============================================================================

def adaptive_binarize(image: np.ndarray, block_size: int = 35, C: int = 10) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )


def load_and_preprocess_image(
    image_path: str,
    target_height: int = 256,
    max_width: int = 3056,
    binarize: bool = True,
    block_size: int = 35,
    C: int = 10,
    augment: bool = False,
) -> np.ndarray:
    """Load image from file, resize, normalize to [0, 1].

    Must match the normalization used during pretraining (data_lmx.py):
    pixels in [0, 1], padding with 1.0 (white), shape (1, H, W).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Augment before binarization (on raw grayscale) -- matches data.py's ordering
    if augment:
        image = augment_image(image)

    h, w = image.shape
    scale = target_height / h
    new_w = int(w * scale)
    if new_w > max_width:
        new_w = max_width

    # Round to multiple of 16 (ConvNeXt encoder downsampling)
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (target_height // 16) * 16)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if binarize:
        image = adaptive_binarize(image, block_size=block_size, C=C)

    # Normalize to [0, 1] — must match pretraining pipeline (data_lmx.py)
    image = image.astype(np.float32) / 255.0

    # Add channel dimension: (1, H, W)
    image = np.expand_dims(image, axis=0)

    return image


def musicxml_to_lmx(musicxml_path: str) -> str:
    """Convert a MusicXML file to linearized MusicXML (LMX) string."""
    import xml.etree.ElementTree as ET
    
    # Parse the .musicxml file directly
    tree = ET.parse(musicxml_path)
    root = tree.getroot()
    
    # Linearize using olimpic-icdar24 linearizer
    linearizer = Linearizer(fail_on_unknown_tokens=False)
    
    # Process each part
    for part in root.findall('.//part'):
        linearizer.process_part(part)
    
    # Join tokens with spaces
    return ' '.join(linearizer.output_tokens)


# =============================================================================
# Dataset discovery (adapted from data.py for Debussy structure)
# =============================================================================

def discover_debussy_samples(
    dataset_dir: str,
    exclude_documents: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Discover samples from Debussy dataset structure.
    
    Returns list of (document_id, image_path, musicxml_path) tuples.
    """
    dataset_path = Path(dataset_dir)
    exclude = set(exclude_documents or [])
    samples = []
    
    for doc_dir in sorted(dataset_path.iterdir()):
        if not doc_dir.is_dir():
            continue
        
        doc_id = doc_dir.name
        if doc_id in exclude:
            logger.info(f"Excluding document: {doc_id}")
            continue
        
        # Find all MusicXML files in this document
        for mxml_file in sorted(doc_dir.rglob('*.musicxml')):
            # Find corresponding image (assumes same name with .png)
            img_file = mxml_file.with_suffix('.png')
            if not img_file.exists():
                img_file = mxml_file.with_suffix('.jpg')
            
            if img_file.exists():
                samples.append((doc_id, str(img_file), str(mxml_file)))
            else:
                logger.warning(f"No image found for {mxml_file}")
    
    logger.info(f"Discovered {len(samples)} samples from {dataset_dir}")
    return samples


def build_lmx_vocab_from_samples(
    samples,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build w2i/i2w by converting all MusicXML samples to LMX and collecting tokens.

    Accepts samples as 2-tuples (img_path, mxml_path) or 3-tuples (doc_id, img_path, mxml_path).
    Special tokens: <pad>=0, <bos>=1, <eos>=2
    """
    all_tokens = set()
    for sample in samples:
        mxml_path = sample[-1]
        try:
            lmx_string = musicxml_to_lmx(mxml_path)
            all_tokens.update(lmx_string.split())
        except Exception as e:
            logger.warning(f"Could not linearize {mxml_path}: {e}")

    sorted_tokens = sorted(all_tokens)
    w2i = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    i2w = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    for idx, token in enumerate(sorted_tokens, start=3):
        w2i[token] = idx
        i2w[idx] = token

    logger.info(f"Vocabulary: {len(w2i)} tokens ({len(sorted_tokens)} LMX + 3 special)")
    return w2i, i2w


def split_by_document(
    samples: List[Tuple[str, str, str]],
    train_ratio: float = 0.75,
    val_ratio: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split samples by document ID."""
    doc_samples = defaultdict(list)
    for sample in samples:
        img_path, mxml_path = sample[-2], sample[-1]
        doc_id = Path(img_path).parts[-3]
        doc_samples[doc_id].append((img_path, mxml_path))
    
    doc_ids = sorted(doc_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    
    if val_ratio is None:
        # 2-way split (train/val)
        n_train = int(len(doc_ids) * train_ratio)
        train_docs = doc_ids[:n_train]
        val_docs = doc_ids[n_train:]
        test_docs = []
    else:
        # 3-way split (train/val/test)
        n_train = int(len(doc_ids) * train_ratio)
        n_val = int(len(doc_ids) * val_ratio)
        train_docs = doc_ids[:n_train]
        val_docs = doc_ids[n_train:n_train + n_val]
        test_docs = doc_ids[n_train + n_val:]
    
    train_samples = [s for doc in train_docs for s in doc_samples[doc]]
    val_samples = [s for doc in val_docs for s in doc_samples[doc]]
    test_samples = [s for doc in test_docs for s in doc_samples[doc]]
    
    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    return train_samples, val_samples, test_samples


# =============================================================================
# Dataset
# =============================================================================

class DebussyLMXDataset(Dataset):
    """Debussy dataset with MusicXML → LMX conversion."""
    
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        w2i: Dict[str, int],
        i2w: Dict[int, str],
        target_height: int = 256,
        max_width: int = 3056,
        max_seq_len: int = 1512,
        binarize: bool = True,
        block_size: int = 35,
        C: int = 10,
        teacher_forcing_error_rate: float = 0.0,
        is_train: bool = False,
    ):
        self.samples = samples
        self.w2i = w2i
        self.i2w = i2w
        self.target_height = target_height
        self.max_width = max_width
        self.max_seq_len = max_seq_len
        self.binarize = binarize
        self.block_size = block_size
        self.C = C
        self.teacher_forcing_error_rate = teacher_forcing_error_rate
        self.is_train = is_train
        self.vocab_size = len(w2i)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mxml_path = self.samples[idx]
        
        # Load and preprocess image
        image = load_and_preprocess_image(
            img_path,
            target_height=self.target_height,
            max_width=self.max_width,
            binarize=self.binarize,
            block_size=self.block_size,
            C=self.C,
            augment=self.is_train,
        )
        
        # Convert MusicXML to LMX tokens
        try:
            lmx_string = musicxml_to_lmx(mxml_path)
            tokens = lmx_string.split()
        except Exception as e:
            logger.error(f"Failed to linearize {mxml_path}: {e}")
            tokens = []
        
        # Convert to token IDs
        bos_id = self.w2i.get('<bos>', 1)
        eos_id = self.w2i.get('<eos>', 2)
        pad_id = self.w2i.get('<pad>', 0)
        
        token_ids = [bos_id]
        for token in tokens:
            if token in self.w2i:
                token_ids.append(self.w2i[token])
            # Unknown tokens are skipped (consistent with Kern and LMX-from-scratch pipelines)
        token_ids.append(eos_id)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [eos_id]
        
        # Teacher forcing: corrupt decoder input tokens (matching Kern pipeline)
        decoder_input_ids = token_ids.copy()
        if self.teacher_forcing_error_rate > 0 and self.is_train:
            for i in range(1, len(decoder_input_ids) - 1):  # skip <bos> and <eos>
                if random.random() < self.teacher_forcing_error_rate:
                    decoder_input_ids[i] = random.randint(0, self.vocab_size - 1)
        
        return {
            'image': torch.from_numpy(image),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'label_ids': torch.tensor(token_ids, dtype=torch.long),
            'lmx_strings': [lmx_string],  # For validation
            'musicxml_path': mxml_path,
        }


def batch_collate(batch):
    """Collate function: pad images (with white=1.0) and tokens (with pad=0) to batch max size.

    Returns:
        encoder_input: (B, 1, max_H, max_W) - padded images
        decoder_input: (B, max_L-1) - corrupted tokens without <eos> (teacher forcing)
        labels: (B, max_L-1) - clean tokens without <bos>
        lmx_strings: list of ground truth LMX strings
    """
    images = [item['image'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lmx_strings = [item['lmx_strings'][0] for item in batch]
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
        'lmx_strings': lmx_strings,
        'musicxml_paths': musicxml_paths,
    }


# =============================================================================
# Lightning DataModule
# =============================================================================

class DebussyLMXDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for Debussy dataset with LMX format."""
    
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
    
    def setup(self, stage: Optional[str] = None):
        # Discover all samples
        all_samples = discover_debussy_samples(
            self.dataset_dir,
            exclude_documents=self.exclude_documents,
        )
        
        # Split by document
        train_samples, val_samples, test_samples = split_by_document(
            all_samples,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        
        # Create datasets
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
        
        self.train_dataset = DebussyLMXDataset(
            samples=train_samples,
            teacher_forcing_error_rate=self.teacher_forcing_error_rate,
            is_train=True,
            **common_args,
        )
        
        self.val_dataset = DebussyLMXDataset(
            samples=val_samples,
            teacher_forcing_error_rate=0.0,
            is_train=False,
            **common_args,
        )
        
        if test_samples:
            self.test_dataset = DebussyLMXDataset(
                samples=test_samples,
                teacher_forcing_error_rate=0.0,
                is_train=False,
                **common_args,
            )
    
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
            batch_size=1,  # For validation, use batch_size=1
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=batch_collate,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=batch_collate,
            pin_memory=True,
        )
