import gin
import os
import cv2
import torch

import numpy as np

from loguru import logger
from rich.progress import track
from torch.utils.data import Dataset
from torchvision import transforms
from utils import check_and_retrieveVocabulary
from datasets import load_dataset as hf_load_dataset
from PIL import Image

@logger.catch
def batch_preparation_ctc(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    L = [sample[2] for sample in data]
    T = [sample[3] for sample in data]
    indices = [sample[4] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        c, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])
    Y_train = torch.zeros(size=[len(gt),max_length_seq])
    for i, seq in enumerate(gt):
        Y_train[i, 0:len(seq)] = torch.from_numpy(np.asarray([char for char in seq]))

    return X_train, Y_train, L, T, indices

@logger.catch
@gin.configurable
def load_data(partition_file, resize_ratio, use_raw_krn=False, load_distorted=False, extension=".bekrn"):
    X = []
    Y = []
    names = []
    with open(partition_file) as partfile:
        part_lines = partfile.read()
        part_lines = part_lines.split("\n")
        for file_path in track(part_lines, description="Loading..."):
            if not file_path.strip():
                continue
            if extension != ".bekrn":
                file_path = file_path.replace(".bekrn", extension)
            # Append extension if path doesn't already end with it
            if not file_path.endswith(extension):
                file_path = file_path + extension
            krn = None
            krnlines = []
            file_path = f"datasets/{file_path}"
            if os.path.isfile(file_path):
                with open(file_path) as krnfile:
                    krn = krnfile.read()
                    krn = krn.replace(" ", " <s> ")
                    krn = krn.replace("·", " ")
                    lines = krn.split("\n")
                    for line in lines:
                        line = line.replace("\t", " <t> ")
                        line = line.split(" ")
                        # Filter out empty strings from the line
                        line = [token for token in line if token]
                        if len(line) > 0:
                            line.append("<b>")
                            krnlines.append(line)
                    # Resolve image path: try .jpg first, then .png
                    base_path = os.path.splitext(file_path)[0]
                    img_path = None
                    for img_ext in (".jpg", ".png"):
                        candidate = f"{base_path}{img_ext}"
                        if load_distorted:
                            candidate = f"{base_path}_distorted{img_ext}"
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is not None:
                        if load_distorted:
                            height = 256
                            img = cv2.imread(img_path, 0)
                            width = int(float(height * img.shape[1]) / img.shape[0])
                            img =  cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                            if (height//8) * (width//16) > len(sum(krnlines, [])):
                                width = int(np.ceil(img.shape[1] * resize_ratio))
                                height = int(np.ceil(img.shape[0] * resize_ratio))
                                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                                X.append(img)
                                Y.append(sum(krnlines, []))
                                names.append(file_path)
                        else:
                            img = cv2.imread(img_path, 0)
                            width = int(np.ceil(img.shape[1] * resize_ratio))
                            height = int(np.ceil(img.shape[0] * resize_ratio))
                            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                            # CTC feasibility: pad width so T >= L after rotation
                            L_tokens = len(sum(krnlines, []))
                            ctc_margin = 2
                            height_factor = height // 8
                            if height_factor > 0:
                                min_width = 16 * int(np.ceil((L_tokens + ctc_margin) / height_factor))
                                if width < min_width:
                                    pad_cols = min_width - width
                                    print(f"CTC padding: {file_path} | shape=({height},{width}) -> ({height},{min_width}) | L={L_tokens}, pad={pad_cols}")
                                    img = np.pad(img, ((0, 0), (0, pad_cols)), mode='constant', constant_values=255)
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            X.append(img)
                            Y.append(sum(krnlines, []))
                            names.append(file_path)

    return X, Y, names


class PoliphonicDataset(Dataset):
    def __init__(self, partition_file) -> None:
        self.x, self.y, self.names = load_data(partition_file)
        self.tensorTransform = transforms.ToTensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image = self.tensorTransform(self.x[index])
        gt = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        return image, gt, (image.shape[2] // 8) * (image.shape[1] // 16), len(gt), index

    def get_max_hw(self):
        max_height = max(img.shape[0] for img in self.x)
        max_width = max(img.shape[1] for img in self.x)
        return max_height, max_width

    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y

    def get_sample_name(self, index):
        return self.names[index] if index < len(self.names) else f"unknown_{index}"

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w


class HuggingFacePoliphonicDataset(Dataset):
    """Dataset class that loads data directly from HuggingFace - MEMORY EFFICIENT VERSION"""
    def __init__(self, hf_dataset_split, resize_ratio=0.125, max_height=None, target_height=None) -> None:
        """
        Args:
            hf_dataset_split: A split from a HuggingFace dataset (e.g., dataset['train'])
            resize_ratio: Ratio to resize images (used only when target_height is None)
            max_height: If set and target_height is None, caps image height BEFORE rotation
                        to prevent LSTM sequence overflow.
            target_height: If set, normalises every image to exactly this height (aspect
                           ratio preserved) BEFORE rotation, matching the local distorted
                           pipeline (height=256). When set, resize_ratio and max_height are
                           ignored, giving consistent sequence lengths across datasets.
        """
        self.hf_dataset = hf_dataset_split
        self.resize_ratio = resize_ratio
        self.max_height = max_height
        self.target_height = target_height
        self.tensorTransform = transforms.ToTensor()
        
        # Pre-process ONLY transcriptions (text is small, images are huge!)
        # Images will be loaded on-demand in __getitem__ to avoid OOM errors
        logger.info(f"Processing {len(hf_dataset_split)} transcriptions from HuggingFace dataset...")
        self.y = []

        # Use a transcription-only view so the HuggingFace dataset does NOT decode
        # images while we iterate. Without this, hf_dataset_split[idx] decodes all
        # features in the row (including the full PIL image) even if we never access
        # item['image'] — causing every image to pass through RAM during init.
        col_names = hf_dataset_split.column_names if hasattr(hf_dataset_split, 'column_names') else None
        if col_names is not None and 'image' in col_names:
            transcription_view = hf_dataset_split.select_columns(['transcription'])
        else:
            transcription_view = hf_dataset_split
        
        for idx in track(range(len(transcription_view)), description="Processing HF transcriptions"):
            item = transcription_view[idx]
            
            # Debug first few samples (access hf_dataset only for key listing, not image decoding)
            if idx < 3:
                logger.info(f"Sample {idx}: keys=['image', 'transcription']")
                logger.info(f"  Has transcription: {'transcription' in item}")
                if 'transcription' in item:
                    trans = item['transcription']
                    logger.info(f"  Transcription type: {type(trans)}, len={len(trans) if trans else 'None'}")
                    if trans:
                        logger.info(f"  First 200 chars: {repr(trans[:200])}")
            
            # Process transcription ONLY - images loaded on-demand
            item = transcription_view[idx]
            transcription = item['transcription']
            transcription = transcription.replace(" ", " <s> ")
            transcription = transcription.replace("·", " ")
            lines = transcription.split("\n")
            krnlines = []
            for line in lines:
                line = line.replace("\t", " <t> ")
                line = line.split(" ")
                # Filter out empty strings from the line
                line = [token for token in line if token]
                if len(line) > 0:
                    line.append("<b>")
                    krnlines.append(line)
            
            # Debug first sample
            if idx == 0:
                logger.info(f"First sample processing debug:")
                logger.info(f"  Transcription length: {len(item['transcription'])}")
                logger.info(f"  Number of lines: {len(lines)}")
                logger.info(f"  Number of krnlines: {len(krnlines)}")
                if krnlines:
                    logger.info(f"  First krnline: {krnlines[0][:10]}")
                logger.info(f"  Total tokens: {len(sum(krnlines, []))}")
            
            # Store only transcription tokens (images loaded on-demand)
            self.y.append(sum(krnlines, []))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Load and process image on-demand to save memory
        item = self.hf_dataset[index]
        image = item['image']
        
        if isinstance(image, Image.Image):
            img = np.array(image.convert('L'))  # Convert to grayscale
        else:
            img = image
        
        # Resize image
        if self.target_height is not None:
            # Normalise to a fixed target height (aspect ratio preserved).
            # Mirrors the local distorted pipeline which always normalises to height=256
            # before rotation, giving consistent LSTM sequence lengths across datasets.
            scale = self.target_height / img.shape[0]
            width = max(1, int(np.ceil(img.shape[1] * scale)))
            height = self.target_height
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            width = int(np.ceil(img.shape[1] * self.resize_ratio))
            height = int(np.ceil(img.shape[0] * self.resize_ratio))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            # Cap height before rotation to bound the LSTM temporal sequence length
            if self.max_height is not None and height > self.max_height:
                scale = self.max_height / height
                width = max(1, int(np.ceil(width * scale)))
                height = self.max_height
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        # Ensure CTC feasibility before rotation.
        # After rotation: T = (img.shape[0]//8) * (img.shape[1]//16)
        # With target_height=256: T = 32 * (width//16).
        # Pad image width (right side, white=255) so that T >= L + ctc_margin.
        L_tokens = len(self.y[index])
        ctc_margin = 2
        height_factor = img.shape[0] // 8  # = 32 when target_height=256
        if height_factor > 0:
            min_width_for_ctc = 16 * int(np.ceil((L_tokens + ctc_margin) / height_factor))
            if img.shape[1] < min_width_for_ctc:
                pad_cols = min_width_for_ctc - img.shape[1]
                logger.warning(
                    f"CTC padding needed for sample {index}: "
                    f"img pre-rotation shape=({img.shape[0]}, {img.shape[1]}), "
                    f"L={L_tokens}, T_before={(img.shape[0]//8)*(img.shape[1]//16)}, "
                    f"T_after={(img.shape[0]//8)*(min_width_for_ctc//16)}, "
                    f"padding {pad_cols} white columns on the right"
                )
                img = np.pad(img, ((0, 0), (0, pad_cols)), mode='constant', constant_values=255)

        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert to tensor
        image = self.tensorTransform(img)
        
        # Handle unknown tokens gracefully
        gt_indices = []
        unknown_tokens = set()
        for token in self.y[index]:
            if token in self.w2i:
                gt_indices.append(self.w2i[token])
            else:
                unknown_tokens.add(token)
                # Use padding token for unknowns
                gt_indices.append(self.padding_token)
        
        if unknown_tokens:
            logger.warning(f"Sample {index} contains unknown tokens: {unknown_tokens}")
        
        gt = torch.from_numpy(np.asarray(gt_indices))
        
        return image, gt, (image.shape[2] // 8) * (image.shape[1] // 16), len(gt), index

    def get_max_hw(self):
        # Calculate max dimensions from ALL images (just reading dimensions is fast - no pixel data loaded)
        max_width = 0
        max_height = 0
        logger.info(f"Calculating max image dimensions from {len(self.hf_dataset)} images...")
        for idx in track(range(len(self.hf_dataset)), description="Scanning image dimensions"):
            item = self.hf_dataset[idx]
            image = item['image']
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                height, width = image.shape[:2]
            
            # Apply the same resize logic as __getitem__
            if self.target_height is not None:
                scale = self.target_height / height
                width = max(1, int(np.ceil(width * scale)))
                height = self.target_height
            else:
                width = int(np.ceil(width * self.resize_ratio))
                height = int(np.ceil(height * self.resize_ratio))
                if self.max_height is not None and height > self.max_height:
                    scale = self.max_height / height
                    width = max(1, int(np.ceil(width * scale)))
                    height = self.max_height
            # Mirror the CTC feasibility padding from __getitem__
            L_tokens = len(self.y[idx])
            ctc_margin = 2
            height_factor = height // 8
            if height_factor > 0:
                min_width_for_ctc = 16 * int(np.ceil((L_tokens + ctc_margin) / height_factor))
                width = max(width, min_width_for_ctc)

            # After rotation, width becomes height and vice versa
            max_width = max(max_width, height)
            max_height = max(max_height, width)
        
        logger.info(f"Max dimensions after resize: height={max_height}, width={max_width}")
        return max_height, max_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w

@gin.configurable
def load_dataset(train_path=None, val_path=None, test_path=None, corpus_name=None):
    train_dataset = PoliphonicDataset(partition_file=train_path)
    val_dataset = PoliphonicDataset(partition_file=val_path)
    test_dataset = PoliphonicDataset(partition_file=test_path)

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{corpus_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset


@gin.configurable
def load_dataset_from_huggingface(dataset_name="PRAIG/grandstaff", corpus_name="GrandStaff_bekrn", resize_ratio=0.125, max_height=None, target_height=None, filter_corrupted_samples=True):
    """
    Load dataset directly from HuggingFace with their official splits
    
    Args:
        dataset_name: HuggingFace dataset name (default: "PRAIG/grandstaff")
        corpus_name: Name for vocabulary files
        resize_ratio: Ratio to resize images
        filter_corrupted_samples: If True, filters out known corrupted samples (default: True)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
    hf_dataset = hf_load_dataset(dataset_name)
    
    logger.info(f"Split sizes - Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['val'])}, Test: {len(hf_dataset['test'])}")
    
    # Filter out known corrupted samples
    if filter_corrupted_samples:
        # Known corrupted samples per dataset
        corrupted_samples = {
            "PRAIG/camera-grandstaff": {
                'train': [29279],  # Sample 29279: 7x256 pixels (too narrow)
                'val': [],
                'test': []
            }
        }
        
        if dataset_name in corrupted_samples:
            for split_name in ['train', 'val', 'test']:
                corrupted_list = corrupted_samples[dataset_name][split_name]
                if corrupted_list and split_name in hf_dataset:
                    original_size = len(hf_dataset[split_name])
                    # Create list of valid indices (excluding corrupted ones)
                    valid_indices = [i for i in range(original_size) if i not in corrupted_list]
                    hf_dataset[split_name] = hf_dataset[split_name].select(valid_indices)
                    filtered_size = len(hf_dataset[split_name])
                    logger.info(f"Filtered {original_size - filtered_size} corrupted sample(s) from {split_name} split: {corrupted_list}")
    
    logger.info(f"Final split sizes - Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['val'])}, Test: {len(hf_dataset['test'])}")
    
    train_dataset = HuggingFacePoliphonicDataset(hf_dataset['train'], resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)
    val_dataset = HuggingFacePoliphonicDataset(hf_dataset['val'], resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)
    test_dataset = HuggingFacePoliphonicDataset(hf_dataset['test'], resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)

    # Debug vocabulary building
    train_gt = train_dataset.get_gt()
    val_gt = val_dataset.get_gt()
    test_gt = test_dataset.get_gt()
    logger.info(f"Ground truth sizes - Train: {len(train_gt)}, Val: {len(val_gt)}, Test: {len(test_gt)}")
    if train_gt:
        logger.info(f"First train sample tokens: {len(train_gt[0])} tokens")
        if train_gt[0]:
            logger.info(f"  Sample tokens: {train_gt[0][:20]}")
    else:
        logger.error("Train ground truth is EMPTY!")
    
    w2i, i2w = check_and_retrieveVocabulary(
        [train_gt, val_gt, test_gt], 
        "vocab/", 
        f"{corpus_name}"
    )

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset


@gin.configurable
def load_mixed_datasets_from_huggingface(dataset_names=None, corpus_name="Mixed_bekrn", resize_ratio=0.125, max_height=None, target_height=None):
    """
    Load and mix multiple HuggingFace datasets while respecting their splits
    
    Args:
        dataset_names: List of HuggingFace dataset names (e.g., ["PRAIG/grandstaff", "PRAIG/camera-grandstaff"])
        corpus_name: Name for vocabulary files
        resize_ratio: Ratio to resize images
    
    Returns:
        train_dataset, val_dataset, test_dataset (concatenated)
    """
    if dataset_names is None or len(dataset_names) == 0:
        raise ValueError("dataset_names must be a non-empty list")
    
    logger.info(f"Loading and mixing {len(dataset_names)} datasets from HuggingFace")
    
    all_train_splits = []
    all_val_splits = []
    all_test_splits = []
    
    # Load each dataset and collect split references (lazy - no data loaded)
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        hf_dataset = hf_load_dataset(dataset_name)
        logger.info(f"  Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['val'])}, Test: {len(hf_dataset['test'])}")
        
        # Store references to splits, not the data itself
        all_train_splits.append(hf_dataset['train'])
        all_val_splits.append(hf_dataset['val'])
        all_test_splits.append(hf_dataset['test'])
    
    # Create a lazy concatenation wrapper that doesn't load data into memory
    class ConcatenatedDataset:
        """Lazy concatenation - accesses original datasets on-demand"""
        def __init__(self, datasets):
            self.datasets = datasets
            self.cumulative_sizes = []
            total = 0
            for ds in datasets:
                total += len(ds)
                self.cumulative_sizes.append(total)
        
        def __len__(self):
            return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        
        def __getitem__(self, idx):
            # Find which dataset contains this index
            for dataset_idx, cumulative_size in enumerate(self.cumulative_sizes):
                if idx < cumulative_size:
                    # Calculate index within that dataset
                    if dataset_idx == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                    return self.datasets[dataset_idx][local_idx]
            raise IndexError(f"Index {idx} out of range")
    
    # Create lazy concatenated views
    train_concat = ConcatenatedDataset(all_train_splits)
    val_concat = ConcatenatedDataset(all_val_splits)
    test_concat = ConcatenatedDataset(all_test_splits)
    
    logger.info(f"Mixed dataset sizes - Train: {len(train_concat)}, Val: {len(val_concat)}, Test: {len(test_concat)}")
    
    # Create datasets from concatenated views (still lazy - images loaded on-demand)
    train_dataset = HuggingFacePoliphonicDataset(train_concat, resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)
    val_dataset = HuggingFacePoliphonicDataset(val_concat, resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)
    test_dataset = HuggingFacePoliphonicDataset(test_concat, resize_ratio=resize_ratio, max_height=max_height, target_height=target_height)
    
    # Build vocabulary from all mixed data
    train_gt = train_dataset.get_gt()
    val_gt = val_dataset.get_gt()
    test_gt = test_dataset.get_gt()
    
    w2i, i2w = check_and_retrieveVocabulary(
        [train_gt, val_gt, test_gt], 
        "vocab/", 
        f"{corpus_name}"
    )
    
    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)
    
    return train_dataset, val_dataset, test_dataset

