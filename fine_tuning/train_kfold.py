"""
K-fold cross-validation training for SMT fine-tuning.

Usage:
    python train_kfold.py --config config_kfold.yaml
    python train_kfold.py --config config_kfold.yaml --n_folds 5
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from smt_model import SMTConfig, SMTModelForCausalLM
from smt_trainer import SMTTrainer
from data import (
    DebussyKernDataset, DebussyDataModule,
    discover_samples, kfold_by_document, batch_collate as kern_batch_collate,
    build_kern_vocab_from_samples,
)
from data_debussy_lmx import DebussyLMXDataset, batch_collate as lmx_batch_collate, build_lmx_vocab_from_samples
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def detect_model_format(model) -> str:
    """
    Detect if the model uses kern or LMX format based on vocabulary.
    
    Returns:
        'kern' or 'lmx'
    """
    # Check for characteristic tokens
    w2i = model.w2i
    
    # LMX has tokens like 'note', 'pitch', 'measure', etc.
    lmx_indicators = ['note', 'measure', 'attributes', 'time', 'key']
    lmx_score = sum(1 for token in lmx_indicators if token in w2i)
    
    # Kern has tokens like 'clefG2', tab characters, etc.
    kern_indicators = ['clefG2', 'clefF4', '<t>', '<b>', '<s>']
    kern_score = sum(1 for token in kern_indicators if token in w2i)
    
    if lmx_score > kern_score:
        return 'lmx'
    else:
        return 'kern'


def is_decomposed_kern(w2i: dict) -> bool:
    """Check if vocabulary uses decomposed (atomic) Kern tokens.

    Decomposed Kern has single-letter pitch names (c, d, e ...) as separate
    tokens, which never appear standalone in compound Kern vocabularies.
    """
    pitch_atoms = {'c', 'd', 'e', 'f', 'g', 'a', 'b'}
    return sum(1 for t in pitch_atoms if t in w2i) >= 5


def extract_vocab_from_checkpoint(checkpoint_path):
    """
    Extract vocabulary (w2i/i2w) from a Lightning checkpoint.
    
    The checkpoint may store the vocab in hyper_parameters, or we can
    at least recover the vocab size from the embedding weight shape.
    
    Returns:
        (w2i, i2w, vocab_size) — w2i/i2w may be None if not stored in checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get vocab from hyper_parameters
    if 'hyper_parameters' in checkpoint and 'model' in checkpoint['hyper_parameters']:
        saved_model = checkpoint['hyper_parameters']['model']
        if hasattr(saved_model, 'w2i') and saved_model.w2i:
            w2i = dict(saved_model.w2i)
            i2w = dict(saved_model.i2w)
            return w2i, i2w, len(w2i)
    
    # Fall back to embedding shape
    for key, value in checkpoint['state_dict'].items():
        if key.endswith('decoder.embedding.weight'):
            return None, None, value.shape[0]
    
    return None, None, None


def merge_vocabularies(ckpt_w2i, ckpt_i2w, dataset_w2i):
    """
    Merge checkpoint vocabulary with dataset vocabulary.
    
    Keeps all checkpoint token→index mappings intact (so pretrained
    embeddings align) and appends any new tokens from the dataset.
    
    Returns:
        (merged_w2i, merged_i2w)
    """
    merged_w2i = dict(ckpt_w2i)
    merged_i2w = dict(ckpt_i2w)
    next_idx = max(merged_w2i.values()) + 1
    
    new_tokens = []
    for token in sorted(dataset_w2i.keys()):
        if token not in merged_w2i:
            merged_w2i[token] = next_idx
            merged_i2w[next_idx] = token
            new_tokens.append(token)
            next_idx += 1
    
    if new_tokens:
        logger.info(f"Added {len(new_tokens)} new tokens from dataset: {new_tokens}")
    else:
        logger.info("Dataset vocabulary is a subset of checkpoint vocabulary — no new tokens")
    
    return merged_w2i, merged_i2w


def load_model_from_checkpoint_or_pretrained(model_path, max_seq_len=1512, w2i=None, i2w=None):
    """
    Load SMT model either from a local Lightning checkpoint or from HuggingFace Hub.
    
    Args:
        model_path: Either a HuggingFace model name (e.g., 'PRAIG/smt-grandstaff')
                    or a local path to a .ckpt file
        max_seq_len: Maximum sequence length to set on the model
        w2i: Optional word-to-index mapping. When provided, the model is constructed
             with this vocabulary. Must be compatible with checkpoint shapes
             (same size, or a superset with new tokens appended at the end).
        i2w: Optional index-to-word mapping
    
    Returns:
        Loaded SMTModelForCausalLM instance
    """
    SMTConfig.register_for_auto_class()
    SMTModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    # Check if it's a local checkpoint file
    if Path(model_path).exists() and model_path.endswith('.ckpt'):
        logger.info(f"Loading model from local checkpoint: {model_path}")
        
        # Load the Lightning checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model state_dict (remove 'model.' prefix from keys)
        model_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                model_state[key[6:]] = value  # Remove 'model.' prefix
        
        # Get model config from checkpoint hyper_parameters or create default
        if 'hyper_parameters' in checkpoint and 'model' in checkpoint['hyper_parameters']:
            # Model was saved with its config
            saved_model = checkpoint['hyper_parameters']['model']
            model = saved_model
            logger.info(f"Loaded model config from checkpoint")
        else:
            # Reconstruct model from state_dict
            logger.info("Reconstructing model from state_dict...")
            
            # Get architecture params from checkpoint weights
            ckpt_vocab_size = model_state['decoder.embedding.weight'].shape[0]
            d_model = model_state['decoder.embedding.weight'].shape[1]
            
            # Load base config from HF for architecture defaults
            base_config = SMTConfig.from_pretrained(
                'PRAIG/smt-grandstaff', trust_remote_code=True
            )
            base_config.d_model = d_model
            base_config.dim_ff = d_model  # typically same as d_model
            
            # Build model with checkpoint vocab size first so all weights load
            if w2i is not None and i2w is not None:
                target_vocab_size = len(w2i)
            else:
                target_vocab_size = ckpt_vocab_size
            
            # Construct model matching checkpoint embedding size to load all layers
            base_config.out_categories = ckpt_vocab_size
            if w2i is not None:
                base_config.w2i = w2i
                base_config.i2w = i2w
            
            model = SMTModelForCausalLM(base_config)
            
            # Load ALL weights from checkpoint (shapes now match)
            model.load_state_dict(model_state, strict=True)
            logger.info(f"Loaded all {len(model_state)} layers from checkpoint (vocab_size={ckpt_vocab_size})")
            
            # If target vocab is larger (new tokens), extend embedding/projection layers
            if target_vocab_size > ckpt_vocab_size:
                logger.info(f"Extending vocab from {ckpt_vocab_size} → {target_vocab_size} "
                            f"({target_vocab_size - ckpt_vocab_size} new tokens)")
                old_emb = model.decoder.embedding
                old_proj_w = model.decoder.vocab_projection.weight
                old_proj_b = model.decoder.vocab_projection.bias
                
                # New embedding: copy old rows, random-init new ones
                new_emb = torch.nn.Embedding(target_vocab_size, d_model, padding_idx=0)
                new_emb.weight.data[:ckpt_vocab_size] = old_emb.weight.data
                model.decoder.embedding = new_emb
                
                # New projection: copy old rows, random-init new ones
                new_proj = torch.nn.Linear(d_model, target_vocab_size)
                new_proj.weight.data[:ckpt_vocab_size] = old_proj_w.data
                new_proj.bias.data[:ckpt_vocab_size] = old_proj_b.data
                model.decoder.vocab_projection = new_proj
                
                model.config.out_categories = target_vocab_size
            
            # Update w2i/i2w on model config
            if w2i is not None:
                model.config.w2i = w2i
                model.config.i2w = i2w
                logger.info(f"Using vocabulary with {len(w2i)} tokens")
        
    else:
        # Load from HuggingFace Hub
        logger.info(f"Loading pretrained model from HuggingFace: {model_path}")
        model = SMTModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # If an extended vocabulary is provided, resize embedding/projection layers
        if w2i is not None and i2w is not None:
            orig_vocab_size = model.config.out_categories
            target_vocab_size = len(w2i)
            if target_vocab_size > orig_vocab_size:
                d_model = model.config.d_model
                logger.info(f"Extending HF model vocab from {orig_vocab_size} → {target_vocab_size} "
                            f"({target_vocab_size - orig_vocab_size} new tokens)")

                old_emb = model.decoder.embedding
                old_proj_w = model.decoder.vocab_projection.weight
                old_proj_b = model.decoder.vocab_projection.bias

                # New embedding: copy pretrained rows, random-init new ones
                new_emb = torch.nn.Embedding(target_vocab_size, d_model, padding_idx=0)
                new_emb.weight.data[:orig_vocab_size] = old_emb.weight.data
                model.decoder.embedding = new_emb

                # New projection: copy pretrained rows, random-init new ones
                new_proj = torch.nn.Linear(d_model, target_vocab_size)
                new_proj.weight.data[:orig_vocab_size] = old_proj_w.data
                new_proj.bias.data[:orig_vocab_size] = old_proj_b.data
                model.decoder.vocab_projection = new_proj

                model.config.out_categories = target_vocab_size

            # Update vocab mappings on model
            model.config.w2i = w2i
            model.config.i2w = i2w
            model.w2i = w2i
            model.i2w = i2w

    # Update max sequence length if needed
    if max_seq_len > model.config.maxlen:
        logger.info(f"Extending model maxlen from {model.config.maxlen} to {max_seq_len}")
        model.config.maxlen = max_seq_len
        model.maxlen = max_seq_len
        model._reinitialize_positional_encodings()
    
    return model


def run_one_fold(
    fold_idx: int,
    train_samples, val_samples, test_samples,
    dataset_w2i, dataset_i2w,
    cfg: dict, n_folds: int,
    run_label: str = None,
) -> dict:
    """Train + test a single fold. Shared by the k-fold CV driver (main()) and the size-ablation
    driver (run_size_ablation.py) -- run_label lets callers distinguish checkpoint/wandb naming
    for the same fold trained at different subset sizes, e.g. 'fold_1_size20'.

    Returns a dict with 'test_results' (metrics dict), 'best_model_path', 'best_model_score'.
    """
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('training', {})
    label = run_label or f'fold_{fold_idx + 1}'

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RUN: {label} ({fold_idx + 1}/{n_folds})")
    logger.info(f"{'=' * 60}")

    # Load a fresh model for each fold (either from checkpoint or HuggingFace)
    model_name = cfg.get('model_name', 'PRAIG/smt-grandstaff')
    local_checkpoint = cfg.get('local_checkpoint', None)

    # Prefer local checkpoint if specified
    model_path = local_checkpoint if local_checkpoint else model_name
    max_seq_len = data_cfg.get('max_seq_len', 1512)

    # Load model using the helper function
    model = load_model_from_checkpoint_or_pretrained(
        model_path, max_seq_len, w2i=dataset_w2i, i2w=dataset_i2w
    )

    w2i = model.w2i
    i2w = {}
    for k, v in model.i2w.items():
        try:
            i2w[int(k)] = v
        except (ValueError, TypeError):
            i2w[k] = v
    model.i2w = i2w

    # ---- Detect format and choose appropriate dataset class ----
    model_format = detect_model_format(model)
    decompose = False
    if model_format == 'kern':
        decompose = is_decomposed_kern(w2i)
    logger.info(f"Detected model format: {model_format.upper()}"
                f"{' (decomposed)' if decompose else ''}")

    if model_format == 'lmx':
        logger.info("Using LMX data pipeline (MusicXML → Linearized MusicXML)")
        DatasetClass = DebussyLMXDataset
        collate_fn = lmx_batch_collate
    else:
        if decompose:
            logger.info("Using decomposed Kern data pipeline (MusicXML → bekrn atoms)")
        else:
            logger.info("Using Kern data pipeline (MusicXML → Kern)")
        DatasetClass = DebussyKernDataset
        collate_fn = kern_batch_collate

    # Build datasets for this fold
    common_args = dict(
        w2i=w2i, i2w=i2w,
        target_height=data_cfg.get('target_height', 256),
        max_width=data_cfg.get('max_width', 3056),
        max_seq_len=max_seq_len,
        binarize=data_cfg.get('binarize', True),
        block_size=data_cfg.get('block_size', 35),
        C=data_cfg.get('C', 10),
    )
    if model_format == 'kern':
        common_args['decompose'] = decompose

    train_ds = DatasetClass(
        samples=train_samples,
        teacher_forcing_error_rate=data_cfg.get('teacher_forcing_error_rate', 0.2),
        is_train=True, **common_args,
    )
    val_ds = DatasetClass(
        samples=val_samples,
        teacher_forcing_error_rate=0.0, is_train=False, **common_args,
    )
    test_ds = DatasetClass(
        samples=test_samples,
        teacher_forcing_error_rate=0.0, is_train=False, **common_args,
    )

    num_workers = data_cfg.get('num_workers', 4)
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.get('batch_size', 4),
        shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Trainer
    lr = float(train_cfg.get('lr', 5e-5))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    label_smoothing = float(train_cfg.get('label_smoothing', 0.1))
    freeze_encoder_epochs = int(train_cfg.get('freeze_encoder_epochs', 0))
    warmup_steps = int(train_cfg.get('warmup_steps', 0))
    compute_tedn = bool(cfg.get('compute_tedn', False))

    smt_trainer = SMTTrainer(model=model, lr=lr, weight_decay=weight_decay,
                             label_smoothing=label_smoothing,
                             freeze_encoder_epochs=freeze_encoder_epochs,
                             warmup_steps=warmup_steps, compute_tedn=compute_tedn)

    checkpoint_dir = f"{train_cfg.get('checkpoint_dir', 'checkpoints_kfold')}/{label}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'smt-{label}-{{epoch:03d}}-{{val_SER:.2f}}',
        monitor='val_SER', mode='min', save_top_k=1, save_last=True, verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_SER', patience=int(train_cfg.get('patience', 10)),
        min_delta=float(train_cfg.get('min_delta', 0.5)),
        mode='min', verbose=True,
    )

    wandb_project = train_cfg.get('wandb_project', 'smt-lmx')
    wandb_run_name = train_cfg.get('wandb_run_name', 'kfold')
    wandb_logger = WandbLogger(
        project=wandb_project, name=f'{wandb_run_name}-{label}', log_model=False,
    )

    trainer = L.Trainer(
        max_epochs=int(train_cfg.get('max_epochs', 500)),
        check_val_every_n_epoch=int(train_cfg.get('val_every_n_epoch', 5)),
        precision=train_cfg.get('precision', '16-mixed'),
        gradient_clip_val=float(train_cfg.get('gradient_clip_val', 1.0)),
        accumulate_grad_batches=int(train_cfg.get('accumulate_grad_batches', 1)),
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=1,
    )

    # Train
    trainer.fit(smt_trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint and test
    best_path = checkpoint_cb.best_model_path
    if best_path:
        logger.info(f"{label} best: {best_path} (val_SER={checkpoint_cb.best_model_score:.2f})")
        smt_trainer = SMTTrainer.load_from_checkpoint(
            best_path, model=model, lr=lr, weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            freeze_encoder_epochs=freeze_encoder_epochs,
            warmup_steps=warmup_steps, compute_tedn=compute_tedn,
        )

    test_results = trainer.test(smt_trainer, dataloaders=test_loader)

    # Finish W&B run for this fold
    import wandb
    wandb.finish()

    best_score = checkpoint_cb.best_model_score
    return {
        'test_results': test_results[0] if test_results else {},
        'best_model_path': best_path,
        'best_model_score': float(best_score) if best_score is not None else None,
    }


def build_folds_and_vocab(cfg: dict, n_folds: int, extend_vocab: bool):
    """Discover samples, build document-level k-fold splits, and build/merge the training
    vocabulary -- exactly the setup `main()` needs once per run. Factored out so
    `run_size_ablation.py` can build the *identical* folds/vocab (same dataset_dir/seed/
    val_ratio/checkpoint) without duplicating this ~90-line block.

    Returns (folds, dataset_w2i, dataset_i2w).
    """
    data_cfg = cfg.get('data', {})

    # ---- Discover samples once ----
    flat_structure = data_cfg.get('flat_structure', False)
    all_samples = discover_samples(
        data_cfg['dataset_dir'],
        exclude_documents=data_cfg.get('exclude_documents', None),
        flat=flat_structure,
    )
    val_ratio = float(data_cfg.get('val_ratio', 0.15))
    folds = kfold_by_document(all_samples, n_folds=n_folds, seed=data_cfg.get('seed', 42),
                              val_ratio=val_ratio, flat=flat_structure)

    logger.info(f"extend_vocab: {extend_vocab}")

    # ---- Build vocabulary: extract from checkpoint and merge with dataset tokens ----
    local_checkpoint = cfg.get('local_checkpoint', None)
    if local_checkpoint and Path(local_checkpoint).exists():
        # Extract vocab from checkpoint (preserves token→index mapping)
        ckpt_w2i, ckpt_i2w, ckpt_vocab_size = extract_vocab_from_checkpoint(local_checkpoint)
        logger.info(f"Checkpoint vocab size: {ckpt_vocab_size}")
        
        # Build dataset vocab to check for new tokens
        logger.info("Building LMX vocabulary from target dataset...")
        dataset_w2i, dataset_i2w = build_lmx_vocab_from_samples(all_samples)
        
        if ckpt_w2i is not None:
            if extend_vocab:
                # Merge: keep checkpoint indices, append any new dataset tokens
                dataset_w2i, dataset_i2w = merge_vocabularies(ckpt_w2i, ckpt_i2w, dataset_w2i)
            else:
                # Use checkpoint vocab only; warn about dropped tokens
                oov_tokens = sorted(t for t in dataset_w2i if t not in ckpt_w2i)
                if oov_tokens:
                    logger.warning(f"extend_vocab=False: {len(oov_tokens)} dataset tokens NOT in "
                                   f"checkpoint vocab will be mapped to <pad>: {oov_tokens}")
                else:
                    logger.info("Dataset vocabulary is a subset of checkpoint vocabulary — no OOV tokens")
                dataset_w2i, dataset_i2w = dict(ckpt_w2i), dict(ckpt_i2w)
        else:
            # Checkpoint didn't store vocab — rebuild from pretraining data files
            pretrain_root = cfg.get('pretrain_dataset_root', None)
            if pretrain_root and Path(pretrain_root).exists():
                logger.info(f"Rebuilding pretraining vocabulary from {pretrain_root}")
                from data_lmx import build_lmx_vocabulary_from_files
                pretrain_splits = []
                for split_name in ['train', 'dev', 'test']:
                    split_file = Path(pretrain_root) / f'samples.{split_name}.txt'
                    if split_file.exists():
                        with open(split_file) as f:
                            pretrain_splits.append([line.strip() for line in f if line.strip()])
                ckpt_w2i, ckpt_i2w = build_lmx_vocabulary_from_files(
                    [(pretrain_root, split) for split in pretrain_splits]
                )
                if len(ckpt_w2i) != ckpt_vocab_size:
                    logger.warning(f"Rebuilt vocab ({len(ckpt_w2i)}) != checkpoint embedding ({ckpt_vocab_size}). "
                                   f"Verify pretrain_dataset_root matches the data used to train the checkpoint.")
                if extend_vocab:
                    dataset_w2i, dataset_i2w = merge_vocabularies(ckpt_w2i, ckpt_i2w, dataset_w2i)
                else:
                    oov_tokens = sorted(t for t in dataset_w2i if t not in ckpt_w2i)
                    if oov_tokens:
                        logger.warning(f"extend_vocab=False: {len(oov_tokens)} dataset tokens NOT in "
                                       f"checkpoint vocab will be mapped to <pad>: {oov_tokens}")
                    dataset_w2i, dataset_i2w = dict(ckpt_w2i), dict(ckpt_i2w)
            else:
                logger.warning("No checkpoint vocab or pretrain_dataset_root found. "
                               "Using dataset-only vocab (embedding layers will be reinitialized).")
    else:
        # HuggingFace model path — optionally extend vocab with dataset tokens
        model_name = cfg.get('model_name', 'PRAIG/smt-grandstaff')
        if extend_vocab:
            max_seq_len = data_cfg.get('max_seq_len', 1512)
            logger.info(f"Loading HF model '{model_name}' to extract vocab for extension...")
            tmp_model = load_model_from_checkpoint_or_pretrained(model_name, max_seq_len)
            model_w2i = dict(tmp_model.w2i)
            model_i2w = {}
            for k, v in tmp_model.i2w.items():
                try:
                    model_i2w[int(k)] = v
                except (ValueError, TypeError):
                    model_i2w[k] = v

            model_format = detect_model_format(tmp_model)
            decompose = is_decomposed_kern(model_w2i) if model_format == 'kern' else False
            del tmp_model  # free memory
            logger.info(f"HF model vocab: {len(model_w2i)} tokens, format: {model_format}"
                        f"{' (decomposed)' if decompose else ''}")

            if model_format == 'kern':
                logger.info("Building Kern vocabulary from dataset samples (with articulations)...")
                ds_w2i, _ = build_kern_vocab_from_samples(
                    all_samples, w2i=model_w2i, decompose=decompose)
            else:
                logger.info("Building LMX vocabulary from dataset samples...")
                ds_w2i, _ = build_lmx_vocab_from_samples(all_samples)

            # Merge: keep HF model indices, append new dataset tokens
            dataset_w2i, dataset_i2w = merge_vocabularies(model_w2i, model_i2w, ds_w2i)
            logger.info(f"Extended vocabulary: {len(dataset_w2i)} tokens "
                        f"(was {len(model_w2i)}, added {len(dataset_w2i) - len(model_w2i)})")
        else:
            dataset_w2i, dataset_i2w = None, None

    return folds, dataset_w2i, dataset_i2w


def main():
    parser = argparse.ArgumentParser(description='SMT K-Fold CV Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--n_folds', type=int, default=None, help='Number of folds (overrides config)')
    parser.add_argument('--extend-vocab', action='store_true', default=None,
                        help='Extend model vocab with new tokens from dataset (overrides config)')
    parser.add_argument('--no-extend-vocab', dest='extend_vocab', action='store_false',
                        help='Use only the checkpoint vocab; OOV tokens in dataset are dropped')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get('training', {})
    n_folds = args.n_folds or cfg.get('n_folds', 5)

    logger.info(f"Config: {args.config} | {n_folds}-fold CV")

    # Safety check: warn if loading from checkpoint in the same dir where we'll save
    local_checkpoint = cfg.get('local_checkpoint', None)
    checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints_kfold')
    if local_checkpoint and Path(local_checkpoint).parent.name == Path(checkpoint_dir).name:
        logger.warning("=" * 70)
        logger.warning("WARNING: Loading checkpoint from a directory with similar name to where")
        logger.warning(f"new checkpoints will be saved ({checkpoint_dir}). Verify this is intentional.")
        logger.warning("=" * 70)

    # ---- Determine extend_vocab flag (CLI > config > default True) ----
    if args.extend_vocab is not None:
        extend_vocab = args.extend_vocab
    else:
        extend_vocab = cfg.get('extend_vocab', True)

    folds, dataset_w2i, dataset_i2w = build_folds_and_vocab(cfg, n_folds, extend_vocab)

    # ---- Run each fold ----
    fold_results = []
    fold_manifest = []
    for fold_idx, (train_samples, val_samples, test_samples) in enumerate(folds):
        result = run_one_fold(
            fold_idx, train_samples, val_samples, test_samples,
            dataset_w2i, dataset_i2w, cfg, n_folds,
        )
        fold_results.append(result['test_results'])
        fold_manifest.append({
            'fold': fold_idx + 1,
            'best_model_path': result['best_model_path'],
            'best_model_score': result['best_model_score'],
            **result['test_results'],
        })

    manifest_path = Path(train_cfg.get('checkpoint_dir', 'checkpoints_kfold')) / 'fold_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        import json
        json.dump(fold_manifest, f, indent=2)
    logger.info(f"Wrote fold manifest (incl. per-fold test metrics) to {manifest_path}")

    # ---- Summary ----
    logger.info(f"\n{'=' * 60}")
    logger.info(f"K-FOLD CROSS-VALIDATION SUMMARY ({n_folds} folds)")
    logger.info(f"{'=' * 60}")

    metrics = ['test_SER', 'test_CER', 'test_LER', 'test_TEDn']
    for m in metrics:
        values = [r.get(m, float('nan')) for r in fold_results]
        valid = [v for v in values if v == v]  # filter NaN
        if valid:
            mean = sum(valid) / len(valid)
            std = (sum((v - mean) ** 2 for v in valid) / len(valid)) ** 0.5
            logger.info(f"{m}: {mean:.4f} ± {std:.4f}  (per fold: {[f'{v:.4f}' for v in valid]})")
        else:
            logger.info(f"{m}: no results")

    logger.info("Done.")


if __name__ == '__main__':
    main()
