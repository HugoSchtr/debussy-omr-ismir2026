# Fine-tuning (reproduces Table 6)

5-fold document-level cross-validation fine-tuning of the Sheet Music Transformer
(SMT) on the Debussy dataset, in two tracks that share the same training driver
and hyperparameters and differ only in starting checkpoint and output vocabulary:

- **`lmx_track/`** -- SMT trained from scratch on GrandStaff LMX (Mayer et al.
  LMX vocabulary), then fine-tuned on Debussy.
- **`kern_track/`** -- the publicly released `PRAIG/smt-grandstaff` checkpoint
  (**kern vocabulary), fine-tuned on Debussy.

## Layout

```
fine_tuning/
├── train_kfold.py          # main entrypoint (both tracks)
├── data.py                 # Kern dataset/dataloader + k-fold-by-document splitting
├── data_debussy_lmx.py      # LMX dataset/dataloader
├── musicxml_to_kern.py      # standalone MusicXML -> **kern converter (verovio + music21 fallback)
├── eval_functions.py        # SER/CER/LER metric implementation (compute_poliphony_metrics)
├── tedn_eval.py             # TEDn (tree edit distance) metric, requires olimpic-icdar24
├── smt_trainer.py            # PyTorch Lightning wrapper around the SMT model
├── smt_model/                # vendored SMT model code (config + modeling), from PRAIG/smt-grandstaff
├── fold_splits.json          # the EXACT document->fold assignment used to produce Table 6
├── lmx_track/config_main_kfold.yaml
├── kern_track/config_main_kfold.yaml
└── requirements.txt
```

This is the complete essential set: `train_kfold.py` only imports `smt_trainer`,
`data`, and `data_debussy_lmx` (which in turn need `musicxml_to_kern.py`,
`eval_functions.py`, `tedn_eval.py`, and `smt_model/`). The original source
repository (`smt-finetune/`) also contains ablation/diagnostic scripts
(`debussy_experiment*/scripts/*.py`: `token_taxonomy.py`, `run_size_ablation.py`,
`error_analysis*.py`, etc.), an unused from-scratch training path
(`train_from_scratch.py`, `smt_trainer_scratch.py`), and a standalone TEDn
evaluation script (`eval_lmx_v2_tedn.py`) -- none of these are imported by
`train_kfold.py` and are deliberately excluded here as internal research
tooling, not part of reproducing Table 6.

## Running

```bash
pip install -r requirements.txt

# LMX track
python train_kfold.py --config lmx_track/config_main_kfold.yaml

# Kern track
python train_kfold.py --config kern_track/config_main_kfold.yaml
```

Both configs use `n_folds: 5`, `data.seed: 42`, `data.val_ratio: 0.15`, and the
same `data.dataset_dir` -- so both tracks are trained/evaluated on **identical**
document-level fold assignments (see below). Each run writes a
`checkpoint_dir/fold_manifest.json` with per-fold test metrics
(`test_SER`, `test_CER`, `test_LER`, `test_TEDn`) and best-checkpoint paths.

## The 5-fold CV split (`fold_splits.json`)

Reviewers asked for the exact CV fold splits used to produce Table 6.
`fold_splits.json` in this directory contains the **real** document-to-fold
assignment used to fine-tune both tracks -- not a re-derivation, but the
literal output of `data.discover_samples()` + `data.kfold_by_document()` run
against the actual Debussy `paired_system_level` dataset (37 documents, 561
system-level samples), with the same `seed=42`, `n_folds=5`, `val_ratio=0.15`
both configs use.

Each entry in `fold_splits.json["folds"]` lists, for one fold, the
`test_documents` (held out entirely -- these form the test set), the
`val_documents` (15% of the remaining documents), and the `train_documents`
(everything else). The test-document lists across all 5 folds are disjoint and
together cover all 37 documents (test sample counts: 109 + 94 + 119 + 85 + 154 =
561), confirming this is a genuine, leak-free 5-fold partition at the document
level.

**Exact method** (see `data.py:kfold_by_document()`): document IDs are sorted
lexicographically, shuffled with `random.Random(seed=42).shuffle(...)`, then
split into 5 contiguous chunks of (nearly) equal size. For fold *i*, chunk *i*
is the test set; among the remaining documents (in the same shuffled order),
the first `floor(len(remaining) * 0.15)` become the validation set and the
rest are training documents. Because this is a pure function of
`(sorted document IDs, seed, n_folds, val_ratio)`, re-running
`kfold_by_document()` against a copy of the dataset with the same 37 documents
reproduces `fold_splits.json` exactly -- no need to trust a cached manifest.

## Data & checkpoints (not included in this repo)

- **Debussy dataset**: released under CC-BY-4.0 on HuggingFace (full-page:
  https://huggingface.co/datasets/HugoSchtr/debussy-omr-fullpage-lvl ,
  system-level: https://huggingface.co/datasets/HugoSchtr/debussy-omr-system-lvl ).
  Expected layout: `dataset_dir/doc_id/page/sample.{jpg,musicxml}`.
- **LMX track base checkpoint**: an SMT trained from scratch on GrandStaff LMX
  (val SER 0.90% per the paper). Not distributed here -- train your own using
  the OLiMPiC/Zeus LMX pipeline (https://github.com/ufal/olimpic-icdar24), or
  substitute a different `local_checkpoint`/`model_name`.
- **Kern track base checkpoint**: `PRAIG/smt-grandstaff` on HuggingFace Hub,
  loaded automatically by `model_name` in the config.
- **olimpic-icdar24** (Mayer et al.): required for the LMX linearizer
  (`data_debussy_lmx.py`) and for `compute_tedn: true` (`tedn_eval.py`). Not
  pip-installable -- clone https://github.com/ufal/olimpic-icdar24 and set
  `OLIMPIC_ICDAR24_DIR` (env var) to point at it, or place it as a sibling
  `../olimpic-icdar24` directory next to this `code/` checkout (the default).
  Set `compute_tedn: false` in a config to skip this dependency for the SER/CER/LER
  metrics only.

## Environment

PyTorch + PyTorch Lightning (see `requirements.txt`). This is a separate
environment from `zero_shot_eval/zeus/` (TensorFlow) but compatible with
`zero_shot_eval/smt/` (also PyTorch).
