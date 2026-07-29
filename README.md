# The Debussy Dataset -- Replication Code

📄 **Paper**: "The Debussy Dataset: Evaluating Optical Music Recognition on Modern Staff Notation Manuscripts" (ISMIR 2026) — *proceedings link to be added*
🎼 **Dataset** (CC-BY-4.0): [full-page](https://huggingface.co/datasets/HugoSchtr/debussy-omr-fullpage-lvl) · [system-level](https://huggingface.co/datasets/HugoSchtr/debussy-omr-system-lvl)

Replication code for **"The Debussy Dataset: Evaluating Optical Music
Recognition on Modern Staff Notation Manuscripts"** (ISMIR 2026). This
repository contains the code needed to reproduce Tables 4, 5, and 6 of the
paper: SOTA OMR benchmark replication, zero-shot evaluation of those models on
the Debussy dataset, and 5-fold cross-validation fine-tuning results. It is a
curated, cleaned subset of the research codebases used to produce the paper's
results -- not a raw dump of the original (much messier) source repositories.
See each subfolder's own README for details specific to that part of the
pipeline.

## Reproducing each table

| Paper table | What it reports | Reproduce with |
|---|---|---|
| **Table 4** | SOTA OMR models on their own training-distribution test sets (GrandStaff, Camera-GrandStaff, OLiMPiC) | `zero_shot_eval/smt/run_smt_evaluation_optimized.py --config config.yaml` (system SMT); `zero_shot_eval/smt/run_smt_evaluation_fullpage.py --config config_fullpage.yaml` (full-page SMT); `zero_shot_eval/zeus/eval_zeus.py --model <checkpoint>` (Zeus, 3 checkpoints); `zero_shot_eval/e2e/eval_e2e.py --checkpoint <ckpt> --arch {FCN,CRNN,CNNT}` |
| **Table 5** | Zero-shot evaluation of the same models on the Debussy dataset (no fine-tuning) | Same scripts as Table 4, pointed at the Debussy `paired_system_level` dataset instead of each model's own benchmark. See `zero_shot_eval/zeus/README.md` for the corrected per-checkpoint binarization policy used for these rows. |
| **Table 6** | Per-fold results, 5-fold document-level CV fine-tuning: SMT-from-scratch on LMX vocabulary (LMX track) vs. pretrained PRAIG SMT **kern checkpoint fine-tuned on **kern vocabulary (Kern track) | `fine_tuning/train_kfold.py --config lmx_track/config_main_kfold.yaml` and `fine_tuning/train_kfold.py --config kern_track/config_main_kfold.yaml`. The exact document-to-fold assignment used to produce the published numbers is in `fine_tuning/fold_splits.json`. |

## Repository layout

```
code/
├── fine_tuning/            # Table 6: SMT fine-tuning, 2 tracks (LMX / Kern), shared k-fold driver
│   ├── lmx_track/
│   ├── kern_track/
│   └── bekern_track/       # future work (see BEKERN.md): decomposed-kern fine-tuning via kernpy
├── zero_shot_eval/          # Tables 4 + 5: zero-shot evaluation of SOTA models
│   ├── smt/                # SMT (system + full-page), PyTorch
│   ├── zeus/                # Zeus (3 checkpoints), TensorFlow -- includes a corrected preprocessing policy
│   └── e2e/                # FCN/CRNN/CNNT, adapted from Ríos-Vila et al.
└── BEKERN.md               # future work: the decomposed-kern encoding experiment
                            #   (negative result — the shipped recipe does not converge; read it
                            #    before running bekern_track/. Does not affect Tables 4/5/6.)
```

## Environments: PyTorch vs. TensorFlow

This repository spans **two incompatible ML frameworks** and is not meant to
be installed into a single environment:

- **PyTorch environment** -- `fine_tuning/`, `zero_shot_eval/smt/`, and
  `zero_shot_eval/e2e/` (evaluation and training). Each has its own
  `requirements.txt`; they are largely compatible with each other (same
  PyTorch major version) but are kept separate since they come from
  independent source repositories with slightly different pinned versions
  (e.g. `zero_shot_eval/e2e/requirements_cluster.txt` pins `torch<2.1.0` for older
  GPU compatibility -- check that file if you plan to train, not just
  evaluate, the e2e baselines).
- **TensorFlow environment** -- `zero_shot_eval/zeus/` (`tensorflow~=2.12.0`).
  Zeus's own `zeus.py` is invoked as a subprocess by `eval_zeus.py`, so this
  environment never needs to coexist with the PyTorch ones in the same
  Python process -- but you do need `pip install -r zero_shot_eval/zeus/requirements.txt`
  in a separate virtualenv/conda env from the PyTorch code.

Each subfolder's `requirements.txt` is deliberately separate rather than one
monolithic file, matching how these were genuinely separate environments in
the original source repositories.

## Data & Checkpoints

**No model checkpoints, dataset images, or other large binary artifacts are
included in this repository** -- only code and small config/text files. To
run any of the scripts here, obtain the following separately:

| Item | Source |
|---|---|
| `PRAIG/smt-grandstaff`, `PRAIG/smt-fp-grandstaff` | HuggingFace Hub (loaded automatically via `model_name` in the relevant config; `trust_remote_code=True`) |
| Zeus checkpoints (`zeus-grandstaff-lmx-1.0-2024-02-12.model`, `zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`, `zeus-olimpic-1.0-2024-02-12.model`) | GitHub releases: https://github.com/ufal/olimpic-icdar24/releases |
| GrandStaff / Camera-GrandStaff / OLiMPiC datasets | Same GitHub releases page, plus https://grfia.dlsi.ua.es (per the `Makefile` in `olimpic-icdar24/Makefile`); GrandStaff is also linked from https://sites.google.com/view/multiscore-project/datasets |
| FCN/CRNN/CNNT checkpoints | Not redistributed by the original `e2e-pianoform-system-based` repo either -- train them yourself using `zero_shot_eval/e2e/main.py` with the provided `config/GrandStaff/*.gin` / `config/Camera_GrandStaff/*.gin` configs, or obtain from the original repository's own release channel if/when available |
| olimpic-icdar24 (Mayer et al.) -- LMX linearizer + TEDn | https://github.com/ufal/olimpic-icdar24 (not pip-installable; clone and set `OLIMPIC_ICDAR24_DIR`, or place as a sibling `../olimpic-icdar24` next to this repo -- see each subfolder's README) |
| SMT base checkpoint pretrained from scratch on GrandStaff LMX (used to start the `fine_tuning/lmx_track`) | Not distributed here -- train your own using the OLiMPiC/Zeus LMX pipeline, or substitute your own `local_checkpoint` |
| **The Debussy dataset itself** | Released under CC-BY-4.0 on HuggingFace: full-page version https://huggingface.co/datasets/HugoSchtr/debussy-omr-fullpage-lvl , system-level version https://huggingface.co/datasets/HugoSchtr/debussy-omr-system-lvl |

## License

This repository mixes original code (MIT) with third-party code that retains
its own original license:

- **MIT** (this repo's own code): `fine_tuning/`, `zero_shot_eval/smt/`, and
  `zero_shot_eval/zeus/eval_zeus.py` -- see the top-level `LICENSE`.
- **`zero_shot_eval/e2e/`**: MIT, Copyright (c) 2023 Antonio Ríos Vila -- adapted
  from the official `e2e-pianoform-system-based` repository, with
  Debussy-specific evaluation code and configs added. See
  `zero_shot_eval/e2e/LICENSE`.
- **`zero_shot_eval/zeus/`** (`zeus.py`, `ser_metric.py`, `tedn_metric.py`):
  MIT, Copyright 2024 Milan Straka -- Mayer et al.'s / UFAL's original Zeus
  code, copied unmodified. See `zero_shot_eval/zeus/LICENSE.txt`. (The trained
  Zeus model checkpoints themselves are CC BY-SA -- obtain them from the
  GitHub releases page, not from this repo.)

Third-party code retains its original license regardless of its location in
this repository -- see the nested `LICENSE` files for the authoritative text.

## Reproducibility notes

- `fine_tuning/fold_splits.json` documents the exact document-to-fold
  assignment used for Table 6's 5-fold cross-validation (both tracks use
  identical folds), extracted from a real run of the k-fold splitting code
  against the actual dataset -- not a re-derivation or approximation. See
  `fine_tuning/README.md` for the exact splitting method (seed, ordering
  logic) so it can be reproduced from scratch if needed.
- `zero_shot_eval/zeus/README.md` documents a genuine methodological
  correction made during the camera-ready revision: the original submission
  applied one binarization method uniformly to all three Zeus checkpoints,
  which was a mismatch for two of them given their actual training-image
  statistics. This is corrected by default in `eval_zeus.py`, with an
  explicit override available to reproduce the original (uncorrected)
  numbers.
- `zero_shot_eval/e2e/README.md` documents a related but smaller and *not fully
  resolved* preprocessing question for the FCN/CRNN/CNNT baselines -- see
  that README for what was verified and what remains an open caveat.
