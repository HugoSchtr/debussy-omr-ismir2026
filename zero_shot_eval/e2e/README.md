# E2E baselines: FCN / CRNN / CNNT (reproduces Table 4 + Table 5 e2e rows)

Zero-shot evaluation of the three convolutional encoder-decoder architectures
(FCN, CRNN, CNNT) introduced by Ríos-Vila et al. for system-level OMR, on both
their own training-distribution test sets (Table 4) and the Debussy dataset
(Table 5), with no fine-tuning.

## Attribution

This directory is adapted from Antonio Ríos-Vila's official e2e OMR codebase
(`e2e-pianoform-system-based`, MIT-licensed, Copyright (c) 2023 Antonio
Ríos Vila -- see `LICENSE`), accompanying:

> Ríos-Vila, A., Rizo, D., Iñesta, J.M., Calvo-Zaragoza, J. *End-to-end optical
> music recognition for pianoform sheet music*. IJDAR (2023).
> https://doi.org/10.1007/s10032-023-00432-z

`main.py`, `data.py`, `ModelManager.py`, `eval_functions.py`, `utils.py`,
`seed_utils.py`, and `model/E2E_Score_Unfolding.py` are copied from that
repository as-is (these are exactly the files `main.py` transitively imports;
nothing else was needed for training/inference to work standalone).
`config/GrandStaff/` and `config/Camera_GrandStaff/` (`.gin` files for
FCN/CRNN/CNNT) are likewise copied from the original repository. `eval_e2e.py`
is **new** code written for this release (see below).

**Not copied**: the dozens of one-off `debug_*.py`, `test_*.py`, `inspect_*.py`,
`check_*.py` diagnostic scripts, wandb logs, and anything under the original
repo's `logs/`, `out/`, `weights/`, `vocab/*.json` (these are generated
artifacts/checkpoints, not code). `config/fp-grandstaff/`,
`config/OLIMPIC_*/` are also omitted -- Table 4/5 only use the GrandStaff and
Camera-GrandStaff configs.

## New: `eval_e2e.py`

A clean, standalone evaluation script extracted from the original exploratory
notebook `e2e_pianoform_eval.ipynb` (mirrors `../zero_shot_eval/zeus/eval_zeus.py`'s
treatment: same reasoning, same scope). Pipeline: load a raw `nn.Module` from a
Lightning checkpoint (no `lightning`/`gin` dependency needed at inference
time) -> preprocess an image (binarize -> grayscale -> resize to fixed height ->
rotate 90 degrees clockwise, matching the training pipeline in `data.py`) ->
CTC greedy decode to bekrn tokens -> kern -> MusicXML (music21, with a 4-level
sanitization fallback matching `../zero_shot_eval/smt/run_smt_evaluation_optimized.py`'s
protocol) -> MusicXML -> LMX (Zeus Linearizer) -> SER/CER/LER and "filtered"
fSER/fCER/fLER (unfair-token-removed) metrics, plus optional TEDn. Dropped:
the notebook's exploratory cells (matplotlib visualizations, per-sample
distribution plots, per-document breakdown tables, prediction-vs-GT image
galleries) -- only the inference -> metrics pipeline needed to reproduce the
final numbers is kept.

```bash
pip install -r requirements.txt

python eval_e2e.py \
    --checkpoint /path/to/CNNT-v5-camera.ckpt --arch CNNT \
    --dataset /path/to/paired_system_level \
    --output ./results/CNNT-v5-camera \
    --binarize adaptive --compute-tedn
```

## Binarization: investigated, found likely low-impact, but NOT corrected the way Zeus was

Unlike the SMT and Zeus zero-shot pipelines -- where a genuine preprocessing
mismatch was found and corrected for two of three Zeus checkpoints (see
`../zero_shot_eval/zeus/README.md`) -- the e2e evaluation was **investigated
but not re-run with the same rigor**. Here is the actual, verified situation:

- The evaluation notebook (`e2e_pianoform_eval.ipynb`) applied
  `BINARIZATION_METHODS = ['adaptive']` **uniformly to every model, including
  the two "camera" variants** (`CNNT-v5-camera`, `CRNN-v7-camera`, trained on
  Camera-GrandStaff's grayscale/photographic "\_distorted" images -- the same
  training-distribution character as Zeus's Camera-GrandStaff-LMX checkpoint,
  which *was* corrected). This is structurally the same potential mismatch as
  Zeus's original (uncorrected) submission.
- The run that produced the paper's Table 5 CNNT numbers was confirmed
  directly: `run_20260417_172922/comparison_summary.csv`'s
  `CNNT-v5-camera` + `adaptive` row gives Global SER = 92.45%, CER = 88.02% --
  matching Table 5's "e2e CNNT / Sys / Cam. GrandStaff" row exactly.
- However, an earlier exploratory sweep across binarization methods
  (`run_20260309_130012/comparison_summary.csv`) found that for
  `CRNN-v7-camera` (also camera-trained), SER varied by **under 1 point**
  across `none` / `adaptive` / `otsu` (96.01 / 96.06 / 96.76). This is a much
  smaller effect than what was found for Zeus, where the equivalent mismatch
  caused a >10-point SER difference. It suggests binarization choice is
  **not a practically significant confound for these convolutional
  architectures**, unlike for Zeus.
- This was **not** re-run to the same rigor as the Zeus correction (no
  systematic per-model per-binarization sweep across the full corpus with
  fully controlled sample-exclusion settings was completed in the time
  available), and the exact run/config behind the final published CRNN/FCN
  numbers could not be pinned down with full certainty among the many
  timestamped exploratory runs. Treat Table 5's FCN/CRNN/CNNT rows with this
  caveat in mind: investigated and found likely low-impact, but not
  conclusively resolved the way the Zeus rows were.

`eval_e2e.py`'s `--binarize` defaults to `adaptive` (matching the run behind
the published CNNT numbers); pass a different value to explore sensitivity
for other checkpoints yourself.

## Sample-set note

The evaluation notebook's run used `EXCLUDE_DOCUMENTS = []` -- i.e. the full
561-sample corpus, with **no** out-of-domain document exclusion. This differs
from the SMT (`../zero_shot_eval/smt/config.yaml`) and Zeus zero-shot
pipelines, both of which exclude 6 documents by default. This means the e2e
Table 5 rows are not drawn from exactly the same sample set as the other
zero-shot rows in that table. This is a small, factual methodological
inconsistency worth being aware of when comparing rows across Table 5 -- it is
documented here, not corrected.

## Data & checkpoints (not included in this repo)

- **Datasets**: GrandStaff / Camera-GrandStaff, publicly available for
  replication purposes -- see the original repository's README
  (https://sites.google.com/view/multiscore-project/datasets).
- **FCN/CRNN/CNNT checkpoints**: not redistributed here; obtain/retrain per
  the original repository's instructions (`config/GrandStaff/*.gin`,
  `config/Camera_GrandStaff/*.gin` define the exact training configurations
  used, matching what produced the Table 4 benchmark numbers).
- **Debussy dataset**: released under CC-BY-4.0; link to be added upon
  camera-ready publication.
- **olimpic-icdar24** (Mayer et al.): required for the LMX linearizer used by
  `eval_e2e.py`. Not pip-installable -- clone
  https://github.com/ufal/olimpic-icdar24 and set `OLIMPIC_ICDAR24_DIR` (env
  var), or place it as a sibling `../olimpic-icdar24` directory next to this
  `code/` checkout (the default).

## License

`LICENSE` in this directory is the original repository's MIT license
(Copyright (c) 2023 Antonio Ríos Vila), covering `main.py`, `data.py`,
`ModelManager.py`, `eval_functions.py`, `utils.py`, `seed_utils.py`,
`model/E2E_Score_Unfolding.py`, and the `config/*.gin` files. `eval_e2e.py` is
new code written for this release, covered by the top-level repo `LICENSE`
(also MIT).

## Environment

PyTorch (see `requirements.txt`). `main.py` (training) additionally needs
`lightning`, `gin-config`, and `wandb` (see the commented-out block in
`requirements.txt` and `requirements_cluster.txt` for the original pinned
training environment, kept for reference -- note the comment there about
PyTorch 2.0.x being required for older GPUs).
