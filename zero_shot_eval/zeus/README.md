# Zero-shot Zeus evaluation (reproduces Table 4 Zeus rows + Table 5 Zeus rows)

Zero-shot evaluation of the three Zeus checkpoints (Mayer et al., OLiMPiC/ICDAR24
2024 -- a CRNN trained on LMX vocabulary) on their own training-distribution
test sets (Table 4) and the Debussy dataset (Table 5), with no fine-tuning.

- `zeus.py`, `ser_metric.py`, `tedn_metric.py` -- Zeus's original code (Mayer et
  al. / UFAL), TensorFlow-based, with one minimal compatibility fix in `zeus.py`
  (see "A compatibility fix" below) -- otherwise unmodified.
- `eval_zeus.py` -- **new** clean, standalone evaluation script written for
  this release. Extracts the core logic of the original exploratory
  `zeus_eval.ipynb` notebook (load model + tags, binarize input images, run
  inference via `zeus.py`, compute SER/CER/TEDn) into an argparse-driven CLI.
  Dropped: the notebook's exploratory/visualization cells (matplotlib
  binarization-method previews, interactive per-sample comparisons, vocabulary
  deep-dives) -- only what's needed to reproduce the final per-model
  SER/CER/TEDn numbers is kept.
- `LICENSE.txt` -- Zeus's own license (see "License" below).

## IMPORTANT: a real methodological correction, not just code cleanup

The camera-ready revision corrected a preprocessing mismatch affecting the
Camera-GrandStaff and Synthetic (OLiMPiC) Zeus zero-shot rows in Table 5 --
see `REVISION_PLAN.md` (item T1.3) and the paper's methodology section for
details.

The **original submission** applied `binarization_method: adaptive` uniformly
to all three Zeus checkpoints. Pixel-statistics evidence gathered during the
camera-ready revision (sampled from
`olimpic-icdar24/datasets/grandstaff/beethoven/piano-sonatas/sonata25-1/`,
comparing `original_*.jpg` vs. `*_distorted.jpg`) showed this uniform choice
was a mismatch for two of the three checkpoints:

| Checkpoint | Training images | Eval-time binarization |
|---|---|---|
| `zeus-grandstaff-lmx-1.0-2024-02-12.model` | near-binary (~87% pixels > brightness 200, ~10% mid-gray) | `adaptive` -- **matches original submission, still correct** |
| `zeus-camera-grandstaff-lmx-1.0-2024-02-12.model` | genuinely grayscale/photographic "\_distorted" images (~68% > 200, ~25% mid-gray) | `none` -- **corrected** (was wrongly `adaptive`) |
| `zeus-olimpic-1.0-2024-02-12.model` | synthetic, trained with no binarization step (confirmed via `zeus.py`'s own dataset-description `threshold:` transform convention -- the official synthetic training command in the Zeus README never sets it) | `none` -- **corrected** (was wrongly `adaptive`) |

`eval_zeus.py` implements this corrected policy as its **default**, via a
small lookup dict (`DEFAULT_BINARIZE_BY_MODEL`) keyed by the checkpoint
directory name, with a code comment explaining the pixel-stat evidence. Pass
`--binarize {none,adaptive,otsu,fixed}` explicitly to override this for any
checkpoint and reproduce the **original (uncorrected)** submission numbers for
comparison.

## A compatibility fix (not a methodological change)

`zeus.py`'s `WithAttention` cell originally subclassed `tf.keras.layers.Layer` with a
hand-added `output_size` property standing in for the RNN-cell API. Against TensorFlow 2.18.1
(the version actually installed in the cluster environment used to run this evaluation --
newer than this repo's own pinned `tensorflow~=2.12.0` in `requirements.txt`, not tested
against 2.12.0 specifically), that raises `AttributeError: 'WithAttention' object has no
attribute 'get_initial_state'` the moment the decoder runs, so the original file could not
actually run end-to-end in that environment -- caught by a 2-sample smoke test before relying
on it for real evaluation runs. Fixed by
subclassing `tf.keras.layers.AbstractRNNCell` instead (its built-in `get_initial_state`
covers what the removed `output_size` property was working around). This is a
TensorFlow-version compatibility fix only: same architecture, same weights, same forward
computation -- verified by running both versions against the same checkpoint and confirming
identical SER/CER on the same samples. If you hit the same `AttributeError` against a
different TensorFlow version, this is the fix to look for.

## Running

```bash
pip install -r requirements.txt

python eval_zeus.py \
    --model models/zeus-grandstaff-lmx-1.0-2024-02-12.model \
    --dataset /path/to/paired_system_level \
    --output ./results/zeus_grandstaff_lmx \
    --compute-tedn --tedn-flavor lmx

python eval_zeus.py \
    --model models/zeus-camera-grandstaff-lmx-1.0-2024-02-12.model \
    --dataset /path/to/paired_system_level \
    --output ./results/zeus_camera_grandstaff_lmx
    # --binarize is NOT passed -- uses the corrected default (none)

python eval_zeus.py \
    --model models/zeus-olimpic-1.0-2024-02-12.model \
    --dataset /path/to/paired_system_level \
    --output ./results/zeus_olimpic
```

To reproduce the original (uncorrected) submission's Table 5 numbers for
comparison, add `--binarize adaptive` to every invocation.

Each run writes `results.json` (SER, SERnotuplets, CER, and TEDn if
`--compute-tedn`) plus the raw Zeus prediction file (`*.predicted.lmx`) under
`--output`.

## Data & checkpoints (not included in this repo)

- **Zeus model checkpoints** (`zeus-grandstaff-lmx-1.0-2024-02-12.model`,
  `zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`,
  `zeus-olimpic-1.0-2024-02-12.model`): GitHub releases at
  https://github.com/ufal/olimpic-icdar24/releases. Each `.model` directory
  should contain `weights.h5`, `tags.txt`, `options.json` (and its own
  `README.md`/`LICENSE`).
- **Datasets** (GrandStaff, Camera-GrandStaff, OLiMPiC): same GitHub releases,
  plus https://grfia.dlsi.ua.es (per the `Makefile` in
  `olimpic-icdar24/Makefile`).
- **Debussy dataset**: released under CC-BY-4.0; link to be added upon
  camera-ready publication.
- **olimpic-icdar24** (Mayer et al.): required for the LMX linearizer
  (`Linearizer`) and TEDn (`TEDn_lmx_xml`). Not pip-installable -- clone
  https://github.com/ufal/olimpic-icdar24 and set `OLIMPIC_ICDAR24_DIR` (env
  var), or place it as a sibling `../../olimpic-icdar24` directory next to this
  `code/` checkout (the default `eval_zeus.py` looks for).

## License

`zeus.py`, `ser_metric.py`, and `tedn_metric.py` are Mayer et al.'s / UFAL's
original code (OLiMPiC/Zeus), preserved as-is under their own license --
see `LICENSE.txt` in this directory (MIT, Copyright 2024 Milan Straka). Per
the upstream `olimpic-icdar24` README: source code is MIT-licensed, while the
trained Zeus model checkpoints and datasets are released under CC BY-SA
(obtain them from the GitHub releases page, not from this repo). `eval_zeus.py`
is new code written for this release, covered by the top-level repo `LICENSE`
(MIT).

## Environment

TensorFlow (`tensorflow~=2.12.0`, see `requirements.txt`) -- this is a
**separate environment** from `../../fine_tuning/` and `../smt/`, which are
both PyTorch. `zeus.py` is invoked as a subprocess by `eval_zeus.py`, so the
two environments never need to coexist in the same Python process.
