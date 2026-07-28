# Zero-shot SMT evaluation (reproduces Table 4 SMT rows + Table 5 SMT rows)

Zero-shot evaluation of the two publicly released Sheet Music Transformer (SMT)
checkpoints (Ríos-Vila et al.) on both their own training-distribution test
sets (Table 4) and the Debussy dataset (Table 5), with no fine-tuning.

- `run_smt_evaluation_optimized.py` -- system-level evaluation, model
  `PRAIG/smt-grandstaff` (**kern vocabulary).
- `run_smt_evaluation_fullpage.py` -- full-page evaluation, model
  `PRAIG/smt-fp-grandstaff`.
- `smt_model/` -- vendored HuggingFace model code (`configuration_smt.py`,
  `modeling_smt.py`) required to load the PRAIG checkpoints via
  `trust_remote_code`.
- `config.yaml` / `config_fullpage.yaml` -- run configuration for the two
  scripts above.

Both scripts were already config-driven and path-agnostic in the source
repository (no machine-specific absolute paths to strip): `dataset_base_dir`,
`output_dir`, `logs_dir`, and `olimpic_path` are all relative to the working
directory by default.

## Running

```bash
pip install -r requirements.txt

# System-level (PRAIG/smt-grandstaff)
python run_smt_evaluation_optimized.py --config config.yaml

# Full-page (PRAIG/smt-fp-grandstaff)
python run_smt_evaluation_fullpage.py --config config_fullpage.yaml
```

Each run writes a timestamped `output_dir/run_YYYYMMDD_HHMMSS/` with
per-sample predictions, converted MusicXML, `evaluation_summary.json`
(SER/CER/LER, and TEDn if `enable_tedn: true`), and a `latest` symlink.

## Binarization: kept as `adaptive` deliberately

Both configs set `binarization_method: adaptive`. Per this session's
camera-ready investigation (see `REVISION_PLAN.md` item T1.3 in the paper
repo), pixel statistics confirmed this is the *correct* choice for both SMT
checkpoints: they are trained on clean, near-binary GrandStaff renders
(~87% of pixels above brightness 200, only ~10% mid-gray), unlike the Zeus
Camera-GrandStaff and Synthetic/OLiMPiC checkpoints (see
`../zeus/README.md` for the correction that *does* apply there). Do not
change this setting for the SMT scripts.

## Data & checkpoints (not included in this repo)

- **Models**: `PRAIG/smt-grandstaff` and `PRAIG/smt-fp-grandstaff`, both on
  HuggingFace Hub, loaded automatically via `model_name` in each config
  (`trust_remote_code=True`).
- **Datasets**: GrandStaff / Camera-GrandStaff (for Table 4) and the Debussy
  dataset (for Table 5, `dataset_base_dir` / `dataset_subdir` in the config).
  The Debussy dataset is released under CC-BY-4.0; link to be added upon
  camera-ready publication.
- **olimpic-icdar24** (Mayer et al.): required for LMX linearization
  (`setup_olimpic()` in both scripts expects `./olimpic-icdar24` relative to
  the working directory, configurable via `olimpic_path` in the config).
  Clone https://github.com/ufal/olimpic-icdar24 next to where you run these
  scripts, or edit `olimpic_path`.

## Environment

PyTorch (see `requirements.txt`: torch, transformers, music21, zss,
python-Levenshtein). Compatible with `../../fine_tuning/`'s environment; a
separate TensorFlow environment is needed for `../zeus/` (see that
subfolder's README).
