# Bekern fine-tuning (decomposed **kern encoding)

Exploratory / future-work experiment referenced in the paper's Conclusion, **not a
paper-table result.** It tests whether a *decomposed* **kern encoding rescues the **kern
fine-tuning track.

## Motivation

The paper's **kern track (Table 6) uses the compound **kern of `PRAIG/smt-grandstaff`, where one
token fuses duration+pitch+accidental+beam+... . On the 561-sample Debussy corpus this yields a
huge, sparse vocabulary (8,598 token types, 43% of them appearing exactly once) — most tokens can't
be learned, and the track scores 63.55% SER (vs. LMX's 50.85%). A *decomposed* encoding (bekern:
`8e-J -> 8 e - J`) collapses that to ~160 dense, reused atoms — the same density as LMX (158) — so
we expect it to close most of the gap.

## Tokenizer: official kernpy, not home-grown

We use **kernpy** (`pip install kernpy`), PRAIG's own package, via
`fine_tuning/bekern_kernpy.py`: MusicXML → verovio kern → `kp.Encoding.bEkern` → the SMT
`<t>`/`<b>`/`<s>` token stream. This is selected in a config with `use_kernpy: true` (a flag added
to `fine_tuning/{data.py,train_kfold.py}`); the default (`false`) path keeps the compound tokenizer
used for the paper's Table 6.

## Usage

Fine-tune the pretrained bekern SMT (`PRAIG/smt-fp-grandstaff`, a 181-token decomposed model), with
the same k-fold driver as Table 6:

```bash
# system-level (directly comparable to Table 6's compound **kern track)
python fine_tuning/train_kfold.py --config fine_tuning/bekern_track/config_system.yaml
# full-page (the fp model's native level)
python fine_tuning/train_kfold.py --config fine_tuning/bekern_track/config_fullpage.yaml
```

Notes:
- `precision: "32-true"` — this fp checkpoint is numerically unstable in fp16/bf16 mixed precision
  (it produces NaN losses); fp32 is stable.
- `--only-fold N` re-runs a single CV fold in isolation.
- The training loop skips any batch whose loss is non-finite, so one degenerate sample cannot
  poison a whole fold.

Adds `kernpy` to `fine_tuning/requirements.txt`.
