# Bekern fine-tuning (decomposed **kern encoding)

Exploratory / future-work experiment referenced in the paper's Conclusion, **not a
paper-table result.** It tests whether a *decomposed* **kern encoding rescues the **kern
fine-tuning track.

> **Status: negative / incomplete result — read this before running anything here.**
> The recipe in `bekern_track/` (fine-tuning `PRAIG/smt-fp-grandstaff`) **does not work**: it
> produces a non-finite loss on the very first batch. The code is included for transparency, and
> because the tokenizer is reusable, not because it reproduces a number. See *Status* below.
>
> **None of this affects the paper's Table 6.** Both published fine-tuning tracks leave
> `use_kernpy` unset (default `false`) and never call kernpy: the **kern track uses the compound
> tokenizer in `fine_tuning/data.py`, the LMX track uses the olimpic-icdar24 linearizer.

## Motivation

The paper's **kern track (Table 6) uses the compound **kern of `PRAIG/smt-grandstaff`, where one
token fuses duration+pitch+accidental+beam+... . On the 561-sample Debussy corpus this yields a
huge, sparse vocabulary (8,598 token types, 43% of them appearing exactly once) — most tokens can't
be learned, and the track scores 63.55% SER (vs. LMX's 50.85%). A *decomposed* encoding (bekern:
`8e-J -> 8 e - J`) collapses that to ~150 dense, reused atoms — comparable to LMX's 158 — so the
hypothesis is that it closes most of the gap.

## Tokenizer: official kernpy, not home-grown

We use **kernpy** (`pip install kernpy`), PRAIG's own package, via
`fine_tuning/bekern_kernpy.py`: MusicXML → verovio kern → `kp.Encoding.bEkern` → the SMT
`<t>`/`<b>`/`<s>` token stream. This is selected in a config with `use_kernpy: true` (a flag added
to `fine_tuning/{data.py,train_kfold.py}`); the default (`false`) path keeps the compound tokenizer
used for the paper's Table 6.

### Important: chord-note loss in kernpy, and the workaround

kernpy's bEkern exporter **silently drops every chord note after the first whenever any note in the
chord carries a signifier** (tie, slur, beam, stem, articulation). No error is raised:

```
4c 4e 4g       ->  4@c 4@e 4@g     # plain chord: correct
[4c [4e [4g    ->  4@c             # tied chord: two pitches silently lost
8cL 8eL 8gL    ->  8@c             # beamed chord: two pitches silently lost
```

Left uncorrected this removes **~17% of all notes** on the Debussy corpus (worst sample: 60%) and
~3% on GrandStaff, making the ground truth musically wrong and the task artificially easier.

`clean_kern_for_kernpy()` in `fine_tuning/bekern_kernpy.py` therefore strips note-level signifiers
*before* handing tokens to kernpy. This is a **correctness fix, not cosmetics.** It is safe because
bEkern discards those signifiers anyway (verified atom-by-atom), so nothing bEkern would have kept
is lost:

| Corpus | note retention (MusicXML → token stream) |
|---|---|
| Debussy, without the workaround | 82.7% |
| Debussy, with the workaround | **99.2%** |
| GrandStaff, without / with | 97.2% / **100.0%** |

**If you reuse `bekern_kernpy.py` on another corpus, keep this behaviour.**

## What bekern does *not* encode

bEkern is **b**asic extended kern: duration + pitch + accidental. It deliberately discards slurs,
ties, beams, stems, articulations and ornaments. That is by design, not a bug, and it survives the
fix above.

Consequence: **a bekern SER is not directly comparable to the LMX or compound-**kern numbers in
Table 6.** Its ground truth is a strictly easier target than LMX's, which retains that content (per
60 Debussy samples: LMX GT has 435 slur-starts, 224 staccato, 91 tenuto; bekern GT has effectively
none). Any comparison must state this explicitly.

TEDn, which would otherwise make the formats comparable, is **also unusable here**: `tedn_eval.py`
converts kern predictions with `music21.converter.parse(..., format='humdrum')`, which cannot read
decomposed bekern atoms, so `test_TEDn` reads ~100% for any bekern run regardless of model quality.

## Status: what worked and what didn't

**Did not work — fine-tuning `PRAIG/smt-fp-grandstaff` (the recipe in `bekern_track/`).**
The checkpoint produces a **non-finite loss at epoch 0, batch 0** — in the forward pass, before any
optimizer step — and the failure is *position-dependent*: a given fold trains when it runs second in
a process but goes all-NaN when run first via `--only-fold N`. In a 5-fold run, folds 1 and 5 were
NaN while folds 2/3/4 reached 52.0 / 53.0 / 59.3% SER. Five hypotheses were tested and **all ruled
out**: flash attention; cuDNN autotuning (`benchmark=False`); embedding-init scale (the trained
embeddings are already ≈N(0,1), so the default init was never oversized); init randomness
(deterministic mean-init); and teacher-forcing corruption (`teacher_forcing_error_rate: 0.0`).
The failure could not be reproduced faithfully on CPU, so it was not bisected further.

**The route we moved to (not included in this repo).** Pretraining an SMT *from scratch* on
GrandStaff bekern and fine-tuning that instead sidesteps the problem structurally: the model then
has the bekern vocabulary natively, so there is no embedding extension onto fragile pretrained
weights. Pretraining code is out of scope for this replication repo, which covers the paper's
tables.

## Usage (for reference — see *Status* first)

```bash
# system-level
python fine_tuning/train_kfold.py --config fine_tuning/bekern_track/config_system.yaml
# full-page
python fine_tuning/train_kfold.py --config fine_tuning/bekern_track/config_fullpage.yaml
```

Notes:
- `--only-fold N` runs a single CV fold in isolation.
- The training loop skips any batch whose loss is non-finite. That is a guard against one bad
  sample, **not** a fix for the failure above: when every batch is non-finite the model simply
  never leaves its initialisation.
- `precision: "32-true"` is set because the checkpoint NaNs in fp16/bf16 mixed precision — but it
  NaNs in fp32 as well, so this setting does not make the recipe work.
- Fine-tuning from a *local* from-scratch checkpoint is supported via `local_checkpoint: <path>.ckpt`.
  If that checkpoint stores only a `state_dict`, put its vocabulary beside it as
  `<path>.ckpt.vocab.json` (`{"w2i": ..., "i2w": ...}`) so token→index mappings stay aligned with
  the pretrained embeddings.

Adds `kernpy` to `fine_tuning/requirements.txt`.
