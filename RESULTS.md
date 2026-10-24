# F-Fold — methods and results

Project page: <https://zhu-algorithm.github.io/f-fold/>

## 1. Hyperparameter alignment

`ffold/config.py` defines two presets and is written by hand against the
AlphaFold3 specification, *not* generated from OpenFold3's config — otherwise
the comparison below would be circular.

`ffold/check_alignment.py` loads the reference extracted from the
`openfold3==0.4.1` source distribution (`model_config.py`, Apache-2.0) and
diffs every shape-bearing field. Result: **203 match, 0 mismatch, 0 missing**;
the script exits non-zero on any disagreement and is CI-ready.

Out of scope by construction (they do not affect tensor shapes): attention
kernel selection, memory offloading, activation checkpointing, and
weight-initialisation policy.

## 2. Parameter audit

Meta-device instantiation, so the full-scale preset costs no memory to audit.

| preset | parameters |
|---|---:|
| `ffold-base` (AF3 dimensions, MSA + templates) | 368,003,859 |
| `ffold-mini` (T4-trainable, protein-only) | 8,816,323 |

The diffusion transformer (198.3M) and Pairformer (147.4M) are 94% of
`ffold-base`. Per-module breakdown: `ffold/param_audit.csv`.

These are F-Fold's own counts. They are **not** a claim about the OpenFold3
checkpoint, which is gated on HuggingFace and was never downloaded or
inspected.

## 3. Teacher dataset

- Source: AlphaFold Protein Structure Database, **model v6**, retrieved
  2026-10-24 (all 1,723 structures came from v6; no fallback was needed).
- Candidates: 34,438 reviewed UniProt entries, 60–256 aa, protein-level
  evidence, with an AlphaFold DB cross-reference.
- Redundancy filter: greedy k-mer (k=5) selection with an inverted index,
  rejecting any candidate sharing >30% of its 5-mers with an accepted one.
- Confidence filter: models with mean pLDDT < 70 discarded (292 dropped) as
  predicted disorder rather than folds. 786 accessions had no retrievable
  prediction.
- Final: **1,723** structures — 1,550 train / 172 validation / 1 test.
- Labels: Cβ–Cβ distances discretised into AF3's 64-bin distogram over
  2–22 Å (glycine uses Cα), plus per-residue pLDDT from the B-factor column.
- Held out: human lysozyme C `P61626` (148 aa, teacher mean pLDDT 94.1),
  excluded by accession *and* by the same k-mer similarity ceiling.

## 4. Training

One Tesla T4 (15 GB). Batch 8 at crop 160, selected automatically by an
OOM probe; AMP (fp16) and gradient checkpointing on both Pairformer stacks.
Peak VRAM 5.9 GB. Loss = distogram cross-entropy + 0.3 × pLDDT
cross-entropy. AF3's optimiser schedule shape (linear warmup, stepwise
exponential decay) with peak lr reduced to 8e-4 because `ffold-mini` is
~40× narrower than `ffold-base`.

The job was stopped by its 2-hour run clock (`exit_code 124`) at
**step 2,000 of 8,000 requested**. The reported model is therefore
quarter-trained. Validation distogram cross-entropy at the saved checkpoint:
**2.211**.

## 5. Results on the held-out protein

| metric | value | reference |
|---|---:|---|
| distogram cross-entropy | 2.633 | 4.159 = uniform over 64 bins |
| contact P@L/5, \|i−j\|≥12 | 0.310 | 0.022 base rate of <8 Å pairs |
| contact P@L/5, \|i−j\|≥24 | 0.034 | 0.018 base rate |
| pLDDT vs teacher, Pearson | 0.958 | Spearman 0.670 |
| Cβ RMSD after Kabsch | 14.31 Å | — |
| TM-score | 0.200 | <0.17 random, >0.5 same fold |
| **control:** RMSD from *teacher* distances | **1.19 Å** | same reconstruction path |

**Interpretation.** The model learned per-residue confidence (r = 0.958,
including the low-confidence N-terminal signal peptide) and near-diagonal
local geometry, but not tertiary packing: long-range contact precision of
0.034 against a 0.018 base rate is barely above chance, and TM-score 0.200
means the fold is wrong.

The control is the load-bearing number. Running the teacher's own distance
matrix through the identical MDS-initialised, gradient-refined reconstruction
recovers the structure to 1.19 Å. The geometry pipeline is therefore correct,
and the 14.31 Å figure is attributable to the model's predicted distances.

## 6. Limitations

1. F-Fold does not produce usable structures. It is an architecture and a
   pipeline, not a deployable predictor.
2. A distilled student cannot exceed its teacher. No claim of superiority over
   AlphaFold is made or supportable.
3. 2,000 steps on 1,550 structures, against AF3's PDB-scale training with deep
   MSAs. The accuracy gap is expected.
4. No MSA, no templates, no all-atom output in the trained path. Those modules
   are implemented and audited but unused during distillation.
5. `ffold-base` has never been trained — only parameter-audited and
   shape-verified. Training it needs hundreds of GPU-days.
6. A single 148-residue monomer is an illustration, not an evaluation set.

## 7. What would move the numbers

In rough order of expected return: run the remaining 6,000 steps; add MSA
features (the module is already built and is the single largest known gap);
widen `ffold-mini` toward `ffold-base` on a 40–80 GB GPU; expand the teacher
set past 10k structures; add AF3's smooth-LDDT and bond losses alongside the
distogram objective.
