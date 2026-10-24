# F-Fold

A compact, config-faithful reimplementation of the AlphaFold3 architecture,
with a knowledge-distillation pipeline that trains on a single consumer GPU.

F-Fold's hyperparameters are aligned, field by field and verified by a script,
against [OpenFold3-preview](https://github.com/aqlaboratory/openfold3) — the
AlQuraishi Lab / OpenFold consortium's Apache-2.0 reproduction of AlphaFold3.

## What this is, and what it is not

**It is** an independently written AF3-shaped model whose dimensions provably
match OpenFold3's, plus a working distillation pipeline that produces your own
weights from AlphaFold DB structures on one 15 GB GPU.

**It is not** a model that competes with AlphaFold on accuracy. The small
preset is distilled *from* AlphaFold, so AlphaFold is its ceiling, not its
competitor. It has no MSA search, no templates, and no all-atom output in the
trained path. If you need accurate structures, run AlphaFold2, OpenFold3,
Boltz-2 or Chai-1 — not this.

Where F-Fold is genuinely useful:

| | F-Fold | OpenFold3-preview |
|---|---|---|
| Minimum VRAM | 15 GB **trains** `ffold-mini`; `ffold-base` is audit-only | 24 GB for *inference*, 80 GB recommended |
| Weights | you train them; nothing gated | Apache-2.0 but **gated** on HuggingFace |
| Parameters | 368.0 M (`ffold-base`) / 8.8 M (`ffold-mini`) | AF3-scale |
| Core model code | ~2,000 lines, no custom CUDA kernels | kernel-accelerated (DeepSpeed / cuEquivariance) |
| Hyperparameter fidelity | 203/203 fields verified against OpenFold3 | reference |

## Hyperparameter alignment

`ffold/config.py` is written independently against the AF3 specification; it
is **not** generated from OpenFold3's config, which is what makes the check
meaningful. `ffold/check_alignment.py` diffs it against
`of3_reference_config.json` (extracted from the `openfold3==0.4.1` sdist) and
exits non-zero on any disagreement:

```console
$ python ffold/check_alignment.py
           match: 203
```

Kernel selection, memory offloading, checkpointing and weight-init policy are
explicitly out of scope — they do not change tensor shapes. Everything that
does is compared.

Base dimensions, all verified:

| | value | | value |
|---|---|---|---|
| `c_s` | 384 | `c_atom` | 128 |
| `c_z` | 128 | `c_atom_pair` | 16 |
| `c_m` | 64 | `c_token_embedder` | 384 |
| `c_t` | 64 | `c_token_diffusion` | 768 |
| `c_s_input` | 449 | `sigma_data` | 16 |

Stack depths: Pairformer 48 blocks (16 pair-bias heads, `c_hidden` 24),
diffusion transformer 24 blocks (16 heads, `c_hidden` 48), MSA module 4,
template pair stack 2, confidence Pairformer 4, atom encoder/decoder 3 each.
Label spaces: distogram 64 bins over 2–22 Å, pLDDT 50 bins, PAE/PDE 64 bins
over 0–32 Å.

## Parameter audit

`python ffold/audit_params.py` instantiates both presets on the meta device
(no allocation) and reports:

| module | `ffold-base` | `ffold-mini` |
|---|---:|---:|
| input embedder | 1,477,381 | 135,189 |
| recycling | 164,864 | 19,040 |
| MSA module | 3,063,488 | — |
| template module | 255,296 | — |
| Pairformer | 147,394,560 | 2,036,352 |
| diffusion conditioning | 2,439,554 | 311,442 |
| diffusion atom encoder | 1,039,536 | 60,624 |
| diffusion transformer | 198,343,680 | 5,516,832 |
| diffusion atom decoder | 937,456 | 43,664 |
| distogram head | 8,256 | 3,136 |
| confidence heads | 12,879,788 | 690,044 |
| **total** | **368,003,859** | **8,816,323** |

This is F-Fold's own count at AF3 dimensions. It is not a claim about the
OpenFold3 checkpoint's parameter count, which is gated and was not inspected.

## Quickstart

```bash
pip install -r requirements.txt

python ffold/check_alignment.py            # verify the config match
python ffold/audit_params.py               # parameter table

python ffold/build_dataset.py --n 2800 --min-plddt 70
python ffold/train.py --auto-batch --amp --grad-checkpoint --crop 160
python ffold/evaluate.py
```

`build_dataset.py` pulls AlphaFold DB predictions for a k-mer-deduplicated
sample of reviewed UniProt sequences (60–256 aa), drops models whose mean
pLDDT is below the threshold, and writes Cβ distance labels plus per-residue
pLDDT.

## Architecture

Implemented in `ffold/layers.py` and `ffold/model.py`:

- SwiGLU transitions, triangular multiplicative update (outgoing/incoming),
  triangular attention (starting/ending node), attention with pair bias
- Adaptive layer norm and conditioned transitions for the diffusion path
- Relative position encoding with chain-aware buckets
- Pairformer stack with optional gradient checkpointing
- MSA module with outer-product mean and pair-weighted averaging
- Template pair embedder and stack
- Diffusion conditioning with Fourier noise embedding, EDM schedule
  (`s_max` 160, `s_min` 4e-4, `p` 7)
- Sequence-local atom attention encoder/decoder
- Five heads: distogram, PAE, PDE, pLDDT, experimentally-resolved

The trained path is protein-only and token-level. AF3's data pipeline — CCD
ligand graphs, MSA search, template search — is not implemented.

## Licensing

Apache-2.0. See `LICENSE` and `NOTICE`. No OpenFold3 weights are
redistributed. Teacher labels derive from the AlphaFold Protein Structure
Database (CC BY 4.0).

## References

- Abramson et al., *Accurate structure prediction of biomolecular interactions
  with AlphaFold 3*, Nature 630 (2024)
- OpenFold3-preview, AlQuraishi Lab / OpenFold consortium,
  <https://github.com/aqlaboratory/openfold3>
- Varadi et al., *AlphaFold Protein Structure Database*, NAR 50 (2022)
- Karras et al., *Elucidating the design space of diffusion-based generative
  models*, NeurIPS 2022
