# F-Fold 2.0 architecture scaffold

F-Fold 2.0 extends the existing static, protein-only F-Fold trunk into a
research architecture for experiment-conditioned conformational dynamics.
The code is executable and tested, but no trained dynamic checkpoint is
included and no molecular-dynamics or discovery claim is made.

## Data flow

1. The existing F-Fold trunk encodes sequence into single and pair features.
2. `DensityEncoder3D` encodes an optional cryo-EM or cryo-ET density crop.
3. `ConformationalPathDecoder` generates intermediate token coordinates
   conditioned on start and end structures. Its envelope preserves both
   supplied endpoints exactly.
4. `EnergyForceHead` predicts diagnostic frame energies and residue-level
   forces. These are learned surrogates, not a replacement for OpenMM,
   AI2BMD or an independently validated force field.
5. `physics_losses` penalises C-alpha bond distortion, non-neighbour clashes,
   path roughness and force discontinuity.
6. `acquisition_score` ranks uncertain or ambiguous candidates for additional
   simulation or experiment.

## Minimal example

```python
from ffold.config import get_config
from ffold.ffold2 import FFold2, DynamicsConfig, physics_losses

cfg = get_config("ffold-mini")
model = FFold2(cfg, DynamicsConfig(path_frames=12))
out = model(sequence_ids, residue_index, start_coords, end_coords,
            density=density_crop, mask=mask)
losses = physics_losses(out["path"], out["forces"], mask)
```

## Scientific boundaries

- Current coordinates are token-level C-alpha representatives, not all-atom
  trajectories.
- Predicted energy and force values require calibration against a licensed,
  independently validated simulation backend.
- A generated path is a hypothesis until enhanced sampling and experimental
  evidence show that it is physically reachable.
- Density conditioning requires registered density/structure training pairs.
- Discovery claims require frozen, independent benchmarks and prospective
  experimental validation.

## Next implementation gates

1. Add all-atom CCD features and atom masks.
2. Add density-to-coordinate registration and local density crops.
3. Train the path model on trajectory and multi-state structural data.
4. Add OpenMM/AI2BMD adapters with versioned force-field provenance.
5. Add a data manifest for simulation and experimental feedback.
6. Evaluate state recovery, path plausibility and force accuracy on independent
   systems before any biological mechanism claim.

