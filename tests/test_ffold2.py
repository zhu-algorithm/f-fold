import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffold.ffold2 import (  # noqa: E402
    ConformationalPathDecoder, DensityEncoder3D, EnergyForceHead,
    acquisition_score, physics_losses,
)


class FFold2ArchitectureTests(unittest.TestCase):
    def test_density_encoder_is_differentiable(self):
        model = DensityEncoder3D(24, width=4)
        density = torch.randn(2, 1, 12, 12, 12, requires_grad=True)
        out = model(density)
        self.assertEqual(out.shape, (2, 24))
        out.sum().backward()
        self.assertIsNotNone(density.grad)

    def test_path_preserves_endpoints_and_has_gradients(self):
        decoder = ConformationalPathDecoder(16, 8, hidden=32, max_step=2.0)
        token = torch.randn(2, 7, 16, requires_grad=True)
        condition = torch.randn(2, 8)
        start = torch.randn(2, 7, 3)
        end = torch.randn(2, 7, 3)
        path = decoder(token, condition, start, end, n_frames=6)
        self.assertEqual(path.shape, (2, 6, 7, 3))
        self.assertTrue(torch.allclose(path[:, 0], start))
        self.assertTrue(torch.allclose(path[:, -1], end))
        path[:, 1:-1].square().mean().backward()
        self.assertIsNotNone(token.grad)

    def test_energy_force_and_physics_contract(self):
        token = torch.randn(2, 5, 12)
        path = torch.randn(2, 4, 5, 3)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
        head = EnergyForceHead(12, hidden=24)
        energy, forces = head(token, path, mask)
        self.assertEqual(energy.shape, (2, 4))
        self.assertEqual(forces.shape, path.shape)
        losses = physics_losses(path, forces, mask)
        self.assertEqual(set(losses), {"bond", "clash", "smoothness", "force_smoothness"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_active_learning_score(self):
        uncertainty = torch.tensor([0.2, 0.8])
        logits = torch.tensor([[0.0, 0.0], [8.0, 8.0]])
        score = acquisition_score(uncertainty, logits)
        self.assertEqual(score.shape, (2,))
        self.assertGreater(float(score[0]), float(score[1]))


if __name__ == "__main__":
    unittest.main()

