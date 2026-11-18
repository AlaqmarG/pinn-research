"""Checks that the batched ZO model matches the cooling network exactly."""
import unittest

import torch

from pinnlab import zo as zp


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.theta = zp.init_theta(7)
        self.model = zp.cp.CoolingNet(zp.C)
        torch.manual_seed(7)
        self.model = zp.cp.CoolingNet(zp.C)
        self.x = torch.rand((50, 1), dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    def test_parameter_count(self):
        self.assertEqual(zp.N_PARAMS, 673)
        self.assertEqual(self.theta.numel(), 673)

    def test_forward_matches_coolingnet(self):
        u, _ = zp.forward(self.theta[None], self.x, False)
        with torch.no_grad():
            ref = self.model(self.x)[:, 0]
        self.assertLess(float((u[0] - ref).abs().max()), 1e-14)

    def test_forward_mode_derivative_matches_autograd(self):
        _, du = zp.forward(self.theta[None], self.x, True)
        r, u = zp.cp.residual(self.model, self.x, zp.C, create_graph=False)
        ref_du = (r - zp.KT * u)[:, 0].detach()
        self.assertLess(float((du[0] - ref_du).abs().max()), 1e-12)

    def test_loss_matches_original(self):
        total, p, ic = zp.batched_loss(self.theta[None], self.x, "exact")
        rp, ric = zp.cp.losses(self.model, self.x, zp.C, create_graph=False)
        self.assertLess(abs(float(p[0] - rp)), 1e-13)
        self.assertLess(abs(float(ic[0] - ric)), 1e-14)

    def test_finite_difference_close(self):
        e = zp.batched_loss(self.theta[None], self.x, "exact")[1][0]
        f = zp.batched_loss(self.theta[None], self.x, "fd")[1][0]
        self.assertLess(abs(float(e - f)) / float(e), 1e-4)

    def test_batch_rows_independent(self):
        batch = torch.stack([self.theta, self.theta * 0.5, self.theta])
        L = zp.batched_loss(batch, self.x, "exact")[0]
        self.assertEqual(float(L[0]), float(L[2]))
        self.assertNotEqual(float(L[0]), float(L[1]))

    def test_load_flat_roundtrip(self):
        m = zp.cp.CoolingNet(zp.C)
        zp.load_flat(m, self.theta)
        self.assertTrue(torch.equal(zp.flatten_model(m), self.theta))


if __name__ == "__main__":
    unittest.main(verbosity=2)
