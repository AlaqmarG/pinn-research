"""Check calculus/scaling and checkpoint serialization independently of optimization."""

import io
import unittest

import torch
from torch import nn

from pinnlab.cooling import Config, CoolingNet, load_checkpoint, residual


class CalculusTests(unittest.TestCase):
    def test_analytic_solution_residual_and_units(self):
        # Use k*end != 1 so a missing time scale cannot cancel accidentally.
        c = Config(k_per_min=0.17, end_min=7.0)

        class Analytic(nn.Module):
            def forward(self, tau):
                return torch.exp(-c.k_per_min * c.end_min * tau)

        tau = torch.tensor([[0.0], [0.13], [0.61], [1.0]], dtype=torch.float64)
        r, _ = residual(Analytic(), tau, c)
        torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-14, rtol=0)
        x = tau.clone().requires_grad_(True)
        temperature = c.ambient_c + (c.initial_c - c.ambient_c) * Analytic()(x)
        actual = torch.autograd.grad(temperature, x, torch.ones_like(x))[0] / c.end_min
        expected = -c.k_per_min * (temperature - c.ambient_c)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)

    def test_residual_backpropagates_through_derivative(self):
        c = Config(k_per_min=0.2, end_min=3.0)
        model = nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.fill_(2.0)
        x = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.float64)
        r, _ = residual(model, x, c)
        r.square().mean().backward()
        # r = a*(1+k*end*tau); d mean(r^2)/da = 2*a*mean((1+k*end*tau)^2).
        expected = 4 * (1 + c.k_per_min * c.end_min * x).square().mean()
        torch.testing.assert_close(model.weight.grad.squeeze(), expected)

    def test_model_derivative_matches_finite_difference(self):
        torch.manual_seed(123)
        c = Config(k_per_min=0.17, end_min=7.0)
        model = CoolingNet(c)
        x = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
        r, u = residual(model, x, c, create_graph=False)
        delta = 1e-5
        numeric = (model(x + delta) - model(x - delta)) / (2 * delta)
        torch.testing.assert_close(r - c.k_per_min * c.end_min * u, numeric, atol=1e-9, rtol=1e-7)

    def test_checkpoint_reload(self):
        c = Config()
        model = CoolingNet(c)
        buffer = io.BytesIO()
        torch.save({"config": c.__dict__, "model_state_dict": model.state_dict()}, buffer)
        buffer.seek(0)
        loaded, config, _ = load_checkpoint(buffer)
        x = torch.linspace(0, 1, 17, dtype=torch.float64)[:, None]
        torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)
        self.assertEqual(config, c)


if __name__ == "__main__":
    unittest.main()
