from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from validate_e2_bridge import (  # noqa: E402
    _load_e2_interfaces,
    _nodal_kernels,
    build_schedule_float64,
    independent_hat_weights,
)


class ScheduleAndBasisFixtureTests(unittest.TestCase):
    def test_seven_step_toy_schedule_and_hat_partition(self) -> None:
        beta, alpha, tau, u = build_schedule_float64(7, 1e-4, 0.02)
        self.assertTrue(np.all(np.diff(beta) > 0.0))
        self.assertTrue(np.all(np.diff(alpha) < 0.0))
        self.assertTrue(np.all(np.diff(tau) < 0.0))
        knot_times = np.asarray((6, 4, 3, 1, 0), dtype=np.int64)
        knots = u[knot_times]
        hats = independent_hat_weights(u, knots)
        np.testing.assert_allclose(np.sum(hats, axis=1), np.ones(7), atol=5e-13, rtol=5e-13)
        np.testing.assert_allclose(independent_hat_weights(knots, knots), np.eye(5), atol=5e-13, rtol=5e-13)

    def test_base_residual_equals_five_kernel_interpolation(self) -> None:
        values = np.linspace(-0.8, 0.7, 9, dtype=np.float64)
        knots = np.asarray((-0.8, -0.45, -0.1, 0.25, 0.7), dtype=np.float64)
        hats = independent_hat_weights(values, knots)
        nodes = np.asarray((0.01, 0.05, 0.11, 0.19, 0.27), dtype=np.float64)
        kernels = _nodal_kernels(nodes)
        base = kernels[0]
        residual = kernels[1:] - base
        base_residual = base[None, ...] + np.einsum("tm,moihw->toihw", hats[:, 1:], residual)
        five_kernel = np.einsum("tm,moihw->toihw", hats, kernels)
        np.testing.assert_allclose(base_residual, five_kernel, atol=5e-13, rtol=5e-13)

    def test_native_e2_time_adapter_executes_frozen_base_residual_route(self) -> None:
        """Non-formal fixture: exercise native float32 E2 adapter semantics directly."""
        module = _load_e2_interfaces()
        knots = np.asarray((-0.8, -0.45, -0.1, 0.25, 0.7), dtype=np.float64)
        values = np.linspace(-0.8, 0.7, 9, dtype=np.float64)
        nodes = np.asarray((0.01, 0.05, 0.11, 0.19, 0.27), dtype=np.float64)
        kernels = _nodal_kernels(nodes)
        base = kernels[0]
        residual = kernels[1:] - base

        adapter = module.TimeAdapter(knots).cpu().eval()
        runtime_u = module.torch.from_numpy(values).to(dtype=module.torch.float32)
        with module.torch.no_grad():
            adapter.base.copy_(module.torch.from_numpy(base).to(dtype=adapter.base.dtype))
            adapter.residual.copy_(
                module.torch.from_numpy(residual).to(dtype=adapter.residual.dtype)
            )
            runtime_hats = module.hat_weights(runtime_u, adapter.knots)
            actual = adapter.kernels(runtime_u)
            expected = adapter.base.unsqueeze(0) + module.torch.einsum(
                "tm,moihw->toihw", runtime_hats[:, 1:], adapter.residual
            )

        self.assertEqual(tuple(adapter.base.shape), (3, 3, 3, 3))
        self.assertEqual(tuple(adapter.residual.shape), (4, 3, 3, 3, 3))
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 405)
        self.assertEqual(actual.dtype, module.torch.float32)
        self.assertTrue(module.torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
