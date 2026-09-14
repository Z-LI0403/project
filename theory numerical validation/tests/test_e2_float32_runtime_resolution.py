from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from validate_e2_bridge import (  # noqa: E402
    _close_array,
    _float32_semantic_comparison,
    _load_e2_interfaces,
    _nodal_kernels,
    _runtime_vs_float64_target_comparison,
    independent_hat_weights,
    independent_hat_weights_float32,
    independent_kernel_interpolation_float32,
)


class E2Float32RuntimeResolutionTests(unittest.TestCase):
    OPERATION_COUNT = 8
    PATH_FACTOR = 2

    def _fixture(self) -> dict[str, object]:
        module = _load_e2_interfaces()
        knots = np.asarray(
            (
                -0.8431394319417529,
                -0.475873267284342,
                -0.2060556053881204,
                0.00714538481207612,
                0.76752003058133,
            ),
            dtype=np.float64,
        )
        values = np.linspace(knots[0], knots[-1], 37, dtype=np.float64)
        nodes = np.asarray(
            (
                4.6118982589253026e-05,
                0.003739872543278163,
                0.07408905806089838,
                0.25457600213472975,
                0.3199910393548336,
            ),
            dtype=np.float64,
        )
        nodal = _nodal_kernels(nodes)
        base = nodal[0]
        residuals = nodal[1:] - base

        adapter = module.TimeAdapter(knots).cpu().eval()
        runtime_u = module.torch.from_numpy(values).to(dtype=module.torch.float32)
        with module.torch.no_grad():
            adapter.base.copy_(module.torch.from_numpy(base).to(dtype=adapter.base.dtype))
            adapter.residual.copy_(
                module.torch.from_numpy(residuals).to(dtype=adapter.residual.dtype)
            )
            actual_hats = module.hat_weights(runtime_u, adapter.knots).cpu().numpy()
            actual_kernels = adapter.kernels(runtime_u).cpu().numpy()

        reference_hats = independent_hat_weights_float32(values, knots)
        reference_kernels = independent_kernel_interpolation_float32(
            reference_hats, base, residuals
        )
        ideal_hats = independent_hat_weights(values, knots)
        ideal_kernels = np.einsum("tm,moihw->toihw", ideal_hats, nodal)
        return {
            "actual_hats": actual_hats,
            "actual_kernels": actual_kernels,
            "reference_hats": reference_hats,
            "reference_kernels": reference_kernels,
            "ideal_hats": ideal_hats,
            "ideal_kernels": ideal_kernels,
            "base": base,
            "residuals": residuals,
            "nodal": nodal,
        }

    def _semantic(self, actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
        return _float32_semantic_comparison(
            actual,
            reference,
            operation_count=self.OPERATION_COUNT,
            path_factor=self.PATH_FACTOR,
        )

    def test_independent_float32_reference_matches_actual_time_adapter(self) -> None:
        fixture = self._fixture()
        hat_check = self._semantic(
            fixture["actual_hats"], fixture["reference_hats"]  # type: ignore[arg-type]
        )
        kernel_check = self._semantic(
            fixture["actual_kernels"], fixture["reference_kernels"]  # type: ignore[arg-type]
        )
        self.assertTrue(hat_check["passed"])
        self.assertTrue(kernel_check["passed"])
        self.assertEqual(hat_check["reference_dtype"], "float32")
        self.assertEqual(kernel_check["reference_dtype"], "float32")
        self.assertEqual(hat_check["operation_count"], 8)
        self.assertEqual(kernel_check["comparison_path_factor"], 2)

    def test_normal_float32_rounding_is_reported_without_structure_failure(self) -> None:
        fixture = self._fixture()
        semantic = self._semantic(
            fixture["actual_kernels"], fixture["reference_kernels"]  # type: ignore[arg-type]
        )
        comparison = _runtime_vs_float64_target_comparison(
            fixture["actual_kernels"],  # type: ignore[arg-type]
            fixture["reference_kernels"],  # type: ignore[arg-type]
            fixture["ideal_kernels"],  # type: ignore[arg-type]
            semantic,
        )
        self.assertGreater(comparison["absolute_max_error"], 0.0)
        self.assertGreater(comparison["derived_rounding_budget"], 0.0)
        self.assertLessEqual(comparison["error_budget_ratio"], 1.0)
        self.assertTrue(comparison["passed"])

    def test_corrupted_hat_and_kernel_beyond_budget_fail_closed(self) -> None:
        fixture = self._fixture()
        bad_hats = np.array(fixture["actual_hats"], dtype=np.float32, copy=True)
        bad_hats[10, 2] += np.float32(1e-3)
        self.assertFalse(
            self._semantic(bad_hats, fixture["reference_hats"])["passed"]  # type: ignore[arg-type]
        )

        bad_kernels = np.array(fixture["actual_kernels"], dtype=np.float32, copy=True)
        bad_kernels[10, 0, 0, 1, 0] += np.float32(1e-3)
        bad_kernel_semantic = self._semantic(
            bad_kernels, fixture["reference_kernels"]  # type: ignore[arg-type]
        )
        self.assertFalse(bad_kernel_semantic["passed"])
        bad_target = _runtime_vs_float64_target_comparison(
            bad_kernels,
            fixture["reference_kernels"],  # type: ignore[arg-type]
            fixture["ideal_kernels"],  # type: ignore[arg-type]
            self._semantic(
                fixture["actual_kernels"],  # type: ignore[arg-type]
                fixture["reference_kernels"],  # type: ignore[arg-type]
            ),
        )
        self.assertFalse(bad_target["passed"])

    def test_float64_algebraic_containment_keeps_original_gate(self) -> None:
        fixture = self._fixture()
        hats = fixture["ideal_hats"]
        base = fixture["base"]
        residuals = fixture["residuals"]
        nodal = fixture["nodal"]
        base_residual = base[None, ...] + np.einsum(  # type: ignore[index]
            "tm,moihw->toihw", hats[:, 1:], residuals  # type: ignore[index]
        )
        five_kernel = np.einsum("tm,moihw->toihw", hats, nodal)
        self.assertTrue(_close_array(base_residual, five_kernel, 5e-13, 5e-13))

        corrupted = np.array(base_residual, copy=True)
        corrupted[10, 0, 0, 1, 0] += 1e-9
        self.assertFalse(_close_array(corrupted, five_kernel, 5e-13, 5e-13))


if __name__ == "__main__":
    unittest.main()
