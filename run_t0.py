"""Formal T0 orchestrator. Running without --smoke starts the approved full suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common_linear_gaussian import (
    NumericalFailure,
    PROJECT_ROOT,
    SourceDriftError,
    base_parser,
    environment_metadata,
    load_config,
    new_run_directory,
    overall_status,
    sha256_file,
    stable_seed,
    verify_source_lock,
    write_json,
)
from validate_boundaries import run_boundaries, smoke_check as smoke_boundaries
from validate_e2_bridge import run_bridge, smoke_check as smoke_bridge, write_profile_csv
from validate_fwv1 import run_fwv1, smoke_check as smoke_fwv1
from validate_theorem1 import run_theorem1, smoke_check as smoke_theorem1


REFERENCE_BY_CASE = {
    "A-GEN": "Theorem 1(b); Appendix A.1",
    "A-INV": "Theorem 1(b), measurement-coordinate invariance",
    "A-HMR": "Corollary 1.1; Appendix A.1.4",
    "A-DST": "Appendix B.1 distinct input/target extension",
    "B-RSK": "Theorem 2(i); Appendix A.2.2-A.2.3",
    "B-ID": "Theorem 2(i); fixed Haar/identity PSG",
    "B-WIT": "Appendix A.2.8 witness comparison",
    "B-OPT": "Theorem 2(ii); Appendix A.2.4",
    "B-REG": "Theorem 2(ii), exact regret identity",
    "B-FULL": "Theorem 2(iii); Appendix A.2.6-A.2.7",
    "B-FAC": "Theorem 2(iii); Appendix A.2.7",
    "B-SNR": "Theorem 2(iv); Appendix A.2.5",
    "B-RNK": "theory.md §3.2; Appendix A.2.1/B.2.5",
    "B-POL": "Corollary 2.1; Appendix A.3",
    "B-POST": "Corollary 2.2; Appendix A.4",
    "C-NOISE": "Appendix B.2.2",
    "C-CLEAN": "Appendix B.2.3",
    "C-SING": "Appendix B.2.4",
    "C-RANK": "Appendix B.2.5",
    "C-COV": "Appendix B.3",
    "C-CTR": "Appendix C corrective counterexample",
}

APPROVED_CASE_IDS = {
    "A": (
        "A-GEN-01", "A-GEN-02", "A-GEN-03", "A-GEN-04", "A-GEN-05",
        "A-INV-01", "A-INV-02",
        "A-HMR-00", "A-HMR-01", "A-HMR-02", "A-HMR-03",
        "A-DST-01", "A-DST-02",
    ),
    "B": (
        "B-RSK-01", "B-ID-01", "B-WIT-01", "B-OPT-01", "B-REG-01",
        "B-FULL-01", "B-FAC-01", "B-SNR-01", "B-RNK-01", "B-RNK-02",
        "B-POL-01", "B-POL-02", "B-POL-03", "B-POL-04", "B-POL-05",
        "B-POST-01", "B-POST-02",
    ),
    "C": (
        "C-NOISE-01", "C-NOISE-02", "C-CLEAN-01", "C-CLEAN-02",
        "C-SING-01", "C-RANK-01", "C-COV-01", "C-CTR-01", "C-CTR-02",
    ),
    "D": ("D-BRIDGE",),
}


def _reference_for_case(case_id: str) -> str:
    for prefix, reference in REFERENCE_BY_CASE.items():
        if case_id.startswith(prefix):
            return reference
    return "approved T0 plan case matrix"


def _case_detail(case: Mapping[str, Any]) -> str:
    details: list[str] = []
    if "dimensions" in case:
        dims = case["dimensions"]
        if "d" in dims:
            details.append(
                f"d={dims.get('d')}, m={dims.get('m')}, q={dims.get('q')}"
            )
        else:
            details.append(f"dimensions={dims}")
    if "points" in case:
        details.append(f"points={case['points']}")
    if "rows" in case and isinstance(case["rows"], list):
        details.append(f"rows={len(case['rows'])}")
    if "tau" in case and isinstance(case["tau"], list):
        details.append(f"law atoms={len(case['tau'])}")
    if "laws" in case and isinstance(case["laws"], list):
        details.append(f"frozen laws={len(case['laws'])}")
    if "binding_pairs" in case and isinstance(case["binding_pairs"], list):
        details.append(f"binding pairs={len(case['binding_pairs'])}")
    mc = case.get("mc")
    if isinstance(mc, Mapping):
        if "final" in mc:
            details.append(f"MC N={mc['final']['checkpoint']}")
        elif mc.get("deduplicated"):
            details.append(f"MC alias={mc.get('alias_of')}")
    return "; ".join(details) if details else "fixed construction/grid in raw JSON"


def _case_rows(
    result: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, str]]:
    if not result:
        return []
    return [
        (
            str(case_id),
            str(case["status"]),
            _case_detail(case),
            _reference_for_case(str(case_id)),
        )
        for case_id, case in result.get("cases", {}).items()
    ]


def _module_execution_failure(
    module: str, error: Exception, *, numerical_failure_is_fail: bool
) -> dict[str, Any]:
    verdict = (
        "FAIL"
        if numerical_failure_is_fail and isinstance(error, NumericalFailure)
        else "INCONCLUSIVE"
    )
    common = {
        "module": module,
        "evidence_type": "fail-closed execution record; no result inferred",
        "claim_status_changed": False,
        "execution_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    if module == "D":
        bridge_verdict = (
            "E2_REVIEW_REQUIRED"
            if isinstance(error, NumericalFailure)
            else "INCONCLUSIVE"
        )
        return {
            **common,
            "D_BRIDGE_INTEGRITY_VERDICT": bridge_verdict,
            "D_APPROXIMATION_ADVISORY": "NOT_EVALUABLE",
            "binding_checks": {},
        }
    return {
        **common,
        "verdict": verdict,
        "cases": {},
        "mc_summaries": {},
    }


def _apply_case_count_gate(
    module: str, result: dict[str, Any], expected: int
) -> None:
    if module == "D":
        actual_ids = (
            (str(result["case_id"]),) if result.get("case_id") is not None else ()
        )
    else:
        actual_ids = tuple(str(item) for item in result.get("cases", {}).keys())
    expected_ids = APPROVED_CASE_IDS[module]
    actual = len(actual_ids)
    count_passed = actual == expected
    id_set_passed = set(actual_ids) == set(expected_ids)
    passed = count_passed and id_set_passed
    result["coverage_check"] = {
        "expected": expected,
        "actual": actual,
        "expected_case_ids": list(expected_ids),
        "actual_case_ids": list(actual_ids),
        "missing_case_ids": sorted(set(expected_ids) - set(actual_ids)),
        "unexpected_case_ids": sorted(set(actual_ids) - set(expected_ids)),
        "count_passed": count_passed,
        "id_set_passed": id_set_passed,
        "passed": passed,
    }
    if (
        module in {"A", "B", "C"}
        and not passed
        and result.get("execution_error") is None
    ):
        result["verdict"] = "FAIL"
    if module == "D" and not passed and result.get("execution_error") is None:
        result["D_BRIDGE_INTEGRITY_VERDICT"] = "E2_REVIEW_REQUIRED"
        result["D_APPROXIMATION_ADVISORY"] = "NOT_EVALUABLE"


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _mc_table_rows(modules: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    lines: list[str] = []
    for module_name in ("A", "B"):
        module = modules.get(module_name)
        if module is None:
            continue
        for case_id, mc in module.get("mc_summaries", {}).items():
            if mc.get("deduplicated"):
                lines.append(
                    f"| {case_id} | {mc['status']} | alias | N/A | N/A | {mc.get('alias_of')} |"
                )
                continue
            final = mc.get("final")
            if not isinstance(final, Mapping):
                lines.append(f"| {case_id} | {mc.get('status')} | N/A | N/A | N/A | missing final summary |")
                continue
            assessment = final["assessment"]
            maximum_half_width = max(
                float(item["five_se_half_width"])
                for item in assessment["checks"].values()
            )
            lines.append(
                f"| {case_id} | {mc['status']} | {final['checkpoint']} | "
                f"{assessment['all_consistent']} | {assessment['all_precision_sufficient']} | "
                f"max 5-SE={maximum_half_width:.6g} |"
            )
    return lines


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    metadata: Mapping[str, Any],
    health: Mapping[str, Any],
    modules: Mapping[str, Mapping[str, Any] | None],
    mathematical_verdict: str,
    bridge_verdict: str,
    advisory: str,
    combined: str,
    stopped_after: str | None,
) -> None:
    lines = [
        "# T0 — Current-Theory Numerical Validation Report",
        "",
        "> This artifact is numerical regression evidence, not a mathematical proof or claim-status update.",
        "",
        "## 1. Layered verdicts and mandatory stop state",
        "",
        f"- `T0_MATHEMATICAL_REGRESSION_VERDICT`: `{mathematical_verdict}`",
        f"- `D_BRIDGE_INTEGRITY_VERDICT`: `{bridge_verdict}`",
        f"- `D_APPROXIMATION_ADVISORY`: `{advisory}`",
        f"- Combined non-substitutive status: `{combined}`",
        f"- Stopped after: `{stopped_after or 'NONE; all approved modules evaluated'}`",
        "- E2 modified: `false`",
        "- Claim status modified: `false`",
        "- Held-out data accessed: `false`",
        "",
        "## 2. Authority and provenance",
        "",
        f"- Plan: `{config['plan_path']}` (`{metadata['plan_sha256']}`)",
        f"- Frozen config: `{metadata['config_path']}` (`{metadata['config_sha256']}`)",
        f"- Protocol: `{config['protocol_id']}`",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Locked source files: `{len(source_lock['files'])}`",
        f"- Executed argv: `{metadata['formal_command_argv']}`",
        f"- Root seeds: `{metadata['rng_stream_registry']['root_seeds']}`",
        "- All locked files passed byte/path-set SHA-256 verification before numerical work.",
        "",
        "| Locked project-relative path | SHA-256 |",
        "|---|---|",
    ]
    for relative_path, digest in sorted(source_lock["files"].items()):
        lines.append(f"| `{relative_path}` | `{digest}` |")
    environment = metadata["environment"]
    lines.extend(
        [
            "",
            "Environment:",
            "",
            f"- Python: `{environment['python']}` ({environment['python_implementation']})",
            f"- NumPy: `{environment['numpy']}`",
            f"- Platform: `{environment['platform']}`",
            f"- Processor: `{environment['processor']}`",
            f"- Dtype: `{environment['dtype']}`; float64 epsilon `{metadata['numpy_float64_epsilon']:.17g}`",
            "- NumPy/BLAS build configuration:",
            "",
            "```text",
            environment["numpy_build_configuration"],
            "```",
            "",
        "## 3. Case coverage matrix",
        "",
            "| Case | Status | Frozen size actually evaluated | Theory reference |",
            "|---|---|---|---|",
        ]
    )
    for module_name in ("A", "B", "C"):
        result = modules.get(module_name)
        if result is None:
            lines.append(f"| {module_name} module | NOT_RUN_DUE_TO_STOP | N/A | dependency stop |")
        else:
            result_rows = _case_rows(result)
            for case_id, status, detail, reference in result_rows:
                lines.append(f"| {case_id} | {status} | {detail} | {reference} |")
            if not result_rows:
                lines.append(
                    f"| {module_name} module | {result.get('verdict', 'INCONCLUSIVE')} | "
                    f"execution failure before case completion | fail-closed module record |"
                )
    d_result = modules.get("D")
    lines.append(
        f"| D-BRIDGE | {d_result['D_BRIDGE_INTEGRITY_VERDICT'] if d_result else 'NOT_RUN_DUE_TO_STOP'} | "
        f"{len(d_result.get('profile_rows', [])) if d_result else 0} schedule points | theory.md FW-v1 → frozen E2 protocol |"
    )

    a_result = modules.get("A")
    b_result = modules.get("B")
    a_route_errors: list[float] = []
    if a_result is not None:
        for case in a_result.get("cases", {}).values():
            routes = case.get("routes")
            if isinstance(routes, Mapping):
                gaps = [
                    float(routes[name]["gap"])
                    for name in ("formula", "direct", "whitened")
                ]
                a_route_errors.extend(abs(left - right) for left in gaps for right in gaps)
    lines.extend(
        [
            "",
            "## 4. A — Theorem 1",
            "",
            f"Module verdict: `{a_result['verdict'] if a_result is not None else 'NOT_RUN'}`. See `A_THEOREM1_RESULTS.json` for all formula/direct/whitened, invariance, Haar/distinct and MC fields.",
            f"Maximum pairwise gap discrepancy across three-route generic cases: `{_format_value(max(a_route_errors) if a_route_errors else None)}`.",
            f"Cases evaluated: `{len(a_result.get('cases', {})) if a_result else 0}`; MC streams evaluated: `{len(a_result.get('mc_summaries', {})) if a_result else 0}`.",
            "",
            "## 5. B — FW-v1",
            "",
            f"Module verdict: `{b_result['verdict'] if b_result is not None else 'NOT_RUN'}`. See `B_FWV1_RESULTS.json` for risk/regret/optimizer/factorization/rank/policy/pre-post and MC fields.",
        ]
    )
    if a_result is not None and "A-INV-02" in a_result.get("cases", {}):
        a_inv_02 = a_result["cases"]["A-INV-02"]
        raw_coordinate = a_inv_02["raw_coordinate_diagnostic"]
        canonical_qr = a_inv_02["canonical_qr"]
        lines[lines.index("## 5. B — FW-v1") : lines.index("## 5. B — FW-v1")] = [
            f"- A-INV-02 amended route: `{a_inv_02['protocol_route']}`; post-result amendment `{a_inv_02['post_result_amendment']}`.",
            f"- A-INV-02 raw diagnostic: `{raw_coordinate['label']}`; M_raw rank `{raw_coordinate['raw_measurement_rank']['rank']}/{raw_coordinate['raw_measurement_rank']['columns']}`, condition `{raw_coordinate['raw_measurement_rank']['condition_number']:.12g}`; G_raw rank `{raw_coordinate['raw_normal_matrix_rank']['rank']}/{raw_coordinate['raw_normal_matrix_rank']['columns']}`, condition `{raw_coordinate['raw_normal_matrix_rank']['condition_number']:.12g}`; raw Cholesky attempted `{a_inv_02['raw_normal_system_cholesky_attempted']}`.",
            f"- A-INV-02 canonical QR: Q rank `{canonical_qr['Q_rank']['rank']}`, T rank `{canonical_qr['T_rank']['rank']}`, reconstruction residual `{canonical_qr['normalized_reconstruction_residual']:.6g}`, raw-column residual `{canonical_qr['raw_column_space_residual']:.6g}`, base/projector residual `{canonical_qr['base_transformed_projector_residual']:.6g}`.",
            f"- A-INV-02 route checks: transformed pairwise `{a_inv_02['transformed_pairwise_checks']}`, direct-risk invariance `{a_inv_02['direct_risk_comparisons']}`, all binding `{all(a_inv_02['binding_checks'].values())}`.",
            "",
        ]

    if b_result is not None and b_result.get("cases"):
        b_cases = b_result["cases"]
        risk = b_cases["B-RSK-01"]
        optimizer = b_cases["B-OPT-01"]
        regret = b_cases["B-REG-01"]
        factor = b_cases["B-FAC-01"]
        snr = b_cases["B-SNR-01"]
        rank_loss = b_cases["B-RNK-01"]
        policy_rows = [b_cases[f"B-POL-0{index}"] for index in range(1, 5)]
        parameter_reference = optimizer["parameter_recovery_reference_summary"]
        dirac = b_cases["B-POL-01"]
        lines.extend(
            [
                f"- Risk grid points: `{risk['points']}`; max ambient↔four-mode error `{risk['max_ambient_four_error']:.6g}`; max ambient↔rational error `{risk['max_ambient_rational_error']:.6g}`; min q `{risk['minimum_q']:.6g}`.",
                f"- Optimizer tau rows: `{len(optimizer['rows'])}`; binding-failed tau count `{len(optimizer['binding_failed_tau'])}`. `{parameter_reference['label']}` is non-binding: misses `{parameter_reference['miss_count']}/{parameter_reference['evaluated_count']}`, max/median absolute rho error `{_format_value(parameter_reference['maximum_absolute_error'])}` / `{_format_value(parameter_reference['median_absolute_error'])}`.",
                f"- B-POL-01 Dirac binding checks: `{dirac.get('binding_checks')}`; risk-difference advantage `{_format_value(dirac.get('risk_difference_advantage'))}`; exact-regret advantage `{_format_value(dirac.get('exact_regret_advantage'))}`; risk-resolution tolerance `{_format_value(dirac.get('risk_resolution_tolerance'))}`; parameter reference `{dirac.get('parameter_recovery_reference')}` (non-binding).",
                f"- Exact regret max identity error: `{regret['max_identity_error']:.6g}`; sampled optimizer/nonoptimizer counts `{regret['sampled_optimizer_count']}/{regret['sampled_nonoptimizer_count']}`.",
                f"- Factorization tau set: `{config['b']['factorization_tau_tokens']}`; max factor residual `{max(row['factorization_residual'] for row in factor['rows']):.6g}`.",
                f"- SNR binding pairs: `{len(snr['binding_pairs'])}`; minimum reported projector distance `{min(row['projector_distance'] for row in snr['binding_pairs']):.6g}`.",
                f"- Rank-loss point: rank `{rank_loss['rank']['rank']}`, label `{rank_loss['structured_result']}`, ordinary risk evaluated `{rank_loss['ordinary_risk_evaluated']}`.",
                "- Static-policy advantages (exact-regret representation): "
                + ", ".join(
                    f"B-POL-0{index}={_format_value(row.get('advantage'))}"
                    for index, row in enumerate(policy_rows, start=1)
                )
                + ".",
            ]
        )

    c_result = modules.get("C")
    lines.extend(
        [
            "",
            "## 6. C — Boundaries and corrective guards",
            "",
            f"Module verdict: `{c_result['verdict'] if c_result else 'NOT_RUN'}`. See `C_BOUNDARY_RESULTS.json` for every fixed endpoint, covariance direction/radius, and corrective guard.",
        ]
    )
    if c_result is not None and c_result.get("cases"):
        c_cases = c_result["cases"]
        covariance_rows = c_cases["C-COV-01"]["rows"]
        noise_distinct = c_cases["C-NOISE-01"]
        noise_original_final = next(
            row for row in noise_distinct["rows"] if row["k"] == 6
        )
        singular = c_cases["C-SING-01"]
        accuracy_eligible = [
            row["k"]
            for row in singular["rows"]
            if row.get("direct_comparison_binding")
        ]
        finite_precision_diagnostics = [
            row["k"]
            for row in singular["rows"]
            if row.get("direct_route")
            == "FINITE_PRECISION_DIRECT_RISK_DIAGNOSTIC"
        ]
        lines.extend(
            [
                f"- Pure-noise distinct profile: `{len(noise_distinct['rows'])}` rows; preserved k=6 error `{noise_original_final['absolute_error']:.12g}`; appended final k=`{noise_distinct['rows'][-1]['k']}` error `{noise_distinct['rows'][-1]['absolute_error']:.12g}`; computed/exact limit `{noise_distinct['computed_subspace_limit']:.17g}` / `{noise_distinct['limit']}`; binding checks `{noise_distinct['binding_checks']}`. Matched final gap `{c_cases['C-NOISE-02']['rows'][-1]['gap']:.6g}`.",
                f"- SPD-clean OLS slope: `{c_cases['C-CLEAN-01']['natural_log_ols_slope']:.12g}` on frozen tail indices `{c_cases['C-CLEAN-01']['tail_indices']}`.",
                f"- Nonuniform clean final ratios: `{c_cases['C-CLEAN-02']['increase_differences']}` increments against coefficients `{c_cases['C-CLEAN-02']['exact_coefficients']}`.",
                f"- Singular-clean interior final gap: `{singular['rows'][-1]['interior_gap']:.12g}`; exact endpoint risks/gap `{singular['exact_endpoint']['risk_retained']}` / `{singular['exact_endpoint']['risk_full']}` / `{singular['exact_endpoint']['gap']}`; accuracy-eligible direct rows `{accuracy_eligible}`; finite-precision diagnostic rows `{finite_precision_diagnostics}`; solver-excluded final route `{singular['rows'][-1]['direct_route']}`; binding checks `{singular['binding_checks']}`; endpoint/interior mixed `{singular['endpoint_and_interior_compared']}`.",
                f"- Covariance-local combinations: `{len(covariance_rows)}`; minimum gain `{min(row['gain'] for row in covariance_rows):.12g}`; maximum nonwhite residual `{max(row['nonwhite_residual'] for row in covariance_rows):.12g}`.",
                f"- Corrective labels: `{c_cases['C-CTR-01']['label']}`; `{c_cases['C-CTR-02']['label']}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## 7. D — E2 bridge",
            "",
            f"Binding integrity: `{bridge_verdict}`.",
            f"Approximation advisory: `{advisory}`.",
            "The advisory thresholds do not determine the A/B/C mathematical-regression verdict and do not automatically require E2 modification.",
        ]
    )
    if d_result is not None and "schedule" in d_result:
        schedule = d_result["schedule"]
        containment = d_result["containment"]
        containment_layers = containment["validation_layers"]
        float64_layer = containment_layers["FLOAT64_ALGEBRAIC_CONTAINMENT"]
        semantic_layer = containment_layers["ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT"]
        hat_semantic = semantic_layer["hat_values"]
        kernel_semantic = semantic_layer["kernel_interpolation"]
        runtime_target = containment_layers["FLOAT32_RUNTIME_VS_IDEAL_FLOAT64_TARGET"]
        approximation = d_result["approximation_metrics"]
        failed_binding = sorted(
            key for key, value in d_result["binding_checks"].items() if not value
        )
        lines.extend(
            [
                f"- Schedule endpoints: tau `{schedule['tau_start']:.12g}` → `{schedule['tau_end']:.12g}`, u `{schedule['u_start']:.12g}` → `{schedule['u_end']:.12g}`.",
                f"- Knots: `{d_result['basis']['knots']}` at timesteps `{d_result['basis']['knot_timesteps']}`.",
                f"- Failed binding fields: `{failed_binding}`.",
                f"- `FLOAT64_ALGEBRAIC_CONTAINMENT`: pass `{float64_layer['passed']}`, max absolute error `{float64_layer['absolute_max_error']:.12g}`, original high-precision tolerance `{float64_layer['absolute_tolerance']:.12g}`.",
                f"- `ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT` hats: pass `{hat_semantic['passed']}`, error `{hat_semantic['absolute_max_error']:.12g}`, budget `{hat_semantic['absolute_error_budget']:.12g}`, scale-ULPs `{hat_semantic['budget_in_scale_ulps']:.12g}`, error/budget `{hat_semantic['error_budget_ratio']:.12g}`.",
                f"- `ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT` kernels: pass `{kernel_semantic['passed']}`, error `{kernel_semantic['absolute_max_error']:.12g}`, budget `{kernel_semantic['absolute_error_budget']:.12g}`, scale-ULPs `{kernel_semantic['budget_in_scale_ulps']:.12g}`, error/budget `{kernel_semantic['error_budget_ratio']:.12g}`.",
                f"- Float32 budget derivation: epsilon `{kernel_semantic['float32_epsilon']:.17g}`, unit roundoff `{kernel_semantic['float32_unit_roundoff']:.17g}`, operations `{kernel_semantic['operation_count']}`, two-path factor `{kernel_semantic['comparison_path_factor']}`, gamma `{kernel_semantic['gamma_n']:.12g}`.",
                f"- `FLOAT32_RUNTIME_VS_IDEAL_FLOAT64_TARGET`: pass `{runtime_target['passed']}`, absolute error `{runtime_target['absolute_max_error']:.12g}`, scale-aware error `{runtime_target['scale_aware_error']:.12g}`, independent-reference displacement `{runtime_target['independent_reference_rounding_displacement']:.12g}`, derived budget `{runtime_target['derived_rounding_budget']:.12g}`, scale-ULPs `{runtime_target['budget_in_scale_ulps']:.12g}`, error/budget `{runtime_target['error_budget_ratio']:.12g}`.",
                f"- Rho max/RMSE: `{approximation['rho_error_max']:.6g}` / `{approximation['rho_error_rmse']:.6g}`; overall capture `{approximation['overall_capture']:.12g}`.",
                f"- Per-coordinate regret max/mean: `{approximation['per_coordinate_regret_max']:.6g}` / `{approximation['per_coordinate_regret_mean']:.6g}`.",
                f"- Integrated risks: exact `{approximation['exact_policy_integrated_risk']:.12g}`, five-knot `{approximation['five_knot_integrated_risk']:.12g}`, best static `{_format_value(approximation['best_static_integrated_risk'])}` (status `{approximation['best_static_computation_status']}`).",
                f"- Best-static diagnostic error: `{approximation.get('best_static_computation_error', 'NONE')}`.",
                f"- `ADVISORY_REFERENCE_THRESHOLDS`: `{d_result['ADVISORY_REFERENCE_THRESHOLDS']}`; checks `{d_result['advisory_checks']}`.",
                "",
                "| Bin | Timesteps | rho max error | rho RMSE | Capture |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in approximation["bins"]:
            lines.append(
                f"| {row['bin']} | [{row['start']},{row['stop_exclusive']}) | "
                f"{row['rho_error_max']:.6g} | {row['rho_error_rmse']:.6g} | {_format_value(row['capture'])} |"
            )

    lines.extend(
        [
            "",
            "## 8. Numerical health and Monte Carlo precision",
            "",
            "All mathematical theory/reference calculations use NumPy float64. D additionally observes the actual frozen E2 TimeAdapter runtime in native torch.float32 and compares it with an independent NumPy-float32 semantic reference under the frozen unit-roundoff/operation budget; the separate ideal float64 algebraic target remains binding at its original high-precision gate. SPD routes are Cholesky-only behind the frozen condition gate; rank diagnostics are SVD-only; full-rank projectors are reduced-QR-only.",
            f"- Solver routes: `{health['solver_routes']}`",
            f"- Module verdicts: `{health['module_verdicts']}`",
            f"- Actual A/B/C/D case counts: `{health['actual_case_counts']}`; frozen-count match `{health['coverage_counts_match']}`.",
            f"- Execution errors: `{health['execution_errors']}`",
            f"- Stop: `{health['stopped_after']}`; NaN/Inf policy: `{health['nan_inf_policy']}`",
            "",
            "| MC case | Status | N | 5-SE consistency | Precision sufficient | Detail |",
            "|---|---|---:|---|---|---|",
        ]
    )
    mc_rows = _mc_table_rows(modules)
    lines.extend(mc_rows or ["| none evaluated | N/A | N/A | N/A | N/A | dependency stop |"])

    unexpected: list[str] = []
    for name in ("A", "B", "C"):
        module = modules.get(name)
        if module is None:
            continue
        for case_id, case in module.get("cases", {}).items():
            if case["status"] not in {
                "PASS",
                "EXPECTED_OUT_OF_DOMAIN",
                "EXPECTED_ILL_CONDITIONED",
            }:
                unexpected.append(f"{case_id}={case['status']}")
    d_failed = []
    if d_result is not None:
        d_failed = [
            key for key, value in d_result.get("binding_checks", {}).items() if not value
        ]
    lines.extend(
        [
            "",
            "## 9. Failures, negative results, and deviations",
            "",
            f"- Unexpected/inconclusive A/B/C cases: `{unexpected}`.",
            f"- Failed D binding fields: `{d_failed}`.",
            f"- Fail-closed stop field: `{stopped_after or 'NONE'}`.",
            "- The fixed witness retains its pre-registered low-SNR negative comparison; Appendix C outputs remain refutation guards, not positive non-Gaussian claims.",
            "- No case, seed, draw, direction, grid point, optimizer, threshold, knot, coefficient, or time region was replaced or filtered after observing output.",
            "- Protocol deviations: `NONE AUTOMATICALLY APPLIED`; any execution error is listed above and left for human review.",
            "",
            "## 10. Evidence boundary",
            "",
            "- A assumes the finite-dimensional jointly Gaussian squared-loss setting, unrestricted Bayes predictors, the stated full-column-rank measurement condition, and interior noise parameter `b>0` except in separately labelled boundary cases.",
            "- B is confined to the exact Gaussian FW-v1 construction and the domains stated in Theorem 2 and Corollaries 2.1–2.2; the rank-loss point is an expected out-of-domain case, not an ordinary-risk evaluation.",
            "- C checks only the frozen boundary limits, covariance-local sanity construction, and Appendix C corrective guards; it does not create a positive non-Gaussian theory.",
            "- D evaluates only the scalar FW-v1 slice against the frozen E2 schedule and five-knot adapter family; it does not establish architecture-wide adequacy.",
            "- Finite-grid agreement does not prove a universal theorem.",
            "- Numerical optimizer agreement does not prove global uniqueness.",
            "- Monte Carlo agreement does not prove Gaussian conditioning identities.",
            "- The frozen post-risk comparison does not re-prove the full deterministic Borel-map statement.",
            "- FW-v1 bridge results do not validate the CIFAR-10 law, network trainability, optimization attainability, FID, sample quality, causality, or architecture-wide generalization.",
            "- Appendix C outputs are corrective refutation guards, not positive non-Gaussian theory.",
            "- No numerical result in T0 upgrades a mathematical claim or modifies `notes/claims.md`.",
            "",
            "## 11. Raw artifact index and reproduction",
            "",
            "- `RUN_METADATA.json`",
            "- `A_THEOREM1_RESULTS.json` when A ran",
            "- `B_FWV1_RESULTS.json` when B ran",
            "- `C_BOUNDARY_RESULTS.json` when C ran",
            "- `D_E2_BRIDGE_RESULTS.json` and `D_E2_BRIDGE_PROFILE.csv` when D ran successfully",
            "- `MC_SUMMARIES.json`",
            "- `NUMERICAL_HEALTH.json`",
            "",
            "Reproduce only after explicit human authorization:",
            "",
            "```powershell",
            "uv run python experiments/theory_validation/t0/run_t0.py --config experiments/theory_validation/t0/t0_config.json",
            "```",
            "",
            "## 12. Human decision required",
            "",
            "No downstream action is automatic. A human must review this report and decide whether T0 is accepted and whether any separate E2 or mathematical review is warranted.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def _combined_status(math_verdict: str, bridge_verdict: str) -> str:
    if math_verdict == "PASS" and bridge_verdict == "PASS":
        return "T0_BINDING_CHECKS_PASS"
    if math_verdict == "PASS" and bridge_verdict == "E2_REVIEW_REQUIRED":
        return "T0_MATHEMATICAL_REGRESSION_PASS_E2_REVIEW_REQUIRED"
    if math_verdict == "FAIL":
        return "T0_BINDING_CHECKS_FAIL"
    return "T0_INCONCLUSIVE"


def _rng_stream_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    a_synthetic_root = int(config["seeds"]["a_synthetic"])
    a_mc_root = int(config["seeds"]["a_monte_carlo"])
    b_mc_root = int(config["seeds"]["b_monte_carlo"])
    synthetic = []
    for spec in config["a"]["generic_cases"]:
        for tag in (
            "sigma-basis",
            "target-basis",
            "measurement-left",
            "measurement-right",
        ):
            synthetic.append(
                {
                    "case_id": spec["id"],
                    "stream_tag": tag,
                    "derived_seed128": stable_seed(a_synthetic_root, spec["id"], tag),
                }
            )
    for case_id, tags in (
        ("A-INV-01", ("coordinate-right",)),
        ("A-INV-02", ("coordinate-left", "coordinate-right")),
    ):
        for tag in tags:
            synthetic.append(
                {
                    "case_id": case_id,
                    "stream_tag": tag,
                    "derived_seed128": stable_seed(a_synthetic_root, case_id, tag),
                }
            )
    a_mc_ids = [
        *[str(spec["id"]) for spec in config["a"]["generic_cases"]],
        "A-HMR-00",
        "A-HMR-01",
        "A-DST-01",
        "A-DST-02",
    ]
    a_mc = [
        {
            "case_id": case_id,
            "stream_tag": "mc-samples",
            "derived_seed128": stable_seed(a_mc_root, case_id, "mc-samples"),
        }
        for case_id in a_mc_ids
    ]
    b_mc = []
    for tau_token in config["b"]["mc_tau_tokens"]:
        for policy_tag in config["b"]["mc_policy_tags"]:
            case_id = f"B-MC/{tau_token}/{policy_tag}"
            b_mc.append(
                {
                    "case_id": case_id,
                    "stream_tag": "mc-samples",
                    "derived_seed128": stable_seed(b_mc_root, case_id, "mc-samples"),
                }
            )
    return {
        "derivation": "SHA256(T0-RNG-v1, root_seed, case_id, stream_tag) first 128 bits little-endian",
        "root_seeds": dict(config["seeds"]),
        "a_synthetic": synthetic,
        "a_monte_carlo": a_mc,
        "b_monte_carlo": b_mc,
    }


def formal_run(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_lock = verify_source_lock(config)
    run_dir = new_run_directory(config)
    (run_dir / "figures").mkdir(exist_ok=False)
    metadata = {
        "protocol_id": config["protocol_id"],
        "plan_path": config["plan_path"],
        "plan_sha256": sha256_file(PROJECT_ROOT / str(config["plan_path"])),
        "config_path": "experiments/theory_validation/t0/t0_config.json",
        "config_sha256": sha256_file(Path(__file__).resolve().parent / "t0_config.json"),
        "formal_command_argv": list(sys.argv),
        "environment": environment_metadata(),
        "numpy_float64_epsilon": float(np.finfo(np.float64).eps),
        "source_lock": source_lock,
        "rng_stream_registry": _rng_stream_registry(config),
        "approved_case_counts": {
            name: len(case_ids) for name, case_ids in APPROVED_CASE_IDS.items()
        },
        "approved_case_ids": {
            name: list(case_ids) for name, case_ids in APPROVED_CASE_IDS.items()
        },
        "formal_run_explicitly_started_by_human_command": True,
        "heldout_accessed": False,
        "network_training_performed": False,
        "gpu_used": False,
    }
    write_json(run_dir / "RUN_METADATA.json", metadata)

    modules: dict[str, Mapping[str, Any] | None] = {"A": None, "B": None, "C": None, "D": None}
    stopped_after: str | None = None
    try:
        modules["A"] = run_theorem1(config)
    except Exception as error:
        modules["A"] = _module_execution_failure(
            "A", error, numerical_failure_is_fail=True
        )
    _apply_case_count_gate("A", modules["A"], metadata["approved_case_counts"]["A"])
    write_json(run_dir / "A_THEOREM1_RESULTS.json", modules["A"])
    if modules["A"]["verdict"] != "PASS":
        stopped_after = "A"
    if stopped_after is None:
        try:
            modules["B"] = run_fwv1(config)
        except Exception as error:
            modules["B"] = _module_execution_failure(
                "B", error, numerical_failure_is_fail=True
            )
        _apply_case_count_gate("B", modules["B"], metadata["approved_case_counts"]["B"])
        write_json(run_dir / "B_FWV1_RESULTS.json", modules["B"])
        if modules["B"]["verdict"] != "PASS":
            stopped_after = "B"
    if stopped_after is None:
        try:
            modules["C"] = run_boundaries(config)
        except Exception as error:
            modules["C"] = _module_execution_failure(
                "C", error, numerical_failure_is_fail=True
            )
        _apply_case_count_gate("C", modules["C"], metadata["approved_case_counts"]["C"])
        write_json(run_dir / "C_BOUNDARY_RESULTS.json", modules["C"])
        if modules["C"]["verdict"] != "PASS":
            stopped_after = "C"
    if stopped_after is None:
        try:
            modules["D"] = run_bridge(config)
        except Exception as error:
            modules["D"] = _module_execution_failure(
                "D", error, numerical_failure_is_fail=False
            )
        _apply_case_count_gate("D", modules["D"], metadata["approved_case_counts"]["D"])
        write_json(run_dir / "D_E2_BRIDGE_RESULTS.json", modules["D"])
        if "profile_rows" in modules["D"]:
            write_profile_csv(
                run_dir / "D_E2_BRIDGE_PROFILE.csv", modules["D"]["profile_rows"]
            )
        if modules["D"]["D_BRIDGE_INTEGRITY_VERDICT"] != "PASS":
            stopped_after = "D_STRUCTURE"

    coverage_checks: dict[str, bool | str] = {}
    for name in ("A", "B", "C"):
        module_result = modules[name]
        coverage_checks[name] = (
            bool(module_result["coverage_check"]["passed"])
            if module_result is not None
            else "NOT_RUN_DUE_TO_STOP"
        )
    coverage_checks["D"] = (
        bool(modules["D"]["coverage_check"]["passed"])
        if modules["D"] is not None
        else "NOT_RUN_DUE_TO_STOP"
    )

    math_statuses: list[str] = []
    for name in ("A", "B", "C"):
        module_result = modules[name]
        math_statuses.append(
            str(module_result["verdict"])
            if module_result is not None
            else "INCONCLUSIVE"
        )
    if any(status == "FAIL" for status in math_statuses):
        mathematical = "FAIL"
    elif all(status == "PASS" for status in math_statuses):
        mathematical = "PASS"
    else:
        mathematical = "INCONCLUSIVE"
    bridge = (
        str(modules["D"]["D_BRIDGE_INTEGRITY_VERDICT"])
        if modules["D"] is not None
        else "INCONCLUSIVE"
    )
    advisory = (
        str(modules["D"]["D_APPROXIMATION_ADVISORY"])
        if modules["D"] is not None
        else "NOT_EVALUABLE"
    )
    combined = _combined_status(mathematical, bridge)
    mc = {
        "A": modules["A"]["mc_summaries"] if modules["A"] is not None else None,
        "B": modules["B"]["mc_summaries"] if modules["B"] is not None else None,
    }
    write_json(run_dir / "MC_SUMMARIES.json", mc)
    health = {
        "dtype": "float64",
        "bridge_runtime_observation_dtype": "torch.float32 (frozen E2 implementation only)",
        "machine_epsilon": float(np.finfo(np.float64).eps),
        "solver_routes": {
            "spd": "Cholesky plus project-local deterministic forward/back triangular substitution",
            "condition_gate": config["tolerances"]["condition_gate"],
            "rank_condition": "SVD",
            "full_rank_projector": "reduced QR",
            "fallback_used": False,
        },
        "module_verdicts": {
            name: (
                value.get("verdict", value.get("D_BRIDGE_INTEGRITY_VERDICT"))
                if value is not None
                else "NOT_RUN"
            )
            for name, value in modules.items()
        },
        "execution_errors": {
            name: value.get("execution_error")
            for name, value in modules.items()
            if value is not None and value.get("execution_error") is not None
        },
        "actual_case_counts": {
            "A": len(modules["A"].get("cases", {})) if modules["A"] is not None else 0,
            "B": len(modules["B"].get("cases", {})) if modules["B"] is not None else 0,
            "C": len(modules["C"].get("cases", {})) if modules["C"] is not None else 0,
            "D": 1 if modules["D"] is not None else 0,
        },
        "stopped_after": stopped_after,
        "nan_inf_policy": "any binding nonfinite value is FAIL",
    }
    health["coverage_counts_match"] = coverage_checks
    write_json(run_dir / "NUMERICAL_HEALTH.json", health)
    _write_report(
        run_dir / "T0_THEORY_NUMERICAL_VALIDATION_REPORT.md",
        config=config,
        source_lock=source_lock,
        metadata=metadata,
        health=health,
        modules=modules,
        mathematical_verdict=mathematical,
        bridge_verdict=bridge,
        advisory=advisory,
        combined=combined,
        stopped_after=stopped_after,
    )
    summary = {
        "run_directory": run_dir,
        "T0_MATHEMATICAL_REGRESSION_VERDICT": mathematical,
        "D_BRIDGE_INTEGRITY_VERDICT": bridge,
        "D_APPROXIMATION_ADVISORY": advisory,
        "combined_status": combined,
        "stopped_after": stopped_after,
    }
    return run_dir, summary


def smoke_run(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "A": smoke_theorem1(config),
        "B": smoke_fwv1(config),
        "C": smoke_boundaries(config),
        "D": smoke_bridge(),
    }
    return {
        "mode": "IMPLEMENTATION_ONLY_SMOKE_NO_FORMAL_CASE_MATRIX",
        "results_directory_created": False,
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }


def main() -> None:
    parser = base_parser(__doc__ or "T0 runner")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.smoke:
            result = smoke_run(config)
            print(result)
            if not result["passed"]:
                raise SystemExit(1)
            return
        run_dir, summary = formal_run(config)
    except SourceDriftError as error:
        print(f"SOURCE_DRIFT_REVIEW_REQUIRED: {error}")
        raise SystemExit(2) from error
    print(run_dir)
    print(summary)
    if summary["combined_status"] != "T0_BINDING_CHECKS_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
