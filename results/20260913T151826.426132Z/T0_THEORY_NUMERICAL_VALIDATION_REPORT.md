# T0 — Current-Theory Numerical Validation Report

> This artifact is numerical regression evidence, not a mathematical proof or claim-status update.

## 1. Layered verdicts and mandatory stop state

- `T0_MATHEMATICAL_REGRESSION_VERDICT`: `PASS`
- `D_BRIDGE_INTEGRITY_VERDICT`: `PASS`
- `D_APPROXIMATION_ADVISORY`: `BRIDGE_APPROXIMATION_REVIEW_RECOMMENDED`
- Combined non-substitutive status: `T0_BINDING_CHECKS_PASS`
- Stopped after: `NONE; all approved modules evaluated`
- E2 modified: `false`
- Claim status modified: `false`
- Held-out data accessed: `false`

## 2. Authority and provenance

- Plan: `T0_THEORY_NUMERICAL_VALIDATION_PLAN.md` (`bbeeb15b09f1117db5d2dd0d3dddc32a86f8084b585c2319f589d04e6b0f7c41`)
- Frozen config: `experiments/theory_validation/t0/t0_config.json` (`9481792706d0f3ec65ec9e65b447ad05900218310fac67ce21f3e8b1b425b0cf`)
- Protocol: `T0-CURRENT-THEORY-NUMERICAL-VALIDATION-v1.4-D-FLOAT32-RUNTIME-RESOLUTION`
- Project root: `D:\1research\ICLR2027`
- Locked source files: `41`
- Executed argv: `['experiments/theory_validation/t0/run_t0.py', '--config', 'experiments/theory_validation/t0/t0_config.json']`
- Root seeds: `{'a_synthetic': 2026091200, 'a_monte_carlo': 2026091202, 'b_monte_carlo': 2026091201}`
- All locked files passed byte/path-set SHA-256 verification before numerical work.

| Locked project-relative path | SHA-256 |
|---|---|
| `T0_THEORY_NUMERICAL_VALIDATION_PLAN.md` | `bbeeb15b09f1117db5d2dd0d3dddc32a86f8084b585c2319f589d04e6b0f7c41` |
| `experiments/controlled_protocol/M1_THEORY_ALIGNED_PROTOCOL.md` | `6d55bfe0a3e50f13bcc3ac5b6871497d6bf252e749bd56a55ff27e3604714019` |
| `experiments/controlled_protocol/e1/e1_impl.py` | `db2e5643d3f95f0bdfa31d484c70a288107ba48e2a3fae4305e22698314e9318` |
| `experiments/controlled_protocol/e2/e2_config.yaml` | `02ae613f16491de9ff41bca04f9fb3ff26bc0951dabf81f2ee3032bb3fc8772a` |
| `experiments/controlled_protocol/e2/e2_impl.py` | `95d01c21f0571123a3e512dda35f3fb6ac5050399c294d1530e0e9b409997a89` |
| `experiments/theory_validation/t0/T0_PROTOCOL_AMENDMENT_A_INV_02.md` | `3b00f64e79b84c3b6f178cd1a4f4a204ec1b1ba14b75ad2c94bc943d04a457bf` |
| `experiments/theory_validation/t0/T0_PROTOCOL_AMENDMENT_B_OPT_FLOAT64_IDENTIFIABILITY.md` | `6810c3ebcabcbd975f83b68ed1db20db698ee962f0e15e4d1f26990dceed428f` |
| `experiments/theory_validation/t0/T0_PROTOCOL_AMENDMENT_C_BOUNDARY_NUMERICAL_RESOLUTION.md` | `a637549971337a1769e39594dbccd810f7e3ec363c89db7b7a8d3b3db7c356d8` |
| `experiments/theory_validation/t0/T0_PROTOCOL_AMENDMENT_D_FLOAT32_RUNTIME_RESOLUTION.md` | `6b292550bf5b42ad5c1b5a35889b6754b826f82c139ded6f53efd865b73002e1` |
| `notes/claims.md` | `7c6f4f54910ecabcc029e46987f2c731112ae0a30a400e3bfdf47dbb3cbeb5f4` |
| `proofs/gaussian_endpoint_boundaries/INDEPENDENT_REVIEW.md` | `28d00064f3d6a5369650cccf2e623e2be1f0fa712822102d8a6717999d8a739a` |
| `proofs/gaussian_endpoint_boundaries/PROOF_PACKAGE.md` | `d48fe04c6ab36b68d6c09e4e11df7476cf8ece7e9aeac0b381398a3f1c56effd` |
| `proofs/gaussian_interface_gap/AGGREGATE_INDEPENDENT_REVIEW.md` | `086b0467ce60c5108a68a6c2eddcd0af6038fa226dc71785020b527164b902d0` |
| `proofs/gaussian_interface_gap/PROOF_AUDIT.html` | `7787e5691e9ae11218943804ba5bf17f65c670f77ed552bc7fe37f354c656c08` |
| `proofs/gaussian_interface_gap/PROOF_AUDIT.html.review.json` | `5c8be28aae0c699744ba15e01bb25e2b4e3ee57526c8b1e9443ca02346fcebf1` |
| `proofs/gaussian_interface_gap/PROOF_AUDIT.json` | `29567b7d9034cc787e1fb1d746cc9ea37c78fa763b607bdd0d505a42617b64c9` |
| `proofs/gaussian_interface_gap/PROOF_AUDIT.md` | `a8330a07e2f70661a9e6e87e662e9a9cd17910da176f8046c15c5a73263b0ec8` |
| `proofs/gaussian_interface_gap/PROOF_CHECK_STATE.json` | `f3d931cedb12e7c488d67f2aec6364be9d25fd9777d63d2976ec1201b081f2ad` |
| `proofs/gaussian_interface_gap/PROOF_PACKAGE.md` | `86475ee4385590c6e35c61da56fa58516a352f242c21375da8f2df2ebf006eac` |
| `proofs/gaussian_interface_gap/PROOF_SKELETON.md` | `5cb067cd3e705bfc8a0b81d1901ba3e53093a511d1e68bcc86dcfc4ce8571be7` |
| `proofs/gaussian_interface_gap/R2_DEEPSEEK_REVIEW.md` | `7261b4d46e5e549a4e22665e42c704a5240893cba467e6f3d23ad0876a725b10` |
| `proofs/gaussian_interface_gap/R2_RECONCILIATION.md` | `1db71268ec351e69bf1135676d09fa3d8d91e4bfe40a0c59279b5752d94c0b76` |
| `proofs/gaussian_interface_gap/proof_audit_report.tex` | `47e6df8937f2ecc439d7a289e7e20bb2fa87ace2c3ba370b09677f21a170ebee` |
| `proofs/h01_fwv1_covariance_robustness/INDEPENDENT_REVIEW.md` | `462a3399a34c20f72dea6f04585fb79f04ba82ccf792c7fe765e88c7e6a736f4` |
| `proofs/h01_fwv1_covariance_robustness/PROOF_PACKAGE.md` | `8c570d2f33d1400c46810c2e10a99ac3abe4514975c2d9630b46e8b0345b7d8f` |
| `proofs/h01_fwv1_final_theory_closure/EXACT_OUTPUT.txt` | `6dd3f89cbbeb8a210d49b5ab5e348abdbd75a2fa7e978b89e500f0dd98310af3` |
| `proofs/h01_fwv1_final_theory_closure/INDEPENDENT_REVIEW.md` | `203b8a8c3cc1bf9b5da2465628078ff15b2ea2a2eae7eaf6a2e79631aba57e8f` |
| `proofs/h01_fwv1_final_theory_closure/PROOF_PACKAGE.md` | `17daee1843fc57075e19a87a1bbc5c56add4247227ec935db13593ba6e5cac38` |
| `proofs/h01_fwv1_final_theory_closure/exact_check.py` | `d6a8c3b5cc8db9d7b9fa5da127e0cd36ee0bf3de7bf7d4d9eb59c842372a477d` |
| `proofs/h01_fwv1_three_tap_integrated_policy/INDEPENDENT_REVIEW.md` | `7840e272a9c9d9c9dd87baa604ed7367b42139efad2d24f07fc050505ee06cf1` |
| `proofs/h01_fwv1_three_tap_integrated_policy/PROOF_PACKAGE.md` | `8d8cbe7cd8454903ff6e63519c1dbf2c72a694fa0bfa93bd9d7f77a8b989871f` |
| `proofs/h01_fwv1_three_tap_snr_optimization/EXACT_OUTPUT.txt` | `967bf1f2bfb661ff106133462538a06a9728b307eb65579aba35c75867d30dd3` |
| `proofs/h01_fwv1_three_tap_snr_optimization/INDEPENDENT_REVIEW.md` | `576f45d077603ae85ae64bd3b63349c34ce393bb74ef3b7a8b83d5384d1d5900` |
| `proofs/h01_fwv1_three_tap_snr_optimization/PROOF_PACKAGE.md` | `a14507d94c9d83434d4064c89b36c9d158320c5bc5be71288780948f0e876116` |
| `proofs/h01_fwv1_three_tap_snr_optimization/exact_check.py` | `f8f69de1a7e8c02b92c383372ae52f2e4e733daff3cf5a58ee098dbaa67c33db` |
| `proofs/h02_post_projection_information_boundary/INDEPENDENT_REVIEW.md` | `ba817e7516d15a2676911fffbf0e789e69247044b52ccc5a966474e697f7ee15` |
| `proofs/h02_post_projection_information_boundary/PROOF_PACKAGE.md` | `b3e15c0892f7eaac2400797cbf9e3068dc2bb0da31a642dc8edbbc018919d50e` |
| `proofs/haar_matched_specialization/INDEPENDENT_REVIEW.md` | `6649362e19847b651dbf3ad33ca71767bc402edfc0ac83c0db14d0cc8e9abe07` |
| `proofs/haar_matched_specialization/PROOF_PACKAGE.md` | `f6abd98fffc11085aceb70758d035f1c110816c5eafe65ae33f382c5e31e5e2e` |
| `research_proposal.md` | `9840c1c12475a766d246f1f1a3593fb7d8ccedae51c52ad498d115ed95c4913e` |
| `theory.md` | `4a7f7417aaf6715441fc9bb369cd1671e61015694180756cc4b5384426ec1ae4` |

Environment:

- Python: `3.11.15` (CPython)
- NumPy: `1.26.4`
- Platform: `Windows-10-10.0.19045-SP0`
- Processor: `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`
- Dtype: `float64`; float64 epsilon `2.2204460492503131e-16`
- NumPy/BLAS build configuration:

```text
Build Dependencies:
  blas:
    detection method: pkgconfig
    found: true
    include directory: /c/opt/64/include
    lib directory: /c/opt/64/lib
    name: openblas64
    openblas configuration: USE_64BITINT=1 DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS=
      NO_LAPACK= NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP= SKYLAKEX MAX_THREADS=2
    pc file directory: C:/opt/64/lib/pkgconfig
    version: 0.3.23.dev
  lapack:
    detection method: internal
    found: true
    include directory: unknown
    lib directory: unknown
    name: dep2270588361616
    openblas configuration: unknown
    pc file directory: unknown
    version: 1.26.4
Compilers:
  c:
    commands: cl
    linker: link
    name: msvc
    version: 19.29.30153
  c++:
    commands: cl
    linker: link
    name: msvc
    version: 19.29.30153
  cython:
    commands: cython
    linker: cython
    name: cython
    version: 3.0.8
Machine Information:
  build:
    cpu: x86_64
    endian: little
    family: x86_64
    system: windows
  host:
    cpu: x86_64
    endian: little
    family: x86_64
    system: windows
Python Information:
  path: C:\Users\runneradmin\AppData\Local\Temp\cibw-run-j442zwj6\cp311-win_amd64\build\venv\Scripts\python.exe
  version: '3.11'
SIMD Extensions:
  baseline:
  - SSE
  - SSE2
  - SSE3
  found:
  - SSSE3
  - SSE41
  - POPCNT
  - SSE42
  - AVX
  - F16C
  - FMA3
  - AVX2
  - AVX512F
  - AVX512CD
  - AVX512_SKX
  - AVX512_CLX
  - AVX512_CNL
  - AVX512_ICL
```

## 3. Case coverage matrix

| Case | Status | Frozen size actually evaluated | Theory reference |
|---|---|---|---|
| A-GEN-01 | PASS | d=4, m=1, q=2; MC N=262144 | Theorem 1(b); Appendix A.1 |
| A-GEN-02 | PASS | d=8, m=3, q=4; MC N=524288 | Theorem 1(b); Appendix A.1 |
| A-GEN-03 | PASS | d=16, m=4, q=8; MC N=524288 | Theorem 1(b); Appendix A.1 |
| A-GEN-04 | PASS | d=16, m=8, q=16; MC N=524288 | Theorem 1(b); Appendix A.1 |
| A-GEN-05 | PASS | d=32, m=8, q=12; MC N=1048576 | Theorem 1(b); Appendix A.1 |
| A-INV-01 | PASS | d=8, m=3, q=4 | Theorem 1(b), measurement-coordinate invariance |
| A-INV-02 | PASS | d=8, m=3, q=4 | Theorem 1(b), measurement-coordinate invariance |
| A-HMR-00 | PASS | d=8, m=4, q=4; MC N=262144 | Corollary 1.1; Appendix A.1.4 |
| A-HMR-01 | PASS | d=8, m=4, q=4; MC N=262144 | Corollary 1.1; Appendix A.1.4 |
| A-HMR-02 | PASS | d=16, m=4, q=4 | Corollary 1.1; Appendix A.1.4 |
| A-HMR-03 | PASS | dimensions={'pair_average_d': 8, 'four_pixel_average_d': 16} | Corollary 1.1; Appendix A.1.4 |
| A-DST-01 | PASS | d=2, m=1, q=1; MC N=262144 | Appendix B.1 distinct input/target extension |
| A-DST-02 | PASS | d=2, m=1, q=1; MC N=262144 | Appendix B.1 distinct input/target extension |
| B-RSK-01 | PASS | points=7085 | Theorem 2(i); Appendix A.2.2-A.2.3 |
| B-ID-01 | PASS | rows=242 | Theorem 2(i); fixed Haar/identity PSG |
| B-WIT-01 | PASS | rows=242 | Appendix A.2.8 witness comparison |
| B-OPT-01 | PASS | rows=242 | Theorem 2(ii); Appendix A.2.4 |
| B-REG-01 | PASS | fixed construction/grid in raw JSON | Theorem 2(ii), exact regret identity |
| B-FULL-01 | PASS | rows=242 | Theorem 2(iii); Appendix A.2.6-A.2.7 |
| B-FAC-01 | PASS | rows=3 | Theorem 2(iii); Appendix A.2.7 |
| B-SNR-01 | PASS | binding pairs=4 | Theorem 2(iv); Appendix A.2.5 |
| B-RNK-01 | EXPECTED_OUT_OF_DOMAIN | fixed construction/grid in raw JSON | theory.md §3.2; Appendix A.2.1/B.2.5 |
| B-RNK-02 | PASS | rows=6 | theory.md §3.2; Appendix A.2.1/B.2.5 |
| B-POL-01 | PASS | law atoms=1 | Corollary 2.1; Appendix A.3 |
| B-POL-02 | PASS | law atoms=2 | Corollary 2.1; Appendix A.3 |
| B-POL-03 | PASS | law atoms=5 | Corollary 2.1; Appendix A.3 |
| B-POL-04 | PASS | law atoms=1000 | Corollary 2.1; Appendix A.3 |
| B-POL-05 | PASS | frozen laws=3 | Corollary 2.1; Appendix A.3 |
| B-POST-01 | PASS | rows=12 | Corollary 2.2; Appendix A.4 |
| B-POST-02 | PASS | rows=242 | Corollary 2.2; Appendix A.4 |
| C-NOISE-01 | PASS | rows=8 | Appendix B.2.2 |
| C-NOISE-02 | PASS | rows=6 | Appendix B.2.2 |
| C-CLEAN-01 | PASS | rows=6 | Appendix B.2.3 |
| C-CLEAN-02 | PASS | fixed construction/grid in raw JSON | Appendix B.2.3 |
| C-SING-01 | PASS | rows=6 | Appendix B.2.4 |
| C-RANK-01 | PASS | fixed construction/grid in raw JSON | Appendix B.2.5 |
| C-COV-01 | PASS | rows=48 | Appendix B.3 |
| C-CTR-01 | PASS | fixed construction/grid in raw JSON | Appendix C corrective counterexample |
| C-CTR-02 | PASS | fixed construction/grid in raw JSON | Appendix C corrective counterexample |
| D-BRIDGE | PASS | 1000 schedule points | theory.md FW-v1 → frozen E2 protocol |

## 4. A — Theorem 1

Module verdict: `PASS`. See `A_THEOREM1_RESULTS.json` for all formula/direct/whitened, invariance, Haar/distinct and MC fields.
Maximum pairwise gap discrepancy across three-route generic cases: `3.10862446895e-14`.
Cases evaluated: `13`; MC streams evaluated: `9`.

- A-INV-02 amended route: `A-INV-02-ONLY-CANONICAL-QR-v1.1`; post-result amendment `True`.
- A-INV-02 raw diagnostic: `EXPECTED_RAW_COORDINATE_ILL_CONDITIONING`; M_raw rank `4/4`, condition `436488.060594`; G_raw rank `3/4`, condition `1.57641768496e+12`; raw Cholesky attempted `False`.
- A-INV-02 canonical QR: Q rank `4`, T rank `4`, reconstruction residual `4.24883e-16`, raw-column residual `3.57082e-16`, base/projector residual `7.33746e-12`.
- A-INV-02 route checks: transformed pairwise `{'formula_vs_direct': True, 'formula_vs_whitened': True, 'direct_vs_whitened': True}`, direct-risk invariance `{'risk_full': True, 'risk_measurement': True, 'gap': True}`, all binding `True`.

## 5. B — FW-v1

Module verdict: `PASS`. See `B_FWV1_RESULTS.json` for risk/regret/optimizer/factorization/rank/policy/pre-post and MC fields.
- Risk grid points: `7085`; max ambient↔four-mode error `1.81544e-12`; max ambient↔rational error `1.81721e-12`; min q `3.50142`.
- Optimizer tau rows: `242`; binding-failed tau count `0`. `PARAMETER_RECOVERY_REFERENCE_CHECK` is non-binding: misses `123/242`, max/median absolute rho error `2.2306879629e-05` / `2.50873237462e-08`.
- B-POL-01 Dirac binding checks: `{'static_solver_completed_and_finite': True, 'static_risk_equals_exact_policy_risk': True, 'risk_difference_advantage_theoretical_zero': True, 'exact_regret_advantage_within_risk_resolution': True, 'dense_crosscheck': True}`; risk-difference advantage `0`; exact-regret advantage `8.22068553153e-16`; risk-resolution tolerance `1.11e-08`; parameter reference `{'label': 'PARAMETER_RECOVERY_REFERENCE_CHECK', 'binding': False, 'absolute_error': 2.1203035743821985e-08, 'threshold': 2e-08, 'passed': False}` (non-binding).
- Exact regret max identity error: `5.32907e-15`; sampled optimizer/nonoptimizer counts `242/6843`.
- Factorization tau set: `['1e-4', '1', '1e4']`; max factor residual `1.76156e-16`.
- SNR binding pairs: `4`; minimum reported projector distance `0.0285844`.
- Rank-loss point: rank `12`, label `OUT_OF_DOMAIN_RANK_12`, ordinary risk evaluated `False`.
- Static-policy advantages (exact-regret representation): B-POL-01=8.22068553153e-16, B-POL-02=0.0210125508933, B-POL-03=0.0260229998506, B-POL-04=0.0311795648489.

## 6. C — Boundaries and corrective guards

Module verdict: `PASS`. See `C_BOUNDARY_RESULTS.json` for every fixed endpoint, covariance direction/radius, and corrective guard.
- Pure-noise distinct profile: `8` rows; preserved k=6 error `2.50022225146e-12`; appended final k=`8` error `2.22044604925e-16`; computed/exact limit `0.99999999999999978` / `1.0`; binding checks `{'computed_limit_matches_exact_limit': True, 'final_point_absolute_tolerance': True, 'profile_finite_nonnegative': True}`. Matched final gap `5e-25`.
- SPD-clean OLS slope: `1.99999976088` on frozen tail indices `[3, 4, 5, 6]`.
- Nonuniform clean final ratios: `[89.99999998038129, 899.9999980220625]` increments against coefficients `[9.0, 99.0, 999.0]`.
- Singular-clean interior final gap: `0.499999999999`; exact endpoint risks/gap `1.0` / `1.0` / `0.0`; accuracy-eligible direct rows `[1, 2, 3]`; finite-precision diagnostic rows `[4, 5]`; solver-excluded final route `ILL_CONDITIONED_FULL_RANK`; binding checks `{'interior_profile_finite_nonnegative': True, 'interior_final_limit': True, 'endpoint_risks_equal_one': True, 'endpoint_gap_zero': True, 'direct_numeric_health': True, 'accuracy_eligible_direct_checks': True, 'solver_admissibility_boundary': True}`; endpoint/interior mixed `False`.
- Covariance-local combinations: `48`; minimum gain `0.166655750016`; maximum nonwhite residual `3.38100954559`.
- Corrective labels: `CORRECTIVE_REFUTATION_GUARD_NOT_POSITIVE_NONGAUSSIAN_THEORY`; `CORRECTIVE_CLASS_MISMATCH_GUARD_NOT_REPRESENTATION_LOSS`.

## 7. D — E2 bridge

Binding integrity: `PASS`.
Approximation advisory: `BRIDGE_APPROXIMATION_REVIEW_RECOMMENDED`.
The advisory thresholds do not determine the A/B/C mathematical-regression verdict and do not automatically require E2 modification.
- Schedule endpoints: tau `9999` → `4.03599265117e-05`, u `0.767520030581` → `-0.843139431942`.
- Knots: `[-0.84313943 -0.47587327 -0.20605561  0.00714538  0.76752003]` at timesteps `[999 750 500 250   0]`.
- Failed binding fields: `[]`.
- `FLOAT64_ALGEBRAIC_CONTAINMENT`: pass `True`, max absolute error `2.22044604925e-16`, original high-precision tolerance `1e-12`.
- `ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT` hats: pass `True`, error `0`, budget `9.53674771154e-07`, scale-ULPs `8.0000038147`, error/budget `0`.
- `ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT` kernels: pass `True`, error `2.98023223877e-08`, budget `9.53674771154e-07`, scale-ULPs `8.0000038147`, error/budget `0.0312499850988`.
- Float32 budget derivation: epsilon `1.1920928955078125e-07`, unit roundoff `5.9604644775390625e-08`, operations `8`, two-path factor `2`, gamma `4.76837385577e-07`.
- `FLOAT32_RUNTIME_VS_IDEAL_FLOAT64_TARGET`: pass `True`, absolute error `3.65724505125e-08`, scale-aware error `3.65724505125e-08`, independent-reference displacement `4.20124339207e-08`, derived budget `9.95687205074e-07`, scale-ULPs `8.35242965399`, error/budget `0.0367308631929`.
- Rho max/RMSE: `0.0409273` / `0.0176017`; overall capture `0.989489005558`.
- Per-coordinate regret max/mean: `0.000202966` / `3.32277e-05`.
- Integrated risks: exact `5.51308006013`, five-knot `5.5136117035`, best static `5.54425962498` (status `PASS`).
- Best-static diagnostic error: `NONE`.
- `ADVISORY_REFERENCE_THRESHOLDS`: `{'overall_capture_min': 0.95, 'bin_capture_min': 0.9, 'max_per_coordinate_regret': 0.0001, 'mean_per_coordinate_regret': 1e-05}`; checks `{'overall_capture': True, 'all_defined_bins_capture': False, 'max_per_coordinate_regret': False, 'mean_per_coordinate_regret': False}`.

| Bin | Timesteps | rho max error | rho RMSE | Capture |
|---:|---:|---:|---:|---:|
| 0 | [0,50) | 0.0382112 | 0.029881 | 0.991804314389 |
| 1 | [50,100) | 0.0409273 | 0.0403084 | 0.988432769631 |
| 2 | [100,150) | 0.0405028 | 0.037993 | 0.989349530237 |
| 3 | [150,200) | 0.0340355 | 0.0282443 | 0.993629937255 |
| 4 | [200,250) | 0.020485 | 0.0125024 | 0.998633365089 |
| 5 | [250,300) | 0.00548436 | 0.0037638 | 0.999815091191 |
| 6 | [300,350) | 0.00571688 | 0.00534959 | 0.999514239145 |
| 7 | [350,400) | 0.00426171 | 0.002536 | 0.999836673454 |
| 8 | [400,450) | 0.00311927 | 0.00213984 | 0.999770007366 |
| 9 | [450,500) | 0.00316468 | 0.0024531 | 0.999409132731 |
| 10 | [500,550) | 0.0148212 | 0.00920476 | 0.979143508831 |
| 11 | [550,600) | 0.0212538 | 0.0189075 | 0.76441458551 |
| 12 | [600,650) | 0.0215752 | 0.0211117 | 0.0790975773396 |
| 13 | [650,700) | 0.0197106 | 0.0165269 | -1.02763292405 |
| 14 | [700,750) | 0.0120295 | 0.00731043 | -0.608010875459 |
| 15 | [750,800) | 0.00135118 | 0.00088538 | 0.892571212118 |
| 16 | [800,850) | 0.00164057 | 0.00156459 | -0.655681651932 |
| 17 | [850,900) | 0.00163497 | 0.00152209 | -7.59312379666 |
| 18 | [900,950) | 0.00133138 | 0.00107006 | -24.8107822775 |
| 19 | [950,1000) | 0.00073467 | 0.000434125 | -27.5779041885 |

## 8. Numerical health and Monte Carlo precision

All mathematical theory/reference calculations use NumPy float64. D additionally observes the actual frozen E2 TimeAdapter runtime in native torch.float32 and compares it with an independent NumPy-float32 semantic reference under the frozen unit-roundoff/operation budget; the separate ideal float64 algebraic target remains binding at its original high-precision gate. SPD routes are Cholesky-only behind the frozen condition gate; rank diagnostics are SVD-only; full-rank projectors are reduced-QR-only.
- Solver routes: `{'spd': 'Cholesky plus project-local deterministic forward/back triangular substitution', 'condition_gate': 1000000000000.0, 'rank_condition': 'SVD', 'full_rank_projector': 'reduced QR', 'fallback_used': False}`
- Module verdicts: `{'A': 'PASS', 'B': 'PASS', 'C': 'PASS', 'D': 'PASS'}`
- Actual A/B/C/D case counts: `{'A': 13, 'B': 17, 'C': 9, 'D': 1}`; frozen-count match `{'A': True, 'B': True, 'C': True, 'D': True}`.
- Execution errors: `{}`
- Stop: `None`; NaN/Inf policy: `any binding nonfinite value is FAIL`

| MC case | Status | N | 5-SE consistency | Precision sufficient | Detail |
|---|---|---:|---|---|---|
| A-GEN-01 | PASS | 262144 | True | True | max 5-SE=0.00763029 |
| A-GEN-02 | PASS | 524288 | True | True | max 5-SE=0.0150273 |
| A-GEN-03 | PASS | 524288 | True | True | max 5-SE=0.0174316 |
| A-GEN-04 | PASS | 524288 | True | True | max 5-SE=0.0145302 |
| A-GEN-05 | PASS | 1048576 | True | True | max 5-SE=0.0186485 |
| A-HMR-00 | PASS | 262144 | True | True | max 5-SE=0.0123889 |
| A-HMR-01 | PASS | 262144 | True | True | max 5-SE=0.01842 |
| A-DST-01 | PASS | 262144 | True | True | max 5-SE=0.0137433 |
| A-DST-02 | PASS | 262144 | True | True | max 5-SE=0.00734721 |
| B-MC/1e-4/identity | PASS | 262144 | True | True | max 5-SE=1.34134e-05 |
| B-MC/1e-4/optimal | PASS | 262144 | True | True | max 5-SE=1.33718e-05 |
| B-MC/1e-4/quarter | PASS | 262144 | True | True | max 5-SE=0.00150634 |
| B-MC/0.1/identity | PASS | 262144 | True | True | max 5-SE=0.0106864 |
| B-MC/0.1/optimal | PASS | 262144 | True | True | max 5-SE=0.0105238 |
| B-MC/0.1/quarter | PASS | 262144 | True | True | max 5-SE=0.0109225 |
| B-MC/1/identity | PASS | 1048576 | True | True | max 5-SE=0.019321 |
| B-MC/1/optimal | PASS | 1048576 | True | True | max 5-SE=0.0189949 |
| B-MC/1/quarter | PASS | alias | N/A | N/A | B-MC/1/optimal |
| B-MC/10/identity | PASS | 1048576 | True | True | max 5-SE=0.0264603 |
| B-MC/10/optimal | PASS | 1048576 | True | True | max 5-SE=0.0263777 |
| B-MC/10/quarter | PASS | 1048576 | True | True | max 5-SE=0.0263888 |
| B-MC/1e4/identity | PASS | 1048576 | True | True | max 5-SE=0.0276171 |
| B-MC/1e4/optimal | PASS | 1048576 | True | True | max 5-SE=0.0276077 |
| B-MC/1e4/quarter | PASS | 1048576 | True | True | max 5-SE=0.0275791 |

## 9. Failures, negative results, and deviations

- Unexpected/inconclusive A/B/C cases: `[]`.
- Failed D binding fields: `[]`.
- Fail-closed stop field: `NONE`.
- The fixed witness retains its pre-registered low-SNR negative comparison; Appendix C outputs remain refutation guards, not positive non-Gaussian claims.
- No case, seed, draw, direction, grid point, optimizer, threshold, knot, coefficient, or time region was replaced or filtered after observing output.
- Protocol deviations: `NONE AUTOMATICALLY APPLIED`; any execution error is listed above and left for human review.

## 10. Evidence boundary

- A assumes the finite-dimensional jointly Gaussian squared-loss setting, unrestricted Bayes predictors, the stated full-column-rank measurement condition, and interior noise parameter `b>0` except in separately labelled boundary cases.
- B is confined to the exact Gaussian FW-v1 construction and the domains stated in Theorem 2 and Corollaries 2.1–2.2; the rank-loss point is an expected out-of-domain case, not an ordinary-risk evaluation.
- C checks only the frozen boundary limits, covariance-local sanity construction, and Appendix C corrective guards; it does not create a positive non-Gaussian theory.
- D evaluates only the scalar FW-v1 slice against the frozen E2 schedule and five-knot adapter family; it does not establish architecture-wide adequacy.
- Finite-grid agreement does not prove a universal theorem.
- Numerical optimizer agreement does not prove global uniqueness.
- Monte Carlo agreement does not prove Gaussian conditioning identities.
- The frozen post-risk comparison does not re-prove the full deterministic Borel-map statement.
- FW-v1 bridge results do not validate the CIFAR-10 law, network trainability, optimization attainability, FID, sample quality, causality, or architecture-wide generalization.
- Appendix C outputs are corrective refutation guards, not positive non-Gaussian theory.
- No numerical result in T0 upgrades a mathematical claim or modifies `notes/claims.md`.

## 11. Raw artifact index and reproduction

- `RUN_METADATA.json`
- `A_THEOREM1_RESULTS.json` when A ran
- `B_FWV1_RESULTS.json` when B ran
- `C_BOUNDARY_RESULTS.json` when C ran
- `D_E2_BRIDGE_RESULTS.json` and `D_E2_BRIDGE_PROFILE.csv` when D ran successfully
- `MC_SUMMARIES.json`
- `NUMERICAL_HEALTH.json`

Reproduce only after explicit human authorization:

```powershell
uv run python experiments/theory_validation/t0/run_t0.py --config experiments/theory_validation/t0/t0_config.json
```

## 12. Human decision required

No downstream action is automatic. A human must review this report and decide whether T0 is accepted and whether any separate E2 or mathematical review is warranted.
