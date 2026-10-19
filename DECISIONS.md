# Locked project decisions

D1. Severity scheme
- Canonical severity scheme: K=2 run `k2_20260914T055946Z` = Mild / Severe / Uncertain(-1).
- Moderate and K=3 are superseded design exploration; archived under `data/processed/deprecated_severity/20260914T055846Z/` and never shown as canonical.
- Rationale: the versioned K=2 run is canonical, and the repo's diagnostic reports a higher silhouette for K=2 than K=3 (0.1774 vs 0.1585).

Sentence to use in slides/reports:
“Severity is reported using the canonical K=2 run (`k2_20260914T055946Z`): Mild, Severe, and Uncertain (-1). Moderate/K=3 is historical exploratory work and is not used as the canonical thesis result.”

D2. Threshold policy
- The headline threshold for every model in the comparison is 0.25, selected on validation only by maximum F2, and applied unchanged to test.
- 0.20 is a FiLM validation-retuned sensitivity row only. 0.50 is a reference column only.
- Rationale: selecting on validation and freezing the threshold prevents test-set leakage and gives the model comparison one common operating point.

Sentence to use in slides/reports:
“The headline comparison uses a threshold of 0.25, selected on validation by maximum F2 and applied unchanged to test. The 0.20 FiLM row is sensitivity analysis only; 0.50 is shown as a reference.”

D3. Synthetic users
- Final synthetic-user method: parametric per-channel gain/offset on SisFall windows: `x_sim = x * g + b`.
- TimeGAN/TimeVAE is future work only, not part of the final canonical experiment.
- Rationale: the current repo-backed FiLM prototype uses parametric user shifts to create controlled user-specific calibration scenarios.

Sentence to use in slides/reports:
“The final synthetic-user protocol uses parametric per-channel gain/offset transformations of SisFall windows to simulate user-specific shifts. TimeGAN/TimeVAE remains a future extension and is not part of the canonical evaluation.”

D4. FiLM wording
- The CNN-LSTM backbone is frozen; a small user encoder is trained once on simulated users; per-user adaptation at deployment is inference-only (no gradient steps).
- Rationale: this describes the implemented separation between offline user-encoder meta-training and deployment-time FiLM inference.

Sentence to use in slides/reports:
“The CNN-LSTM backbone remains frozen. A compact user encoder is trained once on simulated-user calibration data, and per-user adaptation at deployment is inference-only without gradient updates.”

## Known caveats

- FiLM results currently cited are from `results/film_personalization/prototype_run_final_smoke_v2/`, run with `--epochs 2 --episodes-per-epoch 3`; a full-length run is pending.
- Slide-quoted FiLM k=1 standard deviations of 11.21 and 11.26 are not supported for that k=1 comparison. The persisted run reports FiLM k=1 at 0.25 as `12.6909 +/- 9.4835` versus Global `15.1052 +/- 11.2360`.
