# Inference Bugfix Baseline Report

## AASIST-L Input Fix
* **Old Behavior:** `np.tile()` was used to duplicate 2.0s windows up to 4.0375s, creating a phase discontinuity.
* **New Behavior:** Continuous stream up to 4.0375s is extracted without periodic duplication, utilizing `prepare_aasist_context`.
* **Exact Input Length:** 64,600 samples @ 16 kHz.

## Prosody Semantic Fix
* **Old Behavior:** LightGBM output interpreted as P(spoof).
* **New Behavior:** LightGBM output recognized as P(bonafide) due to dataset label inversion. API returns explicit dictionary: `{"bonafide_score": prob, "spoof_score": 1.0 - prob}`.

## Frontend Scaling Fix
* **Old Behavior:** Inconsistent scaling (some `/100`, some direct rendering).
* **New Behavior:** `formatRiskScore(score)` centralizes formatting, converting standard `[0.0, 1.0]` scores consistently (e.g. `0.72 -> 72%`).

## End-to-End Examples

| Audio | AASIST spoof | Prosody spoof | Fused risk | UI |
| --- | --- | --- | --- | --- |
| Real Human (test_libri.wav) | 0.044 | 0.710 | 0.210 | 21% |
| Synthetic (synthetic.wav) | 1.000 | 0.995 | 0.998 | 100% |
| Current 37s Cloned Audio (synthetic_cloned.ogg) | ~0.999 | ~0.997 | ~0.998 | 100% |
