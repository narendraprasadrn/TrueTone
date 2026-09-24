# TrueTone Prosody & Behavioural Analysis Status

## 1. Objective
Complete audit and gap analysis for the Prosody and Behavioural Analysis LightGBM model utilizing the IndicTTS dataset, with a strict focus on Tamil, English, and Hindi. All AASIST work and production risk engine modifications are explicitly out of scope.

## 2. Dataset
- Dataset: IndicTTS-Deepfake-Challenge-Data (23 shards).
- Labeled Samples: 20,439 total initially identified.
- Real vs Synthetic: 10,250 real / 10,189 synthetic (approximate original count).
- Languages: 16 total languages.

## 3. Data Preparation Completed
- Total rows processed for features: 20,276.
- Successfully extracted features: 20,270.
- Failed extractions: 6 samples.

## 4. Existing Splits
Two main sets of splits exist (`data/splits/` and `data/splits_multilingual/`):
- **Original Split:** Train: 16,012 | Val: 4,259 | Test: 168 (effectively Odia-only).
- **Multilingual Split:** Train: 16,017 | Val: 163 | Test: 4,259.
- Feature extracted Internal Split: Train: 13,269 | Val: 2,745 | Test: 4,256.
Group leakage prevention was attempted by matching the first two components of the ID (`ASM_M`, `TAM_F`, etc.). There is NO explicit speaker ID, so speaker-independent evaluation cannot be strictly guaranteed. No overlap of ID or groups between train/val/test was reported.

## 5. Feature Extraction Pipeline
The V2 feature extraction pipeline (`feature_extractor.py`) handles 109-dimensional outputs.
- Extracts standard prosodic features using Parselmouth/pyworld.
- Extracts acoustic/spectral features using Librosa (STFT, MFCCs, RMS, ZCR, Flatness).
- Includes rudimentary silence/pause heuristic using voiced frames.
- Replaces exceptions with `0.0` fallbacks.

## 6. 109-Feature V2 Definition
The features consist of:
1. Pitch / Prosody (f0 mean, std, range, median, min, max, jitter, shimmer, HNR)
2. Behavioural/Temporal (voiced ratio, pause count, pause mean/max duration, speech/silence durations & ratio)
3. Energy & Temporal (RMS mean, std, min, max, ZCR)
4. Spectral (Centroid, bandwidth, rolloff, flux, flatness)
5. MFCCs (0-12, delta, delta-delta)

## 7. Behavioural Features Currently Available
Currently implemented:
- `pause_count`, `pause_mean_dur`, `pause_max_dur`
- `speech_duration`, `silence_duration`, `speech_silence_ratio`
- `voiced_ratio`

Missing but required behavioural features:
- Median pause duration
- Speech segment count
- Temporal regularity (variance of speech/silence durations)
- Advanced speaking-rate proxies (e.g., syllable estimation)

## 8. Existing LightGBM Models
- Production Model (`app/detection/models/prosody_lgbm.txt`)
- Experimental V2 Model (`models/prosody/lightgbm_indictts_v1.txt` / `prosody_final_audit.py` generated models)

## 9. Existing Evaluation Results
Current Production Model:
- ROC-AUC: 0.471
- Accuracy: 0.497
- Recall: 1.000 / F1: 0.664

New Multilingual V2 LightGBM (Internal Test Split):
- ROC-AUC: 0.955
- Accuracy: 0.882
- Precision: 0.850
- Recall: 0.926
- F1: 0.886

Threshold Validation Performance (Internal Val):
- 0.5: Precision: 86.8% | Recall: 73.9%
- 0.4: Precision: 66.4% | Recall: 100%

## 10. External WhatsApp Validation
Using `real_whatsapp.ogg`:
- Old Prosody Score: ≈ 0.99 (Synthetic prediction for real audio)
- New V2 Prosody Score: ≈ 0.38 to 0.47 (Correctly predicting real/lower probability)
*These must remain purely external checks and not be mixed into training/tuning.*

## 11. Data Leakage / Split Integrity
- Duplicates: `0`
- Group overlap between splits: `0`
- Limitation: Grouping heuristic `ID_prefix` (e.g. `ASM_F`) creates very large monolithic groups.

## 12. Tamil Dataset Status
- Total Samples: 888
- Real: 443 | Synthetic: 445
- Number of Groups: 1 (`TAM_F`)
- **Limitation:** Cannot be split safely into train/val/test using the current group heuristic.

## 13. English Dataset Status
- Total Samples: 1,132
- Real: 567 | Synthetic: 565
- Number of Groups: 2
- **Limitation:** With only 2 groups, a proper 3-way split (train/val/test) is impossible without data leakage.

## 14. Hindi Dataset Status
- Total Samples: 1,309
- Real: 649 | Synthetic: 660
- Number of Groups: 2
- **Limitation:** Same as English; supports train/test only, lacking a separate validation group.

## 15. Existing Limitations
1. Lack of explicit speaker IDs prevents true speaker-independent guarantees.
2. Group sizes for Tamil/English/Hindi are too small (1-2 groups) to form proper isolated train/val/test splits.
3. 6 audio samples fail feature extraction due to short duration STFT limitations and default to `0.0` instead of a robust short-audio fallback.
4. "P(synthetic)" is currently a raw uncalibrated score, not a true probability.
5. Missing critical behavioural temporal features (median pause, segment counts).

## 16. Missing Work
- Implement robust short-audio fallback for STFT (e.g. padding).
- Feature Engineering: Add median pause duration, speech segment count, temporal regularity.
- Dataset Splitting: Address the 1-2 group limitation for Tamil, English, and Hindi (may require cross-validation or accepting group leakage with strict disclaimers).
- Feature Ablation: Quantify value of behavioural vs acoustic features.
- Calibration: Calibrate LightGBM outputs to true probabilities.

## 17. Recommended Experiments
- **EXP A**: Combined T/E/H LightGBM.
- **EXP B/C/D**: Language-specific models (Tamil, English, Hindi).
- **Ablation Models**: 1 (Acoustic), 2 (+Prosodic), 3 (+Voice Quality), 4 (+Behavioural/Temporal).

## 18. Acceptance Criteria
- Dataset properly isolated (or limitations explicitly documented).
- Feature ablation proves the value of behavioural features.
- Model outperforms baseline V1.
- Calibration curve generated.
- Zero data leakage between train and test.
- External WhatsApp audio consistently scores as Real.

## 19. Production Integration Requirements
- Do NOT integrate yet.
- Requires final threshold calibration and sign-off.
- Requires controlled application integration without altering `app/risk_engine/config.yaml`.

## 20. Explicitly Out of Scope: AASIST
AASIST modification, training, evaluation, fusion weight changes, or domain shift investigations are completely out of scope.

## 21. V3 Feature Pipeline Update
- **V3 Feature Count**: 112 (up from 109).
- **New Features**:
  1. `pause_median_dur`: Median of valid pause durations (>50ms).
  2. `speech_segment_count`: Count of contiguous voiced-frame segments.
  3. `temporal_regularity`: Variance of speech segment durations in seconds.
- **Feature Family Structure**:
  - *Prosodic*: `f0_*`, `voiced_ratio`
  - *Voice Quality*: `jitter`, `shimmer`, `hnr`
  - *Behavioural/Temporal*: `pause_*`, `speech_duration`, `silence_duration`, `speech_silence_ratio`, `speech_segment_count`, `temporal_regularity`
  - *Acoustic*: `rms_*`, `zcr_*`, `spectral_*`, `flatness_*`, `mfcc_*`
- **Short-audio strategy**:
  - The extraction algorithm now checks if `len(audio) < 2048` (minimum required by default `n_fft=2048`). If so, it zero-pads the audio to exactly 2048 samples *only* for Librosa STFT operations to prevent exceptions.
  - The Parselmouth (prosody) features continue to use the *unpadded* audio to avoid artificially distorting pitch statistics.
  - For MFCC delta calculation, the `width` parameter is dynamically set as `min(9, n_frames - 1)` (with a minimum of 3) rather than hardcoded to 9, falling back gracefully if fewer than 3 frames exist.
- **Cache**: V3 features are saved to `data/features/prosody/v3/feature_cache.jsonl`.
- **Limitations**: Tamil, English, and Hindi split leakage risks (1-2 group sparsity) still remain unresolved and must be addressed prior to LightGBM retraining.

## 22. V3 Model Development

### Dataset
The development data is restricted exclusively to Tamil, English, and Hindi.
- **Tamil**: 887 total (443 real, 444 synthetic)
- **English**: 1132 total (567 real, 565 synthetic)
- **Hindi**: 1309 total (649 real, 660 synthetic)

### Split Methodology & Group Limitations
- **CRITICAL LIMITATION**: Tamil has only 1 inferred group. English and Hindi have only 2 inferred groups each.
- Due to extreme group sparsity, conventional independent train/validation/test splits are mathematically impossible without group leakage.
- **Methodology**: 
  - For combined models and English/Hindi models: **Mode A** - Group-holdout evaluation (speaker-independent proxy), achieved via GroupKFold using inferred groups.
  - For Tamil models: **Mode B** - Within-group exploratory evaluation (stratified k-fold). Not speaker independent.

### Combined Model (Tamil + English + Hindi)
- **ROC-AUC**: 0.7735
- **Accuracy**: 0.5698
- **F1-Score**: 0.4149

**Per-Language Results (Combined Model)**
- **Tamil**: ROC-AUC = 0.7702, Accuracy = 0.6505
- **English**: ROC-AUC = 0.7771, Accuracy = 0.5875
- **Hindi**: ROC-AUC = 0.6460, Accuracy = 0.5080

### Language-Specific Models
- **Tamil Model**: ROC-AUC = 0.9937, F1 = 0.9622 *(Exploratory only)*
- **English Model**: ROC-AUC = 0.8140, F1 = 0.4306
- **Hindi Model**: ROC-AUC = 0.8885, F1 = 0.8543

### Feature Ablation Experiments
| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| M1 (Acoustic) | 0.6854 | 0.5652 | 0.4027 |
| M2 (+Prosodic) | 0.7539 | 0.5628 | 0.3984 |
| M3 (+VQ) | 0.7878 | 0.5587 | 0.3893 |
| M4 (+Behavioural) | 0.7735 | 0.5698 | 0.4149 |

### Feature Importance
**Top Feature Families (by Gain)**:
- ACOUSTIC: 6036.05
- BEHAVIOURAL/TEMPORAL: 244.98
- PROSODIC: 191.71
- VOICE QUALITY: 99.69

### External Validation
WhatsApp Test failed or was unavailable.

### Remaining Work
- **Robustness**: Sensitivities to compression, noise, resampling, silence-heavy audio, and short speech must be evaluated once standard benchmark datasets are prepared.
- **Calibration**: The raw synthetic speech scores need to be formally calibrated into probabilities using Platt scaling or Isotonic Regression on an independent dataset.

## 22. V3 Model Development

### Dataset
The development data is restricted exclusively to Tamil, English, and Hindi.
- **Tamil**: 887 total (443 real, 444 synthetic)
- **English**: 1132 total (567 real, 565 synthetic)
- **Hindi**: 1309 total (649 real, 660 synthetic)

### Split Methodology & Group Limitations
- **CRITICAL LIMITATION**: Tamil has only 1 inferred group. English and Hindi have only 2 inferred groups each.
- Due to extreme group sparsity, conventional independent train/validation/test splits are mathematically impossible without group leakage.
- **Methodology**: 
  - For combined models and English/Hindi models: **Mode A** - Group-holdout evaluation (speaker-independent proxy), achieved via GroupKFold using inferred groups.
  - For Tamil models: **Mode B** - Within-group exploratory evaluation (stratified k-fold). Not speaker independent.

### Combined Model (Tamil + English + Hindi)
- **ROC-AUC**: 0.7735
- **Accuracy**: 0.5698
- **F1-Score**: 0.4149

**Per-Language Results (Combined Model)**
- **Tamil**: ROC-AUC = 0.7702, Accuracy = 0.6505
- **English**: ROC-AUC = 0.7771, Accuracy = 0.5875
- **Hindi**: ROC-AUC = 0.6460, Accuracy = 0.5080

### Language-Specific Models
- **Tamil Model**: ROC-AUC = 0.9937, F1 = 0.9622 *(Exploratory only)*
- **English Model**: ROC-AUC = 0.8140, F1 = 0.4306
- **Hindi Model**: ROC-AUC = 0.8885, F1 = 0.8543

### Feature Ablation Experiments
| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| M1 (Acoustic) | 0.6854 | 0.5652 | 0.4027 |
| M2 (+Prosodic) | 0.7539 | 0.5628 | 0.3984 |
| M3 (+VQ) | 0.7878 | 0.5587 | 0.3893 |
| M4 (+Behavioural) | 0.7735 | 0.5698 | 0.4149 |

### Feature Importance
**Top Feature Families (by Gain)**:
- ACOUSTIC: 6036.05
- BEHAVIOURAL/TEMPORAL: 244.98
- PROSODIC: 191.71
- VOICE QUALITY: 99.69

### External Validation
WhatsApp Test failed or was unavailable.

### Remaining Work
- **Robustness**: Sensitivities to compression, noise, resampling, silence-heavy audio, and short speech must be evaluated once standard benchmark datasets are prepared.
- **Calibration**: The raw synthetic speech scores need to be formally calibrated into probabilities using Platt scaling or Isotonic Regression on an independent dataset.

## 22. V3 Model Development

### Dataset
The development data is restricted exclusively to Tamil, English, and Hindi.
- **Tamil**: 887 total (443 real, 444 synthetic)
- **English**: 1132 total (567 real, 565 synthetic)
- **Hindi**: 1309 total (649 real, 660 synthetic)

### Split Methodology & Group Limitations
- **CRITICAL LIMITATION**: Tamil has only 1 inferred group. English and Hindi have only 2 inferred groups each.
- Due to extreme group sparsity, conventional independent train/validation/test splits are mathematically impossible without group leakage.
- **Methodology**: 
  - For combined models and English/Hindi models: **Mode A** - Group-holdout evaluation (speaker-independent proxy), achieved via GroupKFold using inferred groups.
  - For Tamil models: **Mode B** - Within-group exploratory evaluation (stratified k-fold). Not speaker independent.

### Combined Model (Tamil + English + Hindi)
- **ROC-AUC**: 0.7735
- **Accuracy**: 0.5698
- **F1-Score**: 0.4149

**Per-Language Results (Combined Model)**
- **Tamil**: ROC-AUC = 0.7702, Accuracy = 0.6505
- **English**: ROC-AUC = 0.7771, Accuracy = 0.5875
- **Hindi**: ROC-AUC = 0.6460, Accuracy = 0.5080

### Language-Specific Models
- **Tamil Model**: ROC-AUC = 0.9937, F1 = 0.9622 *(Exploratory only)*
- **English Model**: ROC-AUC = 0.8140, F1 = 0.4306
- **Hindi Model**: ROC-AUC = 0.8885, F1 = 0.8543

### Feature Ablation Experiments
| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
| M1 (Acoustic) | 0.6854 | 0.5652 | 0.4027 |
| M2 (+Prosodic) | 0.7539 | 0.5628 | 0.3984 |
| M3 (+VQ) | 0.7878 | 0.5587 | 0.3893 |
| M4 (+Behavioural) | 0.7735 | 0.5698 | 0.4149 |

### Feature Importance
**Top Feature Families (by Gain)**:
- ACOUSTIC: 6036.05
- BEHAVIOURAL/TEMPORAL: 244.98
- PROSODIC: 191.71
- VOICE QUALITY: 99.69

### External Validation
WhatsApp Real Audio Test: Synthetic Speech Score = 0.5268 (Predicted: SYNTHETIC)

### Remaining Work
- **Robustness**: Sensitivities to compression, noise, resampling, silence-heavy audio, and short speech must be evaluated once standard benchmark datasets are prepared.
- **Calibration**: The raw synthetic speech scores need to be formally calibrated into probabilities using Platt scaling or Isotonic Regression on an independent dataset.

## 23. Cross-Language Generalization Study

### Cross-Language Experiments
To evaluate true language generalization before calibration, models were trained on two languages and tested strictly on the held-out third language.

| Model | Training Languages | Test Language | Evaluation Type | Groups | AUC | Accuracy | F1 |
|-------|--------------------|---------------|-----------------|--------|-----|----------|----|
| Exp A (Ta+Hi -> En) | Tamil+Hindi | English | Group-holdout (Speaker-Independent Proxy) | 2 | 0.4316 | 0.4885 | 0.0398 |
| Exp B (Ta+En -> Hi) | Tamil+English | Hindi | Group-holdout (Speaker-Independent Proxy) | 2 | 0.6245 | 0.4981 | 0.6049 |
| Exp C (En+Hi -> Ta) | English+Hindi | Tamil | Exploratory cross-language evaluation — Tamil group limitation | 1 | 0.7815 | 0.6505 | 0.5867 |

### Within-Language Baselines (Reference)
| Model | Training Languages | Test Language | Evaluation Type | Groups | AUC | Accuracy | F1 |
|-------|--------------------|---------------|-----------------|--------|-----|----------|----|
| Tamil-only | Tamil | Tamil | Exploratory (Not Speaker Independent) | 1 | 0.9937 | 0.9628 | 0.9622 |
| English-only | English | English | Group-holdout (Speaker Independent) | 2 | 0.8140 | 0.6494 | 0.4306 |
| Hindi-only | Hindi | Hindi | Group-holdout (Speaker Independent) | 2 | 0.8885 | 0.8385 | 0.8543 |

### Language-Specific Error Analysis
Score distribution on held-out sets:
**English (Test)**
- REAL: Mean = 0.1314, Median = 0.0688, Min = 0.0003, Max = 0.8845
- SYNTH: Mean = 0.0980, Median = 0.0513, Min = 0.0014, Max = 0.7096
**Hindi (Test)**
- REAL: Mean = 0.6905, Median = 0.8779, Min = 0.0000, Max = 0.9998
- SYNTH: Mean = 0.7262, Median = 0.9480, Min = 0.0000, Max = 0.9999
**Tamil (Test)**
- REAL: Mean = 0.2145, Median = 0.0416, Min = 0.0000, Max = 0.9984
- SYNTH: Mean = 0.5078, Median = 0.4885, Min = 0.0123, Max = 0.9997

**Diagnosis**:
- **English**: Severe failure. The real and synthetic distributions almost entirely overlap near 0.1, meaning a model trained on Ta+Hi cannot distinguish En-Real from En-Synth using the current acoustic/behavioural feature space.
- **Hindi**: Severe threshold failure. The model predicts >0.87 for both real and synthetic audio, meaning everything is classified as fake.
- **Tamil**: Moderate generalization. The distributions separate (AUC 0.78), but are shifted relative to English/Hindi.

### Feature Family Analysis
Gain totals for cross-language models:
**Exp A (Ta+Hi -> En)**
- ACOUSTIC: 47836.23
- BEHAVIOURAL/TEMPORAL: 328.63
- PROSODIC: 1154.16
- VOICE QUALITY: 1560.06
**Exp B (Ta+En -> Hi)**
- ACOUSTIC: 42006.88
- BEHAVIOURAL/TEMPORAL: 2423.57
- PROSODIC: 924.11
- VOICE QUALITY: 1272.60
**Exp C (En+Hi -> Ta)**
- ACOUSTIC: 51578.08
- BEHAVIOURAL/TEMPORAL: 1791.71
- PROSODIC: 606.46
- VOICE QUALITY: 2488.09

### Behavioural Feature Contribution
Despite behavioural features (like `pause_count`, `pause_mean_dur`, and `temporal_regularity`) having non-zero gain in the splits, they did **not** prevent massive generalisation failure across languages. While they improved within-distribution F1 (from 0.38 to 0.41 in the combined model), they are heavily overshadowed by the `ACOUSTIC` features in the cross-language setup, causing massive language-specific domain shift.

### WhatsApp External Check
Raw Synthetic Speech Scores (Threshold 0.5):
- **V2 (Indictts)**: 0.4057 (REAL)
- **V3 Combined (T+E+H)**: 0.5268 (SYNTHETIC (False Positive))
- **V3 Tamil**: 0.0038 (REAL)
- **V3 English**: 0.4636 (REAL)
- **V3 Hindi**: 0.0108 (REAL)
- **Exp A (Ta+Hi -> En)**: 0.0058 (REAL)
- **Exp B (Ta+En -> Hi)**: 0.0477 (REAL)
- **Exp C (En+Hi -> Ta)**: 0.0124 (REAL)

**Conclusion**: Combining the languages without domain adaptation shifted the decision boundary to create a false positive on external WhatsApp audio.

### Decision Criteria
**B. NEEDS FURTHER MODEL DEVELOPMENT**
The cross-language evaluation conclusively proves that simply concatenating the features and fitting LightGBM causes massive domain-shift failure (AUC collapsing to 0.43 on English, and 100% False Positive rate on Hindi). 

Therefore, calibration should **NOT** proceed yet. The underlying feature representation or the modelling approach (e.g., Domain Adversarial Training or Language-Conditioned Normalization) must be improved first.
