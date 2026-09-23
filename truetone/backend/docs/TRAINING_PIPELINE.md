# TrueTone AASIST-L Fine-Tuning & Prosody Retraining Configuration

## 1. Dataset & Splits
* **Dataset:** `SherryT997/IndicTTS-Deepfake-Challenge-Data`
* **Labels:** `is_tts=0` (Real/Bonafide), `is_tts=1` (Synthetic/TTS)
* **Leakage-Safe Splitting:**
  - Grouped by `language` and parsing metadata prefixes from `id` (e.g. speaker/source ID).
  - Train (70%), Validation (15%), Test (15%) strictly disjoint on source/speaker.

## 2. AASIST-L Pretrained Baseline
Prior to fine-tuning, the original pretrained checkpoint will be evaluated on the 15% held-out test split, measuring Accuracy, Precision, Recall, F1, ROC-AUC, EER, FPR, and FNR. 

## 3. AASIST-L Fine-Tuning Strategy
* **Checkpoint:** Existing `third_party/aasist/models/weights/AASIST-L.pth`
* **Stage 1 (Warmup):**
  - **Epochs:** 3
  - **Learning Rate:** 1e-5
  - **Action:** Freeze feature extraction layers, adapt only the final classification layers to the target domain distribution.
* **Stage 2 (Fine-Tuning):**
  - **Epochs:** 15 (Early stopping patience = 3)
  - **Learning Rate:** 1e-6 (with Cosine Annealing scheduler)
  - **Batch Size:** 16
  - **Weight Decay:** 1e-4
  - **Action:** Unfreeze all layers for full-model adaptation with a highly conservative learning rate to prevent catastrophic forgetting.

## 4. Target-Domain Augmentation (AASIST-L)
To bridge the domain mismatch between ASVspoof (FLAC) and TrueTone (WhatsApp Ogg / telecom), augmentation is applied **only to the training set**:
```yaml
augmentation:
  enabled: true
  codec_simulation:
    probability: 0.5
    types: ["ogg_vorbis", "mp3"]
    bitrates: ["16k", "24k", "32k"]
  resample:
    probability: 0.2
    rates: [8000, 16000]
  noise:
    probability: 0.3
    snr_db_range: [15, 30]
  reverberation:
    probability: 0.2
  amplitude_variation:
    probability: 0.3
    range: [0.5, 1.5]
```
Validation and test sets remain strictly clean. A separate external sanity check uses `real_whatsapp.ogg`.

## 5. Prosody LightGBM Retraining
* **Labels:** `is_tts=1` maps explicitly to `P(synthetic)`.
* **Features (`models/prosody/feature_schema.json`):**
  - RMS, ZCR, Spectral Centroid, Spectral Bandwidth, Spectral Flatness
  - MFCC Mean/Std, Delta, Delta-Delta
  - F0 Mean/Std/Range, Pitch Variation
  - Speaking rate proxy, Jitter, Shimmer, HNR
* **Strategy:** All extracted features will be checked for NaN/Inf. Unstable features will be dropped based on feature-importance metrics during cross-validation.
