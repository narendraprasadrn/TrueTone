import json
import pandas as pd

with open("reports/prosody/v3/teh_combined_results.json", "r") as f:
    teh = json.load(f)
    
with open("reports/prosody/v3/tamil_results.json", "r") as f:
    ta = json.load(f)
    
with open("reports/prosody/v3/english_results.json", "r") as f:
    en = json.load(f)
    
with open("reports/prosody/v3/hindi_results.json", "r") as f:
    hi = json.load(f)

ablation = pd.read_csv("reports/prosody/v3/ablation_results.csv")
importance = pd.read_csv("reports/prosody/v3/feature_importance.csv")

try:
    with open("reports/prosody/v3/external_validation.json", "r") as f:
        ext = json.load(f)
except:
    ext = None

markdown = f"""
## 22. V3 Model Development

### Dataset
The development data is restricted exclusively to Tamil, English, and Hindi.
- **Tamil**: {ta['sample_count']} total ({ta['real_count']} real, {ta['synthetic_count']} synthetic)
- **English**: {en['sample_count']} total ({en['real_count']} real, {en['synthetic_count']} synthetic)
- **Hindi**: {hi['sample_count']} total ({hi['real_count']} real, {hi['synthetic_count']} synthetic)

### Split Methodology & Group Limitations
- **CRITICAL LIMITATION**: Tamil has only 1 inferred group. English and Hindi have only 2 inferred groups each.
- Due to extreme group sparsity, conventional independent train/validation/test splits are mathematically impossible without group leakage.
- **Methodology**: 
  - For combined models and English/Hindi models: **Mode A** - Group-holdout evaluation (speaker-independent proxy), achieved via GroupKFold using inferred groups.
  - For Tamil models: **Mode B** - Within-group exploratory evaluation (stratified k-fold). Not speaker independent.

### Combined Model (Tamil + English + Hindi)
- **ROC-AUC**: {teh['overall']['roc_auc']:.4f}
- **Accuracy**: {teh['overall']['accuracy']:.4f}
- **F1-Score**: {teh['overall']['f1']:.4f}

**Per-Language Results (Combined Model)**
- **Tamil**: ROC-AUC = {teh['per_language']['Tamil']['roc_auc']:.4f}, Accuracy = {teh['per_language']['Tamil']['accuracy']:.4f}
- **English**: ROC-AUC = {teh['per_language']['English']['roc_auc']:.4f}, Accuracy = {teh['per_language']['English']['accuracy']:.4f}
- **Hindi**: ROC-AUC = {teh['per_language']['Hindi']['roc_auc']:.4f}, Accuracy = {teh['per_language']['Hindi']['accuracy']:.4f}

### Language-Specific Models
- **Tamil Model**: ROC-AUC = {ta['roc_auc']:.4f}, F1 = {ta['f1']:.4f} *(Exploratory only)*
- **English Model**: ROC-AUC = {en['roc_auc']:.4f}, F1 = {en['f1']:.4f}
- **Hindi Model**: ROC-AUC = {hi['roc_auc']:.4f}, F1 = {hi['f1']:.4f}

### Feature Ablation Experiments
| Model | ROC-AUC | Accuracy | F1 |
|-------|---------|----------|----|
"""
for _, row in ablation.iterrows():
    markdown += f"| {row['Model']} | {row['roc_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} |\n"

markdown += f"""
### Feature Importance
**Top Feature Families (by Gain)**:
"""
fam_imp = importance.groupby('family')['gain'].sum().sort_values(ascending=False)
for fam, val in fam_imp.items():
    markdown += f"- {fam}: {val:.2f}\n"

markdown += f"""
### External Validation
"""
if ext:
    markdown += f"WhatsApp Real Audio Test: Synthetic Speech Score = {ext['synthetic_speech_score']:.4f} (Predicted: {'SYNTHETIC' if ext['predicted_class']==1 else 'REAL'})\n"
else:
    markdown += "WhatsApp Test failed or was unavailable.\n"

markdown += f"""
### Remaining Work
- **Robustness**: Sensitivities to compression, noise, resampling, silence-heavy audio, and short speech must be evaluated once standard benchmark datasets are prepared.
- **Calibration**: The raw synthetic speech scores need to be formally calibrated into probabilities using Platt scaling or Isotonic Regression on an independent dataset.
"""

with open("docs/PROSODY_BEHAVIOURAL_ANALYSIS_STATUS.md", "a") as f:
    f.write(markdown)
