import pandas as pd
import json

with open("reports/prosody/v3/cross_language_summary.json", "r") as f:
    cross = json.load(f)
    
score_dist = pd.read_csv("reports/prosody/v3/score_distributions.csv")
beh = pd.read_csv("reports/prosody/v3/behavioural_feature_analysis.csv")
fam = pd.read_csv("reports/prosody/v3/feature_family_analysis.csv")

with open("reports/prosody/v3/whatsapp_cross_model_results.json", "r") as f:
    wa = json.load(f)

markdown = """
## 23. Cross-Language Generalization Study

### Cross-Language Experiments
To evaluate true language generalization before calibration, models were trained on two languages and tested strictly on the held-out third language.

| Model | Training Languages | Test Language | Evaluation Type | Groups | AUC | Accuracy | F1 |
|-------|--------------------|---------------|-----------------|--------|-----|----------|----|
"""
for exp in cross:
    markdown += f"| {exp['Model']} | {exp['Training Languages']} | {exp['Test Language']} | {exp['Evaluation Type']} | {exp['Groups']} | {exp['AUC']:.4f} | {exp['Accuracy']:.4f} | {exp['F1']:.4f} |\n"

markdown += """
### Within-Language Baselines (Reference)
| Model | Training Languages | Test Language | Evaluation Type | Groups | AUC | Accuracy | F1 |
|-------|--------------------|---------------|-----------------|--------|-----|----------|----|
| Tamil-only | Tamil | Tamil | Exploratory (Not Speaker Independent) | 1 | 0.9937 | 0.9628 | 0.9622 |
| English-only | English | English | Group-holdout (Speaker Independent) | 2 | 0.8140 | 0.6494 | 0.4306 |
| Hindi-only | Hindi | Hindi | Group-holdout (Speaker Independent) | 2 | 0.8885 | 0.8385 | 0.8543 |

### Language-Specific Error Analysis
Score distribution on held-out sets:
"""
for _, row in score_dist.iterrows():
    markdown += f"**{row['Test Language']} (Test)**\n"
    markdown += f"- REAL: Mean = {row['Real_Mean']:.4f}, Median = {row['Real_Median']:.4f}, Min = {row['Real_Min']:.4f}, Max = {row['Real_Max']:.4f}\n"
    markdown += f"- SYNTH: Mean = {row['Synth_Mean']:.4f}, Median = {row['Synth_Median']:.4f}, Min = {row['Synth_Min']:.4f}, Max = {row['Synth_Max']:.4f}\n"

markdown += """
**Diagnosis**:
- **English**: Severe failure. The real and synthetic distributions almost entirely overlap near 0.1, meaning a model trained on Ta+Hi cannot distinguish En-Real from En-Synth using the current acoustic/behavioural feature space.
- **Hindi**: Severe threshold failure. The model predicts >0.87 for both real and synthetic audio, meaning everything is classified as fake.
- **Tamil**: Moderate generalization. The distributions separate (AUC 0.78), but are shifted relative to English/Hindi.

### Feature Family Analysis
Gain totals for cross-language models:
"""
for mod in fam['Model'].unique():
    markdown += f"**{mod}**\n"
    mod_df = fam[fam['Model'] == mod]
    for _, row in mod_df.iterrows():
        markdown += f"- {row['Family']}: {row['Gain']:.2f}\n"

markdown += """
### Behavioural Feature Contribution
Despite behavioural features (like `pause_count`, `pause_mean_dur`, and `temporal_regularity`) having non-zero gain in the splits, they did **not** prevent massive generalisation failure across languages. While they improved within-distribution F1 (from 0.38 to 0.41 in the combined model), they are heavily overshadowed by the `ACOUSTIC` features in the cross-language setup, causing massive language-specific domain shift.

### WhatsApp External Check
Raw Synthetic Speech Scores (Threshold 0.5):
"""
for k, v in wa.items():
    if isinstance(v, float):
        markdown += f"- **{k}**: {v:.4f} ({'SYNTHETIC (False Positive)' if v >= 0.5 else 'REAL'})\n"
    else:
        markdown += f"- **{k}**: {v}\n"

markdown += """
**Conclusion**: Combining the languages without domain adaptation shifted the decision boundary to create a false positive on external WhatsApp audio.

### Decision Criteria
**B. NEEDS FURTHER MODEL DEVELOPMENT**
The cross-language evaluation conclusively proves that simply concatenating the features and fitting LightGBM causes massive domain-shift failure (AUC collapsing to 0.43 on English, and 100% False Positive rate on Hindi). 

Therefore, calibration should **NOT** proceed yet. The underlying feature representation or the modelling approach (e.g., Domain Adversarial Training or Language-Conditioned Normalization) must be improved first.
"""

with open("docs/PROSODY_BEHAVIOURAL_ANALYSIS_STATUS.md", "a") as f:
    f.write(markdown)
