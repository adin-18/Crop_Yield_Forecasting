# Crop Yield Forecasting with Machine Learning & Deep Learning

**Predicting crop yield (tonnes/hectare) from environmental and agronomic data to support farm planning, crop insurance risk assessment, and early shortage forecasting**

## Dataset

Two public Kaggle agricultural datasets, merged on shared crop and soil type categories, combining large-scale environmental/agronomic measurements with farm-level operational detail (irrigation, fertiliser, pesticide use). Stratified sampling reduced the merged ~500,000 records to 100,000, preserving the proportional distribution of crop type, climate conditions, and yield range. The dataset is fully synthetic and contains no personally identifiable information.

## Tech Stack

Python · pandas · scikit-learn (Random Forest, PCA) · TensorFlow/Keras (autoencoder, DNN) · SciPy (ANOVA) · Matplotlib/Seaborn

## Results

| Model | Features | R² | RMSE | MAE | CV R² |
|---|---|---|---|---|---|
| Random Forest (baseline) | 17 | 0.585 | 1.088 | 0.890 | 0.590 |
| Random Forest (EDA-selected) | 8 | 0.586 | 1.087 | 0.889 | 0.591 |
| Autoencoder + DNN | 8 (latent: 5) | 0.586 | 1.087 | 0.889 | — |

## Key Learnings

- Feature engineering (12 → 22 features: per-acre metrics, moisture-heat interaction, growing degree days, risk indices) surfaced climate-driven signal that raw farm inputs alone didn't capture.
- ANOVA showed categorical variables (region, soil type, crop, weather condition) had no statistically significant effect on yield (p > 0.05 across all four), so they were dropped rather than encoded, keeping the model simpler without losing accuracy.
- PCA showed 8 components explained 95% of variance in the 17-feature set, confirming redundancy and validating the reduced 8-feature set used for modelling.
- Reducing 17 features to the 8 EDA-selected ones cost almost nothing in performance (ΔR² = 0.001), confirming the engineered features carried the real signal.
- The autoencoder + DNN pipeline matched but didn't beat Random Forest, showing that extra model complexity added no benefit once the feature space was already high-signal, an important sanity check before defaulting to deep learning on tabular data.
- ~40–42% of yield variance stayed unexplained, most likely a ceiling imposed by the dataset's synthetic, less heterogeneous nature rather than by model choice.

## Repository Structure

```
├── Cropyield_prediction.ipynb    # EDA, feature engineering, Random Forest, autoencoder + DNN
├── Predictive_final_dataset.csv  # Dataset (post-stratified sampling)
├── metadata.pdf                  # Feature/column descriptions
└── README.md
```

## Author's Contribution

Dataset sourcing and merging (combining two Kaggle datasets on shared crop/soil categories) was a collaborative group effort as part of a coursework module. From exploratory data analysis onward, including feature engineering, correlation/ANOVA/PCA analysis, model development, and evaluation, all work in this repository was carried out independently.
