# Practical Deep Learning: Exams, Sleep & Healthy Eating
Three compact deep-learning projects tackling regression (student exam scores) and classification (sleep disorder & healthy vs. unhealthy meals) with clean preprocessing, sensible baselines, and readable Keras MLPs.

## 📦 Datasets
* Analyzing Student Academic Trends — regression: predict exam_score.
* Lifestyle and Sleep Patterns — classification: predict presence of a sleep disorder.
* Healthy Eating — classification: predict whether a meal is healthy.

(All CSVs are assumed to be locally downloaded. Paths are set in each notebook/script.)

## 🧰 What’s in the box

* **Clear preprocessing pipelines**
  * Trim/clean columns, handle missing values
  * Standardize numeric features (fit on train only)
  * One-hot encode categoricals (drop_first to avoid collinearity)
  * Unit normalization for nutrition data (convert to per-100g)

* **Reasonable deep learning baselines (Keras MLPs)**
  * Student Exams (regression): small MLP with ReLU layers
  * Sleep Disorder (binary): MLP + class weights
  * Healthy Eating (binary): MLP + leakage-aware features + class weights

* **Training discipline**
  * Stratified splits for classification tasks
  * EarlyStopping (monitor val metric), ReduceLROnPlateau
  * Fixed seeds for basic reproducibility
  * K-Fold CV available for the Student Exams regression

* **Evaluation & interpretability**
  * Regression: MAE / RMSE / R²
  * Classification: AUC, Accuracy, Precision, Recall, F1, confusion matrix
  * Threshold tuning on validation to optimize F1 / PR trade-off
  * Permutation importance for feature insights (sklearn’s permutation_importance)
  * Simple plots: training curves, ROC & PR curves, correlation heatmaps

## 🔍 Dependencies
```
python>=3.9
numpy
pandas
scikit-learn
tensorflow>=2.12
matplotlib
seaborn
imbalanced-learn   # optional (SMOTE), disabled by default
``` 
## 📈 How to Run (an outline)

* **Load & EDA:** preview columns, check missingness, basic stats, class balance.
* **Preprocess:** impute, encode, scale; for nutrition—normalize to per-100g.
* **Feature engineering:** add safe ratios/densities; avoid leakage.
* **Split:** stratified (for classification); keep a hold-out test set.
* **Train:** compile MLP, set callbacks, fit on train, monitor val metric.
* **Tune threshold:** use validation probabilities to pick cutoff (F1/AUC/PR target).
* **Evaluate:** compute metrics on test; plot confusion matrix, ROC/PR.
* **Interpret:** run permutation importance on the trained model.
