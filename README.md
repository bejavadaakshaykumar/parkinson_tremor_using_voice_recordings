# Parkinson's Voice Analytics

Streamlit research dashboard with patient-disjoint model evaluation and tuned decisions.

## Run

Use Python 3.12. Keep `app.py` and `modeling.py` in the same folder.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Place your actual `parkinsons_dataset.csv` beside `app.py`, or upload it through the initial dataset screen. The dashboard expects the 22 UCI acoustic columns, binary `status` (0 Healthy, 1 Parkinson's), and either explicit `patient_id` or UCI recording names such as `phon_R01_S01_1`. Unknown IDs, inconsistent patient labels, missing/nonfinite feature values, and insufficient patient groups produce an error rather than falling back to row-wise validation.

The first run performs nested validation and may take several minutes. Later runs load a cached model. The cache is invalidated when training data, feature order, patient IDs, configuration, modeling code, or relevant library versions change. The previous `parkinsons_pipeline.pkl` is not reused; the new artifact is `parkinsons_pipeline_v7.pkl`.

## Model changes

- Each base learner uses `StandardScaler → SelectKBest(mutual_info_classif, k=10) → SMOTEENN(random_state=42) → classifier`.
- Mutual-information scoring is seeded for reproducibility. There is no manually dropped feature list. Univariate MI ranks relevance to the target; it does **not** guarantee elimination of correlated features.
- HGB, random forest, SVC, and the logistic meta-classifier use `{0: 3.5, 1: 1.0}` class weights. This is a chosen false-positive cost, not a measured optimum or a performance guarantee.
- A grouped stack replaces ordinary `StackingClassifier(cv=3)`. It trains its logistic meta-classifier on patient-disjoint out-of-fold outputs. Scaling, selection, and SMOTEENN are re-fitted in every internal training fold.
- SVC contributes its decision margin to the meta-classifier. LIBSVM's `probability=True` calibration uses row-wise internal folds, so it is not used. The complete stack still exposes `predict_proba` through its logistic meta-classifier.

## Validation and thresholds

1. The outer evaluation uses five-fold `StratifiedGroupKFold`. No patient's recordings appear in both sides of a split.
2. Each outer training set has an inner grouped cross-validation loop. It produces predictions for tuning Youden's J.
3. Each inner model creates another grouped split for stacking. Both preprocessing and synthetic samples remain inside that split's training side.
4. The selected threshold is applied once to the untouched outer test patients. Outer labels are used only for evaluation, never for selecting the threshold.
5. The final model is fitted on all training data and saves the average of the five thresholds. Both `predict()` and the app's `predict_single()` use this saved value. Reported outer-fold metrics retain their own fold thresholds; they are not recalculated with the final average.

The primary scores are unweighted means over five folds. F1 refers to the Parkinson's class. The confusion matrix and the additional `pooled_metrics` aggregate all held-out recordings. These are recording-level metrics with patient-group separation, not one prediction per patient. The ROC plot uses pooled out-of-fold scores from different fitted models.

When SGKF's approximate stratification produces a fold with only one class, the splitter checks a fixed sequence of seeds for class coverage. It never uses model performance to select splits. Inner fold counts may decrease from three to two when the number of minority patients requires it. Outer validation remains five folds or fails. At least five patients in each class are required, and nested SMOTEENN also needs six recordings per class in each innermost training split. ENN can make a small fold unusable; such a failure must be addressed with suitable data, not a silent change to the method.

Youden's J balances sensitivity and specificity; it does not optimize all four requested metrics simultaneously. The chosen class weights and resampling can improve or worsen results. No >95% performance claim is made.

Reference: scikit-learn explains the need to separate decision-threshold tuning from evaluation in its [threshold-tuning guide](https://scikit-learn.org/stable/modules/classification_threshold.html). See also the official [SMOTEENN API](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html) and [mutual-information feature scoring](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html).

## Dashboard improvements

- Fold thresholds, per-class support, and downloadable validation audit.
- Side-by-side held-out results for tuned thresholds and 0.5, using the same new model. This is not a comparison against the old model.
- All raw features remain available to training; the final selected ten drive the prediction controls and HGB explanations.
- SHAP uses the scaler **and** feature selector. It explains the HGB component, not the full ensemble.
- Prediction sliders are submitted together in a form. Global SHAP computations and model/data caches are bounded.
- Dataset counts are derived from the data. Missing-data and empty-filter states are handled. Existing layout gains mobile sizing and reduced-motion support.
- Scores and reports use research-oriented wording instead of diagnostic confidence or clinical recommendations.

## Audio limitations

The original extractor substituted constants for RPDE, DFA, spread1, spread2, D2, and PPE. These are now marked missing. Audio classification is blocked if any selected feature cannot be measured. Nonselected missing input slots are filled only to preserve the raw input shape, and the report continues to mark the measurements missing. Selected undefined Praat measurements and recordings with no voiced segment are rejected.

Analyze one recording at a time so its score, SHAP explanation, and report describe the same sample. CSV cross-validation does not validate the audio extractor or its performance on new microphones and speakers. End-to-end audio validation remains future work.

## Reproduce the evaluation

```powershell
python evaluate.py "C:\path\to\your\parkinsons_dataset.csv" --report validation_audit.json
```

This runs the same procedure outside Streamlit and writes the thresholds, metrics, configuration fingerprint, and patient-split audit. It does not publish or upload the dataset.

## Verification completed

Twelve automated tests passed using synthetic patient-group data, including nested group separation, fold-specific decisions, average-threshold persistence, prediction after reload, cache invalidation, invalid-data handling, selected-feature dimensions, the default classifier configuration, and Streamlit loading/prediction/filter interactions. Synthetic results are **not** evidence of Parkinson's prediction performance.

The user's actual dataset was not supplied with this revision, so clinical-data Accuracy, F1, Sensitivity, and Specificity remain unmeasured. Microphone recording, visual browser layout, and exported report appearance have not been manually verified.

```powershell
python -m pip install pytest
python -m pytest tests -q
```
