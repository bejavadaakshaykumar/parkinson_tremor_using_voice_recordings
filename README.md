# Parkinson Voice Analytics single-file rewrite

This rewrite uses only the newly supplied Streamlit source. `app.py` is the entire application; it does not import any local helper modules. The original CSS, six-tab layout, hero styling, audio-recorder configuration, and DOCX table/formatting workflow are retained. Dataset counts are calculated from the actual input.

## Run

Use Python 3.12. Put your `parkinsons_dataset.csv` next to `app.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

The first run trains the nested stacking models and caches the result. A change to the CSV invalidates the training cache. No pretrained pickle is required. The supplied code attachment did not contain the training CSV; no dataset or model from previous work is included or used to validate this rewrite.

## Data contract

- The exact 22 acoustic feature names are listed in `ACOUSTIC_FEATURES` in the code.
- `status` is binary: 0 = control, 1 = Parkinson class.
- `patient_id` identifies the person, not the recording. Explicit IDs are never overwritten. If absent, only the strict UCI recording convention `phon_R01_S01_1` is parsed into `phon_R01_S01`.
- Missing/non-numeric/non-finite features, absent IDs, inconsistent patient labels and identical acoustic rows under different patient IDs are rejected.
- At least three patients per class are required, and every grouped training fold must contain both classes. A fold that cannot train is rejected explicitly rather than silently replaced with a random split.

## Model and evaluation

Outer **5-fold GroupKFold**, strictly grouped by `patient_id`, produces one out-of-fold score per recording. Within each outer training subset, a new **3-fold GroupKFold index list** is passed to `StackingClassifier.cv`. Inner indices are rebuilt relative to that subset, not reused from the full dataset.

Each of the three base estimators contains its own `StandardScaler → SelectFromModel → classifier` pipeline. This placement is essential: a supervised selector fitted outside the stack would leak inner validation labels into the meta-training features. Selection uses L1 logistic regression, C=1, nonzero coefficients and at most 12 retained features. In the pinned sklearn version, `l1_ratio=1.0` specifies L1 regularization without its deprecated `penalty` argument. A fold selecting no features fails explicitly. This heuristic subset is not a proven set of clinical biomarkers.

Base models are Random Forest, RBF SVM and histogram gradient boosting. A standardized logistic regression meta-classifier outputs the final probability. The SVM supplies decision margins (`probability=False`) to avoid the hidden recording-wise probability-calibration split in `SVC(probability=True)`. The gradient booster has early stopping disabled to avoid another implicit random validation split. Base models use class balancing; recordings still receive equal training weight, so patients with more recordings have more influence during fitting.

All hyperparameters, feature-selection settings, the 0.50 threshold and seed 42 are fixed before evaluation. There is no outer-fold model search or threshold optimization. The final interactive model is refitted on all patients after CV; performance cards never evaluate that refit against its own training rows.

The ML Model tab reports fold **mean ± sample SD** for F1, sensitivity, specificity and MCC. Patient-level evaluation is the default: a person's held-out recording probabilities are averaged, then thresholded. Recording-level results are also available. Undefined metrics in single-class test folds are excluded from the corresponding mean, with the number of defined folds displayed. SD is not a confidence interval; overlapping training folds are not independent experiments. ROC and confusion-matrix plots pool OOF predictions and are labeled accordingly. Download buttons export fold metrics, OOF predictions, split IDs, selected features, versions and the dataset fingerprint.

## SHAP

Both prediction tabs replace gauges with `st_shap` force plots and ranked signed contribution tables. The default is bounded, approximate permutation SHAP for the **whole stacking model**, using at most 16 representative patient recordings and four permutations. Additivity is checked against that exact sample's ensemble probability. Values are in probability units.

Users can select a fast tree fallback. If unified SHAP fails, the app automatically tries TreeExplainer on the tree base model with the largest absolute standardized meta coefficient (RF or histogram booster), then the other tree if necessary. The app explicitly identifies the model and its score: a base-model explanation is **not** presented as an explanation of the whole stack. Tree SHAP shape differences across versions are handled. A static waterfall is used if the HTML component fails synchronously; the feature table remains available. If all explainers fail, a visible error is shown and no invented explanation is substituted.

Acoustic features are correlated. SHAP allocations depend on the background and masking assumptions, and do not establish biological causes. Four permutations provide a bounded UI approximation, not a high-precision publication attribution study; increase the declared budget and assess stability for that purpose.

## Voice prediction and DOCX

The original recorder call and WAV upload controls are preserved. The supplied extractor filled 18 acoustic features with constants. These placeholders have been removed. WAV analysis measures only the four quantities the original code actually measured: mean/min/max pitch and HNR, with duration, sample-rate, silence and voiced-frame validation. These Praat measurements are not asserted to match the original MDVP extraction pipeline.

To enable voice prediction, SHAP and DOCX export, upload a matching acoustic CSV containing `recording_id` plus all 22 columns. Each ID must exactly match a current WAV filename; microphone audio uses `live_recording.wav`. Confirm that the supplied rows use compatible training definitions and units. This confirmation records the user's assertion; the app cannot independently validate extraction provenance. Multiple recordings are explained and reported individually, never indiscriminately averaged across people.

Without compatible feature rows, audio analysis remains available and downloadable, but no fabricated disease prediction is made. To predict reliably directly from WAV alone, collect raw training audio and retrain/validate a classifier on a reproducible compatible extractor.

DOCX generation retains the in-memory python-docx workflow, colored tables, patient fields and feature table. Hardcoded accuracy/AUC/sensitivity/specificity values were replaced with the actual patient-fold metrics. Unsupported diagnosis, severity and treatment recommendations were replaced with research wording. The report refers to the selected recording's supplied feature row; the SHAP visualization is displayed in the app.

## Scope and verification

The implementation strengthens research evaluation; it is **not clinically validated**. External cohorts, calibration, prospective evaluation, uncertainty analysis and a prespecified statistical protocol remain necessary before clinical use. Do not tune repeatedly against the displayed OOF results and continue reporting them as an untouched validation result.

Tests use generated synthetic data only, to verify behavior and leakage boundaries. They do not establish Parkinson detection performance. See `VERIFICATION.md` for test coverage and limitations.

Pinned Streamlit 1.60.0 satisfies the retained recorder's Altair dependency; the tested environment passes `pip check`. Do not independently upgrade packages for a publication run without repeating validation. Audio is processed in temporary files which are deleted in a `finally` block. Uploaded voice data and reports otherwise remain in session memory; the static training model/data are cached by Streamlit. Hosting logs, access controls and retention are deployment responsibilities. No public deployment or repository update is included.

Primary API references: [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html), [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).
