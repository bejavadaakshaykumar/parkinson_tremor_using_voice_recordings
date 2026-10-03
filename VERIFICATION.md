# Verification

Verified on 2026-10-03 with Python 3.12.14 on Windows, using the pinned requirements.

**20 tests passed in 36.10 seconds.** The three remaining warnings are pending deprecations inside SHAP's Matplotlib color-map setup. No convergence warnings or model-fit failures occurred. `python -m pip check` reports no broken requirements.

Checks cover:

- Disjoint patient IDs in all outer, inner and final-refit stacking folds.
- Every recording receives exactly one outer held-out prediction, and all samples from each patient share a fold.
- Recalculation of fold F1, sensitivity, specificity and MCC from OOF probabilities.
- Instrumented scaler fits: each inner training subset is fitted separately for each base learner, with no outer test rows present.
- L1 selection retains at most 12 features; the SVM uses decision margins and histogram boosting has early stopping disabled.
- Full-stack and tree-fallback SHAP values sum to their explicitly identified model scores.
- A simulated unsupported unified explainer triggers the correctly labeled tree fallback.
- Missing/blank IDs, inconsistent labels, non-finite features, missing columns and duplicate acoustic vectors across patient IDs are rejected.
- Explicit patient IDs are preserved; strict UCI name parsing is used only when IDs are absent.
- Single-class fold metrics are represented as undefined where appropriate, and the metric formulas are checked against a hand-calculated example.
- A generated 150 Hz WAV returns the expected measured pitch; silent and invalid inputs are rejected. No fabricated acoustic features are returned.
- Acoustic feature rows are matched by recording ID, with mismatches and duplicate filenames rejected.
- The original CSS is unchanged after normalizing the pasted non-breaking spaces. The recorder configuration is preserved. Random sample splitting and Plotly gauges are absent.
- DOCX bytes open as a valid OOXML ZIP, retain all five tables, include actual metrics, and omit hardcoded performance and unsupported diagnosis/treatment text.
- Streamlit AppTest loads all six tabs, reports missing training data clearly, runs prediction, changes SHAP mode without losing the prediction, clears stale results after an input change, and tolerates empty filters.

All model tests use a generated 20-person, 60-recording synthetic fixture. These are software verification results, not clinical performance estimates. No previous task's training data or pretrained model was used.

The DOCX renderer was attempted but could not run because LibreOffice is not installed. DOCX generation and XML/table content were verified; page-level visual rendering is unverified. No real microphone recording, physical device comparison, browser microphone-permission flow, external clinical validation or public deployment was performed. Streamlit custom components were exercised through AppTest; browser-side SHAP JavaScript and recorder hardware were not independently tested in this rewrite.
