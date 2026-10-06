"""Patient-disjoint training, threshold selection, and inference.

Keep this module beside app.py so saved estimators have a stable import path.
"""
from functools import partial
from pathlib import Path
import hashlib
import os
import pickle
import tempfile
import warnings

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

MODEL_VERSION = "v7_nested_group_smoteenn_mi10"
CLASS_WEIGHTS = {0: 3.5, 1: 1.0}
RANDOM_STATE = 42


def _build_pipeline(estimator=None, class_weights=None):
    """Every learner fits scaling, MI selection, and SMOTEENN inside its fold.

    Univariate MI ranks relevance; it does not guarantee uncorrelated features.
    """
    weights = dict(CLASS_WEIGHTS if class_weights is None else class_weights)
    if estimator is None:
        estimator = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.03, max_depth=4,
            min_samples_leaf=4, l2_regularization=0.3,
            early_stopping=False, class_weight=weights, random_state=RANDOM_STATE,
        )
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=10,
        )),
        ("smoteenn", SMOTEENN(random_state=RANDOM_STATE)),
        ("clf", estimator),
    ])


def _base_pipelines(class_weights=None):
    weights = dict(CLASS_WEIGHTS if class_weights is None else class_weights)
    return [
        ("hgb", _build_pipeline(class_weights=weights)),
        ("rf", _build_pipeline(RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight=weights, random_state=RANDOM_STATE, n_jobs=-1,
        ))),
        # probability=True uses LIBSVM's row-wise calibration CV. Feed the raw
        # margin to the grouped logistic meta-classifier instead.
        ("svc", _build_pipeline(SVC(
            C=1.0, kernel="rbf", gamma="scale",
            class_weight=weights, random_state=RANDOM_STATE,
        ))),
    ]


def prepare_data(df):
    """Reject ambiguous patient IDs and invalid inputs before splitting."""
    data = df.copy()
    if "status" not in data or not data["status"].isin([0, 1]).all():
        raise ValueError("status must contain only 0 (Healthy) and 1 (Parkinson's).")
    if data["status"].nunique() != 2:
        raise ValueError("Training requires both Healthy and Parkinson's recordings.")
    if "patient_id" not in data:
        if "name" not in data:
            raise ValueError("Supply patient_id or UCI recording names in a name column.")
        data["patient_id"] = data["name"].astype("string").str.extract(
            r"^(phon_R\d+_S\d+)_\d+$", expand=False,
        )
    if data["patient_id"].isna().any() or data["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing/unrecognized patient IDs; recording-level CV is not allowed.")
    if data.groupby("patient_id")["status"].nunique().max() != 1:
        raise ValueError("Each patient must have a consistent status across recordings.")
    features = [c for c in data if c not in {"name", "status", "label", "patient_id"}]
    if len(features) < 10:
        raise ValueError("At least 10 numeric acoustic features are required for SelectKBest.")
    try:
        X = data[features].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Acoustic features must be numeric.") from exc
    if not np.isfinite(X).all():
        raise ValueError("Acoustic features contain missing or infinite values.")
    data["label"] = data["status"].map({0: "Healthy", 1: "Parkinson's"})
    return data, features, X, data["status"].to_numpy(dtype=int), data["patient_id"].astype(str).to_numpy()


def grouped_splits(X, y, groups, n_splits, *, strict=False):
    """Find class-complete SGKF folds using labels/counts only, never scores.

    Inner CV may reduce its fold count when few minority patients remain.
    A fixed seed sequence handles SGKF's approximate stratification; it is not
    searched for model performance. The outer evaluation always uses 5 folds.
    """
    group_labels = pd.DataFrame({"g": groups, "y": y}).drop_duplicates()
    if group_labels.groupby("g")["y"].nunique().max() != 1:
        raise ValueError("Inconsistent patient labels.")
    available = int(group_labels["y"].value_counts().min())
    if strict and available < n_splits:
        raise ValueError(f"{n_splits}-fold CV needs at least {n_splits} patients in each class.")
    for count in range(min(n_splits, available), 1, -1):
        for seed in range(RANDOM_STATE, RANDOM_STATE + 32):
            splits = list(StratifiedGroupKFold(
                n_splits=count, shuffle=True, random_state=seed,
            ).split(X, y, groups))
            if all(len(np.unique(y[tr])) == len(np.unique(y[va])) == 2 for tr, va in splits):
                for tr, va in splits:
                    if set(groups[tr]) & set(groups[va]):
                        raise RuntimeError("Patient leakage detected in CV split.")
                return splits
        if strict:
            break
    raise ValueError("Too few patients per class for nested patient-group validation.")


def _base_score(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


class GroupedStackingClassifier(ClassifierMixin, BaseEstimator):
    """Fit the logistic stack on patient-disjoint OOF base-learner outputs.

    Preprocessing and resampling belong to each base pipeline, not outside the
    stack: synthetic rows cannot cross an internal validation boundary.
    """
    def __init__(self, estimators=None, class_weights=None, n_splits=3):
        self.estimators = estimators
        self.class_weights = class_weights
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        templates = _base_pipelines(self.class_weights) if self.estimators is None else self.estimators
        splits = grouped_splits(X, y, groups, self.n_splits)
        self.cv_group_audit_ = []
        meta_X = np.full((len(y), len(templates)), np.nan)
        for tr, va in splits:
            if np.bincount(y[tr], minlength=2).min() < 6:
                raise ValueError("SMOTEENN needs at least 6 recordings per class in each inner training fold.")
            self.cv_group_audit_.append((np.unique(groups[tr]).tolist(), np.unique(groups[va]).tolist()))
            for col, (_, template) in enumerate(templates):
                fitted = clone(template).fit(X[tr], y[tr])
                meta_X[va, col] = _base_score(fitted, X[va])
        if not np.isfinite(meta_X).all():
            raise RuntimeError("Incomplete out-of-fold features for the stacking classifier.")
        weights = CLASS_WEIGHTS if self.class_weights is None else self.class_weights
        self.final_estimator_ = LogisticRegression(
            C=0.5, max_iter=10000, class_weight=weights, random_state=RANDOM_STATE,
        ).fit(meta_X, y)
        self.base_pipelines_ = [(name, clone(template).fit(X, y)) for name, template in templates]
        self.estimators_ = [p.named_steps["clf"] for _, p in self.base_pipelines_]
        self.hgb_pipeline_ = dict(self.base_pipelines_)["hgb"]
        self.decision_threshold_ = 0.5
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "base_pipelines_")
        meta_X = np.column_stack([_base_score(p, X) for _, p in self.base_pipelines_])
        return self.final_estimator_.predict_proba(meta_X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold_).astype(int)

    def transform_for_explanation(self, X):
        check_is_fitted(self, "hgb_pipeline_")
        p = self.hgb_pipeline_
        return p.named_steps["selector"].transform(p.named_steps["scaler"].transform(X))


def youden_threshold(y, probabilities):
    """Use held-out training predictions only; finite ties favor specificity."""
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if np.unique(y).size != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Threshold tuning requires both labels and finite probabilities.")
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, objective = thresholds[valid], (tpr - fpr)[valid]
    optimal_idx = np.argmax(objective)
    return float(thresholds[optimal_idx])


def tune_threshold(X, y, groups, estimator):
    probabilities = np.full(len(y), np.nan)
    audit = []
    for tr, va in grouped_splits(X, y, groups, 3):
        model = clone(estimator).fit(X[tr], y[tr], groups=groups[tr])
        probabilities[va] = model.predict_proba(X[va])[:, 1]
        audit.append({"train_patients": np.unique(groups[tr]).tolist(),
                      "validation_patients": np.unique(groups[va]).tolist(),
                      "stack_splits": model.cv_group_audit_})
    return youden_threshold(y, probabilities), audit


def _metric_values(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)}


def train_model(df, version=MODEL_VERSION, cache_path=None, estimator=None, progress=None):
    """Evaluate on untouched outer patients; tune J on inner OOF predictions.

    Fold metrics use that fold's training-only threshold. The average threshold
    is saved for the final model; it is never reapplied to inflate OOF metrics.
    """
    data, features, X, y, groups = prepare_data(df)
    template = GroupedStackingClassifier() if estimator is None else estimator
    fingerprint = joblib.hash((version, data[features + ["status", "patient_id"]],
                               template.get_params(deep=True), sklearn.__version__,
                               imblearn.__version__, np.__version__,
                               hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            saved = joblib.load(cache_path)
            if saved.get("fingerprint") == fingerprint:
                return saved["result"]
        except (OSError, ValueError, EOFError, KeyError, AttributeError, ImportError, IndexError, TypeError, pickle.UnpicklingError):
            warnings.warn("Model cache could not be loaded; retraining.", RuntimeWarning)

    oof_preds, oof_probs = np.zeros(len(y), dtype=int), np.full(len(y), np.nan)
    thresholds, audits, fold_rows, baseline_rows = [], [], [], []
    splits = grouped_splits(X, y, groups, 5, strict=True)
    for fold, (tr, te) in enumerate(splits, 1):
        if progress is not None:
            progress(f"Fold {fold}/5: tuning on training patients, then evaluating held-out patients…")
        threshold, inner_audit = tune_threshold(X[tr], y[tr], groups[tr], template)
        model = clone(template).fit(X[tr], y[tr], groups=groups[tr])
        y_prob = model.predict_proba(X[te])[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        oof_probs[te], oof_preds[te] = y_prob, y_pred
        thresholds.append(threshold)
        fold_rows.append(_metric_values(y[te], y_pred))
        baseline_rows.append(_metric_values(y[te], (y_prob >= 0.5).astype(int)))
        audits.append({"fold": fold, "threshold": threshold,
                       "train_patients": np.unique(groups[tr]).tolist(),
                       "test_patients": np.unique(groups[te]).tolist(),
                       "healthy_recordings": int((y[te] == 0).sum()),
                       "parkinsons_recordings": int((y[te] == 1).sum()),
                       "threshold_splits": inner_audit,
                       "stack_splits": model.cv_group_audit_})
    if not np.isfinite(oof_probs).all():
        raise RuntimeError("Every recording must receive exactly one held-out prediction.")
    cv_metrics = {name: float(np.mean([r[name] for r in fold_rows]) * 100) for name in fold_rows[0]}
    for name, short in [("accuracy", "acc"), ("f1", "f1"), ("sensitivity", "sens"), ("specificity", "spec")]:
        cv_metrics[f"fold_{short}"] = [r[name] for r in fold_rows]
    cv_metrics.update({
        "fold_thresholds": thresholds, "decision_threshold": float(np.mean(thresholds)),
        "fold_audit": audits, "default_threshold_metrics": {
            name: float(np.mean([r[name] for r in baseline_rows]) * 100) for name in fold_rows[0]},
        "pooled_metrics": {k: float(v * 100) for k, v in _metric_values(y, oof_preds).items()},
        "validation": "5-fold outer StratifiedGroupKFold with nested grouped threshold tuning and stacking",
        "fingerprint": fingerprint,
    })
    if progress is not None:
        progress("Refitting the final model on all training patients…")
    pipeline = clone(template).fit(X, y, groups=groups)
    pipeline.decision_threshold_ = cv_metrics["decision_threshold"]
    hgb_pipeline = pipeline.hgb_pipeline_
    selected_features = np.asarray(features)[hgb_pipeline.named_steps["selector"].get_support()].tolist()
    result = (pipeline, hgb_pipeline.named_steps["scaler"], features, selected_features,
              cv_metrics, oof_preds, oof_probs, y, pipeline, pipeline.estimators_[0])
    if cache_path is not None:
        temp_path = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as handle:
                temp_path = Path(handle.name)
            joblib.dump({"fingerprint": fingerprint, "version": version, "result": result}, temp_path)
            os.replace(temp_path, cache_path)
        except OSError as exc:
            warnings.warn(f"Model trained but disk cache could not be saved: {exc}", RuntimeWarning)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    return result


def predict_single(pipeline, x_raw):
    """Return the tuned label, class scores, and selected HGB feature vector."""
    x_raw = np.asarray(x_raw, dtype=float)
    if x_raw.shape != (1, pipeline.n_features_in_) or not np.isfinite(x_raw).all():
        raise ValueError(f"Expected one finite row with {pipeline.n_features_in_} raw features.")
    prob = pipeline.predict_proba(x_raw)[0]
    pred = int(prob[1] >= pipeline.decision_threshold_)
    return pred, prob, pipeline.transform_for_explanation(x_raw)
