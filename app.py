
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from pathlib import Path
import hashlib, json, re, importlib.metadata
import soundfile as sf
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, matthews_corrcoef
import warnings
# Keep convergence and compatibility warnings visible during research.
import random
import parselmouth
from datetime import datetime
from io import BytesIO
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Voice Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background:#ffffff !important; }
[data-testid="stHeader"] { background:transparent !important; }
* { font-family:'Inter',sans-serif !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg,#0d1b2a 0%,#1b2d4f 55%,#162340 100%) !important;
}
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#f1f5f9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color:#94a3b8 !important; font-size:11px;
    letter-spacing:.08em; text-transform:uppercase; font-weight:600;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg,#0d1b2a 0%,#1e3a8a 45%,#7c3aed 100%);
    border-radius:24px; padding:48px 56px; margin-bottom:32px;
    position:relative; overflow:hidden;
    animation: fadeSlideDown .8s cubic-bezier(.22,1,.36,1) both;
    box-shadow: 0 24px 64px rgba(30,58,138,.25);
}
.hero::before {
    content:""; position:absolute; inset:0;
    background:radial-gradient(ellipse at 80% 20%,rgba(124,58,237,.35) 0%,transparent 60%);
}
.hero::after {
    content:""; position:absolute; right:-60px; top:-60px;
    width:260px; height:260px; border-radius:50%;
    background:rgba(255,255,255,.04);
}
.hero-emoji { font-size:3rem; display:block; margin-bottom:10px; animation:pulse 2.5s ease infinite; }
.hero-title { color:#fff; font-size:2.5rem; font-weight:900; margin:0 0 8px; letter-spacing:-.03em; }
.hero-sub   { color:#bfdbfe; font-size:1.05rem; margin:0; font-weight:400; }
.hero-badge {
    display:inline-block; margin-top:16px;
    background:rgba(255,255,255,.12); backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,.2); border-radius:999px;
    padding:6px 18px; color:#e0f2fe; font-size:.8rem; font-weight:600;
    letter-spacing:.06em; text-transform:uppercase;
}

/* ── Pulse bar ── */
.pulse-bar {
    height:3px; border-radius:999px;
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4,#6366f1);
    background-size:300% 100%;
    animation:pulseSlide 3.5s linear infinite;
    margin:0 0 28px;
}
@keyframes pulseSlide { to { background-position:-300% 0; } }

/* ── KPI grid ── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:28px; }
.kpi-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:18px;
    padding:24px 22px; box-shadow:0 4px 24px rgba(0,0,0,.06);
    animation:fadeUp .6s cubic-bezier(.22,1,.36,1) both;
    transition:transform .3s,box-shadow .3s; cursor:default;
    position:relative; overflow:hidden;
}
.kpi-card::before {
    content:""; position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--accent,#6366f1);
    transform:scaleX(0); transform-origin:left;
    transition:transform .4s ease;
}
.kpi-card:hover::before { transform:scaleX(1); }
.kpi-card:hover { transform:translateY(-6px); box-shadow:0 16px 40px rgba(0,0,0,.12); }
.kpi-icon  { font-size:2rem; margin-bottom:10px; display:block; }
.kpi-label { font-size:10.5px; font-weight:700; letter-spacing:.1em;
             text-transform:uppercase; color:#94a3b8; margin-bottom:6px; }
.kpi-value { font-size:2.1rem; font-weight:900; color:#0f172a; line-height:1; }
.kpi-sub   { font-size:12px; color:#64748b; margin-top:6px; }

/* ── Section header ── */
.sec-hdr {
    display:flex; align-items:center; gap:10px;
    font-size:1.1rem; font-weight:800; color:#0f172a;
    margin:28px 0 14px; padding-bottom:12px;
    border-bottom:2px solid #ede9fe;
}

/* ── Chart card ── */
.chart-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:18px;
    padding:12px 16px 16px; box-shadow:0 4px 20px rgba(0,0,0,.05);
    animation:fadeUp .7s cubic-bezier(.22,1,.36,1) both;
    margin-bottom:18px; transition:box-shadow .3s;
}
.chart-card:hover { box-shadow:0 10px 36px rgba(99,102,241,.1); }

/* ── Prediction box ── */
.pred-box {
    border-radius:18px; padding:28px 32px; text-align:center;
    animation:zoomIn .5s cubic-bezier(.22,1,.36,1) both;
    margin-bottom:16px;
}
.pred-pos { background:linear-gradient(135deg,#fef2f2,#ffe4e6); border:2px solid #fca5a5; }
.pred-neg { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:2px solid #86efac; }
.pred-title { font-size:1.6rem; font-weight:900; margin-bottom:8px; }
.pred-conf  { font-size:1rem; font-weight:500; color:#475569; }

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity:0; transform:translateY(-28px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes zoomIn {
    from { opacity:0; transform:scale(.9); }
    to   { opacity:1; transform:scale(1); }
}
@keyframes pulse {
    0%,100% { transform:scale(1); }
    50%      { transform:scale(1.08); }
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight:600; font-size:.88rem; color:#64748b;
    border-radius:10px 10px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#7c3aed; border-bottom:3px solid #7c3aed;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#f8fafc; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:999px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DATA & MODEL
# ══════════════════════════════════════════════
# Fixed, declared protocol. Do not tune these constants against the displayed OOF results.
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
DECISION_THRESHOLD = 0.50
ACOUSTIC_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]


def validate_features(frame):
    """Accept the fixed acoustic schema only; never include identifiers or target data."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(ACOUSTIC_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing acoustic features: " + ", ".join(missing))
    X = frame.loc[:, ACOUSTIC_FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if X.empty or not np.isfinite(X.to_numpy()).all():
        raise ValueError("All 22 acoustic features must contain finite numeric values; no placeholders or imputation.")
    return X


def prepare_dataset(frame):
    data = frame.copy().reset_index(drop=True)
    validate_features(data)
    if "patient_id" not in data:
        if "name" not in data:
            raise ValueError("Supply patient_id for every recording. Row numbers are not valid patient IDs.")
        # Explicit compatibility with the supplied UCI recording-name convention.
        data["patient_id"] = data["name"].astype("string").str.extract(
            r"^(phon_R\d+_S\d+)_\d+$", expand=False)
    if data["patient_id"].isna().any():
        raise ValueError("Missing patient_id or unrecognized recording name. Supply explicit patient IDs.")
    data["patient_id"] = data["patient_id"].astype(str).str.strip()
    if data["patient_id"].eq("").any():
        raise ValueError("Blank patient_id is not allowed.")
    if "status" not in data:
        raise ValueError("The training dataset requires status: 0=control, 1=Parkinson class.")
    data["status"] = pd.to_numeric(data["status"], errors="raise")
    if data["status"].isna().any() or set(data["status"].unique()) != {0, 1}:
        raise ValueError("status must contain both binary classes 0 and 1, without missing values.")
    data["status"] = data["status"].astype(int)
    if data.groupby("patient_id")["status"].nunique().gt(1).any():
        raise ValueError("A patient has inconsistent class labels. Resolve labels before training.")
    subject_counts = data.drop_duplicates("patient_id")["status"].value_counts()
    if subject_counts.min() < 3 or data.patient_id.nunique() < OUTER_FOLDS:
        raise ValueError("Need at least 3 distinct patients in each class and 5 patients overall.")
    # Identical acoustic vectors under different IDs may be duplicated patients.
    duplicates = data.groupby(ACOUSTIC_FEATURES, dropna=False)["patient_id"].nunique()
    if duplicates.gt(1).any():
        raise ValueError("Identical acoustic rows occur under different patient IDs. Audit these duplicates first.")
    if "name" not in data:
        data["name"] = [f"recording_{i + 1}" for i in range(len(data))]
    data["label"] = data["status"].map({1: "Parkinson's", 0: "Healthy"})
    return data


@st.cache_data
def load_data(csv_bytes):
    return prepare_dataset(pd.read_csv(BytesIO(csv_bytes)))


def patient_folds(X, y, groups, n_splits):
    """Materialize indices relative to *this* training subset, for StackingClassifier.cv."""
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    splits = list(splitter.split(X, y, groups))
    seen = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Patient overlap in cross-validation.")
        if len(np.unique(y[train])) != 2:
            raise ValueError("A grouped training fold lacks one class. More patients or a prespecified alternative grouped protocol is required.")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("Cross-validation must predict every recording exactly once.")
    return splits


def base_pipeline(estimator):
    # This pipeline is INSIDE each base learner. A selector outside the stack would
    # see the inner validation labels used to train the meta-learner.
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0,
                               class_weight="balanced", max_iter=5000, random_state=SEED),
            threshold=1e-8, max_features=12)),
        ("model", estimator),
    ])


def make_stack(X, y, groups):
    inner = patient_folds(X, y, groups, INNER_FOLDS)
    stack = StackingClassifier(
        estimators=[
            ("rf", base_pipeline(RandomForestClassifier(
                n_estimators=160, min_samples_leaf=2, class_weight="balanced",
                random_state=SEED, n_jobs=1))),
            # Avoid SVC(probability=True)'s hidden recording-wise calibration CV.
            # The meta-learner consumes this margin and supplies the final probability.
            ("svm", base_pipeline(SVC(C=1.0, kernel="rbf", gamma="scale",
                                      class_weight="balanced", random_state=SEED))),
            ("hgb", base_pipeline(HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=7, min_samples_leaf=8,
                l2_regularization=1.0, class_weight="balanced",
                early_stopping=False, random_state=SEED))),
        ],
        final_estimator=Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, class_weight="balanced",
                                              max_iter=5000, random_state=SEED)),
        ]),
        cv=inner, stack_method="auto", passthrough=False, n_jobs=1,
    )
    return stack, inner


def clinical_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    prediction = (np.asarray(probability) >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "F1-Score": f1_score(y, prediction, zero_division=0) if np.any(y == 1) else np.nan,
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "MCC": matthews_corrcoef(y, prediction) if both_classes else np.nan,
        "ROC-AUC": roc_auc_score(y, probability) if both_classes else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def subject_predictions(recordings):
    return recordings.groupby(["fold", "patient_id"], as_index=False).agg(
        status=("status", "first"), probability=("probability", "mean"),
        recordings=("status", "size"))


@st.cache_resource(show_spinner="Training patient-separated stacking models…")
def train_model(df):
    """Outer GroupKFold estimates generalization; inner GroupKFold trains stacking.

    Fixed parameters and 0.50 threshold are declared above. No outer score selects
    models/features/thresholds. The final deployable estimator is refitted on all
    available patients AFTER OOF evaluation, and never used to report CV metrics.
    """
    data = prepare_dataset(df)
    features = list(ACOUSTIC_FEATURES)
    X = validate_features(data)
    y = data.status.to_numpy()
    groups = data.patient_id.to_numpy()
    outer = patient_folds(X, y, groups, OUTER_FOLDS)
    oof = data[["patient_id", "name", "status"]].copy()
    oof["row_id"] = np.arange(len(data))
    oof["probability"] = np.nan
    oof["fold"] = -1
    folds, audit, selections = [], [], []
    for fold, (train, test) in enumerate(outer, start=1):
        Xtr, ytr, gtr = X.iloc[train], y[train], groups[train]
        estimator, inner = make_stack(Xtr, ytr, gtr)
        with threadpool_limits(limits=2):
            estimator.fit(Xtr, ytr)
            probability = estimator.predict_proba(X.iloc[test])[:, 1]
        oof.loc[test, "probability"] = probability
        oof.loc[test, "fold"] = fold
        for unit, scores in [("Recordings", oof.loc[test]),
                             ("Patients", subject_predictions(oof.loc[test]))]:
            folds.append({"fold": fold, "unit": unit, "n": len(scores),
                          **clinical_metrics(scores.status, scores.probability)})
        selected = np.array(features)[estimator.named_estimators_["rf"].named_steps["select"].get_support()].tolist()
        selections.append({"fold": fold, "features": selected, "count": len(selected)})
        audit.append({"fold": fold, "train_patients": sorted(set(gtr)),
                      "test_patients": sorted(set(groups[test])),
                      "inner": [{"train_patients": sorted(set(gtr[a])),
                                 "test_patients": sorted(set(gtr[b]))} for a, b in inner]})
    if oof.probability.isna().any():
        raise RuntimeError("Incomplete out-of-fold predictions.")
    final_model, final_inner = make_stack(X, y, groups)
    with threadpool_limits(limits=2):
        final_model.fit(X, y)
    # One representative recording per patient; deterministic, bounded explanation background.
    background = data.drop_duplicates("patient_id").sample(
        min(16, data.patient_id.nunique()), random_state=SEED)[features]
    cv = pd.DataFrame(folds)
    summary = cv.groupby("unit")[["F1-Score", "Sensitivity", "Specificity", "MCC", "ROC-AUC"]].agg(["mean", "std", "count"])
    return {"model": final_model, "features": features, "oof": oof,
            "patients": subject_predictions(oof), "fold_metrics": cv, "summary": summary,
            "audit": audit, "selections": selections, "background": background,
            "minimum": X.min(), "maximum": X.max(),
            "metadata": {"seed": SEED, "outer_cv": "GroupKFold", "outer_folds": len(outer),
                         "inner_folds": INNER_FOLDS, "threshold": DECISION_THRESHOLD,
                         "selection": "L1 logistic, C=1; nonzero coefficients, at most 12 features",
                         "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
                         "versions": {p: importlib.metadata.version(p) for p in
                                      ["scikit-learn", "numpy", "pandas", "streamlit", "shap"]},
                         "final_inner_groups": [{"train_patients": sorted(set(groups[a])),
                                                 "test_patients": sorted(set(groups[b]))} for a, b in final_inner]}}


def explain_prediction(bundle, row, method="Full ensemble"):
    """Explain exactly one feature row; return an explicit scope and additivity check."""
    row = validate_features(row)
    if len(row) != 1:
        raise ValueError("Explain one recording at a time.")
    model = bundle["model"]
    reason = None
    if method == "Full ensemble":
        try:
            def predict(values):
                return model.predict_proba(pd.DataFrame(values, columns=ACOUSTIC_FEATURES))[:, 1]
            with threadpool_limits(limits=2):
                explainer = shap.PermutationExplainer(
                    predict, shap.maskers.Independent(bundle["background"], max_samples=16), seed=SEED)
                explanation = explainer(row, max_evals=4 * (2 * len(ACOUSTIC_FEATURES) + 1), silent=True)[0]
                score = float(model.predict_proba(row)[0, 1])
            residual = float(explanation.base_values + np.sum(explanation.values) - score)
            if not np.isclose(residual, 0, atol=1e-5):
                raise ValueError("Ensemble SHAP additivity check failed.")
            return explanation, "Full stacking ensemble (approximate permutation SHAP)", score, None
        except Exception as exc:
            reason = f"Full-ensemble explanation unavailable ({type(exc).__name__}); showing an explicitly labeled base-model explanation."
    # Largest absolute standardized meta coefficient among tree-based learners.
    # This is a fallback choice, not evidence of causal or clinical importance.
    coefficients = model.final_estimator_.named_steps["logistic"].coef_[0]
    candidates = sorted([(abs(coefficients[0]), "rf"), (abs(coefficients[2]), "hgb")], reverse=True)
    errors = []
    for _, name in candidates:
        try:
            pipe = model.named_estimators_[name]
            transformed = pipe[:-1].transform(row)
            background = pipe[:-1].transform(bundle["background"])
            tree = pipe.named_steps["model"]
            with threadpool_limits(limits=2):
                explainer = shap.TreeExplainer(tree, data=background,
                    feature_perturbation="interventional", model_output="probability")
                raw = explainer.shap_values(transformed, check_additivity=True)
                score = float(tree.predict_proba(transformed)[0, 1])
            values = raw[1][0] if isinstance(raw, list) else (
                np.asarray(raw)[0, :, 1] if np.asarray(raw).ndim == 3 else np.asarray(raw)[0])
            expected = np.asarray(explainer.expected_value).reshape(-1)
            base = float(expected[1] if len(expected) > 1 else expected[0])
            support = pipe.named_steps["select"].get_support()
            explanation = shap.Explanation(values=values, base_values=base,
                data=row.to_numpy()[0, support], feature_names=np.array(ACOUSTIC_FEATURES)[support].tolist())
            if not np.isclose(base + np.sum(values), score, atol=1e-5):
                raise ValueError("Tree SHAP additivity check failed.")
            return explanation, f"{name.upper()} base learner ONLY (not the ensemble)", score, reason
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No compatible SHAP explainer available. " + "; ".join(errors))


def render_shap(bundle, row, key):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("**Why this prediction? — local SHAP explanation**")
    method = st.selectbox("Explanation method", ["Full ensemble", "Fast tree fallback"], key=key)
    try:
        with st.spinner("Computing local feature contributions…"):
            explanation, scope, explained_score, reason = explain_prediction(bundle, row, method)
        if reason:
            st.warning(reason)
        st.caption(f"Explained model: {scope}. Its Parkinson-class score is {explained_score:.3f}.")
        force = shap.force_plot(float(explanation.base_values), explanation.values,
                                explanation.data, feature_names=explanation.feature_names,
                                link="identity", matplotlib=False, show=False)
        try:
            st_shap(force, height=220)
        except Exception:
            # Keep the same SHAP values if the JavaScript component is incompatible.
            plt.figure()
            shap.plots.waterfall(explanation, max_display=12, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
        contributions = pd.DataFrame({"Feature": explanation.feature_names,
            "Input value": explanation.data, "SHAP contribution": explanation.values})
        contributions = contributions.iloc[np.argsort(-np.abs(contributions["SHAP contribution"].to_numpy()))]
        st.dataframe(contributions, hide_index=True)
        st.caption("Red raises and blue lowers the explained model's class-1 score relative to its background. "
                   "Base value + contributions = explained score. Correlated acoustic features share attribution; "
                   "SHAP describes this model, not biological causes or a diagnosis.")
    except Exception as exc:
        st.error(f"SHAP could not be computed: {exc}. No substitute explanation is displayed.")
    st.markdown('</div>', unsafe_allow_html=True)


def predict_input(bundle, row):
    X = validate_features(row)
    outside = ((X < bundle["minimum"]) | (X > bundle["maximum"])).sum(axis=1)
    with threadpool_limits(limits=2):
        probability = bundle["model"].predict_proba(X)[:, 1]
    return probability, outside


data_path = Path(__file__).resolve().with_name("parkinsons_dataset.csv")
if not data_path.is_file():
    st.error("Place parkinsons_dataset.csv beside this app.py. It must contain the 22 acoustic features, "
             "binary status and patient_id (or UCI-format recording names).")
    st.stop()
try:
    df = load_data(data_path.read_bytes())
    model_bundle = train_model(df)
except (ValueError, TypeError, RuntimeError) as exc:
    st.error(f"Dataset or model validation failed: {exc}")
    st.stop()
clf = model_bundle["model"]
features = model_bundle["features"]


JITTER_COLS    = ["MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP"]
SHIMMER_COLS   = ["MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA"]
FREQ_COLS      = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)"]
NONLINEAR_COLS = ["RPDE","DFA","spread1","spread2","D2","PPE"]
RATIO_COLS     = ["NHR","HNR"]

PL  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(family="Inter,sans-serif", color="#1e293b"),
           margin=dict(l=8,r=8,t=44,b=8))
PAL = {"Parkinson's":"#7c3aed","Healthy":"#10b981"}

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color to rgba string for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Parkinson's Analytics")
    st.markdown("---")
    st.markdown("#### 🎛️ Dashboard Filters")
    status_filter = st.multiselect(
        "Status", ["Parkinson's","Healthy"],
        default=["Parkinson's","Healthy"])
    patient_filter = st.multiselect(
        "Patient ID", sorted(df["patient_id"].unique()), default=[])
    st.markdown("---")
    st.markdown("#### 🔬 Feature Group")
    feat_group = st.selectbox("Explore group", [
        "Jitter (Frequency Variation)",
        "Shimmer (Amplitude Variation)",
        "Fundamental Frequency",
        "Nonlinear Dynamics",
        "Noise Ratios"])
    st.markdown("---")
    st.caption(f"Voice Dataset\n{len(df)} samples · {df.patient_id.nunique()} patients · 22 features")

mask = df["label"].isin(status_filter)
if patient_filter:
    mask &= df["patient_id"].isin(patient_filter)
dff = df[mask].copy()

# ══════════════════════════════════════════════
#  HERO + PULSE BAR
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <span class="hero-emoji">🧠</span>
  <div class="hero-title">Parkinson's Voice Analytics</div>
  <div class="hero-sub">Biomedical voice signal analysis · 22 acoustic features · Patient-wise ML research evaluation</div>
  <span class="hero-badge">📊 {len(df)} Recordings · {df.patient_id.nunique()} Patients · Voice Dataset</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="pulse-bar"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════
total  = len(dff)
pk_cnt = int((dff["status"]==1).sum())
hl_cnt = int((dff["status"]==0).sum())
pk_pct = round(pk_cnt/total*100,1) if total else 0
avg_hnr= round(float(dff["HNR"].mean()),2) if total else 0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card" style="--accent:#6366f1">
    <span class="kpi-icon">👥</span>
    <div class="kpi-label">Total Recordings</div>
    <div class="kpi-value">{total}</div>
    <div class="kpi-sub">voice samples in view</div>
  </div>
  <div class="kpi-card" style="--accent:#dc2626">
    <span class="kpi-icon">🔴</span>
    <div class="kpi-label">Parkinson's</div>
    <div class="kpi-value">{pk_cnt}</div>
    <div class="kpi-sub">{pk_pct}% of dataset</div>
  </div>
  <div class="kpi-card" style="--accent:#10b981">
    <span class="kpi-icon">🟢</span>
    <div class="kpi-label">Healthy</div>
    <div class="kpi-value">{hl_cnt}</div>
    <div class="kpi-sub">{round(100-pk_pct,1)}% of dataset</div>
  </div>
  <div class="kpi-card" style="--accent:#f59e0b">
    <span class="kpi-icon">🎙️</span>
    <div class="kpi-label">Avg HNR</div>
    <div class="kpi-value">{avg_hnr}</div>
    <div class="kpi-sub">Harmonics-to-Noise Ratio</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔬 Feature Deep Dive",
    "🤖 ML Model", "🧬 Predict", "📋 Data Explorer", "🎤 Voice Predict"])

# ── TAB 1: OVERVIEW ──────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pie = dff["label"].value_counts().reset_index()
        pie.columns = ["Status","Count"]
        fig = px.pie(pie, names="Status", values="Count",
                     title="Class Distribution",
                     color="Status", color_discrete_map=PAL, hole=.52)
        fig.update_traces(textinfo="percent+label", pull=[0.04,0])
        fig.update_layout(**PL, legend=dict(orientation="h",y=-.1))
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pt = dff.groupby(["patient_id","label"])["status"].count().reset_index()
        pt.columns = ["Patient","Status","Recordings"]
        fig2 = px.bar(pt, x="Patient", y="Recordings", color="Status",
                      title="Recordings per Patient",
                      color_discrete_map=PAL, barmode="stack")
        fig2.update_layout(**PL, xaxis_tickangle=45, xaxis_title="")
        st.plotly_chart(fig2, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Radar chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    key_feats = ["MDVP:Fo(Hz)","HNR","RPDE","DFA","PPE","spread1","spread2","D2"]
    h_vals = dff[dff["status"]==0][key_feats].mean().tolist()
    p_vals = dff[dff["status"]==1][key_feats].mean().tolist()
    mms = MinMaxScaler()
    mms.fit(df[key_feats].to_numpy())
    mat = mms.transform(np.array([h_vals, p_vals]))
    h_n, p_n = mat[0].tolist() + [mat[0][0]], mat[1].tolist() + [mat[1][0]]
    cats = key_feats + [key_feats[0]]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(r=h_n, theta=cats, fill="toself",
        name="Healthy", line_color="#10b981", fillcolor="rgba(16,185,129,.15)"))
    fig3.add_trace(go.Scatterpolar(r=p_n, theta=cats, fill="toself",
        name="Parkinson's", line_color="#7c3aed", fillcolor="rgba(124,58,237,.15)"))
    fig3.update_layout(**PL, title="Feature Radar — Normalised Mean Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        legend=dict(orientation="h",y=-.12))
    st.plotly_chart(fig3, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    num_cols = [c for c in dff.columns if dff[c].dtype in [np.float64,np.int64] and c!="status"]
    corr = dff[num_cols+["status"]].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title="Feature Correlation Matrix", aspect="auto", zmin=-1, zmax=1)
    fig4.update_layout(**PL, height=520)
    st.plotly_chart(fig4, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: FEATURE DEEP DIVE ─────────────────
with tab2:
    group_map = {
        "Jitter (Frequency Variation)":  JITTER_COLS,
        "Shimmer (Amplitude Variation)": SHIMMER_COLS,
        "Fundamental Frequency":         FREQ_COLS,
        "Nonlinear Dynamics":            NONLINEAR_COLS,
        "Noise Ratios":                  RATIO_COLS,
    }
    cols_sel = group_map[feat_group]

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    n = len(cols_sel)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig5 = make_subplots(rows=nrows, cols=ncols,
                         subplot_titles=cols_sel, vertical_spacing=.12)
    for idx, col in enumerate(cols_sel):
        r, c = divmod(idx, ncols)
        for lbl, color in PAL.items():
            sub = dff[dff["label"]==lbl]
            fig5.add_trace(go.Violin(
                y=sub[col], name=lbl, fillcolor=hex_to_rgba(color, 0.2), line_color=color,
                box_visible=True, meanline_visible=True, showlegend=(idx==0),
            ), row=r+1, col=c+1)
    fig5.update_layout(**PL, title=f"{feat_group} — Distributions by Status",
                       height=380*nrows, violingap=.3)
    st.plotly_chart(fig5, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    melt = dff[cols_sel+["label"]].melt(id_vars="label", var_name="Feature", value_name="Value")
    fig6 = px.box(melt, x="Feature", y="Value", color="label",
                  title=f"{feat_group} — Grouped Box Plot",
                  color_discrete_map=PAL, points="outliers")
    fig6.update_layout(**PL)
    st.plotly_chart(fig6, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    if len(cols_sel) >= 2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pair_x = st.selectbox("X axis", cols_sel, index=0)
        pair_y = st.selectbox("Y axis", cols_sel, index=min(1,len(cols_sel)-1))
        fig7 = px.scatter(dff, x=pair_x, y=pair_y, color="label",
                          marginal_x="histogram", marginal_y="histogram",
                          color_discrete_map=PAL, opacity=.75,
                          title=f"{pair_x} vs {pair_y}")
        fig7.update_layout(**PL)
        st.plotly_chart(fig7, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: ML MODEL ──────────────────────────
with tab3:
    unit = st.radio("Evaluation unit", ["Patients", "Recordings"], horizontal=True, key="ml_unit")
    st.caption("Patient results average each patient's held-out recording scores before applying the fixed 0.50 threshold. "
               "Cards show unweighted fold mean ± sample SD, not a confidence interval. All evaluation scores are out of fold.")
    summary = model_bundle["summary"].loc[unit]
    cols = st.columns(4)
    for col, label, color in zip(cols, ["F1-Score", "Sensitivity", "Specificity", "MCC"],
                                  ["#6366f1", "#10b981", "#f59e0b", "#7c3aed"]):
        mean, std, count = summary[label]["mean"], summary[label]["std"], int(summary[label]["count"])
        value = f"{mean:.3f}" if label == "MCC" else f"{mean:.1%}"
        deviation = f"{std:.3f}" if label == "MCC" else f"{std:.1%}"
        with col:
            st.markdown(f'''<div class="kpi-card" style="--accent:{color}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color}">{value}</div>
              <div class="kpi-sub">± {deviation} SD · {count}/{OUTER_FOLDS} defined folds</div>
            </div>''', unsafe_allow_html=True)
    st.info("Outer 5-fold GroupKFold evaluates unseen patients. Inner 3-fold GroupKFold trains the stack. "
            "Scaling and L1 selection are refitted within every base learner's training fold. "
            "Hyperparameters and threshold are fixed in advance; the final interactive model is subsequently refitted on all patients.")
    evaluated = model_bundle["patients"] if unit == "Patients" else model_bundle["oof"]
    y_te = evaluated.status.to_numpy()
    y_prob = evaluated.probability.to_numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        fig8 = px.imshow(cm, text_auto=True, x=["Pred Control", "Pred Parkinson's"],
                         y=["Act Control", "Act Parkinson's"], color_continuous_scale="Purples",
                         title=f"Pooled out-of-fold confusion matrix · {unit.lower()}")
        fig8.update_layout(**PL, height=340)
        st.plotly_chart(fig8, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
            name=f"Pooled OOF ROC (AUC={roc_auc_score(y_te, y_prob):.3f})",
            line=dict(color="#7c3aed", width=3)))
        fig9.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#94a3b8", dash="dash"), showlegend=False))
        fig9.update_layout(**PL, title="Out-of-fold ROC curve", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", height=340)
        st.plotly_chart(fig9, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    selection_counts = {name: sum(name in item["features"] for item in model_bundle["selections"])
                        for name in features}
    imp = pd.DataFrame({"Feature": list(selection_counts), "Outer folds selected": list(selection_counts.values())})
    fig10 = px.bar(imp.sort_values("Outer folds selected"), y="Feature", x="Outer folds selected",
                   orientation="h", title="Feature-selection stability across outer training folds",
                   color="Outer folds selected", color_continuous_scale="Purples")
    fig10.update_layout(**PL, coloraxis_showscale=False, height=480)
    st.plotly_chart(fig10, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Fold results, selected features and reproducibility"):
        st.dataframe(model_bundle["fold_metrics"], hide_index=True)
        selected = np.array(features)[clf.named_estimators_["rf"].named_steps["select"].get_support()].tolist()
        st.write(f"Final refit: {len(selected)}/22 features selected", selected)
        st.caption("Undefined sensitivity/specificity/AUC in single-class test folds is NaN, not zero; "
                   "means exclude undefined folds and show the denominator. MCC follows sklearn's zero-denominator "
                   "convention when truth contains both classes. Fold SD is descriptive because training folds overlap.")
        st.download_button("Download fold metrics", model_bundle["fold_metrics"].to_csv(index=False),
                           "grouped_cv_metrics.csv", "text/csv")
        st.download_button("Download out-of-fold predictions", model_bundle["oof"].to_csv(index=False),
                           "grouped_oof_predictions.csv", "text/csv")
        audit_json = {"protocol": model_bundle["metadata"], "folds": model_bundle["audit"],
                      "selected_features": model_bundle["selections"]}
        st.download_button("Download split and feature-selection audit", json.dumps(audit_json, indent=2),
                           "grouped_cv_audit.json", "application/json")
    st.warning("Research evaluation only. A small single-dataset CV experiment and SHAP explanations do not "
               "establish clinical validity. External cohorts, calibration and a prespecified clinical study are still required.")


# ── TAB 4: PREDICT ───────────────────────────
with tab4:
    st.markdown('<div class="sec-hdr">🧬 Interactive Prediction — Adjust Voice Features</div>',
                unsafe_allow_html=True)
    st.info("Use the sliders to simulate a voice recording and inspect the stacking model's research prediction and feature contributions. Simulated features are not patient measurements.")

    defaults = df[features].mean().to_dict()
    mins     = df[features].min().to_dict()
    maxs     = df[features].max().to_dict()

    input_vals = {}
    groups = [
        ("🎵 Fundamental Frequency", FREQ_COLS),
        ("〰️ Jitter Features",        JITTER_COLS),
        ("📶 Shimmer Features",        SHIMMER_COLS),
        ("📡 Noise Ratios",            RATIO_COLS),
        ("🌀 Nonlinear Dynamics",      NONLINEAR_COLS),
    ]
    for grp_name, grp_cols in groups:
        st.markdown(f'<div class="sec-hdr">{grp_name}</div>', unsafe_allow_html=True)
        gcols = st.columns(min(3, len(grp_cols)))
        for i, feat in enumerate(grp_cols):
            with gcols[i % len(gcols)]:
                lo   = float(mins[feat])
                hi   = float(maxs[feat])
                dv   = float(defaults[feat])
                if hi <= lo:
                    input_vals[feat] = st.number_input(feat, value=dv, disabled=True, format="%.5f")
                else:
                    step = max((hi-lo)/200, 1e-6)
                    input_vals[feat] = st.slider(feat, lo, hi, dv, step=step, format="%.5f")

    x_in = pd.DataFrame([[input_vals[f] for f in features]], columns=features)
    fingerprint = hashlib.sha256(x_in.to_numpy().tobytes() + model_bundle["metadata"]["dataset_sha256"].encode()).hexdigest()
    if st.session_state.get("prediction_fingerprint") != fingerprint:
        st.session_state.pop("prediction_result", None)
        st.session_state["prediction_fingerprint"] = fingerprint
    if st.button("🔍 Run Prediction", width="stretch", type="primary"):
        probability, outside = predict_input(model_bundle, x_in)
        st.session_state.prediction_result = float(probability[0])
    if "prediction_result" in st.session_state:
        probability = st.session_state.prediction_result
        positive = probability >= DECISION_THRESHOLD
        title = "Parkinson-class pattern" if positive else "Control-class pattern"
        box = "pred-pos" if positive else "pred-neg"
        st.markdown(f'''<div class="pred-box {box}"><div class="pred-title">{title}</div>
          <div class="pred-conf">Model score: <strong>{probability:.1%}</strong> for Parkinson class</div>
          <div class="pred-conf">Decision threshold: 0.50 · Not calibrated clinical risk</div></div>''', unsafe_allow_html=True)
        render_shap(model_bundle, x_in, "manual_shap_method")

    st.caption("⚠️ Educational/research use only. Not a clinical diagnostic tool.")

# ── TAB 5: DATA EXPLORER ─────────────────────
with tab5:
    st.markdown(f"**{len(dff):,}** records match current filters.")
    c1, c2 = st.columns([3,1])
    with c1:
        search = st.text_input("🔍 Filter by patient ID", "")
    with c2:
        n_rows = st.slider("Rows to show", 1, max(2, len(df)), min(50, len(df)))

    disp = dff.copy()
    if search:
        disp = disp[disp["patient_id"].str.contains(search, case=False)]

    show_cols = ["name","label"] + features[:10]
    st.dataframe(disp[show_cols].head(n_rows).reset_index(drop=True),
                 width="stretch", height=440)

    st.markdown("**📊 Summary Statistics**")
    st.dataframe(dff[features].describe().T, width="stretch")

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="parkinsons_filtered.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════
#  DOCX HELPERS
# ══════════════════════════════════════════════
def _set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def _add_divider(doc, color="1e3a8a", sz="6"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(4)
    pPr  = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot  = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    sz)
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), color)
    pBdr.append(bot)
    pPr.append(pBdr)

def _sec_hdr(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(f"{num}.  {text}")
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a)

def generate_voice_docx(name, age, gender, healthy_prob, parkinson_prob,
                        level, fo, hnr, advice, feature_names, feature_values,
                        source_label="WAV Upload", cv_summary=None):
    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Cm(2)
        sec.bottom_margin = Cm(2)
        sec.left_margin   = Cm(2.5)
        sec.right_margin  = Cm(2.5)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    # BANNER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("PARKINSON VOICE RESEARCH")
    r.bold = True; r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(4)
    r2 = p2.add_run("AI-Powered Voice Research Report")
    r2.italic = True; r2.font.size = Pt(11)
    r2.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    _add_divider(doc)

    # META
    mt = doc.add_table(rows=1, cols=4)
    mt.style = "Table Grid"
    meta_vals = [
        f"Report ID: PD-{random.randint(10000,99999)}",
        f"Date: {datetime.now().strftime('%d %b %Y  %H:%M')}",
        f"Source: {source_label}",
        "Research model — not a diagnosis"
    ]
    for i, v in enumerate(meta_vals):
        mt.rows[0].cells[i].text = v
        _set_cell_bg(mt.rows[0].cells[i], "EFF6FF" if i % 2 == 0 else "FFFFFF")
        for run in mt.rows[0].cells[i].paragraphs[0].runs:
            run.font.size = Pt(9)
    doc.add_paragraph()

    # 1. PATIENT INFO
    _sec_hdr(doc, "1", "Patient Information")
    pt = doc.add_table(rows=2, cols=6)
    pt.style = "Table Grid"
    for i, h in enumerate(["Name","Age","Gender","Model Class","Fo (Hz)","HNR"]):
        pt.rows[0].cells[i].text = h
        _set_cell_bg(pt.rows[0].cells[i], "1e3a8a")
        for run in pt.rows[0].cells[i].paragraphs[0].runs:
            run.bold = True; run.font.color.rgb = RGBColor(255,255,255); run.font.size = Pt(9)
    for i, v in enumerate([name, str(age), gender, level, f"{fo:.2f}", f"{hnr:.2f}"]):
        pt.rows[1].cells[i].text = v
        for run in pt.rows[1].cells[i].paragraphs[0].runs:
            run.font.size = Pt(10)
    doc.add_paragraph()

    # 2. RESULT
    _sec_hdr(doc, "2", "Research Model Result")
    rt = doc.add_table(rows=1, cols=3)
    rt.style = "Table Grid"
    pk_clr = RGBColor(0xDC,0x26,0x26) if parkinson_prob > 0.5 else RGBColor(0x16,0xa3,0x4a)
    hl_clr = RGBColor(0x16,0xa3,0x4a) if healthy_prob >= 0.5 else RGBColor(0xDC,0x26,0x26)
    rh = {"Low Risk":"16a34a","Moderate Risk":"d97706","High Risk":"dc2626"}.get(level,"475569")
    rsk_clr = RGBColor(int(rh[0:2],16), int(rh[2:4],16), int(rh[4:6],16))
    for ci, (txt, clr, bg) in enumerate([
        (f"Parkinson's\n{parkinson_prob*100:.1f}%", pk_clr, "FEE2E2" if parkinson_prob>0.5 else "DCFCE7"),
        (f"Model Class\n{level}",                    rsk_clr,"F8FAFC"),
        (f"Healthy\n{healthy_prob*100:.1f}%",        hl_clr, "DCFCE7" if healthy_prob>=0.5 else "FEE2E2"),
    ]):
        cell = rt.rows[0].cells[ci]
        pp   = cell.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr   = pp.add_run(txt)
        rr.bold = True; rr.font.size = Pt(15); rr.font.color.rgb = clr
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        pp.paragraph_format.space_before = Pt(8)
        pp.paragraph_format.space_after  = Pt(8)
        _set_cell_bg(cell, bg)
    doc.add_paragraph()

    # 3. FEATURES
    _sec_hdr(doc, "3", "Complete Voice Feature Analysis")
    half = len(feature_names)//2 + len(feature_names)%2
    ft   = doc.add_table(rows=half+1, cols=4)
    ft.style = "Table Grid"
    for j, h in enumerate(["Feature","Value","Feature","Value"]):
        ft.rows[0].cells[j].text = h
        _set_cell_bg(ft.rows[0].cells[j], "1e3a8a")
        for run in ft.rows[0].cells[j].paragraphs[0].runs:
            run.bold = True; run.font.color.rgb = RGBColor(255,255,255); run.font.size = Pt(9)
    for i in range(half):
        row = ft.rows[i+1]
        row.cells[0].text = feature_names[i]
        row.cells[1].text = f"{feature_values[i]:.5f}"
        ri = i + half
        if ri < len(feature_names):
            row.cells[2].text = feature_names[ri]
            row.cells[3].text = f"{feature_values[ri]:.5f}"
        bg = "EFF6FF" if i%2==0 else "FFFFFF"
        for c in row.cells:
            _set_cell_bg(c, bg)
            for run in c.paragraphs[0].runs:
                run.font.size = Pt(9)
    doc.add_paragraph()

    # 4. INTERPRETATION
    _sec_hdr(doc, "4", "Research Interpretation")
    interp = ("The reported score is the stacking model's class-1 probability for the supplied "
              "complete acoustic feature row. It is not a calibrated estimate of disease risk. "
              "This report does not infer elevated jitter, reduced HNR or clinical severity from "
              "the score alone. The app's SHAP plot explains the explicitly identified model; "
              "feature attributions do not establish biological causation.")
    doc.add_paragraph(interp).paragraph_format.space_after = Pt(6)

    # 5. RECOMMENDATIONS
    _sec_hdr(doc, "5", "Research Use and Limitations")
    for rec in [advice, "Confirm recording-to-feature correspondence and extractor compatibility.",
                "External validation and probability calibration are required before any clinical use."]:
        p_ = doc.add_paragraph(style="List Bullet")
        p_.add_run(rec).font.size = Pt(10)
    doc.add_paragraph()

    # 6. MODEL METRICS
    _sec_hdr(doc, "6", "Model Performance Metrics")
    mm = doc.add_table(rows=2, cols=4)
    mm.style = "Table Grid"
    for j, h in enumerate(["Metric","Value","Metric","Value"]):
        mm.rows[0].cells[j].text = h
        _set_cell_bg(mm.rows[0].cells[j], "1e3a8a")
        for run in mm.rows[0].cells[j].paragraphs[0].runs:
            run.bold = True; run.font.color.rgb = RGBColor(255,255,255); run.font.size = Pt(9)
    for j, metric in enumerate(["F1-Score", "Sensitivity", "Specificity", "MCC"]):
        mm.rows[0].cells[j].paragraphs[0].runs[0].text = metric
        if cv_summary is None:
            value = "Not evaluated"
        else:
            mean, std = cv_summary[(metric, "mean")], cv_summary[(metric, "std")]
            value = f"{mean:.3f} ± {std:.3f}" if metric == "MCC" else f"{mean:.1%} ± {std:.1%}"
        mm.rows[1].cells[j].text = value
        for run in mm.rows[1].cells[j].paragraphs[0].runs:
            run.font.size = Pt(9)
    doc.add_paragraph("Patient-wise outer GroupKFold; fold mean ± SD, not a confidence interval. "
                      "Threshold 0.50. Interactive model refitted on all patients after evaluation.").runs[0].font.size = Pt(8)
    if cv_summary is not None:
        counts = ", ".join(f"{metric}: {int(cv_summary[(metric, 'count')])}/{OUTER_FOLDS}" for metric in
                           ["F1-Score", "Sensitivity", "Specificity", "MCC"])
        doc.add_paragraph("Defined-fold counts — " + counts).runs[0].font.size = Pt(8)
    doc.add_paragraph()

    # DISCLAIMER
    _add_divider(doc, color="94a3b8", sz="4")
    disc = doc.add_paragraph()
    dr   = disc.add_run(
        "\u26a0  DISCLAIMER: This report is generated by an AI screening tool for "
        "research/educational purposes only. It does not constitute a clinical diagnosis. "
        "Always consult a qualified medical professional.")
    dr.italic = True; dr.font.size = Pt(8)
    dr.font.color.rgb = RGBColor(0x64, 0x74, 0x8b)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

def decode_wav(audio_bytes):
    """Decode PCM and IEEE-float WAV/WAVEX safely without Python wave's PCM-only limit."""
    if not audio_bytes:
        raise ValueError("The uploaded recording is empty.")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Each WAV must be at most 20 MB.")
    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as audio:
            if audio.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("This file is not a WAV recording. Export it as WAV; changing the extension is not enough.")
            sr, frames, channels = audio.samplerate, audio.frames, audio.channels
            if not 16000 <= sr <= 96000:
                raise ValueError(f"Sample rate is {sr} Hz; use a WAV between 16000 and 96000 Hz.")
            if channels not in (1, 2):
                raise ValueError(f"This recording has {channels} channels; use mono or stereo WAV.")
            duration = frames / sr
            if not 1.0 <= duration <= 30.0:
                raise ValueError(f"Recording duration is {duration:.2f} seconds; use 1–30 seconds (5–10 recommended).")
            # Check header bounds before allocating/decoding audio. No resampling,
            # padding, amplitude normalization or conversion to lossy integer PCM.
            samples = audio.read(frames=frames, dtype="float64", always_2d=True)
            if samples.shape != (frames, channels) or not np.isfinite(samples).all():
                raise ValueError("The recording is incomplete or contains non-finite samples. Re-export it as WAV.")
            quality_warnings = []
            if duration < 3:
                quality_warnings.append("Short recording: fewer than 3 seconds. Prefer a steady 5–10 second vowel for feature analysis.")
            if np.any(np.abs(samples) >= 0.999):
                quality_warnings.append("Some samples reach full scale. Check for clipping or excessive recording gain.")
            info = {"duration_seconds": duration, "sample_rate_hz": sr,
                    "channels": channels, "format": audio.format, "subtype": audio.subtype,
                    "warnings": quality_warnings}
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError("This WAV could not be decoded. Re-export it as PCM or IEEE-float WAV and try again.") from exc
    return samples, sr, info


def extract_voice_features(samples, sample_rate):
    """Measure four real descriptors. Never fabricate the other 18 classifier inputs."""
    # SoundFile returns samples x channels; Praat expects channels x samples.
    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64).T, sampling_frequency=sample_rate)
    if not 1 <= sound.duration <= 30:
        raise ValueError("Use a WAV recording between 1 and 30 seconds.")
    if not 16000 <= sound.sampling_frequency <= 96000:
        raise ValueError("Use a sample rate between 16 and 96 kHz.")
    if sound.n_channels > 2 or not np.isfinite(sound.values).all():
        raise ValueError("Use a finite mono or stereo recording.")
    sound = sound.convert_to_mono()
    if np.sqrt(np.mean(sound.values ** 2)) < 1e-5:
        raise ValueError("The recording is silent or too quiet.")
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    pv = pitch.selected_array["frequency"]
    pv = pv[pv > 0]
    if len(pv) < 10:
        raise ValueError("Insufficient voiced frames; record a steady sustained vowel.")
    harmonicity = sound.to_harmonicity()
    harmonics = harmonicity.values[harmonicity.values > -200]
    if not len(harmonics):
        raise ValueError("Harmonicity could not be measured for this recording.")
    fo, fhi, flo, hnr = float(pv.mean()), float(pv.max()), float(pv.min()), float(harmonics.mean())
    return {"MDVP:Fo(Hz)": fo, "MDVP:Fhi(Hz)": fhi, "MDVP:Flo(Hz)": flo, "HNR": hnr}


def measure_wav(audio_bytes, include_metadata=False):
    samples, sample_rate, info = decode_wav(audio_bytes)
    measured = extract_voice_features(samples, sample_rate)
    return (measured, info) if include_metadata else measured


def match_voice_features(csv_bytes, filenames):
    frame = pd.read_csv(BytesIO(csv_bytes))
    validate_features(frame)
    if "recording_id" not in frame or frame.recording_id.isna().any():
        raise ValueError("Feature CSV requires recording_id matching each WAV filename (or live_recording.wav).")
    if frame.recording_id.duplicated().any() or len(set(filenames)) != len(filenames):
        raise ValueError("Recording IDs and WAV filenames must be unique.")
    if set(frame.recording_id) != set(filenames):
        raise ValueError("CSV recording_id values must match exactly the current uploaded/recorded audio files.")
    return frame.set_index("recording_id").loc[filenames, ACOUSTIC_FEATURES]


# ══════════════════════════════════════════════
#  TAB 6 — VOICE PREDICT
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-hdr">🎤 Voice-Based Prediction & Research Report</div>',
                unsafe_allow_html=True)
    st.info("Upload a WAV file or record live voice. Analyze measured descriptors; supply matched complete acoustic features for model prediction, SHAP and a DOCX research report.")

    # Patient info
    v1, v2, v3 = st.columns(3)
    with v1: v_name   = st.text_input("Patient Name", key="v_name", placeholder="Full name")
    with v2: v_age    = st.number_input("Age", 1, 120, 50, key="v_age")
    with v3: v_gender = st.selectbox("Gender", ["Male","Female","Other"], key="v_gender")

    st.markdown("---")

    # Two input columns
    col_upload, col_mid, col_record = st.columns([10, 1, 10])

    with col_upload:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**📂 Upload WAV File(s)**")
        st.caption("PCM or IEEE-float WAV, including WAVEX · 1–30 seconds · 16–96 kHz · mono/stereo · up to 20 MB each")
        v_files = st.file_uploader("wav files", type=["wav"],
                                   accept_multiple_files=True,
                                   label_visibility="collapsed",
                                   key="v_uploader")
        if v_files:
            for f in v_files:
                st.audio(f)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_mid:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:1.2rem;font-weight:800;color:#94a3b8;">OR</div>',
                    unsafe_allow_html=True)

    with col_record:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**🎤 Record Live Voice**")
        st.caption("Click the mic, speak clearly for 5–10 seconds, then stop")
        v_recorded = None
        if RECORDER_AVAILABLE:
            rb = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#7c3aed",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=3.0,
                key="v_recorder"
            )
            if rb:
                st.audio(rb, format="audio/wav")
                v_recorded = rb
                st.success("✅ Recording captured!")
        else:
            st.warning("Install `audio-recorder-streamlit` to enable live recording.")
            st.code("pip install audio-recorder-streamlit", language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    has_v_upload = bool(v_files)
    has_v_record = v_recorded is not None

    feature_file = st.file_uploader("Matched complete acoustic features (CSV)", type=["csv"], key="v_features")
    st.caption("For voice prediction, provide recording_id plus all 22 feature columns, produced by an extractor "
               "compatible with the training dataset. Use each WAV filename as recording_id; use live_recording.wav "
               "for microphone audio. Praat pitch/HNR alone cannot reproduce this feature schema.")
    compatible = st.checkbox("I confirm these feature rows belong to these recordings and use the training feature definitions/units.",
                             key="v_compatible")
    recordings = [(f.name, f.getvalue()) for f in (v_files or [])]
    if v_recorded:
        recordings.append(("live_recording.wav", v_recorded))
    signature = hashlib.sha256()
    signature.update(b"wav-decoder-soundfile-v2")
    signature.update(model_bundle["metadata"]["dataset_sha256"].encode())
    for filename, content in recordings:
        signature.update(filename.encode()); signature.update(content)
    signature.update(feature_file.getvalue() if feature_file else b"")
    signature.update(str(compatible).encode())
    if st.session_state.get("voice_signature") != signature.hexdigest():
        st.session_state.pop("voice_result", None)
        st.session_state["voice_signature"] = signature.hexdigest()
    if not recordings:
        st.info("⬆️ Upload a WAV file or record your voice above, then click Analyze.")
    if recordings and st.button("🔍 Analyze & Generate Report", width="stretch", type="primary", key="v_analyze"):
        st.session_state.pop("voice_result", None)
        try:
            with st.spinner("Analyzing voice…"):
                measured, audio_info = [], []
                for filename, content in recordings:
                    try:
                        descriptors, info = measure_wav(content, include_metadata=True)
                    except (ValueError, RuntimeError) as exc:
                        raise ValueError(f"{filename}: {exc}") from exc
                    measured.append(descriptors)
                    audio_info.append(info)
                matched = None
                scores = None
                if feature_file is not None:
                    if not compatible:
                        raise ValueError("Confirm the feature definitions and recording matches before prediction.")
                    matched = match_voice_features(feature_file.getvalue(), [name for name, _ in recordings])
                    scores, outside = predict_input(model_bundle, matched)
                st.session_state.voice_result = {"measured": measured, "matched": matched,
                    "scores": scores, "names": [name for name, _ in recordings], "audio_info": audio_info}
        except Exception as exc:
            st.error(f"Recording or feature validation failed: {exc}")
    if "voice_result" in st.session_state:
        result = st.session_state.voice_result
        for filename, info in zip(result["names"], result.get("audio_info", [])):
            st.caption(f"{filename}: {info['duration_seconds']:.2f} s · {info['sample_rate_hz']} Hz · "
                       f"{info['channels']} channel(s) · {info['format']} / {info['subtype']}")
            for message in info["warnings"]:
                st.warning(f"{filename}: {message}")
        measured_table = pd.DataFrame(result["measured"], index=result["names"])
        st.dataframe(measured_table.rename_axis("Recording"))
        st.caption("These four Praat descriptors are measured from audio and shown for quality review. "
                   "They are not substituted into the uploaded model feature rows or claimed to be MDVP-equivalent.")
        if result["matched"] is None:
            st.warning("No disease prediction or SHAP plot is generated without complete matched features. "
                       "The original constant-filled feature vector has been removed.")
            st.download_button("Download measured voice descriptors", measured_table.to_csv(),
                               "measured_voice_descriptors.csv", "text/csv")
        else:
            sample_index = st.selectbox("Recording to explain and report", range(len(result["names"])),
                                         format_func=lambda i: result["names"][i], key="v_sample")
            row = result["matched"].iloc[[sample_index]].reset_index(drop=True)
            pk_prob = float(result["scores"][sample_index]); hl_prob = 1.0 - pk_prob
            outside = int(((row < model_bundle["minimum"]) | (row > model_bundle["maximum"])).sum(axis=1).iloc[0])
            if outside:
                st.warning(f"{outside} feature(s) fall outside the training ranges; this prediction may be out of distribution.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Parkinson-class model score", f"{pk_prob:.1%}")
            m2.metric("Control-class model score", f"{hl_prob:.1%}")
            m3.metric("CV patient MCC (mean)", f"{model_bundle['summary'].loc['Patients'][('MCC', 'mean')]:.3f}")
            st.caption("The score and explanation refer to the selected recording's supplied feature row. "
                       "Scores are not calibrated clinical risk estimates and are not averaged across unrelated people.")
            render_shap(model_bundle, row, "voice_shap_method")
            level = "Parkinson-class pattern" if pk_prob >= DECISION_THRESHOLD else "Control-class pattern"
            advice = "Research model output only; do not use this score for diagnosis or treatment."
            box_css = "pred-pos" if pk_prob >= DECISION_THRESHOLD else "pred-neg"
            st.markdown(f'''<div class="pred-box {box_css}"><div class="pred-title">{level}</div>
              <div class="pred-conf">{advice}</div></div>''', unsafe_allow_html=True)
            st.markdown("---")
            doc_buf = generate_voice_docx(
                v_name, v_age, v_gender, hl_prob, pk_prob, level,
                float(row["MDVP:Fo(Hz)"].iloc[0]), float(row["HNR"].iloc[0]), advice,
                features, row.iloc[0].tolist(),
                source_label=f"Matched acoustic CSV: {result['names'][sample_index]}",
                cv_summary=model_bundle["summary"].loc["Patients"])
            safe_name = re.sub(r"[^\w-]", "_", v_name)[:60] or "Anonymous"
            fname = f"PD_Research_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.docx"
            st.download_button("📄 Download Research Report (.docx)", data=doc_buf,
                file_name=fname, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width="stretch", key="v_download")

