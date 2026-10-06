import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
import modeling as m


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(23)
    groups = np.repeat(np.arange(32), 6)
    y = (groups >= 8).astype(int)
    patient_effect = rng.normal(0, 0.3, (32, 22))
    X = rng.normal(0, 0.25, (len(y), 22)) + patient_effect[groups] + 2 * y[:, None]
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(22)])
    frame["name"] = [f"phon_R01_S{g + 1:02d}_{i % 6 + 1}" for i, g in enumerate(groups)]
    frame["status"] = y
    return frame


def small_stack():
    bases = m._base_pipelines()
    bases[0][1].set_params(clf__max_iter=10)
    bases[1][1].set_params(clf__n_estimators=10, clf__n_jobs=1)
    return m.GroupedStackingClassifier(estimators=bases)


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    cache = tmp_path_factory.mktemp("model") / "test.pkl"
    with threadpool_limits(limits=1):
        result = m.train_model(data, cache_path=cache, estimator=small_stack())
    return result, cache


def test_patient_boundaries_and_threshold_decisions(trained, data):
    result, _ = trained
    _, _, _, _, metrics, predictions, probabilities, y, _, _ = result
    groups = m.prepare_data(data)[4]
    held_out = []
    for fold in metrics["fold_audit"]:
        train, test = set(fold["train_patients"]), set(fold["test_patients"])
        assert train.isdisjoint(test)
        held_out.extend(test)
        for inner in fold["threshold_splits"]:
            a, b = set(inner["train_patients"]), set(inner["validation_patients"])
            assert a.isdisjoint(b) and a | b == train
            for c, d in inner["stack_splits"]:
                assert set(c).isdisjoint(d) and set(c) | set(d) == a
        for a, b in fold["stack_splits"]:
            assert set(a).isdisjoint(b) and set(a) | set(b) == train
        mask = np.isin(groups, list(test))
        np.testing.assert_array_equal(predictions[mask], (probabilities[mask] >= fold["threshold"]).astype(int))
    assert len(held_out) == len(set(held_out)) == 32
    assert np.isfinite(probabilities).all()
    assert metrics["decision_threshold"] == np.mean(metrics["fold_thresholds"])


def test_reload_prediction_and_selected_shape(trained, data, monkeypatch):
    result, cache = trained
    model, _, features, selected, *_ = result
    restored = joblib.load(cache)["result"][0]
    X = data[features].iloc[[0]].to_numpy()
    pred, prob, transformed = m.predict_single(restored, X)
    assert transformed.shape == (1, 10)
    assert len(selected) == 10
    np.testing.assert_allclose(prob, model.predict_proba(X)[0])
    assert pred == int(prob[1] >= model.decision_threshold_)
    # A non-default production threshold must drive both entry points.
    restored.decision_threshold_ = max(0.0, prob[1] - 0.01)
    assert m.predict_single(restored, X)[0] == restored.predict(X)[0] == 1
    restored.decision_threshold_ = min(1.0, prob[1] + 0.01)
    assert m.predict_single(restored, X)[0] == restored.predict(X)[0] == 0
    def no_retrain(*args, **kwargs):
        pytest.fail("Matching dataset/config should load cache without fitting")
    monkeypatch.setattr(m.GroupedStackingClassifier, "fit", no_retrain)
    cached = m.train_model(data, cache_path=cache, estimator=small_stack())
    np.testing.assert_allclose(cached[6], result[6])


def test_dataset_change_invalidates_cache(trained, data, monkeypatch):
    _, cache = trained
    modified = data.copy()
    modified.loc[0, "feature_0"] += 0.123
    def fitting_requested(*args, **kwargs):
        raise RuntimeError("retrain requested")
    monkeypatch.setattr(m, "tune_threshold", fitting_requested)
    with pytest.raises(RuntimeError, match="retrain requested"):
        m.train_model(modified, cache_path=cache, estimator=small_stack())


def test_missing_patients_and_inconsistent_labels_rejected(data):
    bad = data.copy()
    bad.loc[0, "name"] = "unknown"
    with pytest.raises(ValueError, match="patient IDs"):
        m.prepare_data(bad)
    bad = data.copy()
    bad.loc[0, "status"] = 1
    with pytest.raises(ValueError, match="consistent status"):
        m.prepare_data(bad)
    bad = data.copy()
    bad.loc[0, "feature_0"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        m.prepare_data(bad)


def test_too_few_healthy_patients_fails(data):
    frame, _, X, y, groups = m.prepare_data(data)
    keep = ~frame["patient_id"].isin([f"phon_R01_S{i:02d}" for i in range(1, 6)])
    with pytest.raises(ValueError, match="at least 5 patients"):
        m.grouped_splits(X[keep], y[keep], groups[keep], 5, strict=True)


@pytest.mark.parametrize("scores", [[0.5, 0.5, 0.5, 0.5], [0.1, 0.8, 0.2, 0.9], [0.9, 0.8, 0.2, 0.1]])
def test_youden_threshold_is_finite_and_reproducible(scores):
    threshold = m.youden_threshold([0, 1, 0, 1], scores)
    assert np.isfinite(threshold) and 0 <= threshold <= 1
    assert threshold == m.youden_threshold([0, 1, 0, 1], scores)


def test_selected_transform_matches_base_pipeline(trained, data):
    model, _, features, *_ = trained[0]
    X = data[features].iloc[:3].to_numpy()
    p = model.hgb_pipeline_
    actual = model.transform_for_explanation(X)
    expected = p.named_steps["selector"].transform(p.named_steps["scaler"].transform(X))
    np.testing.assert_allclose(actual, expected)
    # Identical deterministic preprocessing on the full fit makes the UI's
    # selected-feature controls valid for every learner.
    supports = [p.named_steps["selector"].get_support() for _, p in model.base_pipelines_]
    for support in supports[1:]:
        np.testing.assert_array_equal(support, supports[0])


def test_real_default_pipeline_fits(data):
    _, _, X, y, groups = m.prepare_data(data)
    with threadpool_limits(limits=1):
        fitted = m.GroupedStackingClassifier().fit(X, y, groups)
    assert fitted.predict_proba(X[:2]).shape == (2, 2)


def test_streamlit_missing_data(tmp_path):
    import shutil
    from streamlit.testing.v1 import AppTest
    project = Path(__file__).resolve().parents[1]
    for name in ["app.py", "modeling.py"]:
        shutil.copy2(project / name, tmp_path / name)
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("upload" in item.value for item in app.info)


def test_streamlit_model_prediction_and_empty_filters(trained, data, tmp_path, monkeypatch):
    import shutil
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    columns = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
        "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
    ]
    mapping = dict(zip(trained[0][2], columns))
    renamed = data.rename(columns=mapping)
    renamed.to_csv(tmp_path / "parkinsons_dataset.csv", index=False)
    result = list(trained[0])
    result[2] = columns
    result[3] = [mapping[f] for f in result[3]]
    monkeypatch.setattr(m, "train_model", lambda *a, **k: tuple(result))
    project = Path(__file__).resolve().parents[1]
    for name in ["app.py", "modeling.py"]:
        shutil.copy2(project / name, tmp_path / name)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Saved decision threshold" for metric in app.metric)
    prediction_button = next(button for button in app.button if "Run Prediction" in button.label)
    prediction_button.click().run()
    assert not app.exception
    assert any("Model classification:" in item.value for item in app.markdown)
    status = next(widget for widget in app.multiselect if widget.label == "Status")
    status.set_value([]).run()
    assert not app.exception
    assert any("No recordings match" in item.value for item in app.info)
