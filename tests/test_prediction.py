import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import ARTIFACT_VERSION, load_model, predict_traffic
from src.preprocessing import build_preprocessor


@pytest.fixture
def model_bundle(tmp_path):
    training = pd.DataFrame(
        {
            "duration": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
            "service": ["web", "web", "dns", "dns", "web", "dns"],
        }
    )
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(random_state=7)),
        ]
    )
    pipeline.fit(training, ["Benign", "Benign", "DOS", "DOS", "Benign", "DOS"])
    bundle = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": pipeline,
        "feature_columns": ["duration", "service"],
        "feature_types": {"duration": "numeric", "service": "categorical"},
        "class_names": ["Benign", "DOS"],
        "selected_model": "logistic_regression",
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path, bundle


def test_model_load_and_prediction_preserve_feature_order(model_bundle):
    path, _ = model_bundle
    model = load_model(path)

    result = predict_traffic(model, {"service": "web", "duration": 0.05})

    assert result["prediction"] == "Benign"
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"Benign", "DOS"}


def test_prediction_rejects_missing_and_unexpected_features(model_bundle):
    _, model = model_bundle

    with pytest.raises(ValueError, match="missing features"):
        predict_traffic(model, {"duration": 0.1})
    with pytest.raises(ValueError, match="unexpected features"):
        predict_traffic(model, {"duration": 0.1, "service": "web", "extra": 1})


def test_prediction_rejects_invalid_values_and_multiple_rows(model_bundle):
    _, model = model_bundle

    with pytest.raises(ValueError, match="must be numeric"):
        predict_traffic(model, {"duration": "not-a-number", "service": "web"})
    with pytest.raises(ValueError, match="exactly one"):
        predict_traffic(
            model,
            pd.DataFrame({"duration": [0.1, 0.2], "service": ["web", "dns"]}),
        )


def test_model_load_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model(tmp_path / "missing.joblib")
