import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.feature_selection import select_features
from src.preprocessing import build_preprocessor, normalize_labels, prepare_features


def test_loads_labeled_csv_and_drops_duplicate_records(tmp_path):
    source = tmp_path / "flows.csv"
    pd.DataFrame(
        {
            "duration": [1, 1, 4],
            "service": ["http", "http", "dns"],
            " Label": ["BENIGN", "BENIGN", "DoS"],
        }
    ).to_csv(source, index=False)

    features, labels = load_dataset(source)

    assert len(features) == 2
    assert labels.tolist() == ["BENIGN", "DoS"]


def test_load_dataset_reports_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
        load_dataset(tmp_path / "missing")


def test_loads_headerless_nsl_kdd_records(tmp_path):
    source = tmp_path / "KDDTrain+.txt"
    record = ["0", "tcp", "http", "SF"] + ["0"] * 37 + ["normal", "0"]
    source.write_text(",".join(record) + "\n", encoding="utf-8")

    features, labels = load_dataset(source)

    assert features.shape == (1, 42)
    assert "difficulty" in features.columns
    assert labels.tolist() == ["normal"]


def test_preprocessor_fits_on_training_and_ignores_unseen_categories():
    training = pd.DataFrame(
        {
            "bytes": [1.0, 3.0, None, 4.0],
            "service": ["web", "web", "dns", "dns"],
        }
    )
    validation = pd.DataFrame({"bytes": [1000.0], "service": ["unseen"]})
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression()),
        ]
    )
    pipeline.fit(training, ["Benign", "DOS", "Benign", "DOS"])

    numeric_imputer = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    transformed = pipeline.named_steps["preprocessor"].transform(validation)

    assert numeric_imputer.statistics_[0] == 3.0
    assert transformed.shape[0] == 1
    assert pipeline.predict(validation).tolist() in (["Benign"], ["DOS"])


def test_feature_selection_drops_identifiers_and_infinite_values():
    features = pd.DataFrame(
        {
            " Flow ID ": ["id-1"],
            "duration": [float("inf")],
            "service": ["http"],
        }
    )

    selected = select_features(features)
    prepared = prepare_features(features)

    assert " Flow ID " not in selected
    assert prepared["duration"].isna().all()
    assert list(prepared.columns) == ["duration", "service"]


def test_nsl_kdd_labels_map_to_standard_attack_categories():
    labels = normalize_labels(
        pd.Series(["normal.", "neptune", "ipsweep", "warezclient"])
    )

    assert labels.tolist() == ["Benign", "DOS", "PROBE", "R2L"]
