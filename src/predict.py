"""Load trained artifacts and make validated traffic predictions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ARTIFACT_VERSION = 1


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local joblib model bundle."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    try:
        bundle = joblib.load(path)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Could not load model artifact {path}: {exc}") from exc
    required = {"pipeline", "feature_columns", "feature_types", "class_names"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Model artifact must contain: {', '.join(sorted(required))}.")
    if bundle.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {bundle.get('artifact_version')}."
        )
    if not hasattr(bundle["pipeline"], "predict"):
        raise ValueError("Model artifact does not contain a prediction pipeline.")
    return bundle


def _input_frame(
    features: Mapping[str, Any] | pd.DataFrame,
    expected_columns: list[str],
    feature_types: Mapping[str, str],
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("Prediction accepts exactly one traffic record.")
        frame = features.copy()
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    else:
        raise TypeError("Traffic features must be a mapping or a one-row DataFrame.")

    missing = [column for column in expected_columns if column not in frame.columns]
    unexpected = [
        str(column) for column in frame.columns if column not in expected_columns
    ]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing features: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected features: {', '.join(unexpected)}")
        raise ValueError("Invalid feature set (" + "; ".join(problems) + ").")

    frame = frame.loc[:, expected_columns]
    for column, kind in feature_types.items():
        if kind == "numeric":
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {column!r} must be numeric.") from exc
            values = frame[column].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Feature {column!r} cannot be infinite.")
        elif kind == "categorical":
            frame[column] = frame[column].astype("object")
        else:
            raise ValueError(f"Unsupported feature type for {column!r}: {kind!r}.")
    return frame


def predict_traffic(
    model: dict[str, Any],
    features: Mapping[str, Any] | pd.DataFrame,
) -> dict[str, Any]:
    """Predict a traffic class and provide confidence when the model supports it."""
    expected_columns = list(model["feature_columns"])
    frame = _input_frame(features, expected_columns, model["feature_types"])
    pipeline = model["pipeline"]
    prediction = str(pipeline.predict(frame)[0])
    result: dict[str, Any] = {"prediction": prediction, "confidence": None}

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[0]
        classes = [str(value) for value in pipeline.classes_]
        result["confidence"] = float(max(probabilities))
        result["probabilities"] = {
            label: float(probability)
            for label, probability in zip(classes, probabilities, strict=True)
        }
    return result
