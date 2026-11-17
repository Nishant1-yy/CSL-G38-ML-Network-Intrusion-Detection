"""Metrics and plots for held-out intrusion detection evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def _score_matrix(pipeline: Any, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features))
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(features))
        if scores.ndim == 1:
            return np.column_stack([-scores, scores])
        return scores
    return None


def evaluate_model(
    pipeline: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    class_names: list[str],
    output_dir: str | Path,
    model_name: str = "selected_model",
) -> dict[str, Any]:
    """Write held-out metrics and visualizations; return a JSON-safe summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = pipeline.predict(features)
    scores = _score_matrix(pipeline, features)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }

    matrix = confusion_matrix(labels, predictions, labels=class_names)
    figure, axis = plt.subplots(figsize=(max(6, len(class_names) * 1.1), 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Held-out confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(destination / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(figure)

    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        name: values
        for name, values in report.items()
        if isinstance(values, dict) and name in class_names
    }
    metrics["classification_report"] = report

    if scores is not None and scores.shape[1] == len(class_names):
        roc_figure, roc_axis = plt.subplots(figsize=(7, 6))
        pr_figure, pr_axis = plt.subplots(figsize=(7, 6))
        binary_truth = np.column_stack(
            [np.asarray(labels) == class_name for class_name in class_names]
        )
        auc_values = []
        for index, name in enumerate(class_names):
            if len(np.unique(binary_truth[:, index])) < 2:
                continue
            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_truth[:, index], scores[:, index]
            )
            area = float(auc(false_positive_rate, true_positive_rate))
            auc_values.append(area)
            roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{name} (AUC {area:.3f})",
            )
            precision, recall, _ = precision_recall_curve(
                binary_truth[:, index], scores[:, index]
            )
            pr_axis.plot(recall, precision, label=name)
        if auc_values:
            metrics["roc_auc_ovr_macro"] = float(np.mean(auc_values))
            roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
            roc_axis.set(
                xlabel="False positive rate",
                ylabel="True positive rate",
                title="One-vs-rest ROC curves",
            )
            roc_axis.legend(loc="lower right", fontsize="small")
            roc_figure.tight_layout()
            roc_figure.savefig(destination / f"{model_name}_roc.png", dpi=150)
            pr_axis.set(
                xlabel="Recall",
                ylabel="Precision",
                title="One-vs-rest precision-recall curves",
            )
            pr_axis.legend(loc="best", fontsize="small")
            pr_figure.tight_layout()
            pr_figure.savefig(
                destination / f"{model_name}_precision_recall.png", dpi=150
            )
        plt.close(roc_figure)
        plt.close(pr_figure)

    metrics_path = destination / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics
