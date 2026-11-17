"""Run binary and five-class NSL-KDD experiments and write reproducible outputs."""

from __future__ import annotations

import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dataset, load_official_test
from src.preprocessing import (
    ATTACK_GROUPS,
    build_preprocessor,
    normalize_labels,
    prepare_features,
)

RANDOM_STATE = 42
CLASS_NAMES = ["Benign", "DOS", "PROBE", "R2L", "U2R"]
BINARY_NAMES = ["Benign", "Attack"]
KNN_TRAIN_LIMIT = 10_000
CPU_WORKERS = max(1, min(2, os.cpu_count() or 1))

FEATURE_GROUPS = {
    "Basic connection features": [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
    ],
    "Content features": [
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
    ],
    "Time-based traffic features": [
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
    ],
    "Host-based traffic features": [
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ],
}


def _environment() -> dict[str, Any]:
    cpu_name = platform.processor()
    if not cpu_name and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.casefold().startswith("model name"):
                cpu_name = line.split(":", 1)[-1].strip()
                break

    memory_gib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_gib = round(int(line.split()[1]) / 1024**2, 2)
                break

    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "python-docx",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {
        "os": platform.platform(),
        "cpu": cpu_name or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gib": memory_gib,
        "python": platform.python_version(),
        "libraries": packages,
        "scikit_learn": sklearn.__version__,
    }


def _model_specs() -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"max_iter": 500, "class_weight": "balanced"},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "max_depth": 20,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=80,
                max_depth=20,
                class_weight="balanced_subsample",
                n_jobs=CPU_WORKERS,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 80,
                "max_depth": 20,
                "class_weight": "balanced_subsample",
                "n_jobs": CPU_WORKERS,
            },
        ),
        "Linear SVM": (
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            {"C": 0.1, "class_weight": "balanced", "max_iter": 5000},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=CPU_WORKERS,
            ),
            {
                "n_neighbors": 5,
                "weights": "distance",
                "n_jobs": CPU_WORKERS,
                "training_row_limit": KNN_TRAIN_LIMIT,
            },
        ),
        "Gaussian Naive Bayes": (
            GaussianNB(var_smoothing=1e-9),
            {"var_smoothing": 1e-9},
        ),
        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=31,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "learning_rate": 0.1,
                "class_weight": "balanced",
            },
        ),
    }


def _scores(estimator: Any, features: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = [str(value) for value in estimator.classes_]
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features)), classes
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features))
        if scores.ndim == 1:
            return np.column_stack([-scores, scores]), classes
        return scores, classes
    raise ValueError(f"{type(estimator).__name__} does not provide classification scores.")


def _binary_metrics(
    truth: np.ndarray, predicted: np.ndarray, scores: np.ndarray, score_classes: list[str]
) -> dict[str, float]:
    positive_index = score_classes.index("Attack")
    positive_scores = scores if scores.ndim == 1 else scores[:, positive_index]
    false_alarm_denominator = int(np.sum(truth == "Benign"))
    false_alarms = int(np.sum((truth == "Benign") & (predicted == "Attack")))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(
            precision_score(truth, predicted, pos_label="Attack", zero_division=0)
        ),
        "recall": float(
            recall_score(truth, predicted, pos_label="Attack", zero_division=0)
        ),
        "f1": float(f1_score(truth, predicted, pos_label="Attack", zero_division=0)),
        "auc": float(roc_auc_score(truth == "Attack", positive_scores)),
        "far": float(false_alarms / false_alarm_denominator)
        if false_alarm_denominator
        else 0.0,
        "false_alarms": false_alarms,
        "true_negatives": false_alarm_denominator - false_alarms,
    }


def _multiclass_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    score_classes: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=CLASS_NAMES, zero_division=0
    )
    class_metrics = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    ordered_scores = scores
    if scores.ndim == 1:
        ordered_scores = np.column_stack([-scores, scores])
    index = [score_classes.index(name) for name in CLASS_NAMES]
    ordered_scores = ordered_scores[:, index]
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(
            precision_score(truth, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, predicted, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(truth, predicted, average="weighted", zero_division=0)
        ),
        "per_class": class_metrics,
    }
    try:
        class_auc = [
            roc_auc_score(truth == name, ordered_scores[:, index])
            for index, name in enumerate(CLASS_NAMES)
            if len(np.unique(truth == name)) == 2
        ]
        result["auc_ovr_macro"] = float(np.mean(class_auc)) if class_auc else None
    except ValueError:
        result["auc_ovr_macro"] = None
    return result


def _knn_training_indices(labels: np.ndarray) -> np.ndarray:
    indices = np.arange(len(labels))
    if len(indices) <= KNN_TRAIN_LIMIT:
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=KNN_TRAIN_LIMIT,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return np.sort(selected)


def _plot_distributions(
    train_labels: pd.Series, test_labels: pd.Series, output: Path
) -> None:
    train_counts = train_labels.value_counts().reindex(CLASS_NAMES, fill_value=0)
    test_counts = test_labels.value_counts().reindex(CLASS_NAMES, fill_value=0)
    positions = np.arange(len(CLASS_NAMES))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - 0.2, train_counts, width=0.4, label="Training")
    axis.bar(positions + 0.2, test_counts, width=0.4, label="Official test")
    axis.set_xticks(positions, CLASS_NAMES)
    axis.set_ylabel("Records")
    axis.set_title("NSL-KDD five-category class distribution")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_workflow(output: Path) -> None:
    labels = [
        "Official NSL-KDD\ntrain/test files",
        "Train-fitted\nimputation, encoding\nand scaling",
        "Seven classifiers\nbinary + five-class",
        "Official test\nmetrics and checks",
        "JSON, CSV and\nfigure outputs",
    ]
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.set_xlim(0, len(labels))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, label in enumerate(labels):
        x = index + 0.08
        axis.text(
            x + 0.42,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#D9EAF7",
                "edgecolor": "#4472C4",
            },
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(x + 0.94, 0.52),
                xytext=(x + 0.86, 0.52),
                arrowprops={"arrowstyle": "->", "color": "#404040", "lw": 1.5},
            )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_binary_results(
    binary_results: dict[str, Any],
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    names = list(binary_results)
    metric_names = ["accuracy", "precision", "recall", "f1", "auc", "far"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, metric in zip(axes.flat, metric_names, strict=True):
        values = [binary_results[name]["metrics"][metric] for name in names]
        bars = axis.barh(names, values, color="#4472C4")
        axis.set_title(metric.upper())
        axis.set_xlim(0, 1)
        axis.grid(axis="x", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    figure.suptitle("Binary experiment: held-out test metrics")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_binary_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, name in zip(axes.flat, names, strict=False):
        matrix = np.asarray(binary_results[name]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name, fontsize=9)
        axis.set_xticks(range(len(BINARY_NAMES)), BINARY_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(len(BINARY_NAMES)), BINARY_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes.flat[-1].axis("off")
    figure.suptitle("Binary experiment: confusion matrices")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_binary_confusion_matrices.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for name in names:
        truth, scores = roc_data[name]
        truth = truth == "Attack"
        false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
        area = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC {area:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Binary experiment: ROC curves",
    )
    axis.legend(loc="lower right", fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "fig_binary_roc.png", dpi=180)
    plt.close(figure)


def _plot_multiclass_confusion(
    matrix: list[list[int]], model_name: str, output: Path
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set(xlabel="Predicted label", ylabel="True label", title=model_name)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_feature_importance(
    estimator: Any,
    feature_names: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output: Path,
) -> list[dict[str, Any]]:
    classifier = estimator
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
        method = "model impurity importance"
    elif hasattr(classifier, "coef_"):
        importance = np.mean(np.abs(classifier.coef_), axis=0)
        method = "mean absolute coefficient"
    else:
        sample_count = min(500, len(test_labels))
        importance = permutation_importance(
            classifier,
            test_features[:sample_count],
            test_labels[:sample_count],
            scoring="f1_macro",
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=CPU_WORKERS,
        ).importances_mean
        method = f"permutation importance on {sample_count} held-out rows"

    order = np.argsort(importance)[-15:][::-1]
    top = [
        {"feature": str(feature_names[index]), "importance": float(importance[index])}
        for index in order
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [item["feature"] for item in reversed(top)],
        [item["importance"] for item in reversed(top)],
        color="#70AD47",
    )
    axis.set_xlabel(method.title())
    axis.set_title("Top transformed-feature importance values")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return top


def _test_cases(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    test_y: pd.Series,
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
    preprocessor: Any,
) -> list[dict[str, Any]]:
    checks = [
        (
            "TC-01",
            "Official training data loaded",
            125_000 <= len(train_y) <= 126_000,
            f"{len(train_y)} labeled training rows",
        ),
        (
            "TC-02",
            "Official test data loaded",
            22_000 <= len(test_y) <= 23_000,
            f"{len(test_y)} labeled test rows",
        ),
        (
            "TC-03",
            "Training and test feature schemas match",
            list(train_x.columns) == list(test_x.columns),
            f"{len(train_x.columns)} matching predictor columns",
        ),
        (
            "TC-04",
            "NSL-KDD labels normalize to five categories",
            set(train_y.unique()) == set(CLASS_NAMES),
            f"Observed training labels: {sorted(train_y.unique().tolist())}",
        ),
        (
            "TC-05",
            "Test categories were represented during training",
            set(test_y.unique()).issubset(set(train_y.unique())),
            f"Observed test labels: {sorted(test_y.unique().tolist())}",
        ),
        (
            "TC-06",
            "Train-fitted preprocessing yields finite, aligned matrices",
            transformed_train.shape[0] == len(train_y)
            and transformed_test.shape[0] == len(test_y)
            and transformed_train.shape[1] == transformed_test.shape[1]
            and np.isfinite(transformed_train).all()
            and np.isfinite(transformed_test).all(),
            f"Transformed shapes: {transformed_train.shape} and {transformed_test.shape}",
        ),
        (
            "TC-07",
            "Numeric preprocessing was fit on the training partition",
            list(getattr(preprocessor, "feature_names_in_", []))
            == list(train_x.columns)
            and getattr(preprocessor, "n_features_in_", None) == len(train_x.columns),
            f"Fitted input schema has {getattr(preprocessor, 'n_features_in_', 0)} columns",
        ),
    ]
    return [
        {
            "id": case_id,
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
        }
        for case_id, name, passed, observed in checks
    ]


def run_pipeline(
    data_dir: str | Path = "data/raw", output_dir: str | Path = "results"
) -> dict[str, Any]:
    """Train both experiments on NSL-KDD train and evaluate on official test."""
    data_path = Path(data_dir).expanduser()
    destination = Path(output_dir).expanduser()
    figure_dir = destination / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_features_raw, train_labels_raw = load_dataset(data_path)
    official_test = load_official_test(data_path)
    if official_test is None:
        raise FileNotFoundError(
            f"Official KDDTest+.txt was not found beside {data_path}; "
            "both NSL-KDD files are required."
        )
    test_features_raw, test_labels_raw = official_test
    train_x = prepare_features(train_features_raw)
    train_y = normalize_labels(train_labels_raw)
    test_y = normalize_labels(test_labels_raw)
    prepared_test_x = prepare_features(test_features_raw)
    if list(train_x.columns) != list(prepared_test_x.columns):
        raise ValueError("Training and official test feature schemas do not match.")
    test_x = prepared_test_x
    if set(train_y.unique()) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected training categories: {sorted(train_y.unique())}")
    if not set(test_y.unique()).issubset(set(train_y.unique())):
        raise ValueError(f"Unexpected official test categories: {sorted(test_y.unique())}")

    preprocessor = build_preprocessor()
    train_matrix = preprocessor.fit_transform(train_x)
    test_matrix = preprocessor.transform(test_x)
    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    test_matrix = np.asarray(test_matrix, dtype=np.float64)
    feature_names = preprocessor.get_feature_names_out()

    binary_train_y = np.where(train_y.to_numpy() == "Benign", "Benign", "Attack")
    binary_test_y = np.where(test_y.to_numpy() == "Benign", "Benign", "Attack")
    binary_results: dict[str, Any] = {}
    binary_roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    multiclass_results: dict[str, Any] = {}
    fitted_multiclass: dict[str, Any] = {}
    training_rows: dict[str, int] = {}
    settings: dict[str, Any] = {}

    model_specs = _model_specs()
    knn_indices = _knn_training_indices(train_y.to_numpy())
    for model_name, (template, model_settings) in model_specs.items():
        settings[model_name] = model_settings
        for experiment, target_train, target_test, store in (
            ("binary", binary_train_y, binary_test_y, binary_results),
            ("five_class", train_y.to_numpy(), test_y.to_numpy(), multiclass_results),
        ):
            if model_name == "K-Nearest Neighbors":
                fit_indices = knn_indices
            else:
                fit_indices = np.arange(len(target_train))

            estimator = clone(template)
            started = time.perf_counter()
            estimator.fit(train_matrix[fit_indices], target_train[fit_indices])
            training_seconds = time.perf_counter() - started
            predictions = estimator.predict(test_matrix)
            score_values, score_classes = _scores(estimator, test_matrix)
            training_rows[f"{experiment}:{model_name}"] = int(len(fit_indices))
            print(
                f"{experiment} | {model_name}: trained on {len(fit_indices)} rows "
                f"in {training_seconds:.2f}s",
                flush=True,
            )

            if experiment == "binary":
                metrics = _binary_metrics(
                    target_test, predictions, score_values, score_classes
                )
                positive_index = score_classes.index("Attack")
                positive_scores = (
                    score_values
                    if score_values.ndim == 1
                    else score_values[:, positive_index]
                )
                store[model_name] = {
                    "metrics": metrics,
                    "training_seconds": float(training_seconds),
                    "confusion_matrix": confusion_matrix(
                        target_test, predictions, labels=BINARY_NAMES
                    ).tolist(),
                }
                binary_roc_data[model_name] = (target_test, positive_scores)
            else:
                store[model_name] = {
                    "metrics": _multiclass_metrics(
                        target_test, predictions, score_values, score_classes
                    ),
                    "training_seconds": float(training_seconds),
                    "confusion_matrix": confusion_matrix(
                        target_test, predictions, labels=CLASS_NAMES
                    ).tolist(),
                }
                fitted_multiclass[model_name] = estimator

    binary_comparison = pd.DataFrame(
        [
            {"model": name, **row["metrics"], "training_seconds": row["training_seconds"]}
            for name, row in binary_results.items()
        ]
    ).sort_values(["f1", "auc"], ascending=False)
    binary_comparison.to_csv(destination / "metrics_binary.csv", index=False)
    multiclass_comparison = pd.DataFrame(
        [
            {
                "model": name,
                **{
                    key: value
                    for key, value in row["metrics"].items()
                    if key != "per_class"
                },
                "training_seconds": row["training_seconds"],
            }
            for name, row in multiclass_results.items()
        ]
    ).sort_values(["f1_macro", "accuracy"], ascending=False)
    multiclass_comparison.to_csv(destination / "metrics_five_class.csv", index=False)

    _plot_distributions(train_y, test_y, figure_dir / "fig_class_distribution.png")
    _plot_workflow(figure_dir / "fig_workflow.png")
    _plot_binary_results(binary_results, binary_roc_data, figure_dir)

    best_multiclass = max(
        multiclass_results,
        key=lambda name: (
            multiclass_results[name]["metrics"]["f1_macro"],
            multiclass_results[name]["metrics"]["accuracy"],
        ),
    )
    best_binary = max(
        binary_results,
        key=lambda name: (
            binary_results[name]["metrics"]["f1"],
            binary_results[name]["metrics"]["auc"],
        ),
    )
    best_multi_result = multiclass_results[best_multiclass]
    _plot_multiclass_confusion(
        best_multi_result["confusion_matrix"],
        best_multiclass,
        figure_dir / "fig_five_class_confusion_matrix.png",
    )
    top_features = _plot_feature_importance(
        fitted_multiclass[best_multiclass],
        feature_names,
        test_matrix,
        test_y.to_numpy(),
        figure_dir / "fig_feature_importance.png",
    )

    feature_groups = [
        {"group": name, "count": len(columns), "features": columns}
        for name, columns in FEATURE_GROUPS.items()
    ]
    for group in feature_groups:
        unknown = set(group["features"]) - set(train_x.columns)
        if unknown:
            raise ValueError(f"Unknown features in {group['group']}: {sorted(unknown)}")
    class_distribution = {
        split_name: {
            str(name): int(count)
            for name, count in labels.value_counts().reindex(CLASS_NAMES, fill_value=0).items()
        }
        for split_name, labels in (("train", train_y), ("test", test_y))
    }
    test_cases = _test_cases(
        train_x,
        test_x,
        train_y,
        test_y,
        train_matrix,
        test_matrix,
        preprocessor,
    )
    if any(case["status"] != "PASS" for case in test_cases):
        raise RuntimeError(f"Pipeline validation failed: {test_cases}")

    result = {
        "meta": {
            "synthetic": False,
            "dataset_name": "NSL-KDD",
            "dataset_source": (
                "https://github.com/defcom17/NSL_KDD raw KDDTrain+.txt and "
                "KDDTest+.txt files"
            ),
            "random_state": RANDOM_STATE,
            "training_test_note": (
                "The official KDDTrain+ file was used for fitting; the official "
                "KDDTest+ file was held out for evaluation. All preprocessing "
                "was fit on KDDTrain+ only."
            ),
            "environment": _environment(),
        },
        "dataset": {
            "train_rows": int(len(train_y)),
            "test_rows": int(len(test_y)),
            "feature_count": int(len(train_x.columns)),
            "raw_feature_columns": int(len(train_features_raw.columns)),
            "target_classes": CLASS_NAMES,
            "class_distribution": class_distribution,
            "feature_groups": feature_groups,
            "attack_categories": {
                "Benign": ["normal"],
                **{
                    name: sorted(ATTACK_GROUPS[name.casefold()])
                    for name in CLASS_NAMES
                    if name.casefold() in ATTACK_GROUPS
                },
            },
            "difficulty_field_excluded": "difficulty"
            in train_features_raw.columns,
        },
        "settings": settings,
        "model_training_rows": training_rows,
        "experiment_a": {
            "target": "Benign versus Attack",
            "positive_class": "Attack",
            "comparison": binary_results,
            "best_model": best_binary,
            "best_metrics": binary_results[best_binary]["metrics"],
        },
        "experiment_b": {
            "target": "Benign, DOS, PROBE, R2L, and U2R",
            "class_names": CLASS_NAMES,
            "comparison": multiclass_results,
            "best_model": best_multiclass,
            "best_metrics": best_multi_result["metrics"],
            "top_features": top_features,
        },
        "test_cases": test_cases,
        "figure_files": [
            "figures/fig_workflow.png",
            "figures/fig_class_distribution.png",
            "figures/fig_binary_comparison.png",
            "figures/fig_binary_confusion_matrices.png",
            "figures/fig_binary_roc.png",
            "figures/fig_five_class_confusion_matrix.png",
            "figures/fig_feature_importance.png",
        ],
    }
    (destination / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(
        "Completed: "
        f"train_rows={result['dataset']['train_rows']}, "
        f"test_rows={result['dataset']['test_rows']}, "
        f"best_binary={best_binary}, best_five_class={best_multiclass}",
        flush=True,
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="NSL-KDD data directory.")
    parser.add_argument("--output", default="results", help="Results output directory.")
    args = parser.parse_args()
    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
