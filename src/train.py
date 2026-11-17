"""Train, compare, and serialize leakage-safe intrusion classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dataset, load_official_test
from src.evaluate import evaluate_model
from src.predict import ARTIFACT_VERSION
from src.preprocessing import build_preprocessor, normalize_labels, prepare_features

RANDOM_STATE = 42


def _model_searches(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=random_state
            ),
            {"classifier__C": [0.1, 1.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                class_weight="balanced", random_state=random_state
            ),
            {"classifier__max_depth": [None, 20]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            {"classifier__max_depth": [None, 20]},
        ),
        "linear_svm": (
            LinearSVC(class_weight="balanced", random_state=random_state),
            {"classifier__C": [0.1, 1.0]},
        ),
        "knn": (
            KNeighborsClassifier(weights="distance", n_jobs=-1),
            {"classifier__n_neighbors": [5, 9]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"classifier__n_estimators": [100], "classifier__max_depth": [2, 3]},
        ),
    }


def _split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    external_test: tuple[pd.DataFrame, pd.Series] | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if external_test is None:
        if int(labels.value_counts().min()) < 4:
            raise ValueError(
                "At least four records per class are needed for stratified "
                "train/validation/test splitting and cross-validation."
            )
        train_x, remainder_x, train_y, remainder_y = train_test_split(
            features,
            labels,
            test_size=0.40,
            random_state=random_state,
            stratify=labels,
        )
        validation_x, test_x, validation_y, test_y = train_test_split(
            remainder_x,
            remainder_y,
            test_size=0.50,
            random_state=random_state,
            stratify=remainder_y,
        )
        return train_x, validation_x, test_x, train_y, validation_y, test_y

    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )
    test_x, test_y = external_test
    unknown_classes = set(test_y.unique()) - set(train_y.unique())
    if unknown_classes:
        raise ValueError(
            "The official test set contains target classes absent from training: "
            f"{sorted(unknown_classes)}."
        )
    missing = set(train_x.columns) - set(test_x.columns)
    extra = set(test_x.columns) - set(train_x.columns)
    if missing or extra:
        raise ValueError(
            "Official test features do not match training features "
            f"(missing: {sorted(missing)}, unexpected: {sorted(extra)})."
        )
    return (
        train_x,
        validation_x,
        test_x.loc[:, train_x.columns],
        train_y,
        validation_y,
        test_y,
    )


def train(
    data_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    cv: int = 3,
) -> dict[str, Any]:
    """Run stratified model selection and evaluate once on held-out test data."""
    if cv < 2:
        raise ValueError("Cross-validation folds must be at least 2.")
    features, raw_labels = load_dataset(data_path)
    features = prepare_features(features)
    labels = normalize_labels(raw_labels)
    external = load_official_test(data_path)
    if external is not None:
        external_x, external_y = external
        external_x = prepare_features(external_x)
        external_y = normalize_labels(external_y)
        external = (external_x, external_y)

    train_x, validation_x, test_x, train_y, validation_y, test_y = _split_data(
        features, labels, external, RANDOM_STATE
    )
    min_class_count = int(train_y.value_counts().min())
    folds = min(cv, min_class_count)
    if folds < 2:
        raise ValueError(
            "At least two training examples per class are needed for stratified "
            "cross-validation."
        )

    validation_results = []
    fitted_models = {}
    searches = _model_searches(RANDOM_STATE)
    min_fold_class_count = max(1, min_class_count * (folds - 1) // folds)
    searches["knn"][1]["classifier__n_neighbors"] = sorted(
        {min(5, min_fold_class_count), min(9, min_fold_class_count)}
    )
    for name, (estimator, parameter_grid) in searches.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=folds,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(train_x, train_y)
        training_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        validation_predictions = search.best_estimator_.predict(validation_x)
        inference_seconds = (time.perf_counter() - validation_started) / max(
            len(validation_x), 1
        )
        validation_results.append(
            {
                "model": name,
                "cv_f1_macro": float(search.best_score_),
                "validation_accuracy": float(
                    accuracy_score(validation_y, validation_predictions)
                ),
                "validation_precision_macro": float(
                    precision_score(
                        validation_y,
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "validation_recall_macro": float(
                    recall_score(
                        validation_y,
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "validation_f1_macro": float(
                    f1_score(
                        validation_y,
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "training_seconds": training_seconds,
                "inference_seconds_per_record": inference_seconds,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        fitted_models[name] = (search.best_estimator_, search.best_params_)

    results = pd.DataFrame(validation_results).sort_values(
        ["validation_f1_macro", "cv_f1_macro"], ascending=False
    )
    selected_name = str(results.iloc[0]["model"])
    selected_pipeline, selected_parameters = fitted_models[selected_name]

    final_pipeline = clone(selected_pipeline)
    final_pipeline.fit(
        pd.concat([train_x, validation_x]),
        pd.concat([train_y, validation_y]),
    )

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path / "model_comparison.csv", index=False)
    class_names = sorted(labels.unique().tolist())
    metrics = evaluate_model(
        final_pipeline,
        test_x,
        test_y,
        class_names,
        report_path / "figures",
    )
    numeric_columns = train_x.select_dtypes(include="number").columns
    feature_types = {
        column: "numeric" if column in numeric_columns else "categorical"
        for column in train_x.columns
    }
    bundle = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": final_pipeline,
        "feature_columns": list(train_x.columns),
        "feature_types": feature_types,
        "class_names": class_names,
        "selected_model": selected_name,
        "best_parameters": selected_parameters,
        "dataset": str(data_path),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, model_path / "best_model.joblib")

    report = {
        "selected_model": selected_name,
        "best_parameters": selected_parameters,
        "class_names": class_names,
        "test_records": int(len(test_y)),
        "metrics": metrics,
        "comparison_file": str(report_path / "model_comparison.csv"),
        "model_file": str(model_path / "best_model.joblib"),
        "test_source": "KDDTest+.txt" if external is not None else "stratified holdout",
    }
    (report_path / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_path / "evaluation_report.md").write_text(
        "# Held-out model evaluation\n\n"
        f"- **Selected model:** {selected_name}\n"
        f"- **Test source:** {report['test_source']}\n"
        f"- **Test records:** {len(test_y)}\n"
        f"- **Accuracy:** {metrics['accuracy']:.4f}\n"
        f"- **Macro precision / recall / F1:** "
        f"{metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / "
        f"{metrics['f1_macro']:.4f}\n"
        f"- **Weighted F1:** {metrics['f1_weighted']:.4f}\n"
        + (
            f"- **Macro one-vs-rest ROC-AUC:** "
            f"{metrics['roc_auc_ovr_macro']:.4f}\n"
            if "roc_auc_ovr_macro" in metrics
            else "- **ROC-AUC:** not available for this test split.\n"
        )
        + "\nMetrics are from the held-out test set; model selection used only "
        "training cross-validation and validation macro F1.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="Dataset file or directory.")
    parser.add_argument(
        "--models", default="models", help="Output directory for models."
    )
    parser.add_argument(
        "--reports", default="reports", help="Output directory for reports."
    )
    parser.add_argument(
        "--cv", type=int, default=3, help="Stratified CV folds (default: 3)."
    )
    arguments = parser.parse_args()
    report = train(arguments.data, arguments.models, arguments.reports, arguments.cv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
