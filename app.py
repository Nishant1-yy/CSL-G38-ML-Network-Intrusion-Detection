"""Streamlit dashboard for offline NSL-KDD/network-flow model analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.predict import load_model, predict_traffic
from src.preprocessing import normalize_labels, prepare_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS = ROOT / "reports"
DATA_PATH = ROOT / "data" / "raw"

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)
st.title("Network Intrusion Detection System")
st.caption(
    "Offline machine-learning analysis of labeled network traffic — "
    "not live packet monitoring."
)

home, dataset_tab, performance_tab, detection_tab, about_tab = st.tabs(
    ["Home", "Dataset analysis", "Model performance", "Intrusion detection", "About"]
)

with home:
    st.header("Project overview")
    st.write(
        "This application evaluates labeled network-traffic records with a trained "
        "classifier. Preprocessing is bundled with the model so training and inference "
        "use the same feature schema."
    )
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        model = None
        st.info(
            "No trained model found. Add the dataset and run `python -m src.train`."
        )
    except ValueError as error:
        model = None
        st.error(f"Could not load the saved model: {error}")
    if model:
        left, middle, right = st.columns(3)
        left.metric("Selected model", model["selected_model"])
        middle.metric("Input features", len(model["feature_columns"]))
        right.metric("Traffic classes", len(model["class_names"]))
        st.write("Classes:", ", ".join(model["class_names"]))
    st.markdown(
        "Dataset: **NSL-KDD** by default. Labeled CSV exports with a recognized "
        "target column are also supported."
    )

with dataset_tab:
    st.header("Dataset analysis")
    try:
        features, labels = load_dataset(DATA_PATH)
        display_features = prepare_features(features)
        display_labels = normalize_labels(labels)
        st.metric("Labeled records", f"{len(display_labels):,}")
        st.metric("Predictive features", len(display_features.columns))
        st.subheader("Traffic class distribution")
        distribution = (
            display_labels.value_counts().rename_axis("class").to_frame("records")
        )
        st.bar_chart(distribution)
        numeric = display_features.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            st.subheader("Numeric feature correlation")
            top_features = numeric.var().nlargest(20).index
            st.dataframe(
                numeric.loc[:, top_features].corr().round(2),
                use_container_width=True,
            )
        else:
            st.info("At least two numeric features are needed for a correlation view.")
    except (FileNotFoundError, ValueError) as error:
        st.info(f"Dataset analysis is unavailable: {error}")

with performance_tab:
    st.header("Model performance")
    comparison_path = REPORTS / "model_comparison.csv"
    metrics_path = REPORTS / "evaluation_report.json"
    if comparison_path.is_file() and metrics_path.is_file():
        comparison = pd.read_csv(comparison_path)
        st.subheader("Validation model comparison")
        st.dataframe(comparison, use_container_width=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = metrics["metrics"]
        st.subheader(f"Held-out test metrics — {metrics['selected_model']}")
        columns = st.columns(4)
        for column, key, label in zip(
            columns,
            ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
            ("Accuracy", "Macro precision", "Macro recall", "Macro F1"),
            strict=True,
        ):
            column.metric(label, f"{test[key]:.3f}")
        st.caption(
            "Macro averages weight each class equally; accuracy alone can hide "
            "poor detection of minority attack classes."
        )
        if "roc_auc_ovr_macro" in test:
            st.metric("Macro one-vs-rest ROC-AUC", f"{test['roc_auc_ovr_macro']:.3f}")
        for title, image in (
            ("Confusion matrix", "selected_model_confusion_matrix.png"),
            ("ROC curves", "selected_model_roc.png"),
            ("Precision-recall curves", "selected_model_precision_recall.png"),
        ):
            image_path = REPORTS / "figures" / image
            if image_path.is_file():
                st.subheader(title)
                st.image(str(image_path), use_container_width=True)
    else:
        st.info("Train and evaluate a model before viewing performance reports.")

with detection_tab:
    st.header("Classify one traffic record")
    if model is None:
        st.info("Train a model before using prediction.")
    else:
        with st.form("traffic_record"):
            st.write(
                "Enter one record using the exact feature schema used during training."
            )
            values = {}
            columns = st.columns(2)
            for index, name in enumerate(model["feature_columns"]):
                column = columns[index % 2]
                if model["feature_types"][name] == "numeric":
                    values[name] = column.number_input(name, value=0.0, format="%.6f")
                else:
                    values[name] = column.text_input(name, value="")
            submitted = st.form_submit_button("Classify traffic")
        if submitted:
            try:
                result = predict_traffic(model, values)
            except (TypeError, ValueError) as error:
                st.error(f"Invalid record: {error}")
            else:
                prediction = result["prediction"]
                st.subheader(f"Predicted class: {prediction}")
                if result["confidence"] is not None:
                    st.metric(
                        "Maximum class probability",
                        f"{result['confidence']:.1%}",
                    )
                    st.caption(
                        "Model probability; not a calibrated operational risk score."
                    )
                if result.get("probabilities"):
                    st.dataframe(
                        pd.DataFrame(
                            result["probabilities"].items(),
                            columns=["Class", "Probability"],
                        ).sort_values("Probability", ascending=False),
                        use_container_width=True,
                    )

with about_tab:
    st.header("About")
    st.markdown(
        """
        **Objective:** demonstrate reproducible, offline multiclass classification
        of labeled intrusion-detection traffic.

        **Technologies:** Python, pandas, scikit-learn, matplotlib, Streamlit.

        **Dataset:** NSL-KDD. See `data/README.txt` for sources and setup.

        **Limitations:** this is a benchmark/educational classifier, not a live
        network sensor, blocking control, or production-ready security product.
        Dataset age, label quality, distribution shift, and class imbalance limit
        real-world generalization. Predictions require the same features and
        conventions as the training data.
        """
    )
