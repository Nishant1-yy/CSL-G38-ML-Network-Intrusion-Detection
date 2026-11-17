"""Leakage-safe feature preparation and target normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_selection import select_features

ATTACK_GROUPS = {
    "dos": {
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "apache2",
        "udpstorm",
        "processtable",
        "mailbomb",
        "worm",
    },
    "probe": {
        "ipsweep",
        "nmap",
        "portsweep",
        "satan",
        "mscan",
        "saint",
    },
    "r2l": {
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "spy",
        "warezclient",
        "warezmaster",
        "snmpgetattack",
        "snmpguess",
        "named",
        "xlock",
        "xsnoop",
        "sendmail",
    },
    "u2r": {
        "buffer_overflow",
        "loadmodule",
        "perl",
        "rootkit",
        "httptunnel",
        "ps",
        "sqlattack",
        "xterm",
    },
}


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Normalize labels and group NSL-KDD attacks into standard categories."""
    normalized = labels.astype(str).str.strip().str.rstrip(".").str.casefold()
    grouped = {"normal": "Benign", "benign": "Benign"}
    for category, attacks in ATTACK_GROUPS.items():
        grouped.update({attack: category.upper() for attack in attacks})
    return normalized.map(grouped).fillna(
        normalized.map(lambda label: label.upper() if label == "benign" else label)
    ).replace({"normal": "Benign", "benign": "Benign"})


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Apply name-based feature exclusions and normalize invalid numbers."""
    prepared = select_features(features)
    prepared.columns = [str(column).strip() for column in prepared.columns]
    if prepared.columns.has_duplicates:
        raise ValueError("Feature column names must remain unique after trimming.")
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    return prepared


def build_preprocessor() -> ColumnTransformer:
    """Build train-fitted imputers, categorical encoding, and numeric scaling."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical,
                make_column_selector(
                    dtype_include=["object", "category", "bool", str]
                ),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
