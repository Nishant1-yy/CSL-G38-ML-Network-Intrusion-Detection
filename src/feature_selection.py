"""Remove known identifiers and metadata that are not traffic predictors."""

from __future__ import annotations

import re

import pandas as pd

NON_PREDICTIVE_COLUMNS = {
    "id",
    "flowid",
    "sourceip",
    "srcip",
    "destinationip",
    "dstip",
    "timestamp",
    "difficulty",
}


def _normalized_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def select_features(features: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/metadata columns using names, not held-out values."""
    excluded = [
        column
        for column in features.columns
        if _normalized_name(column) in NON_PREDICTIVE_COLUMNS
        or _normalized_name(column).startswith("unnamed")
    ]
    selected = features.drop(columns=excluded).copy()
    if selected.empty:
        raise ValueError("No predictive features remain after excluding metadata.")
    if selected.columns.has_duplicates:
        raise ValueError("Feature column names must be unique.")
    return selected
