"""Load NSL-KDD files and labeled network-flow CSV datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
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
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
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
    "label",
    "difficulty",
]
LABEL_COLUMNS = {"label", "class", "attack", "target"}
NSL_TRAIN_NAME = "KDDTrain+.txt"
NSL_TEST_NAME = "KDDTest+.txt"


def _read_nsl_kdd(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            header=None,
            names=NSL_KDD_COLUMNS,
            skipinitialspace=True,
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read NSL-KDD file {path}: {exc}") from exc
    return frame


def _read_labeled_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, skipinitialspace=True)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV file {path}: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if not any(column.casefold() in LABEL_COLUMNS for column in frame.columns):
        raise ValueError(
            f"{path} has no recognized target column. Expected one of: "
            f"{', '.join(sorted(LABEL_COLUMNS))}."
        )
    return frame


def _split_target(frame: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    label_column = next(
        (column for column in frame.columns if column.casefold() in LABEL_COLUMNS),
        None,
    )
    if label_column is None:
        raise ValueError(f"No target column was found in {source}.")

    labels = frame.pop(label_column)
    valid = labels.notna()
    frame = frame.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(str).str.strip().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No labeled traffic records were found in {source}.")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    records = pd.concat([frame, labels.rename("__label")], axis=1)
    duplicates = records.duplicated()
    duplicate_count = int(duplicates.sum())
    if duplicate_count:
        keep = ~duplicates
        frame = frame.loc[keep].reset_index(drop=True)
        labels = labels.loc[keep].reset_index(drop=True)
    if frame.shape[1] == 0:
        raise ValueError(f"No usable feature columns were found in {source}.")
    return frame, labels


def _load_nsl_file(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    return _split_target(_read_nsl_kdd(path), path)


def _csv_files(path: Path) -> Iterable[Path]:
    return sorted(item for item in path.rglob("*.csv") if item.is_file())


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load labeled traffic features and labels from a dataset file or folder.

    Directories containing NSL-KDD use ``KDDTrain+.txt`` as the training
    source; official test records are loaded separately with
    :func:`load_official_test`. Other directories may contain one or more
    headered CSV files, which are concatenated by column name.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {source}. See data/README.txt for setup."
        )

    if source.is_file():
        if source.name.casefold().endswith(".txt"):
            return _load_nsl_file(source)
        if source.suffix.casefold() == ".csv":
            return _split_target(_read_labeled_csv(source), source)
        raise ValueError(f"Unsupported dataset file: {source}. Use .txt or .csv.")

    nsl_train = source / NSL_TRAIN_NAME
    if nsl_train.is_file():
        return _load_nsl_file(nsl_train)

    files = list(_csv_files(source))
    if not files:
        raise FileNotFoundError(
            f"No labeled CSV files or {NSL_TRAIN_NAME} found under {source}."
        )
    frames = [_read_labeled_csv(file) for file in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return _split_target(combined, source)


def load_official_test(path: str | Path) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load the optional NSL-KDD official test file next to the training data."""
    source = Path(path).expanduser()
    directory = source if source.is_dir() else source.parent
    test_path = directory / NSL_TEST_NAME
    if not test_path.is_file():
        return None
    return _load_nsl_file(test_path)
