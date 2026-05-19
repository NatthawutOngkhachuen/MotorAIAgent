from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


USAGE_FEATURES = [
    "city",
    "daily",
    "delivery",
    "family",
    "long_distance",
    "rough_road",
    "shopping",
    "storage_heavy",
    "trip",
    "work",
]

STYLE_FEATURES = [
    "adventure",
    "beauty",
    "classic",
    "compact",
    "cute",
    "modern",
    "premium",
    "sporty",
]

LEVEL_FEATURES = [
    "budget_level",
    "performance",
    "comfort",
    "safety_level",
    "technology_level",
]

BOOL_FEATURES = [
    "easy_to_ride",
    "fuel_saving",
    "storage_need",
    "maintenance_easy",
]

LEVEL_SCORE = {
    "unknown": 0.0,
    "": 0.0,
    "none": 0.0,
    "low": 0.33,
    "medium": 0.66,
    "high": 1.0,
}

FEATURE_NAMES = (
    [f"usage_fit_{name}" for name in USAGE_FEATURES]
    + [f"style_{name}" for name in STYLE_FEATURES]
    + [f"{name}_score" for name in LEVEL_FEATURES]
    + [f"{name}_flag" for name in BOOL_FEATURES]
)


def empty_preference_state() -> dict[str, Any]:
    return {
        "budget_level": "unknown",
        "usage_fit": [],
        "style": [],
        "performance": "unknown",
        "comfort": "unknown",
        "safety_level": "unknown",
        "technology_level": "unknown",
        "easy_to_ride": "unknown",
        "fuel_saving": "unknown",
        "storage_need": "unknown",
        "maintenance_easy": "unknown",
    }


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    tokens = []
    for raw in raw_values:
        token = str(raw).strip().lower()
        if token and token not in {"unknown", "nan", "none"} and token not in tokens:
            tokens.append(token)
    return tokens


def normalize_level(value: Any) -> str:
    token = str(value or "unknown").strip().lower()
    if token in {"low", "medium", "high"}:
        return token
    return "unknown"


def normalize_bool(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    token = str(value if value is not None else "unknown").strip().lower()
    if token in {"true", "1", "yes", "y", "ใช่", "เอา", "ต้องการ"}:
        return True
    if token in {"false", "0", "no", "n", "ไม่", "ไม่เอา", "ไม่ต้องการ"}:
        return False
    return "unknown"


def merge_preferences(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = {**empty_preference_state(), **(base or {})}

    for key in ["usage_fit", "style"]:
        existing = normalize_tokens(merged.get(key))
        incoming = normalize_tokens(update.get(key))
        for token in incoming:
            if token not in existing:
                existing.append(token)
        merged[key] = existing

    for key in LEVEL_FEATURES:
        value = normalize_level(update.get(key))
        if value != "unknown":
            merged[key] = value

    for key in BOOL_FEATURES:
        value = normalize_bool(update.get(key))
        if value != "unknown":
            merged[key] = value

    return merged


def preference_to_vector(preference: dict[str, Any]) -> list[float]:
    pref = {**empty_preference_state(), **(preference or {})}
    usage_tokens = set(normalize_tokens(pref.get("usage_fit")))
    style_tokens = set(normalize_tokens(pref.get("style")))

    vector: list[float] = []
    vector.extend(1.0 if token in usage_tokens else 0.0 for token in USAGE_FEATURES)
    vector.extend(1.0 if token in style_tokens else 0.0 for token in STYLE_FEATURES)

    for key in LEVEL_FEATURES:
        vector.append(LEVEL_SCORE[normalize_level(pref.get(key))])

    for key in BOOL_FEATURES:
        value = normalize_bool(pref.get(key))
        vector.append(1.0 if value is True else 0.0)

    return vector


def row_to_preference(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return {
        "budget_level": normalize_level(data.get("budget_level")),
        "usage_fit": normalize_tokens(data.get("usage_fit")),
        "style": normalize_tokens(data.get("style")),
        "performance": normalize_level(data.get("performance")),
        "comfort": normalize_level(data.get("comfort")),
        "safety_level": normalize_level(data.get("safety_level")),
        "technology_level": normalize_level(data.get("technology_level")),
        "easy_to_ride": normalize_bool(data.get("easy_to_ride")),
        "fuel_saving": normalize_bool(data.get("fuel_saving")),
        "storage_need": normalize_bool(data.get("storage_need")),
        "maintenance_easy": normalize_bool(data.get("maintenance_easy")),
    }


def build_feature_map() -> pd.DataFrame:
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        if name.startswith("usage_fit_"):
            rows.append(
                {
                    "index": index,
                    "source_column": "usage_fit",
                    "feature_name": name,
                    "encoding": "multi_hot",
                    "value": name.replace("usage_fit_", ""),
                }
            )
        elif name.startswith("style_"):
            rows.append(
                {
                    "index": index,
                    "source_column": "style",
                    "feature_name": name,
                    "encoding": "multi_hot",
                    "value": name.replace("style_", ""),
                }
            )
        elif name.endswith("_score"):
            rows.append(
                {
                    "index": index,
                    "source_column": name.replace("_score", ""),
                    "feature_name": name,
                    "encoding": "ordinal_score",
                    "value": "Unknown=0, low=0.33, medium=0.66, high=1.0",
                }
            )
        else:
            rows.append(
                {
                    "index": index,
                    "source_column": name.replace("_flag", ""),
                    "feature_name": name,
                    "encoding": "binary",
                    "value": "TRUE=1, Unknown/blank/FALSE=0",
                }
            )
    return pd.DataFrame(rows)


def vectorize_user_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_cols = [col for col in ["user_id", "item_id", "mapped_model"] if col in df.columns]
    records = []
    compact_records = []

    for _, row in df.iterrows():
        preference = row_to_preference(row)
        vector = preference_to_vector(preference)
        record = {col: row.get(col) for col in metadata_cols}
        record.update(dict(zip(FEATURE_NAMES, vector)))
        records.append(record)

        compact = {col: row.get(col) for col in metadata_cols}
        compact["vector_dim"] = len(FEATURE_NAMES)
        compact["user_feature_vector"] = json.dumps(vector, ensure_ascii=False)
        compact_records.append(compact)

    return pd.DataFrame(records), pd.DataFrame(compact_records)


def vector_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in FEATURE_NAMES if name in frame.columns]


def matrix_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return frame[vector_columns(frame)].astype(float).to_numpy()


def write_vector_outputs(source_csv: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_csv)
    expanded, compact = vectorize_user_dataframe(df)
    feature_map = build_feature_map()

    paths = {
        "expanded": output_dir / "user_preference_vectors_expanded.csv",
        "compact": output_dir / "user_preference_vectors_compact.csv",
        "feature_map": output_dir / "user_preference_feature_map.csv",
    }
    expanded.to_csv(paths["expanded"], index=False, encoding="utf-8-sig")
    compact.to_csv(paths["compact"], index=False, encoding="utf-8-sig")
    feature_map.to_csv(paths["feature_map"], index=False, encoding="utf-8-sig")
    return paths
