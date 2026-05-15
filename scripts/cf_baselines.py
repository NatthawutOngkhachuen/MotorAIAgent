from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ncf import read_xlsx_sheet


def make_item_based_similarity(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    numeric_cols = ["cc", "price_est_thb", "easy_to_ride", "fuel_saving", "storage_need"]
    cat_cols = ["brand", "type", "budget_level", "usage_fit", "style", "performance", "comfort"]

    for col in numeric_cols:
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0)
    for col in cat_cols:
        items[col] = items[col].fillna("unknown").astype(str)

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    x = transformer.fit_transform(items[numeric_cols + cat_cols])
    sim = cosine_similarity(x)

    rows = []
    for i, source in items.iterrows():
        ranked = np.argsort(-sim[i])
        for j in ranked:
            if i == j:
                continue
            target = items.iloc[j]
            rows.append(
                {
                    "source_item_id": source["item_id"],
                    "source_model": source["model"],
                    "similar_item_id": target["item_id"],
                    "similar_model": target["model"],
                    "similarity": round(float(sim[i, j]), 4),
                }
            )
    return pd.DataFrame(rows)


def make_user_based_proxy(user_features: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    users = user_features.copy()
    feature_cols = [
        "age_group",
        "gender",
        "budget_level",
        "usage_type",
        "style",
        "performance",
        "comfort",
        "easy_to_ride",
        "fuel_saving",
        "storage_need",
    ]
    for col in feature_cols:
        users[col] = users[col].fillna("unknown").astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore")
    x = encoder.fit_transform(users[feature_cols])
    sim = cosine_similarity(x)

    chosen = interactions[["user_id", "item_id", "mapped_model"]].drop_duplicates("user_id")
    users = users.merge(chosen, on="user_id", how="left", suffixes=("", "_chosen"))

    rows = []
    for i, source in users.iterrows():
        ranked = np.argsort(-sim[i])
        added_items = set()
        for j in ranked:
            if i == j:
                continue
            neighbor = users.iloc[j]
            item_id = neighbor["item_id_chosen"] if "item_id_chosen" in neighbor else neighbor["item_id"]
            model = (
                neighbor["mapped_model_chosen"]
                if "mapped_model_chosen" in neighbor
                else neighbor["mapped_model"]
            )
            if pd.isna(item_id) or item_id in added_items:
                continue
            rows.append(
                {
                    "user_id": source["user_id"],
                    "source_chosen_item": source.get("item_id_chosen", source.get("item_id")),
                    "neighbor_user_id": neighbor["user_id"],
                    "recommended_item_id": item_id,
                    "recommended_model": model,
                    "user_similarity": round(float(sim[i, j]), 4),
                }
            )
            added_items.add(item_id)
            if len(added_items) >= 3:
                break
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, default=Path("CF_NCF_Phase1_Backbone.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase2_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = read_xlsx_sheet(args.excel, "Items_Feature")
    interactions = read_xlsx_sheet(args.excel, "Interactions")
    user_features = read_xlsx_sheet(args.excel, "User_Text_Features")

    item_sim = make_item_based_similarity(items)
    user_proxy = make_user_based_proxy(user_features, interactions)

    item_sim.to_csv(args.output_dir / "item_based_similarity_baseline.csv", index=False, encoding="utf-8-sig")
    user_proxy.to_csv(args.output_dir / "user_based_proxy_recommendations.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print(args.output_dir / "item_based_similarity_baseline.csv")
    print(args.output_dir / "user_based_proxy_recommendations.csv")


if __name__ == "__main__":
    main()
