from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from ncf import make_training_pairs, read_xlsx_sheet, set_seed, train_ncf


def summarize_ranks(ranks: list[int]) -> dict:
    ranks_arr = np.asarray(ranks, dtype=float)
    return {
        "hit_at_1": float(np.mean(ranks_arr <= 1)),
        "hit_at_3": float(np.mean(ranks_arr <= 3)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
    }


def rank_of_true_item(scores: np.ndarray, true_item_idx: int) -> int:
    ranked_items = np.argsort(-scores)
    return int(np.where(ranked_items == true_item_idx)[0][0]) + 1


def evaluate_popularity(model_input: pd.DataFrame, num_items: int) -> tuple[dict, pd.DataFrame]:
    counts = model_input["item_idx"].value_counts().to_dict()
    scores = np.array([counts.get(i, 0) for i in range(num_items)], dtype=float)
    rows = []
    ranks = []
    for row in model_input.itertuples(index=False):
        rank = rank_of_true_item(scores, int(row.item_idx))
        ranks.append(rank)
        rows.append({"method": "Popularity Baseline", "user_id": row.user_id, "true_item_id": row.item_id, "rank": rank})
    return summarize_ranks(ranks), pd.DataFrame(rows)


def evaluate_user_based_proxy(
    model_input: pd.DataFrame,
    user_features: pd.DataFrame,
    num_items: int,
) -> tuple[dict, pd.DataFrame]:
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
    users = model_input[["user_id", "user_idx", "item_id", "item_idx"]].merge(
        user_features[["user_id"] + feature_cols], on="user_id", how="left"
    )
    for col in feature_cols:
        users[col] = users[col].fillna("unknown").astype(str)

    encoded = OneHotEncoder(handle_unknown="ignore").fit_transform(users[feature_cols])
    sim = cosine_similarity(encoded)
    popularity = model_input["item_idx"].value_counts().to_dict()

    rows = []
    ranks = []
    for i, user in users.iterrows():
        scores = np.zeros(num_items, dtype=float)
        for j, neighbor in users.iterrows():
            if i == j:
                continue
            scores[int(neighbor["item_idx"])] += sim[i, j]
        if scores.sum() == 0:
            scores = np.array([popularity.get(k, 0) for k in range(num_items)], dtype=float)
        rank = rank_of_true_item(scores, int(user["item_idx"]))
        ranks.append(rank)
        rows.append(
            {
                "method": "User-Based Proxy Similarity",
                "user_id": user["user_id"],
                "true_item_id": user["item_id"],
                "rank": rank,
            }
        )
    return summarize_ranks(ranks), pd.DataFrame(rows)


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def evaluate_item_feature_fit(
    model_input: pd.DataFrame,
    user_features: pd.DataFrame,
    items: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    item_table = items.copy()
    item_table["item_idx"] = range(len(item_table))
    feature_cols = [
        "user_id",
        "budget_level",
        "usage_type",
        "style",
        "performance",
        "comfort",
        "easy_to_ride",
        "fuel_saving",
        "storage_need",
    ]
    users = model_input[["user_id", "item_id", "item_idx"]].merge(
        user_features[feature_cols], on="user_id", how="left"
    )

    rows = []
    ranks = []
    for _, user in users.iterrows():
        scores = []
        for _, item in item_table.iterrows():
            score = 0.0
            compared = 0.0

            if str(user.get("budget_level", "unknown")) != "unknown":
                compared += 1
                score += float(user["budget_level"] == item["budget_level"])

            if str(user.get("usage_type", "unknown")) != "unknown":
                compared += 1
                usage_values = {x.strip() for x in str(item["usage_fit"]).split(",")}
                score += float(str(user["usage_type"]) in usage_values)

            for col in ["style", "performance", "comfort"]:
                if str(user.get(col, "unknown")) != "unknown":
                    compared += 1
                    score += float(user[col] == item[col])

            for col in ["easy_to_ride", "fuel_saving", "storage_need"]:
                if str(user.get(col, "unknown")) != "unknown":
                    compared += 1
                    score += float(_is_true(user[col]) == _is_true(item[col]))

            scores.append(score / compared if compared else 0.0)

        scores_arr = np.asarray(scores, dtype=float)
        rank = rank_of_true_item(scores_arr, int(user["item_idx"]))
        ranks.append(rank)
        rows.append(
            {
                "method": "Item-Based Feature Fit",
                "user_id": user["user_id"],
                "true_item_id": user["item_id"],
                "rank": rank,
            }
        )
    return summarize_ranks(ranks), pd.DataFrame(rows)


def evaluate_ncf(
    model_input: pd.DataFrame,
    num_users: int,
    num_items: int,
    negatives: int,
    epochs: int,
) -> tuple[dict, pd.DataFrame]:
    pairs = make_training_pairs(model_input, num_items, negatives_per_positive=negatives)
    model, _ = train_ncf(pairs, model_input, num_users, num_items, epochs=epochs)

    rows = []
    ranks = []
    model.eval()
    with torch.no_grad():
        all_items = torch.arange(num_items, dtype=torch.long)
        for row in model_input.itertuples(index=False):
            users = torch.full((num_items,), int(row.user_idx), dtype=torch.long)
            scores = torch.sigmoid(model(users, all_items)).numpy()
            rank = rank_of_true_item(scores, int(row.item_idx))
            ranks.append(rank)
            rows.append({"method": "NCF + Negative Sampling", "user_id": row.user_id, "true_item_id": row.item_id, "rank": rank})
    return summarize_ranks(ranks), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, default=Path("CF_NCF_Phase1_Backbone.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase2_outputs"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--negatives", type=int, default=4)
    args = parser.parse_args()

    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = read_xlsx_sheet(args.excel, "Items_Feature")
    model_input = read_xlsx_sheet(args.excel, "Model_Input_NCF")
    user_features = read_xlsx_sheet(args.excel, "User_Text_Features")
    model_input["user_idx"] = model_input["user_idx"].astype(int)
    model_input["item_idx"] = model_input["item_idx"].astype(int)
    num_users = int(model_input["user_idx"].max()) + 1
    num_items = int(model_input["item_idx"].max()) + 1

    evaluations = [
        ("Popularity Baseline", evaluate_popularity(model_input, num_items)),
        ("User-Based Proxy Similarity", evaluate_user_based_proxy(model_input, user_features, num_items)),
        ("Item-Based Feature Fit", evaluate_item_feature_fit(model_input, user_features, items)),
        ("NCF + Negative Sampling", evaluate_ncf(model_input, num_users, num_items, args.negatives, args.epochs)),
    ]

    summary_rows = []
    rank_tables = []
    for method, (metrics, rank_table) in evaluations:
        summary_rows.append({"method": method, **metrics})
        rank_tables.append(rank_table)

    summary = pd.DataFrame(summary_rows).sort_values(["hit_at_3", "mrr"], ascending=False)
    all_ranks = pd.concat(rank_tables, ignore_index=True)

    summary.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    all_ranks.to_csv(args.output_dir / "model_comparison_user_ranks.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
