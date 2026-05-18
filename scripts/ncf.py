from __future__ import annotations

import argparse
import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    out = 0
    for ch in letters:
        out = out * 26 + ord(ch.upper()) - 64
    return out - 1


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Small xlsx reader for simple tables. Avoids requiring openpyxl."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rid_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append(
                    "".join(
                        t.text or ""
                        for t in si.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    )
                )

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for rel in rels.findall("pr:Relationship", rel_ns):
            target = rel.attrib["Target"].lstrip("/")
            targets[rel.attrib["Id"]] = target if target.startswith("xl/") else f"xl/{target}"

        target = None
        sheet_ns = {
            "a": ns["a"],
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        for sheet in wb.findall("a:sheets/a:sheet", sheet_ns):
            if sheet.attrib["name"] == sheet_name:
                target = targets[sheet.attrib[rid_attr]]
                break
        if target is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        root = ET.fromstring(zf.read(target))
        rows = []
        for row in root.findall("a:sheetData/a:row", ns):
            vals = {}
            for cell in row.findall("a:c", ns):
                idx = _col_to_idx(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                v = cell.find("a:v", ns)
                inline = cell.find("a:is", ns)
                value = ""
                if cell_type == "s" and v is not None:
                    value = shared[int(v.text)]
                elif cell_type == "inlineStr" and inline is not None:
                    value = "".join(
                        t.text or ""
                        for t in inline.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    )
                elif v is not None:
                    value = v.text or ""
                vals[idx] = value
            if vals:
                rows.append([vals.get(i, "") for i in range(max(vals) + 1)])

    header = rows[0]
    data = []
    for row in rows[1:]:
        data.append((row + [""] * len(header))[: len(header)])
    return pd.DataFrame(data, columns=header)


def make_training_pairs(model_input: pd.DataFrame, num_items: int, negatives_per_positive: int = 4) -> pd.DataFrame:
    positives = model_input[["user_idx", "item_idx"]].copy()
    positives["label"] = 1
    positives["source"] = "positive"

    user_to_positive = {
        int(row.user_idx): int(row.item_idx)
        for row in model_input[["user_idx", "item_idx"]].itertuples(index=False)
    }
    negatives = []
    for user_idx, positive_item in user_to_positive.items():
        candidates = [item for item in range(num_items) if item != positive_item]
        sampled = random.sample(candidates, k=min(negatives_per_positive, len(candidates)))
        for item_idx in sampled:
            negatives.append(
                {"user_idx": user_idx, "item_idx": item_idx, "label": 0, "source": "negative"}
            )

    return pd.concat([positives, pd.DataFrame(negatives)], ignore_index=True)


class NCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 8) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        return self.mlp(torch.cat([user_vec, item_vec], dim=1)).squeeze(1)


def train_ncf(
    pairs: pd.DataFrame,
    positives: pd.DataFrame,
    num_users: int,
    num_items: int,
    epochs: int = 120,
) -> tuple[NCF, dict]:
    mask = np.random.rand(len(pairs)) < 0.8
    train_df = pairs[mask].reset_index(drop=True)
    test_df = pairs[~mask].reset_index(drop=True)

    def loader_from(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(df["user_idx"].astype(int).to_numpy(), dtype=torch.long),
            torch.tensor(df["item_idx"].astype(int).to_numpy(), dtype=torch.long),
            torch.tensor(df["label"].astype(float).to_numpy(), dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    model = NCF(num_users, num_items)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for user_idx, item_idx, label in loader_from(train_df, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(user_idx, item_idx), label)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        users = torch.tensor(test_df["user_idx"].astype(int).to_numpy(), dtype=torch.long)
        items = torch.tensor(test_df["item_idx"].astype(int).to_numpy(), dtype=torch.long)
        labels = test_df["label"].astype(int).to_numpy()
        probs = torch.sigmoid(model(users, items)).numpy()
        preds = (probs >= 0.5).astype(int)

    with torch.no_grad():
        hits = []
        ranks = []
        for row in positives[["user_idx", "item_idx"]].itertuples(index=False):
            user_idx = int(row.user_idx)
            true_item = int(row.item_idx)
            all_items = torch.arange(num_items, dtype=torch.long)
            users = torch.full((num_items,), user_idx, dtype=torch.long)
            scores = torch.sigmoid(model(users, all_items)).numpy()
            ranked_items = np.argsort(-scores)
            rank = int(np.where(ranked_items == true_item)[0][0]) + 1
            ranks.append(rank)
            hits.append(rank <= 3)

    metrics = {
        "pair_accuracy_sanity": float(accuracy_score(labels, preds)),
        "pair_auc_sanity": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else np.nan,
        "reconstruction_hit_rate_at_3": float(np.mean(hits)),
        "reconstruction_mean_rank": float(np.mean(ranks)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "note": "Use metrics as small-data sanity checks, not real recommender performance.",
    }
    return model, metrics


def cluster_and_plot(user_vectors: pd.DataFrame, output_dir: Path, k: int = 3) -> pd.DataFrame:
    vec_cols = [col for col in user_vectors.columns if col.startswith("emb_")]
    x = user_vectors[vec_cols].to_numpy()
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(x)
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(x)

    clustered = user_vectors.copy()
    clustered["cluster"] = labels
    clustered["pca_x"] = xy[:, 0]
    clustered["pca_y"] = xy[:, 1]

    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(clustered["cluster"].unique()):
        part = clustered[clustered["cluster"] == cluster_id]
        plt.scatter(part["pca_x"], part["pca_y"], label=f"Cluster {cluster_id}", s=55)
    plt.title("User Embedding Clusters from NCF")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "user_embedding_clusters.png", dpi=160)
    plt.close()
    return clustered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, default=Path("CF_NCF_Phase1_Backbone.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase2_outputs"))
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_input = read_xlsx_sheet(args.excel, "Model_Input_NCF")
    interactions = read_xlsx_sheet(args.excel, "Interactions")
    user_text = read_xlsx_sheet(args.excel, "User_Text_Features")

    model_input["user_idx"] = model_input["user_idx"].astype(int)
    model_input["item_idx"] = model_input["item_idx"].astype(int)
    num_users = int(model_input["user_idx"].max()) + 1
    num_items = int(model_input["item_idx"].max()) + 1

    pairs = make_training_pairs(model_input, num_items, args.negatives)
    pairs.to_csv(args.output_dir / "ncf_training_pairs_with_negatives.csv", index=False, encoding="utf-8-sig")

    model, metrics = train_ncf(pairs, model_input, num_users, num_items, epochs=args.epochs)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "ncf_metrics.csv", index=False, encoding="utf-8-sig")

    with torch.no_grad():
        emb = model.user_embedding.weight.detach().cpu().numpy()
    user_vectors = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    user_lookup = model_input[["user_id", "user_idx", "item_id", "item_idx"]].drop_duplicates("user_idx")
    user_vectors.insert(0, "user_idx", range(len(user_vectors)))
    user_vectors = user_lookup.merge(user_vectors, on="user_idx", how="left")
    user_vectors = user_vectors.merge(
        interactions[["user_id", "mapped_model", "age_group", "gender"]],
        on="user_id",
        how="left",
    )
    user_vectors = user_vectors.merge(
        user_text[["user_id", "usage_type", "style", "performance", "comfort", "fuel_saving"]],
        on="user_id",
        how="left",
    )
    user_vectors.to_csv(args.output_dir / "user_vectors.csv", index=False, encoding="utf-8-sig")

    clustered = cluster_and_plot(user_vectors, args.output_dir, k=args.clusters)
    clustered.to_csv(args.output_dir / "user_vectors_clustered.csv", index=False, encoding="utf-8-sig")

    print("Done")
    print(metrics)
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
