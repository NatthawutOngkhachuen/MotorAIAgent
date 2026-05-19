from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.recommendation.cluster_pipeline import run_user_vector_clustering
from app.services.recommendation.vectorizer import FEATURE_NAMES, write_vector_outputs


DEFAULT_SOURCE = Path(
    r"E:\Final Project-PIM\เอกสารที่เกี่ยวข้อง\Data\Data for CF\Backbone\csv for vector\CF_NCF_Phase1_Backbone_SRO_user_feature_final_with_vector_input.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/recommendation_output/final"))
    args = parser.parse_args()

    vector_paths = write_vector_outputs(args.source_csv, args.output_dir)
    cluster_outputs = run_user_vector_clustering(vector_paths["expanded"], args.output_dir)

    print("Generated user preference assets")
    print(f"rows_source={args.source_csv}")
    print(f"vector_dim={len(FEATURE_NAMES)}")
    for name, path in vector_paths.items():
        print(f"{name}={path}")
    for name, value in cluster_outputs.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
