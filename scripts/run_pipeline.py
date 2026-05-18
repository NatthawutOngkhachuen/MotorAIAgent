from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parents[1]

EXCEL_PATH = BASE_DIR / "data" / "backbone" / "CF_NCF_Backbone.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "recommendation_output"

COMMANDS = [
    [
        "scripts/ncf.py",
        "--excel", str(EXCEL_PATH),
        "--output-dir", str(OUTPUT_DIR),
        "--epochs", "120",
        "--negatives", "4",
        "--clusters", "3",
    ],
    [
        "scripts/cf_baselines.py",
        "--excel", str(EXCEL_PATH),
        "--output-dir", str(OUTPUT_DIR),
    ],
    [
        "scripts/compare_models.py",
        "--excel", str(EXCEL_PATH),
        "--output-dir", str(OUTPUT_DIR),
        "--epochs", "120",
        "--negatives", "4",
    ],
    [
        "scripts/cluster_analysis.py",
        "--excel", str(EXCEL_PATH),
        "--output-dir", str(OUTPUT_DIR),
    ],
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for command in COMMANDS:
        print(f"\nRunning: python {' '.join(command)}")
        subprocess.run(
            [sys.executable, *command],
            cwd=BASE_DIR,
            check=True,
        )

    print("\nRecommendation pipeline completed.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
