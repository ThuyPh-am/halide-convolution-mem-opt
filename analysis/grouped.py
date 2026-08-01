import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW = PROJECT_ROOT / "results" / "raw"
PROCESSED =  PROJECT_ROOT / "results" / "processed"

def groupData(csv_path:str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW/ csv_path)
    df.sort_values(by=["model", "layer", "schedule"], inplace=True)

    baseline = df[df["schedule"] == "baseline"]
    scheduled = df[df["schedule"] != "baseline"]

    baseline.to_csv(PROCESSED / "baseline_data.csv", index=False)
    scheduled.to_csv(PROCESSED/ "grouped_data.csv", index=False)
    print(f"baseline_data.csv: {len(baseline)} rows")
    print(f"grouped_data.csv: {len(scheduled)} rows")

if __name__ == "__main__":
    groupData("clean_results.csv")