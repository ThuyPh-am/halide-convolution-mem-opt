import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def groupData(csv_path:str) -> None:
    grouped_path = PROJECT_ROOT/ "results"/ "raw"
    grouped_path.mkdir(parents=True, exist_ok=True)
    raw_data = PROJECT_ROOT/"results"/"raw"/csv_path
    df = pd.read_csv(raw_data)
    grouped_df = df.sort_values(by=["model","schedule"])
    grouped_df[grouped_df["schedule"] == "baseline"].to_csv(grouped_path/"baseline_data.csv", index=False,header=True)
    grouped_df[grouped_df["schedule"] != "baseline"].to_csv(grouped_path/"grouped_data.csv", index=False,header=True)

if __name__ == "__main__":
    groupData("clean_results.csv")