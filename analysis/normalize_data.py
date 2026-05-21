import numpy as np
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT/ "results"/ "processed"

def describe(baseline_path:str, schedule_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = PROCESSED/baseline_path
    schedule_path = PROCESSED/schedule_path
  
    description = PROCESSED/ "description.csv"
    
    scheduled_df = pd.read_csv(schedule_path).drop(columns=['timestamp', 'l2_write'])
    baseline_df = pd.read_csv(baseline_path).drop(columns=['timestamp', 'l2_write'])
    scheduled_df.describe().to_csv(description, index=True,header=True)
    
    return baseline_df, scheduled_df

def normalize(baseline_df:pd.DataFrame, scheduled_df:pd.DataFrame) -> pd.DataFrame:
    merge = scheduled_df.merge(baseline_df, on=["model", "layer"], suffixes=("","_baseline"))

    normalized_df = merge.copy()
    numeric_cols = scheduled_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        baseline_col = f"{col}_baseline"
        if baseline_col in merge.columns:
            normalized_df[col] = normalized_df[col]/normalized_df[baseline_col]

    baseline_cols = [col for col in normalized_df.columns if col.endswith("_baseline")]
    normalized_df.drop(columns=baseline_cols, inplace=True)
    normalized_df.to_csv(PROCESSED/"normalized.csv", index=False)
    return normalized_df

def scaled(scheduled_df:pd.DataFrame) -> pd.DataFrame:
    # using min max feature scaling
    scaled = scheduled_df.copy()
    numeric_cols = scheduled_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        min_val = scaled[col].min()
        max_val = scaled[col].max()

        if min_val != max_val:
            scaled[col] = (scaled[col] - min_val) / (max_val - min_val)
    scaled.to_csv(PROCESSED/"scaled.csv", index=False, header=True)
    return scaled

if __name__ == "__main__":
    baseline_df, scheduled_df = describe("baseline_data.csv", "grouped_data.csv")
    normalized_df = normalize(baseline_df, scheduled_df)
    scaled(normalized_df)