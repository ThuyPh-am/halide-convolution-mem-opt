import numpy as np
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT/ "results"/ "processed"

DROP_COLS = ['timestamp', 'l2_write']

def describe(scheduled_df:pd.DataFrame) -> None:
    description = PROCESSED / "description.csv"
    scheduled_df.describe().to_csv(description, index=True,header=True)

def normalize() -> pd.DataFrame:
    baseline_df = pd.read_csv(PROCESSED / "baseline_data.csv").drop(columns=DROP_COLS)
    scheduled_df = pd.read_csv(PROCESSED / "grouped_data.csv").drop(columns=DROP_COLS)

    describe(scheduled_df
             )
    baseline_key = baseline_df[["model","layer", "llc_misses", "llc_access", "l2_refill", "l2_read", "l2_write_back"]]

    baseline_key = baseline_key.rename(columns={
        "llc_misses": "baseline_llc_misses",
        "llc_access": "baseline_llc_access",
        "l2_refill":  "baseline_l2_refill",
        "l2_read" :   "baseline_l2_read",
        "l2_write_back" : "baseline_l2_wb",
    })

    merged = scheduled_df.merge(baseline_key, on=["model", "layer"])

    merged["norm_llc_misses"] = merged["llc_misses"] / merged["baseline_llc_misses"]
    merged["norm_llc_access"] = merged["llc_access"] / merged["baseline_llc_access"]
    merged["norm_l2_refill"] = merged["l2_refill"] / merged["baseline_l2_refill"]
    merged["norm_l2_wb"] = merged["l2_write_back"] / merged["baseline_l2_wb"]

    merged.to_csv(PROCESSED / "normalized.csv", index=False)
    print(f"Saved normalized.csv: {len(merged)} rows")
    print(merged[["model", "layer", "schedule", "norm_llc_misses"]].to_string())
    return merged

if __name__ == "__main__":
    normalize()