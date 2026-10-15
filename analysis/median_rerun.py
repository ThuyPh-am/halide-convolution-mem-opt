# Collapse the repeated runs in rerun_results.csv into one median row per
# (model, layer, schedule) and write it as clean_results.csv for the pipeline.
import pandas as pd
from config import RAW

COLS = ["llc_access", "llc_misses", "l2_refill", "l2_read", "l2_write", "l2_write_back"]

def median_rerun(src: str = "rerun_results.csv", dst: str = "clean_results.csv") -> None:
    d = pd.read_csv(RAW / src)
    d = d[d.model != "Test"]
    m = (d.groupby(["model", "layer", "schedule"], sort=False)
           .agg({"timestamp": "first", **{c: "median" for c in COLS}})
           .reset_index()[["timestamp", "model", "layer", "schedule"] + COLS])
    m[COLS] = m[COLS].round().astype("int64")
    m.to_csv(RAW / dst, index=False)
    print(f"{dst}: {len(m)} median rows from {len(d)} runs")

if __name__ == "__main__":
    median_rerun()
