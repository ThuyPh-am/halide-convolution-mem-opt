# Run-to-run spread of l3d_cache_refill per layer and schedule:
# (max - min) / median across the repeated runs.
import pandas as pd
from config import RAW, PROCESSED, LAYER_RENAME

def spread(src: str = "rerun_results.csv") -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RAW / src)
    d = d[d.model != "Test"].copy()
    d["layer"] = d.layer.replace(LAYER_RENAME)
    g = d.groupby(["model", "layer", "schedule"], sort=False).llc_misses
    s = ((g.max() - g.min()) / g.median()).unstack()
    s["runs"] = g.size().unstack().min(axis=1)
    s.round(3).to_csv(PROCESSED / "spread.csv")
    print(s.round(2).to_string())

if __name__ == "__main__":
    spread()
