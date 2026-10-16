# Is each schedule clearly better or worse than the baseline on a layer?
# "better": every run of the schedule is below every baseline run (max < baseline min)
# "worse":  every run of the schedule is above every baseline run (min > baseline max)
# otherwise "no difference": the run ranges overlap.
import pandas as pd
from config import RAW, PROCESSED, LAYER_RENAME, LAYER_ORDER

def significance(src: str = "rerun_results.csv") -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RAW / src)
    d = d[d.model != "Test"].copy()
    d["layer"] = d.layer.replace(LAYER_RENAME)
    g = d.groupby(["model", "layer", "schedule"]).llc_misses.agg(["min", "median", "max", "size"])

    rows = []
    for (model, layer), sub in g.groupby(level=[0, 1]):
        base = sub.loc[(model, layer, "baseline")]
        for sched in sub.index.get_level_values(2):
            if sched == "baseline":
                continue
            s = sub.loc[(model, layer, sched)]
            if s["max"] < base["min"]:
                verdict = "better"
            elif s["min"] > base["max"]:
                verdict = "worse"
            else:
                verdict = "no difference"
            rows.append({"model": model, "layer": layer, "schedule": sched,
                         "ratio_median": s["median"] / base["median"],
                         "ratio_min": s["min"] / base["median"],
                         "ratio_max": s["max"] / base["median"],
                         "runs": int(s["size"]), "verdict": verdict})
    out = pd.DataFrame(rows)
    out["order"] = out.layer.map({l: i for i, l in enumerate(LAYER_ORDER)})
    out = out.sort_values(["order", "schedule"]).drop(columns="order")
    out.round(3).to_csv(PROCESSED / "significance.csv", index=False)
    print(out.pivot(index="layer", columns="schedule", values="verdict")
             .reindex(LAYER_ORDER).to_string())
    print("\ncounts:\n", out.groupby(["schedule", "verdict"]).size().unstack(fill_value=0).to_string())
    return out

if __name__ == "__main__":
    significance()
