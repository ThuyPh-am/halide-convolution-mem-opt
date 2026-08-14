import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import PROJECT_ROOT, PROCESSED, PLOT, EYERISS, SCHEDULES, LABELS, COLORS

def compute_reuse(csv_path: str) -> pd.DataFrame:
    """Compute reuse factor for each row: l3d_cache_rd / l3d_cache_refill."""
    df = pd.read_csv(PROCESSED / csv_path)
    df["reuse_factor"] = df["llc_access"] / df["llc_misses"]
    return df

def gap_analysis(schedule_path: str, eyeriss_path: str) -> None:
    PLOT.mkdir(parents=True, exist_ok=True)

    sched_df  = compute_reuse(schedule_path)
    eyeriss   = pd.read_csv(EYERISS / eyeriss_path)

    # merge on layer — keeps only layers present in both datasets
    merged = sched_df.merge(eyeriss[["layer", "eyeriss_reuse"]], on="layer")

    # ratio: how close your reuse gets to Eyeriss (100% = matches Eyeriss)
    merged["reuse_pct_of_eyeriss"] = (merged["reuse_factor"] / merged["eyeriss_reuse"]) * 100

    print(merged[["model", "layer", "schedule", "reuse_factor",
                  "eyeriss_reuse", "reuse_pct_of_eyeriss"]].to_string())

    # plot
    layers   = merged["layer"].unique()
    x        = np.arange(len(layers))
    width    = 0.2
    offsets  = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Eyeriss bars
    eyeriss_vals = [
        float(merged[merged["layer"] == l]["eyeriss_reuse"].iloc[0])
        for l in layers
    ]
    ax.bar(x + offsets[0], eyeriss_vals, width,
           label="Eyeriss (spatial, RS)", color="#8B0000", alpha=0.8)

    # schedule bars
    for i, schedule in enumerate(SCHEDULES):
        vals = [
            float(merged[(merged["layer"] == l) &
                         (merged["schedule"] == schedule)]["reuse_factor"].iloc[0])
            if len(merged[(merged["layer"] == l) &
                          (merged["schedule"] == schedule)]) > 0 else 0.0
            for l in layers
        ]
        ax.bar(x + offsets[i+1], vals, width,
               label=LABELS[schedule], color=COLORS[schedule], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Reuse Factor  (L3_reads / DRAM_fetches)", fontsize=10)
    ax.set_title("Gap Analysis: Reuse Factor — Temporal Approximation vs Eyeriss Spatial Guarantee\n"
                 "(higher = better reuse of each DRAM fetch)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT / "gap_analysis_reuse.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT / "gap_analysis_reuse.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved gap_analysis_reuse.png/.pdf")

if __name__ == "__main__":
    gap_analysis("grouped_data.csv", "extracted_data.csv")