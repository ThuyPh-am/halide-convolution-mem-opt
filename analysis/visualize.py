# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT/ "results"/ "processed"
PLOT = PROJECT_ROOT/"results"/"plot"

SCHEDULES = [
    "Row-stationary inspired",
    "weight_stationary_inspired",
    "output_stationary_inspired",
]
LABELS = {
    "Row-stationary inspired":      "RS-Inspired",
    "weight_stationary_inspired":   "WS-Inspired",
    "output_stationary_inspired":   "OS-Inspired",
}
COLORS = {
    "Row-stationary inspired":      "#4C72B0",
    "weight_stationary_inspired":   "#DD8452",
    "output_stationary_inspired":   "#55A868",
}
WIDTH = 0.25
Y_MAX = 7.5
OFFSETS = [-WIDTH, 0, WIDTH]


def visualize(csv_path:str) -> None:
    PLOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / csv_path)

    for model in ["AlexNet", "VGG16", "MobileNet"]:
        model_df = df[df['model'] == model]
        layers = model_df['layer'].unique()
        x = np.arange(len(layers))

        fig, ax = plt.subplots(figsize=(12,6))

        for i, schedule in enumerate(SCHEDULES):
            sched_df = model_df[model_df['schedule'] == schedule].set_index("layer")

            # align ratios to layer order - fill 0 if missing
            ratios = [
                float(sched_df.loc[l, "norm_llc_misses"]) if l in sched_df.index else 0.0
                for l in layers
            ]

            bars = ax.bar(
                x + OFFSETS[i], ratios, WIDTH,
                label=LABELS[schedule],
                color=COLORS[schedule],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )

            for bar, ratio in zip(bars, ratios):
                if ratio > Y_MAX:
                    # clipped bar 
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, Y_MAX - 0.25,
                        f"{ratio:.1f}x", ha="center", va="top",
                        fontsize = 7, color="#CC0000", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="#CC0000", alpha=0.85),
                    )
                elif ratio > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(ratio, Y_MAX) + 0.07,
                        f"{ratio:.2f}", ha="center", va="bottom", fontsize=7,
                    )
            # baseline reference line
        ax.axhline(y=1.0, color="black", linestyle="--",
                   linewidth=1.2, label="Baseline (1.0)", zorder=5)
 
        ax.set_ylabel("Normalized L3 Refills  (lower = better)", fontsize=10)
        ax.set_title(
            f"{model} — Normalized LLC Refills per Layer\n"
            "(Raspberry Pi 5, ARM Cortex-A76, metric: l3d_cache_refill)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, Y_MAX)
        ax.yaxis.grid(True, alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="upper right")
 
        plt.tight_layout()
        plt.savefig(PLOT / f"{model}_normalized_llc.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOT / f"{model}_normalized_llc.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {model}_normalized_llc.png/.pdf")

if __name__ == "__main__":
    visualize("normalized.csv")
    