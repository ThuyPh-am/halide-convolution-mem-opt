import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import PROJECT_ROOT, PROCESSED, RAW, PLOT, SCHEDULES, LABELS, MODELS

def heatmap(csv_path:str) -> None:
    PLOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / csv_path)

    pivot=df.pivot_table(
        index="layer",
        columns="schedule",
        values="norm_llc_access",
        aggfunc="mean"
    )

    pivot = pivot[SCHEDULES]

    layer_order = [
         "Alex_conv1","Alex_conv2","Alex_conv3","Alex_conv4","Alex_conv5",
        "VGG_conv1_1","VGG_conv1_2","VGG_conv2_1","VGG_conv3_1","VGG_conv4_1",
        "Mobilenet_conv1","MobileNet_dw2","MobileNet_pw2","MobileNet_dw3",
    ]

    pivot = pivot.reindex([l for l in layer_order if l in pivot.index])

    fig, ax = plt.subplots(figsize=(9,8))

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=6.0)

    winner_idx = pivot.values.argmin(axis=1)
    for r in range(len(pivot.index)):
        for c in range(len(SCHEDULES)):
            val = pivot.values[r,c]
            star = "★ " if c == winner_idx[r] else ""
            weight = "bold" if c == winner_idx[r] else "normal"
            ax.text(c, r, f"{star}{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight=weight, 
                    color="white" if val > 4 else "black")

    ax.set_xticks(range(len(SCHEDULES)))
    ax.set_xticklabels([LABELS[s] for s in SCHEDULES], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_title("Normalized LLC Refills Heatmap\n"
                 "(★ = best per layer | green < 1.0 beats baseline)",
                 fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04,
                 label="Noralized LLC Refills")

    plt.tight_layout()
    plt.savefig(PLOT/ "winner_heatmap.png", dpi = 300, bbox_inches="tight")
    plt.savefig(PLOT/ "winner_heatmap.pdf", dpi = 300, bbox_inches="tight")
    plt.close()
    print("save winner heatmap .pnd/.pdf")

if __name__ == "__main__":
    heatmap("normalized.csv")