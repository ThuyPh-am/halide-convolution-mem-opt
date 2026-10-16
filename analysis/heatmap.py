# Heatmap of normalized L3 refills (schedule / baseline) per layer.
# Colour is centred on 1.0 (= baseline) on a log scale: blue = fewer refills,
# red = more. A star marks only results that are clearly better than the
# baseline across all runs (see significance.py).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from config import PROCESSED, PLOT, SCHEDULES, LABELS, LAYER_RENAME, LAYER_ORDER

def heatmap(csv_path: str = "normalized.csv", sig_path: str = "significance.csv") -> None:
    PLOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / csv_path)
    df["layer"] = df.layer.replace(LAYER_RENAME)
    pivot = (df.pivot_table(index="layer", columns="schedule", values="norm_llc_misses")
               [SCHEDULES].reindex([l for l in LAYER_ORDER if l in set(df.layer)]))

    sig = pd.read_csv(PROCESSED / sig_path)
    better = set(zip(sig[sig.verdict == "better"].layer, sig[sig.verdict == "better"].schedule))

    logv = np.log2(pivot.values)
    lim = 5.0   # colour saturates at 1/32 and 32x
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.clip(logv, -lim, lim), cmap="RdBu_r", norm=norm, aspect="auto")

    for r, layer in enumerate(pivot.index):
        for c, sched in enumerate(SCHEDULES):
            val = pivot.values[r, c]
            star = (layer, sched) in better
            ax.text(c, r, f"{'★ ' if star else ''}{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if star else "normal",
                    color="white" if abs(np.log2(val)) > 3 else "black")

    ax.set_xticks(range(len(SCHEDULES)))
    ax.set_xticklabels([LABELS[s] for s in SCHEDULES], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_title("Normalized L3 refills (schedule / baseline, median of 5 runs)\n"
                 "blue = fewer refills, red = more  |  ★ = clearly better than baseline in all runs",
                 fontsize=11, fontweight="bold")

    ticks = [1/32, 1/8, 1/2, 1, 2, 8, 32]
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04, ticks=np.log2(ticks))
    cb.ax.set_yticklabels(["1/32", "1/8", "1/2", "1", "2", "8", "32"])
    cb.set_label("Normalized L3 refills (log scale)")

    plt.tight_layout()
    plt.savefig(PLOT / "winner_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT / "winner_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved winner_heatmap.png/.pdf")

if __name__ == "__main__":
    heatmap()
