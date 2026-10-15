# Gap to Eyeriss as DRAM data per image.
# Pi:      best schedule per layer, l3d_cache_refill x 64 B (reads only, fp32)
# Eyeriss: reported DRAM MB per batch / batch size (16-bit data)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import RAW, PROCESSED, PLOT, EYERISS

LINE = 64  # bytes per cache line

def dram_gap(csv_path: str = "clean_results.csv", eyeriss_path: str = "extracted_data.csv") -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PLOT.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RAW / csv_path)
    d["pi_mb"] = d.llc_misses * LINE / 1e6
    mb = d.pivot_table(index="layer", columns="schedule", values="pi_mb", sort=False)

    e = pd.read_csv(EYERISS / eyeriss_path).set_index("layer")
    e["eyeriss_mb_per_image"] = e.eyeriss_dram_mb / e.batch_size

    t = pd.DataFrame({
        "best_schedule": mb.idxmin(axis=1),
        "pi_mb_per_image": mb.min(axis=1),
        "baseline_mb_per_image": mb["baseline"],
    }).join(e[["eyeriss_dram_mb", "batch_size", "eyeriss_mb_per_image"]], how="inner")
    t = t.reindex([l for l in e.index if l in t.index])
    t["gap_fp32"] = t.pi_mb_per_image / t.eyeriss_mb_per_image
    t["gap_16bit_equiv"] = t.gap_fp32 / 2   # Eyeriss moves 16-bit values, the Pi 32-bit
    t.round(3).to_csv(PROCESSED / "dram_gap.csv")
    print(t.round(2).to_string())

    x = np.arange(len(t))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    raw = ax.bar(x - w / 2, t.gap_fp32, w, color="#4C72B0", label="Pi (32-bit) / Eyeriss")
    ax.bar(x + w / 2, t.gap_16bit_equiv, w, color="#A9BEDF", label="Pi at 16-bit equivalent / Eyeriss")
    for bar, v in zip(raw, t.gap_fp32):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.1f}x",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Parity with Eyeriss")
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("DRAM data per image, Pi / Eyeriss  (lower = closer)", fontsize=10)
    ax.set_title("Gap to Eyeriss: DRAM data moved per image, best Pi schedule per layer",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT / "dram_gap.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT / "dram_gap.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved dram_gap.png/.pdf")

if __name__ == "__main__":
    dram_gap()
