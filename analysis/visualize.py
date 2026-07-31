# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT/ "results"/ "processed"
PLOT = PROJECT_ROOT/"results"/"plot"
width = 0.25

def visualize(csv_path:str):
    df = pd.read_csv(PROCESSED/csv_path)
    if not Path(PLOT).exists():
        PLOT.mkdir()
    df.drop(columns=['llc_access', 'llc_misses', 'l2_refill', 'l2_read', 'l2_write', 'l2_write_back'])
    schedules = df['schedule'].unique()

    for model in ["AlexNet", "VGG16", "MobileNet"]:
        model_df = df[df['model'] == model]
        layers = model_df['layer'].unique()
        x = np.arange(len(layers))
        fig, ax = plt.subplots(figsize=(12,6))

        multiplier = 0
        for schedule in schedules:
            schedule_df = model_df[model_df['schedule'] == schedule]
            offset = x + width * multiplier
            rects = ax.bar(offset, schedule_df['llc_access'], width, label=schedule)
            ax.bar_label(rects, padding=3, fontsize=8)
            multiplier += 1

        ax.set_ylabel("Normalized LLC Misses")
        ax.set_title(f"Normalized LLC Misses per layer ({model})")
        ax.set_xticks(x + width, layers, rotation=45)
        ax.legend()
        ax.set_ylim(0,3)
        plt.tight_layout()
        plt.savefig(PLOT/f"{model}_llc_refill.png", dpi=300, bbox_inches="tight")
        plt.show()



if __name__ == "__main__":
    visualize("normalized.csv")
    