import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT / "results" / "processed"
RAW = PROJECT_ROOT / "results" / "raw"
PLOT = PROJECT_ROOT / "results" / "plot"
SCHEDULES = ["Row-stationary inspired", "weight_stationary_inspired", "output_stationary_inspired"]
MODELS = ["ALEXNET", "VGG16", "MOBILENET"]
LABELS = {
        "Row-stationary inspired":      "RS-Inspired",
    "weight_stationary_inspired":   "WS-Inspired",
    "output_stationary_inspired":   "OS-Inspired",
}