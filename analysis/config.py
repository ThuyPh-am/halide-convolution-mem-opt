import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED = PROJECT_ROOT / "results" / "processed"
RAW = PROJECT_ROOT / "results" / "raw"
PLOT = PROJECT_ROOT / "results" / "plot"
EYERISS = PROJECT_ROOT / "eyeriss_reference" 
SCHEDULES = ["Row-stationary inspired", "weight_stationary_inspired", "output_stationary_inspired"]
MODELS = ["AlexNet", "VGG16", "MobileNet"]
LABELS = {
        "Row-stationary inspired":      "RS-Inspired",
    "weight_stationary_inspired":   "WS-Inspired",
    "output_stationary_inspired":   "OS-Inspired",
}

COLORS = {
    "Row-stationary inspired" : "#4C72B0",
    "weight_stationary_inspired" : "#DD8452",
    "output_stationary_inspired": "#55A868",
}

DROP_COLS = ['timestamp', 'l2_write']
WIDTH = 0.25
Y_MAX = 7.5