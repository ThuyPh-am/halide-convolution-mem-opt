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
# raw layer names that are spelled inconsistently in the C++ layer list
LAYER_RENAME = {"Mobilenet_conv1": "MobileNet_conv1"}
LAYER_ORDER = [
    "Alex_conv1", "Alex_conv2", "Alex_conv3", "Alex_conv4", "Alex_conv5",
    "VGG_conv1_1", "VGG_conv1_2", "VGG_conv2_1", "VGG_conv3_1", "VGG_conv4_1",
    "MobileNet_conv1", "MobileNet_dw2", "MobileNet_pw2", "MobileNet_dw3",
]
