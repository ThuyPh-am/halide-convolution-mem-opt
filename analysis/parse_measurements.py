import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class modelResult:
    model : str
    layer_name : str
    schedule : str
    llc_access : int 
    llc_misses : int
    l2_refill : int 
    l2_read : int 
    l2_write : int = 0

# alexnet, vgg16, mobilenet
def load_results(csv_path: Path, model: str, schedule: str) -> list[modelResult]:
    results: list[LayerResult] = []

    with csv_path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            if row["model"] == model and row["schedule"] == schedule:
                try:
                    results.append(modelResult(
                        model = row["model"],
                        layer_name = row["layer"],
                        schedule = row["schedule"],
                        llc_access = row["llc_access"],
                        llc_misses = row["llc_misses"],
                        l2_refill = row["l2_refill"],
                        l2_read = row["l2_read"],
                        l2_write = row["l2_write"],
                    ))
                except (KeyError, ValueError) as err :
                    print(f"Warning: skipping row{i} in {csv_path.name}: {err}")
    return results

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    res_path = PROJECT_ROOT/"results"/"raw"/"raw_results.csv"
    for model in ["AlexNet", "VGG16", "MobileNet"]:
        for schedule in ["baseline", "Row-stationary inspired", "weight_stationary_inspired", "output_stationary_inspired"]:
            if load_results(res_path, model, schedule):
                print("==================================")
                print(f"Finish loading {model} {schedule}")
                print(load_results(res_path, model, schedule))
                