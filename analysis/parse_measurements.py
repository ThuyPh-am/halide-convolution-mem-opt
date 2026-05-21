import csv
from dataclasses import dataclass, asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

@dataclass
class layerResult:
    model : str
    layer_name : str
    schedule : str
    llc_access : int 
    llc_misses : int
    l2_refill : int 
    l2_read : int 
    l2_write : int = 0

# alexnet, vgg16, mobilenet
def load_results(csv_path: Path, model: str, schedule: str) -> list[layerResult]:
    results: list[layerResult] = []

    with csv_path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            if row["model"] == model and row["schedule"] == schedule:
                try:
                    results.append(layerResult(
                        model = row["model"],
                        layer_name = row["layer"],
                        schedule = row["schedule"],
                        llc_access = int(row["llc_access"]),
                        llc_misses = int(row["llc_misses"]),
                        l2_refill = int(row["l2_refill"]),
                        l2_read = int(row["l2_read"]),
                        l2_write = int(row["l2_write"]),
                    ))
                except (KeyError, ValueError) as err :
                    print(f"Warning: skipping row{i} in {csv_path.name}: {err}")
    return results

def parse_res(layer_res: list[layerResult]) -> None:
    if not layer_res:
        return

    csv_name = Path(f"{layer_res[0].model}_{layer_res[0].schedule}.csv")

    # make csv path for each csv
    res_path = Path(PROJECT_ROOT/ "results"/ "raw" / csv_name)
    
    # write data
    try:
        with open(res_path, 'w', newline="") as f:
            fieldnames = asdict(layer_res[0]).keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(r) for r in layer_res)
            print(res_path)
    except (KeyError, ValueError) as err:
        print(f"Warning: {err} in {res_path}")


if __name__ == "__main__":
    res_path = PROJECT_ROOT/"results"/"raw"/"raw_results.csv"
    for schedule in ["baseline", "Row-stationary inspired", "weight_stationary_inspired", "output_stationary_inspired"]:
        for model in ["AlexNet", "VGG16", "MobileNet"]:
            res = load_results(res_path, model, schedule)
            parse_res(res)
                