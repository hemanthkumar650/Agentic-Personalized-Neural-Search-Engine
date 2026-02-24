import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Dict


SUMMARY_PATH = Path("runs") / "experiment_summary.csv"
BASE_COLUMNS = [
    "timestamp",
    "run_type",
    "NDCG@10",
    "MRR",
    "Recall@10",
    "Precision@10",
    "metadata",
]


def log_experiment(run_type: str, metrics: Dict[str, float], metadata: Dict[str, str] | None = None) -> None:
    Path("runs").mkdir(exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_type": run_type,
        "NDCG@10": f"{metrics.get('NDCG@10', float('nan')):.6f}" if "NDCG@10" in metrics else "",
        "MRR": f"{metrics.get('MRR', float('nan')):.6f}" if "MRR" in metrics else "",
        "Recall@10": f"{metrics.get('Recall@10', float('nan')):.6f}" if "Recall@10" in metrics else "",
        "Precision@10": f"{metrics.get('Precision@10', float('nan')):.6f}" if "Precision@10" in metrics else "",
        "metadata": json.dumps(metadata or {}, ensure_ascii=True),
    }
    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
