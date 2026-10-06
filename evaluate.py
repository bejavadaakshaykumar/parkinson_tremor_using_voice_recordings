"""Run the same strict evaluation without launching the dashboard."""
import argparse
import json
from pathlib import Path
import pandas as pd
from modeling import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, default=Path("validation_audit.json"))
    args = parser.parse_args()
    result = train_model(pd.read_csv(args.dataset), progress=print)
    metrics = result[4]
    args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name in ["accuracy", "f1", "sensitivity", "specificity"]:
        print(f"{name}: {metrics[name]:.2f}%")
    print(f"Saved threshold: {metrics['decision_threshold']:.4f}")
    print(f"Full audit: {args.report.resolve()}")
