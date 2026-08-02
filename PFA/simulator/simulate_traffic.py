"""
Network traffic simulator — 'Simulation' layer of the architecture
=====================================================================
Replays network connections (one by one) against the API (/predict), and
compares the result to the true label if available, to track accuracy live.

Supports 2 data sources:

1) CSV (raw dataset, named columns):
    python simulate_traffic.py --csv dataset_ids2025.csv
    python simulate_traffic.py --csv dataset_ids2025.csv --speed 5 --loop
    python simulate_traffic.py --csv dataset_ids2025.csv --limit 500 --label-col Label

2) .npy (your real test set, exported from the ML notebook — e.g. the 20% test split):
    python simulate_traffic.py --npy-features X_test_fe.npy --npy-labels y_test.npy
    python simulate_traffic.py --npy-features X_test_fe.npy --npy-labels y_test.npy --speed 5 --limit 300

⚠️ Make sure to use the "_fe" file (features AFTER feature engineering / selection,
43 columns) — not X_test.npy or X_test_scaled.npy (78 columns), which are
BEFORE feature selection and don't match what the model expects.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_vector_from_row(row, feature_order, missing_report):
    """Builds the feature list (CSV) in the same order the model expects."""
    values = []
    for feat in feature_order:
        if feat in row.index:
            val = row[feat]
            try:
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
        else:
            val = 0.0
            missing_report.add(feat)
        values.append(val)
    return values


def load_from_csv(args, feature_order):
    """Returns a list of (features: list[float], true_label: str|None)"""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print(f"📂 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    has_ground_truth = args.label_col is not None and args.label_col in df.columns
    missing_report = set()

    samples = []
    for _, row in df.iterrows():
        features = build_feature_vector_from_row(row, feature_order, missing_report)
        true_label = str(row[args.label_col]) if has_ground_truth else None
        samples.append((features, true_label))

    if missing_report:
        print(f"⚠️  Columns missing from the CSV (filled with 0.0): {sorted(missing_report)}")

    return samples, has_ground_truth


def load_from_npy(args, feature_order):
    """Returns a list of (features: list[float], true_label: str|None)"""
    feat_path = Path(args.npy_features)
    if not feat_path.exists():
        print(f"❌ File not found: {feat_path}")
        sys.exit(1)

    print(f"📂 Reading .npy: {feat_path}")
    X = np.load(feat_path, allow_pickle=True)

    if X.ndim != 2:
        print(f"❌ {feat_path} must be a 2D array (n_samples, n_features), found shape: {X.shape}")
        sys.exit(1)

    if X.shape[1] != len(feature_order):
        print(f"❌ Feature mismatch: {feat_path} has {X.shape[1]} columns, "
              f"but the model expects {len(feature_order)} (selected_features.json).")
        print("   → Did you point to the '_fe' file (after feature engineering),")
        print("     not X_test.npy / X_test_scaled.npy (usually BEFORE feature selection)?")
        sys.exit(1)

    print(f"✅ {X.shape[0]} rows × {X.shape[1]} features loaded")

    y = None
    has_ground_truth = False
    if args.npy_labels:
        labels_path = Path(args.npy_labels)
        if not labels_path.exists():
            print(f"❌ File not found: {labels_path}")
            sys.exit(1)
        y = np.load(labels_path, allow_pickle=True)
        if len(y) != X.shape[0]:
            print(f"❌ Mismatch: {feat_path.name} has {X.shape[0]} rows but "
                  f"{labels_path.name} has {len(y)}. They must be aligned row by row.")
            sys.exit(1)

        class_names = load_json(args.class_names)
        print(f"✅ {len(y)} labels loaded (mapped via {args.class_names})")
        has_ground_truth = True

    n = X.shape[0]
    idx = np.arange(n)
    if args.shuffle:
        np.random.shuffle(idx)
    if args.limit:
        idx = idx[:args.limit]

    samples = []
    for i in idx:
        features = np.nan_to_num(X[i].astype(float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        true_label = None
        if has_ground_truth:
            label_int = int(y[i])
            if 0 <= label_int < len(class_names):
                true_label = class_names[label_int]
            else:
                true_label = f"UNKNOWN_LABEL_{label_int}"
        samples.append((features, true_label))

    return samples, has_ground_truth


def main():
    parser = argparse.ArgumentParser(description="IDS 2025 network traffic simulator")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to an IDS 2025 dataset (.csv)")
    source.add_argument("--npy-features", help="Path to X_test_fe.npy (features, after feature engineering)")

    parser.add_argument("--npy-labels", default=None,
                         help="Path to y_test.npy (integer labels, aligned with --npy-features)")
    parser.add_argument("--class-names", default="../backend/class_names.json",
                         help="Path to class_names.json (to map y_test.npy's integer labels)")
    parser.add_argument("--api", default="http://localhost:5000", help="Backend API URL")
    parser.add_argument("--features", default="../backend/selected_features.json",
                         help="Path to selected_features.json (order expected by the model)")
    parser.add_argument("--speed", type=float, default=2.0,
                         help="Connections sent per second (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of rows to replay (default: the whole file)")
    parser.add_argument("--label-col", default=None,
                         help="[--csv mode only] Name of the ground-truth label column")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before replaying them")
    parser.add_argument("--loop", action="store_true", help="Replay the dataset in an infinite loop")
    args = parser.parse_args()

    feature_order = load_json(args.features)
    print(f"✅ {len(feature_order)} expected features (from {args.features})")

    if args.csv:
        samples, has_ground_truth = load_from_csv(args, feature_order)
    else:
        samples, has_ground_truth = load_from_npy(args, feature_order)

    print(f"✅ {len(samples)} connections to replay against {args.api}/predict")
    print(f"⏱  Speed: {args.speed} connections/second")
    if has_ground_truth:
        print("📊 Ground truth available → accuracy computed live")
    if args.loop:
        print("🔁 Infinite loop mode enabled (Ctrl+C to stop)")
    print("-" * 55)

    delay = 1.0 / args.speed if args.speed > 0 else 0
    total_sent, total_alerts, correct = 0, 0, 0

    try:
        while True:
            for features, true_label in samples:
                try:
                    resp = requests.post(f"{args.api}/predict",
                                          json={"features": features}, timeout=5)
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.RequestException as e:
                    print(f"⚠️  API error: {e}")
                    time.sleep(delay)
                    continue

                total_sent += 1
                pred = result.get("prediction", "?")
                conf = result.get("confidence", 0)
                is_alert = result.get("alert", False)
                severity = result.get("severity", "?")

                if is_alert:
                    total_alerts += 1

                status_icon = "🚨" if is_alert else "✅"
                line = (f"{status_icon} [{total_sent:>5}] {pred:<12} "
                        f"conf={conf:.2%}  severity={severity:<8}")

                if has_ground_truth and true_label is not None:
                    match = (true_label == pred)
                    correct += int(match)
                    acc_so_far = correct / total_sent
                    line += f"  | true={true_label:<12} {'✓' if match else '✗'}  acc={acc_so_far:.2%}"

                print(line)
                time.sleep(delay)

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\n⏹  Simulation stopped by user")

    print("-" * 55)
    print(f"Summary: {total_sent} connections sent, {total_alerts} alerts "
          f"({total_alerts / max(total_sent, 1):.2%})")
    if has_ground_truth and total_sent:
        print(f"Accuracy vs ground truth: {correct / total_sent:.2%}")


if __name__ == "__main__":
    main()
