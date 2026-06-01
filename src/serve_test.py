"""Objective 3: exercise the deployed model.

First serve the champion model in another terminal:

    mlflow models serve -m "models:/fraud_detector@champion" \
        -p 5001 --no-conda

Then send transactions to its REST endpoint:

    python src/serve_test.py --n 5            # real-time, a few rows
    python src/serve_test.py --batch          # score a whole batch

MLflow's scoring server expects the `dataframe_split` JSON orientation at
POST /invocations and returns a list of predictions.
"""
import argparse
import json

import requests

import config
import data

ENDPOINT = "http://127.0.0.1:5001/invocations"


def _payload(frame):
    return {"dataframe_split": {
        "columns": list(frame.columns),
        "data": frame.values.tolist(),
    }}


def score(frame):
    resp = requests.post(ENDPOINT, data=json.dumps(_payload(frame)),
                         headers={"Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main(n=5, batch=False):
    df = data.load_data()
    X = df[config.FEATURES]
    sample = X.head(200) if batch else X.sample(n, random_state=1)
    try:
        preds = score(sample)
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "Could not reach the model server at "
            f"{ENDPOINT}\nStart it with:\n  mlflow models serve "
            f'-m "models:/{config.REGISTERED_MODEL_NAME}@champion" -p 5001 --no-conda')
    result = preds.get("predictions", preds)
    print(f"[serve] sent {len(sample)} rows -> {len(result)} predictions")
    if not batch:
        for i, p in enumerate(result):
            print(f"  row {i}: prediction={p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    main(n=a.n, batch=a.batch)
