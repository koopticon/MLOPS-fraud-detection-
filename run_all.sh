#!/usr/bin/env bash
# Run the whole MLflow lifecycle end-to-end.
# Assumes the tracking server is already running (see README), or scripts
# will fall back to a local ./mlruns store automatically.
set -e
cd "$(dirname "$0")"

echo "==> 1. Experiment tracking + training"
python src/train.py

echo "==> 2. Hyperparameter tuning (Hyperopt, nested runs)"
python src/tune.py --max-evals 25

echo "==> 3. Register best model + set @staging/@champion aliases"
python src/register.py

echo "==> 4. Monitoring + drift simulation"
python src/monitor.py --inject-drift

echo "==> Done. Open the MLflow UI to explore (http://127.0.0.1:5000)."
echo "    To serve the model:"
echo '    mlflow models serve -m "models:/fraud_detector@champion" -p 5001 --no-conda'
echo "    Then: python src/serve_test.py --n 5"
