# Fraud Detection — ML Lifecycle Management with MLflow

AIN-3009 MLOps term project. An end-to-end machine-learning lifecycle for
**credit-card fraud detection**, managed with MLflow: experiment tracking,
model training and tuning, deployment, monitoring, and a model registry.

## Why fraud detection
The dataset is extremely imbalanced (~0.2% frauds), which makes every MLOps
decision meaningful: accuracy is useless, so we track **PR-AUC**, compare
imbalance strategies, tune a threshold, deploy a champion, and watch it drift.

## Repository layout
```
fraud-detection-mlops/
├── README.md
├── requirements.txt
├── run_all.sh                # runs the whole lifecycle in order
├── data/                     # put the real creditcard.csv here (gitignored)
├── report/                   # written report + outline
└── src/
    ├── config.py             # shared paths, tracking URI, model name
    ├── data.py               # loader + synthetic fallback + splits
    ├── models.py             # pipeline factory (model × imbalance strategy)
    ├── metrics.py            # imbalanced metrics + diagnostic plots
    ├── train.py              # Obj 1 & 2: experiment tracking + training
    ├── tune.py               # Obj 4: Hyperopt tuning with nested runs
    ├── register.py           # Obj 5: registry + @staging/@champion aliases
    ├── serve_test.py         # Obj 3: REST client for the served model
    └── monitor.py            # Obj 4: performance + drift over time
```

## The dataset
Uses the **ULB Credit Card Fraud Detection** dataset (Kaggle): download
`creditcard.csv` and place it in `data/`. If it is absent, the code generates
a synthetic dataset with the same schema and class imbalance so the pipeline
runs immediately — drop in the real file and everything works unchanged.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start the MLflow tracking server (backend DB + artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 --port 5000
```
The UI is then at http://127.0.0.1:5000. (If no server is running, the scripts
fall back to a local `./mlruns` store automatically.)

## Run the full lifecycle
```bash
bash run_all.sh
```
Or step by step:
```bash
python src/train.py            # Obj 1+2: log all model/imbalance combos
python src/tune.py --max-evals 30   # Obj 4: Hyperopt search (nested runs)
python src/register.py         # Obj 5: register best + set aliases
python src/monitor.py --inject-drift   # Obj 4: drift + perf over time
```

## Deploy and query the model (Objective 3)
Serve the promoted champion as a REST service:
```bash
mlflow models serve -m "models:/fraud_detector@champion" -p 5001 --no-conda
```
Then send transactions:
```bash
python src/serve_test.py --n 5      # real-time scoring of a few rows
python src/serve_test.py --batch    # batch scoring
```

## How the code maps to the five objectives
1. **Experiment tracking** — `train.py` logs params, the full metric set, PR
   curve + confusion-matrix artifacts, and the model for every run.
2. **Training & tuning** — six (model × imbalance-strategy) combos in
   `train.py`; `tune.py` runs Hyperopt with each trial as a nested MLflow run.
3. **Deployment** — `mlflow models serve` exposes a REST endpoint;
   `serve_test.py` exercises it for real-time and batch predictions.
4. **Monitoring** — `monitor.py` replays data in chronological batches,
   logging PR-AUC and drift (PSI / KS) over time and flagging drift.
5. **Model Registry** — `register.py` registers the best model and manages its
   lifecycle with aliases.

## A note on "stages" vs aliases
The brief asks for stage transitions (staging → production). MLflow's four
fixed stages are **deprecated** in current MLflow (3.x) in favour of *aliases*
and *tags*. This project implements the same lifecycle using aliases —
`@staging` for a version under validation and `@champion` for the promoted
production version — which is the current recommended approach.

## Submission
Zip the project as `PRJ-yourname-number.zip` per the assignment guidelines.
Code follows PEP 8 (`black` / `flake8` clean).
