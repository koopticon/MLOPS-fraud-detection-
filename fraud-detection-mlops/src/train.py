"""Objective 1 & 2: experiment tracking + model training across combos.

Trains every (model, imbalance-strategy) combination as its own MLflow run,
logging parameters, the full metric set, diagnostic plots and the model
artifact. Compare them afterwards in the MLflow UI (sort by average_precision).

    python src/train.py            # run all combos
    python src/train.py --only logreg:none random_forest:smote
"""
import argparse
import warnings

import mlflow
from mlflow.models import infer_signature

import config
import data
import models
from metrics import compute_metrics, save_diagnostics

warnings.filterwarnings("ignore", category=UserWarning)


def _setup_mlflow():
    try:
        mlflow.set_tracking_uri(config.TRACKING_URI)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    except Exception:
        # No server running -> fall back to a local ./mlruns store.
        print("[mlflow] tracking server unreachable; using local ./mlruns")
        mlflow.set_tracking_uri(f"file:{config.PROJECT_ROOT}/mlruns")
        mlflow.set_experiment(config.EXPERIMENT_NAME)


def run_combo(model_name, imbalance, X_tr, X_te, y_tr, y_te, spw):
    run_name = f"{model_name}-{imbalance}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_family": model_name,
                         "imbalance_strategy": imbalance,
                         "stage": "experiment"})
        mlflow.log_params({"model": model_name, "imbalance": imbalance,
                           "scale_pos_weight": spw if imbalance ==
                           "scale_pos_weight" else 1.0})

        pipe = models.build_pipeline(model_name, imbalance, spw)
        pipe.fit(X_tr, y_tr)

        y_prob = pipe.predict_proba(X_te)[:, 1]
        metrics = compute_metrics(y_te, y_prob)
        mlflow.log_metrics(metrics)

        for path in save_diagnostics(y_te, y_prob, config.ARTIFACTS_DIR,
                                     prefix=run_name):
            mlflow.log_artifact(path, artifact_path="diagnostics")

        signature = infer_signature(X_te, y_prob)
        mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                                 input_example=X_te.head(3))

        print(f"  {run_name:32s} "
              f"PR-AUC={metrics['average_precision']:.4f} "
              f"recall={metrics['recall']:.3f} "
              f"precision={metrics['precision']:.3f}")
        return run_name, metrics[config.PRIMARY_METRIC]


def main(only=None):
    _setup_mlflow()
    df = data.load_data()
    X_tr, X_te, y_tr, y_te = data.split_xy(df)
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    combos = models.COMBOS
    if only:
        wanted = {tuple(s.split(":")) for s in only}
        combos = [c for c in combos if c in wanted]

    print(f"[train] running {len(combos)} combos "
          f"(scale_pos_weight={spw:.1f})")
    results = []
    for model_name, imbalance in combos:
        if model_name == "xgboost" and not models.HAS_XGB:
            print(f"  skipping {model_name}-{imbalance} (xgboost not installed)")
            continue
        results.append(run_combo(model_name, imbalance,
                                 X_tr, X_te, y_tr, y_te, spw))

    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\n[train] best by {config.PRIMARY_METRIC}: "
              f"{best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset like logreg:none xgboost:smote")
    main(**vars(ap.parse_args()))
