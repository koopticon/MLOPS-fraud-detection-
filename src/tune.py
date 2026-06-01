"""Objective 4: hyperparameter tuning with Hyperopt + nested MLflow runs.

A parent run wraps the whole search; every Hyperopt trial is logged as a
child (nested) run, so the MLflow UI shows the search as an expandable tree.
We optimise validation PR-AUC (Hyperopt minimises, so we return -PR-AUC).

    python src/tune.py --max-evals 30
"""
import argparse
import warnings

import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
import data
import models
from metrics import compute_metrics, save_diagnostics

warnings.filterwarnings("ignore")

SPACE = {
    "n_estimators": hp.quniform("n_estimators", 100, 600, 50),
    "max_depth": hp.quniform("max_depth", 3, 9, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
}


def _setup():
    try:
        mlflow.set_tracking_uri(config.TRACKING_URI)
    except Exception:
        mlflow.set_tracking_uri(f"file:{config.PROJECT_ROOT}/mlruns")
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def main(max_evals=30):
    if not models.HAS_XGB:
        raise SystemExit("This tuner targets XGBoost; please `pip install xgboost`.")
    _setup()
    df = data.load_data()
    X_tr, X_te, y_tr, y_te = data.split_xy(df)
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    def objective(params):
        params = {**params,
                  "n_estimators": int(params["n_estimators"]),
                  "max_depth": int(params["max_depth"])}
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            from xgboost import XGBClassifier
            from imblearn.pipeline import Pipeline
            pipe = Pipeline([("prep", models._preprocessor()),
                             ("clf", XGBClassifier(
                                 **params, eval_metric="aucpr",
                                 scale_pos_weight=spw, n_jobs=-1,
                                 random_state=config.RANDOM_STATE))])
            score = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                    scoring="average_precision", n_jobs=-1).mean()
            mlflow.log_metric("cv_average_precision", float(score))
            return {"loss": -score, "status": STATUS_OK}

    with mlflow.start_run(run_name="hyperopt-xgboost-search") as parent:
        mlflow.set_tags({"model_family": "xgboost", "stage": "tuning"})
        trials = Trials()
        best = fmin(objective, SPACE, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials,
                    rstate=np.random.default_rng(config.RANDOM_STATE))
        best = {**best, "n_estimators": int(best["n_estimators"]),
                "max_depth": int(best["max_depth"])}
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_average_precision",
                          float(-min(t["result"]["loss"] for t in trials.trials)))

        # Refit best config on full train set, evaluate on test, log final model.
        from xgboost import XGBClassifier
        from imblearn.pipeline import Pipeline
        final = Pipeline([("prep", models._preprocessor()),
                          ("clf", XGBClassifier(
                              **best, eval_metric="aucpr", scale_pos_weight=spw,
                              n_jobs=-1, random_state=config.RANDOM_STATE))])
        final.fit(X_tr, y_tr)
        y_prob = final.predict_proba(X_te)[:, 1]
        test_metrics = compute_metrics(y_te, y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                            if isinstance(v, (int, float))})
        for p in save_diagnostics(y_te, y_prob, config.ARTIFACTS_DIR, "tuned"):
            mlflow.log_artifact(p, artifact_path="diagnostics")
        mlflow.sklearn.log_model(final, name="model",
                                 signature=infer_signature(X_te, y_prob),
                                 input_example=X_te.head(3))
        print(f"\n[tune] best params: {best}")
        print(f"[tune] test PR-AUC={test_metrics['average_precision']:.4f}  "
              f"parent run_id={parent.info.run_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-evals", type=int, default=30)
    main(max_evals=ap.parse_args().max_evals)
