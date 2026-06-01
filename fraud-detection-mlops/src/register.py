"""Objective 5: Model Registry lifecycle with aliases.

NOTE on the assignment wording: the brief asks for "stage transitions like
staging and production". MLflow's four fixed stages (None/Staging/Production/
Archived) are now DEPRECATED in favour of *aliases* and *tags*. This script
demonstrates the same staging->production lifecycle the brief intends, using
the current alias-based API:

    @staging   -> a version under validation
    @champion  -> the version promoted to production

    python src/register.py
"""
import argparse

import mlflow
from mlflow import MlflowClient

import config


def _client() -> MlflowClient:
    try:
        mlflow.set_tracking_uri(config.TRACKING_URI)
        MlflowClient().search_experiments()  # connectivity probe
    except Exception:
        mlflow.set_tracking_uri(f"file:{config.PROJECT_ROOT}/mlruns")
    return MlflowClient()


def best_run(client: MlflowClient):
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit("No experiment found; run train.py first.")
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{config.PRIMARY_METRIC} DESC"],
        max_results=1,
        filter_string="attributes.status = 'FINISHED'",
    )
    if not runs:
        raise SystemExit("No finished runs with the primary metric found.")
    return runs[0]


def main(promote=True):
    client = _client()
    run = best_run(client)
    score = run.data.metrics.get(config.PRIMARY_METRIC)
    print(f"[registry] best run {run.info.run_id[:8]} "
          f"{config.PRIMARY_METRIC}={score:.4f}")

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    print(f"[registry] registered {config.REGISTERED_MODEL_NAME} v{mv.version}")

    # Describe + tag the version, then mark it as staging.
    client.update_model_version(
        config.REGISTERED_MODEL_NAME, mv.version,
        description=f"Best model by {config.PRIMARY_METRIC} ({score:.4f}).")
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version,
                                 "validation_metric", f"{score:.4f}")
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, "staging", mv.version)
    print(f"[registry] set alias @staging -> v{mv.version}")

    if promote:
        # Promotion gate: only champion it if it clears a quality bar.
        if score is not None and score >= 0.5:
            client.set_registered_model_alias(
                config.REGISTERED_MODEL_NAME, "champion", mv.version)
            client.set_model_version_tag(config.REGISTERED_MODEL_NAME,
                                         mv.version, "promoted", "true")
            print(f"[registry] promoted @champion -> v{mv.version} "
                  f"(serve with models:/{config.REGISTERED_MODEL_NAME}@champion)")
        else:
            print("[registry] score below promotion gate; left in @staging")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-promote", action="store_true")
    main(promote=not ap.parse_args().no_promote)
