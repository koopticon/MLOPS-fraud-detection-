"""Objective 4 (monitoring): track performance and data drift over time.

Simulates production traffic by splitting the data into chronological batches.
For each batch we (a) score it with the champion model and log live metrics,
and (b) measure feature drift versus a reference window using PSI and the
Kolmogorov-Smirnov statistic -- the same ideas Evidently reports, implemented
with scipy so the repo has no fragile version dependencies.

To make drift observable, later batches are perturbed (`--inject-drift`).

    python src/monitor.py --inject-drift
"""
import argparse
import warnings

import mlflow
import numpy as np
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

import config  # noqa: E402
import data  # noqa: E402
from metrics import compute_metrics  # noqa: E402

warnings.filterwarnings("ignore")


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_pct = np.histogram(reference, bins=quantiles)[0] / len(reference)
    cur_pct = np.histogram(current, bins=quantiles)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_summary(reference_df, batch_df, features) -> dict:
    """Average PSI and max KS across monitored features."""
    psis, kss = [], []
    for f in features:
        psis.append(psi(reference_df[f].values, batch_df[f].values))
        kss.append(ks_2samp(reference_df[f].values, batch_df[f].values).statistic)
    return {"mean_psi": float(np.mean(psis)),
            "max_psi": float(np.max(psis)),
            "max_ks": float(np.max(kss))}


def load_champion():
    try:
        mlflow.set_tracking_uri(config.TRACKING_URI)
        return mlflow.pyfunc.load_model(
            f"models:/{config.REGISTERED_MODEL_NAME}@champion")
    except Exception:
        mlflow.set_tracking_uri(f"file:{config.PROJECT_ROOT}/mlruns")
        return mlflow.pyfunc.load_model(
            f"models:/{config.REGISTERED_MODEL_NAME}@champion")


def main(n_batches=6, inject_drift=False):
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    model = load_champion()
    df = data.load_data()
    reference = df.sample(min(5000, len(df)), random_state=0)
    monitored = config.PCA_FEATURES[:8] + ["Amount"]

    rows = []
    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.set_tags({"stage": "monitoring", "drift_injected": inject_drift})
        for step, batch in data.time_ordered_batches(df, n_batches):
            batch = batch.copy()
            if inject_drift and step >= n_batches // 2:
                # Gradually shift the distribution of later batches.
                factor = 1 + 0.4 * (step - n_batches // 2 + 1)
                batch["Amount"] *= factor
                for f in config.PCA_FEATURES[:5]:
                    batch[f] += 0.3 * (step - n_batches // 2 + 1)

            X = batch[config.FEATURES]
            y = batch[config.TARGET]
            y_prob = np.asarray(model.predict(X)).ravel()
            # pyfunc may return labels or probs; coerce to float scores.
            perf = compute_metrics(y, y_prob)
            drift = drift_summary(reference, batch, monitored)

            mlflow.log_metrics({
                "batch_average_precision": perf["average_precision"],
                "batch_recall": perf["recall"],
                "batch_precision": perf["precision"],
                **drift,
            }, step=step)
            rows.append((step, perf["average_precision"], drift["mean_psi"]))
            flag = "  <-- DRIFT" if drift["mean_psi"] > 0.2 else ""
            print(f"  batch {step}: PR-AUC={perf['average_precision']:.3f} "
                  f"mean_PSI={drift['mean_psi']:.3f}{flag}")

        # Plot performance vs drift over time and log it.
        steps = [r[0] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(steps, [r[1] for r in rows], "o-", color="tab:blue",
                 label="PR-AUC")
        ax1.set(xlabel="time batch", ylabel="PR-AUC")
        ax2 = ax1.twinx()
        ax2.plot(steps, [r[2] for r in rows], "s--", color="tab:red",
                 label="mean PSI")
        ax2.axhline(0.2, color="tab:red", ls=":", alpha=0.5)
        ax2.set_ylabel("mean PSI (drift)")
        fig.suptitle("Model performance vs feature drift over time")
        fig.tight_layout()
        path = config.ARTIFACTS_DIR / "monitoring_over_time.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(path), artifact_path="monitoring")
        print(f"[monitor] logged {len(rows)} batches + chart to MLflow")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--inject-drift", action="store_true")
    a = ap.parse_args()
    main(n_batches=a.n_batches, inject_drift=a.inject_drift)
