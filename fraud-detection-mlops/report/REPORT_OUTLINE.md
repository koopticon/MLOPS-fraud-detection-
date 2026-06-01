# Project Report — Outline

Fill each section in as you run the pipeline. Target ~6–10 pages.

## 1. Introduction
- Problem: credit-card fraud detection as an MLOps lifecycle case study.
- Why it suits MLflow: severe class imbalance forces real tracking/monitoring
  decisions rather than a toy accuracy chase.

## 2. Domain & Dataset
- ULB Credit Card Fraud dataset: ~284,807 transactions, ~0.172% fraud.
- Features: V1–V28 (PCA-anonymised), Time, Amount, Class.
- Note the synthetic fallback used for development.

## 3. MLflow Setup (architecture)
- Tracking server + SQLite backend store + local artifact store.
- Diagram: client scripts → tracking server → DB + artifact store → UI.

## 4. Methodology
### 4.1 Preprocessing
- Scaling Time/Amount inside the pipeline; V* left as-is.
### 4.2 Models & imbalance strategies
- Logistic Regression, Random Forest, XGBoost.
- none / class_weight / scale_pos_weight / SMOTE.
### 4.3 Metrics
- Why accuracy is rejected; PR-AUC as the primary metric; recall/precision
  trade-off; threshold selection.

## 5. Experiment Tracking (Objective 1)
- Screenshot: runs comparison table sorted by average_precision.
- Discuss what each parameter/metric tells you.

## 6. Training Results (Objective 2)
- Comparison table of the six combos.
- Insight: how class_weight trades precision for recall, etc.

## 7. Hyperparameter Tuning (Objective 4)
- Hyperopt search space and objective (maximise CV PR-AUC).
- Screenshot: nested-run tree in the MLflow UI.
- Best params + improvement over baseline.

## 8. Deployment (Objective 3)
- `mlflow models serve`; sample request/response.
- Real-time vs batch scoring.

## 9. Monitoring & Drift (Objective 4)
- Time-batch replay; PSI / KS drift detection.
- Screenshot/figure: performance vs drift over time (monitoring_over_time.png).
- When drift crosses threshold → retraining trigger discussion.

## 10. Model Registry (Objective 5)
- Registration, @staging → @champion promotion, tags.
- Explain the stages-vs-aliases deprecation and why aliases are used.

## 11. Conclusion & Reflection
- What MLflow made easy/hard; what you'd add for production (CI/CD, auth,
  scheduled retraining, a real feature store).

## 12. References
- MLflow docs, dataset source, libraries used.
