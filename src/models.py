"""Model + imbalance-strategy pipeline factory.

Each pipeline bundles preprocessing (scaling Time/Amount; the V* features are
already PCA-scaled) with an optional resampler and a classifier, so the logged
model accepts *raw* transaction rows at serving time.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import config

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                       # keep the repo runnable without xgboost
    HAS_XGB = False


# (model, imbalance) combinations that get tracked as separate experiments.
COMBOS = [
    ("logreg", "none"),
    ("logreg", "class_weight"),
    ("random_forest", "class_weight"),
    ("random_forest", "smote"),
    ("xgboost", "scale_pos_weight"),
    ("xgboost", "smote"),
]


def _preprocessor() -> ColumnTransformer:
    """Scale only the raw columns; pass PCA features through untouched."""
    return ColumnTransformer(
        transformers=[("scale_raw", StandardScaler(), config.RAW_FEATURES)],
        remainder="passthrough",
    )


def _classifier(model_name: str, imbalance: str, scale_pos_weight: float):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000, n_jobs=-1,
            class_weight="balanced" if imbalance == "class_weight" else None,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=config.RANDOM_STATE,
            class_weight="balanced" if imbalance == "class_weight" else None,
        )
    if model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed; `pip install xgboost`")
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="aucpr", random_state=config.RANDOM_STATE, n_jobs=-1,
            scale_pos_weight=(scale_pos_weight
                              if imbalance == "scale_pos_weight" else 1.0),
        )
    raise ValueError(f"unknown model: {model_name}")


def build_pipeline(model_name: str, imbalance: str,
                   scale_pos_weight: float = 1.0) -> ImbPipeline:
    """Assemble preprocess -> (optional SMOTE) -> classifier."""
    steps = [("prep", _preprocessor())]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("clf", _classifier(model_name, imbalance, scale_pos_weight)))
    return ImbPipeline(steps)
