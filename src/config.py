"""Central configuration shared across the project.

Everything that more than one script needs to agree on lives here so the
pipeline, tuner, registry script and monitor all point at the same tracking
server, experiment and model name.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "creditcard.csv"          # drop the real Kaggle CSV here
ARTIFACTS_DIR = PROJECT_ROOT / "local_artifacts"  # scratch dir for plots etc.

# --- MLflow ----------------------------------------------------------------
# Point at the local tracking server started by `mlflow server ...`.
# Fallback to a local ./mlruns dir if no server is running.
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fraud-detection"
REGISTERED_MODEL_NAME = "fraud_detector"

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Schema (ULB credit-card dataset) --------------------------------------
TARGET = "Class"                       # 1 = fraud, 0 = legitimate
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", "Amount"]      # these get scaled inside the pipeline
FEATURES = PCA_FEATURES + RAW_FEATURES

# The headline metric for this imbalanced problem.
PRIMARY_METRIC = "average_precision"   # = area under the precision-recall curve

ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
