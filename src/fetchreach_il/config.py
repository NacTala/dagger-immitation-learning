from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data_collect"
LEGACY_PLOT_DATA_DIR = REPO_ROOT / "Data_for_plot"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"
PLOTS_DIR = ARTIFACT_DIR / "plots"

MANUAL_DATASET_PATH = DATA_DIR / "fetchreach_manual_dataset.pkl"
FETCHREACH_ENV_ID = "FetchReach-v4"
MAX_EPISODE_STEPS = 100
ACTION_DIM = 4
DEFAULT_STEP_SIZE = 0.2
DEFAULT_EVAL_EPISODES = 20
DEFAULT_BC_EPOCHS = 50
DEFAULT_DAGGER_ITERS = 50
DEFAULT_DAGGER_FIT_EPOCHS = 20
DEFAULT_SAC_TIMESTEPS = [500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000]
