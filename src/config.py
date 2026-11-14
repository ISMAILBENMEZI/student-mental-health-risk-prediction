
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student_mental_health.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "df_clean.csv"

TARGET = "Mental_Health_Risk"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model_pipeline.joblib"

REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"

