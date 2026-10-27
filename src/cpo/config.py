from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"


# Create generated folders automatically
for directory in [
    PROCESSED_DATA_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------

ENV_FILE = (
    RAW_DATA_DIR
    / "17082026_HTS_NN_Analysis_ENV_cleaned_NTUteam.xlsx"
)

WGS_FILE = (
    RAW_DATA_DIR
    / "17082026_HTS_NN_Analysis_WGS_cleaned_NTUteam.xlsx"
)


# ---------------------------------------------------------
# 8 TARGET ACQUISITION PATIENTS
# ---------------------------------------------------------

TARGET_PATIENTS = [
    "T0016",
    "T0036",
    "T0052",
    "T0058",
    "T0080",
    "T0081",
    "T0124",
    "T0125",
]