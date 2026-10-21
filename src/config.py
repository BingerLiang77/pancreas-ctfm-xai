"""Central path configuration.

Override the raw-data location without editing code:
    export PANTS_DATA_ROOT="/path/to/ct-fm-classifyer/data"
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_ROOT = Path(os.environ.get(
    "PANTS_DATA_ROOT",
    "/Users/bingerliang/Library/Mobile Documents/com~apple~CloudDocs/ct-fm-classifyer/data",
))
CT_DIR = DATA_ROOT / "PanTSMini_ImageTr_00000001_00001000"
LABEL_DIR = DATA_ROOT / "PanTSMini_Label 2"
CACHE_DIR = SPLIT_DIR / "cache_roi_pt"
METADATA_PATH = SPLIT_DIR / "metadata.xlsx"
