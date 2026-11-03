import random
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42

# Commit fijo del mirror: si el repo cambia, los datos no cambian sin aviso.
DATA_REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
HF_URL = (
    "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/"
    + DATA_REVISION
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_massive(lang: str = "es") -> DatasetDict:
    data_files = {
        s: f"{HF_URL}/{s}/{lang}.json.gz" for s in ("train", "validation", "test")
    }
    ds = load_dataset(
        "json", data_files=data_files, cache_dir=str(DATA_DIR / "hf_cache")
    )
    return ds.select_columns(["text", "label"])
