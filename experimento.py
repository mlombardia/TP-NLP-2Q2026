"""Entrena (o carga) los modelos y los evalúa sobre test y sobre validation con ruido.

uv run python experimento.py                          # todos los modelos
uv run python experimento.py --modelos tfidf_word,fasttext
uv run python experimento.py --modelos beto --reentrenar
uv run python experimento.py --modelos beto --semillas 2   # menos semillas de ruido
uv run python experimento.py --modelos beto --semillas-entrenamiento 1
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib import metadata

import pandas as pd

from tp.data import (
    DATA_REVISION,
    MODELS_DIR,
    RESULTS_DIR,
    ROOT,
    SEED,
    load_massive,
    set_seed,
)
from tp.evaluation import evaluate
from tp.models import MODELS, STOCHASTIC, cache_path
from tp.perturbations import N_SEEDS, make_perturbed_sets

METRICS_FILE = RESULTS_DIR / "metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.parquet"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"
N_TRAIN_SEEDS = 3
PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "fasttext-numpy2-wheel",
    "torch",
    "transformers",
]

def merge(new: pd.DataFrame, path, models):
    if path.exists():
        old = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(models)], new], ignore_index=True)
    return new

def git_commit() -> str:
    """Commit actual; "-dirty" si hay cambios sin commitear en archivos versionados."""

    def git(*cmd):
        return subprocess.run(
            ["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short=10", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return sha + ("-dirty" if dirty else "")

def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", default=",".join(MODELS))
    parser.add_argument("--reentrenar", action="store_true")
    parser.add_argument(
        "--semillas", type=int, default=N_SEEDS, help="semillas de ruido por conjunto"
    )
    parser.add_argument(
        "--semillas-entrenamiento",
        type=int,
        default=N_TRAIN_SEEDS,
        help="semillas de entrenamiento para FastText y BETO (TF-IDF usa una)",
    )
    args = parser.parse_args()
    names = args.modelos.split(",")

    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    commit = git_commit()
    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "data_revision": DATA_REVISION,
        "args": vars(args),
        "platform": platform.platform(),
        "versions": versions(),
        "models": [],
    }

    ds = load_massive()
    train, test, val = (
        {c: list(ds[s][c]) for c in ("text", "label")}
        for s in ("train", "test", "validation")
    )
    test_sets = {("clean", 0.0, SEED): test["text"]}
    val_sets = make_perturbed_sets(val["text"], args.semillas)

    all_metrics, all_preds = [], []
    for name in names:
        n_train = args.semillas_entrenamiento if name in STOCHASTIC else 1
        for train_seed in range(SEED, SEED + n_train):
            model = MODELS[name](train_seed)
            path = cache_path(name, model)
            tag = f"[{name} semilla={train_seed}]"
            t = time.time()
            trained = not model.exists(path) or args.reentrenar
            if trained:
                model.fit(train["text"], train["label"])
                model.save(path)
                print(f"{tag} entrenado en {time.time() - t:.0f}s")
            else:
                model.load(path)
                print(f"{tag} cargado de {path}")
            run["models"].append(
                {
                    "model": name,
                    "train_seed": train_seed,
                    "model_id": path.name,
                    "config": model.config,
                    "trained": trained,
                }
            )

            for split, sets, labels in (
                ("test", test_sets, test["label"]),
                ("validation", val_sets, val["label"]),
            ):
                m, p = evaluate(name, model, sets, labels, split)
                all_metrics.append(
                    m.assign(train_seed=train_seed, model_id=path.name, commit=commit)
                )
                all_preds.append(p.assign(train_seed=train_seed))
            for split, m in (
                ("test (selección)", all_metrics[-2]),
                ("validation", all_metrics[-1]),
            ):
                c = m[m["perturbation"] == "clean"].iloc[0]
                print(f"{tag} {split} limpio: acc={c.accuracy:.4f} f1={c.f1_macro:.4f}")

    merge(pd.concat(all_metrics), METRICS_FILE, names).to_csv(METRICS_FILE, index=False)
    merge(pd.concat(all_preds), PREDICTIONS_FILE, names).to_parquet(
        PREDICTIONS_FILE, index=False
    )
    with RUNS_FILE.open("a") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(f"Resultados en {METRICS_FILE} y {PREDICTIONS_FILE}; registro en {RUNS_FILE}")

if __name__ == "__main__":
    main()
