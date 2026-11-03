"""Latencia de inferencia de los modelos entrenados, en proceso y de a una frase.

uv run python latencia.py

Para comparar con el LLM (`llm.py`) hay que tener en cuenta que no es la misma
medición: acá no hay servidor ni HTTP, y el LLM sí los tiene. Correr con la GPU libre.
"""

import json
import statistics
import time
from datetime import datetime

import pandas as pd

from experimento import git_commit
from llm import GpuPower
from tp.data import RESULTS_DIR, SEED, load_massive
from tp.models import MODELS, cache_path

N_SINGLE = 300  # frases medidas de a una
LATENCY_FILE = RESULTS_DIR / "latencia.csv"
RUNS_FILE = RESULTS_DIR / "llm_runs.jsonl"


def main():
    ds = load_massive()
    texts = list(ds["validation"]["text"])
    rows = []
    for name, build in MODELS.items():
        model = build(SEED)
        model.load(cache_path(name, model))
        model.predict(texts[:8])  # calentamiento (carga a GPU, cachés)

        lat = []
        for t in texts[:N_SINGLE]:
            start = time.perf_counter()
            model.predict([t])
            lat.append(time.perf_counter() - start)
        lat.sort()

        with GpuPower() as power:
            model.predict(texts)  # todo el conjunto de una vez
        row = {
            "model": name,
            "single_p50_ms": 1000 * statistics.median(lat),
            "single_p95_ms": 1000 * lat[int(0.95 * len(lat))],
            "single_mean_ms": 1000 * statistics.mean(lat),
            "batch_texts_per_s": len(texts) / power.seconds,
            "batch_gpu_w": power.watts,
            "batch_gpu_wh_per_1000": 1000 * power.wh / len(texts),
        }
        rows.append(row)
        print(
            f"{name:14s} de a una: p50={row['single_p50_ms']:.2f} ms "
            f"p95={row['single_p95_ms']:.2f} ms | todo junto: "
            f"{row['batch_texts_per_s']:.0f} frases/s, GPU {row['batch_gpu_w']:.0f} W",
            flush=True,
        )
        del model

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(LATENCY_FILE, index=False)
    with RUNS_FILE.open("a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit(),
                    "tipo": "latencia_modelos",
                    "n_single": N_SINGLE,
                    "rows": rows,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
