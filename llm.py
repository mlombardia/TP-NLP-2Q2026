"""Clasifica intenciones con un LLM local (Ollama) y lo evalúa como a los demás modelos.

uv run python llm.py --ajustar              # elige el prompt sobre test
uv run python llm.py --evaluar --k 0        # evalúa sobre validation con ruido
uv run python llm.py --evaluar --k 2 --modelo qwen3.5:4b

El LLM no se entrena: recibe las 60 etiquetas y, con `--k`, k ejemplos de train por
intención. El prompt se ajusta sobre test, igual que los hiperparámetros de los demás
modelos, y la robustez se mide sobre validation. Los resultados van a archivos propios
(`results/llm_*`) y no tocan `metrics.csv`.
"""

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from experimento import git_commit, merge
from tp.data import RESULTS_DIR, SEED, load_massive
from tp.evaluation import compute_metrics, evaluate
from tp.perturbations import make_perturbed_sets

METRICS_FILE = RESULTS_DIR / "llm_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "llm_predictions.parquet"
RUNS_FILE = RESULTS_DIR / "llm_runs.jsonl"
INTENSITIES = (0.25, 1.0)  # el subconjunto de la grilla que se evalúa
N_AJUSTE = 500  # frases de test para comparar prompts


class GpuPower:
    """Muestrea la potencia de la GPU (W) para estimar la energía de una corrida."""

    def __init__(self, period: float = 0.5):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                self.samples.append(float(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seconds = time.perf_counter() - self.start

    @property
    def watts(self) -> float:
        return statistics.mean(self.samples) if self.samples else float("nan")

    @property
    def wh(self) -> float:
        return self.watts * self.seconds / 3600


class OllamaClassifier:
    """Mismo `predict(textos)` que los demás modelos, pero llamando a Ollama."""

    def __init__(
        self,
        labels,
        train,
        model="llama3.1",
        k=0,
        host="127.0.0.1:11434",
        ctx=4096,
        think=None,
    ):
        self.labels, self.model, self.k, self.ctx = labels, model, k, ctx
        self.think = think  # False apaga el razonamiento de modelos que lo traen
        self.url = f"http://{host}/api/chat"
        self.system = self._system(train)
        self.schema = {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": labels}},
            "required": ["intent"],
        }
        self.latencies: list[float] = []
        self.prompt_tokens: list[int] = []

    def _system(self, train) -> str:
        text = (
            "Clasificás la intención de frases en español que un usuario le dice a un "
            "asistente virtual. Respondé con la etiqueta de intención más adecuada, "
            "usando solo estas etiquetas: " + ", ".join(self.labels) + "."
        )
        if self.k:
            rng = random.Random(SEED)
            by_label: dict[str, list[str]] = {lab: [] for lab in self.labels}
            for t, lab in zip(train["text"], train["label"], strict=True):
                by_label[lab].append(t)
            lines = [
                f'"{t}" -> {lab}'
                for lab in self.labels
                for t in rng.sample(by_label[lab], min(self.k, len(by_label[lab])))
            ]
            rng.shuffle(lines)
            text += "\n\nEjemplos:\n" + "\n".join(lines)
        return text

    def _classify(self, text: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": "30m",
            "format": self.schema,
            "options": {
                "temperature": 0,
                "seed": SEED,
                "num_predict": 24,
                "num_ctx": self.ctx,
            },
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": text},
            ],
        }
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        for attempt in range(3):
            t = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.load(r)
                intent = json.loads(out["message"]["content"])["intent"]
            except (OSError, ValueError, KeyError):
                if attempt == 2:
                    raise
                continue
            self.latencies.append(time.perf_counter() - t)
            self.prompt_tokens.append(out.get("prompt_eval_count", 0))
            return intent
        raise RuntimeError("inalcanzable")

    def predict(self, texts, workers: int = 4):
        done, total = [0], len(texts)

        def one(text):
            intent = self._classify(text)
            done[0] += 1
            if done[0] % 1000 == 0:
                print(f"  {done[0]}/{total}", flush=True)
            return intent

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, texts))

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        return {
            "requests": len(lat),
            "latency_mean_ms": 1000 * statistics.mean(lat),
            "latency_p50_ms": 1000 * statistics.median(lat),
            "latency_p95_ms": 1000 * lat[int(0.95 * len(lat))],
            "prompt_tokens_mean": statistics.mean(self.prompt_tokens),
        }


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC 95 % de una proporción."""
    p = correct / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def ajustar(args, labels, train, test):
    """Compara prompts sobre una muestra fija de test (limpio, nunca con ruido)."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test["text"]), N_AJUSTE, replace=False)
    texts = [test["text"][i] for i in idx]
    gold = [test["label"][i] for i in idx]
    rows = []
    for k in args.ks:
        clf = OllamaClassifier(
            labels, train, args.modelo, k, args.host, args.ctx, args.think
        )
        clf.predict(["pon una alarma"], workers=1)  # carga el modelo y llena la caché
        clf.latencies.clear()
        clf.prompt_tokens.clear()
        with GpuPower() as power:
            pred = clf.predict(texts, args.workers)
        m = compute_metrics(gold, pred)
        correct = round(m["accuracy"] * len(gold))
        lo, hi = wilson(correct, len(gold))
        row = {
            "modelo": args.modelo,
            "k": k,
            "n": len(gold),
            "accuracy": m["accuracy"],
            "ic95": [lo, hi],
            "f1_macro": m["f1_macro"],
            "segundos": power.seconds,
            "frases_por_s": len(gold) / power.seconds,
            "gpu_w": power.watts,
        } | clf.summary()
        rows.append(row)
        print(
            f"k={k}: acc={m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"f1={m['f1_macro']:.3f} {row['frases_por_s']:.1f} frases/s "
            f"p50={row['latency_p50_ms']:.0f} ms "
            f"tokens={row['prompt_tokens_mean']:.0f}",
            flush=True,
        )
    with RUNS_FILE.open("a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit(),
                    "tipo": "ajuste_prompt_test",
                    "rows": rows,
                }
            )
            + "\n"
        )


def evaluar(args, labels, train, val):
    name = f"llm_{args.modelo.replace(':', '_')}_k{args.k}"
    clf = OllamaClassifier(
        labels, train, args.modelo, args.k, args.host, args.ctx, args.think
    )
    all_sets = make_perturbed_sets(val["text"], n_seeds=1)  # solo la semilla SEED
    sets = {
        key: texts
        for key, texts in all_sets.items()
        if key[0] == "clean" or key[1] in INTENSITIES
    }
    n_unique = len({t for texts in sets.values() for t in texts})
    print(f"{name}: {len(sets)} conjuntos, {n_unique} textos distintos", flush=True)
    clf.predict(["pon una alarma"], workers=1)
    clf.latencies.clear()
    clf.prompt_tokens.clear()
    with GpuPower() as power:
        metrics, preds = evaluate(name, clf, sets, val["label"], "validation")
    commit = git_commit()
    metrics = metrics.assign(train_seed=SEED, commit=commit)
    preds = preds.assign(train_seed=SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    merge(metrics, METRICS_FILE, [name]).to_csv(METRICS_FILE, index=False)
    merge(preds, PREDICTIONS_FILE, [name]).to_parquet(PREDICTIONS_FILE, index=False)
    summary = clf.summary() | {
        "segundos": power.seconds,
        "frases_por_s": n_unique / power.seconds,
        "gpu_w": power.watts,
        "gpu_wh": power.wh,
        "gpu_wh_por_1000": 1000 * power.wh / n_unique,
    }
    with RUNS_FILE.open("a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "commit": commit,
                    "tipo": "evaluacion_validation",
                    "model": name,
                    "args": vars(args),
                    "system_prompt": clf.system,
                    "textos_distintos": n_unique,
                }
                | summary
            )
            + "\n"
        )
    clean = metrics[metrics["perturbation"] == "clean"].iloc[0]
    print(
        f"limpio: acc={clean['accuracy']:.3f} f1={clean['f1_macro']:.3f}; "
        f"{summary['frases_por_s']:.1f} frases/s; "
        f"p50={summary['latency_p50_ms']:.0f} ms; "
        f"{summary['gpu_wh_por_1000']:.2f} Wh/1000 frases",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="llama3.1")
    ap.add_argument("--host", default="127.0.0.1:11434")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4096, help="num_ctx de Ollama")
    ap.add_argument(
        "--no-think",
        dest="think",
        action="store_const",
        const=False,
        help="think=false (qwen3.5 sin esto devuelve contenido vacío)",
    )
    ap.add_argument("--ajustar", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--k", type=int, default=0)
    args = ap.parse_args()
    if args.ajustar == args.evaluar:
        ap.error("elegí exactamente una de --ajustar o --evaluar")

    ds = load_massive()
    train, test, val = (
        {c: list(ds[s][c]) for c in ("text", "label")}
        for s in ("train", "test", "validation")
    )
    labels = sorted(set(train["label"]))
    if args.ajustar:
        ajustar(args, labels, train, test)
    else:
        evaluar(args, labels, train, val)


if __name__ == "__main__":
    main()
