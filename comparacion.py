"""Tabla final (accuracy, robustez, latencia, energía) y figura del cruce con los LLM.

uv run python comparacion.py

Lee los resultados ya calculados (`predictions.parquet`, `llm_predictions.parquet`,
`latencia.csv`, `llm_runs.jsonl`); no entrena ni llama a ningún modelo.
Escribe `results/comparacion.csv` y `results/figures/6_cruce_llm.png`.
"""

import json
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

import tp.plots as P  # fija el estilo de matplotlib del resto de las figuras
from tp.data import FIGURES_DIR, RESULTS_DIR
from tp.evaluation import paired_bootstrap

MODELS = {
    "tfidf_word": "TF-IDF palabras",
    "tfidf_char": "TF-IDF caracteres",
    "fasttext_word": "FastText sin subpalabras",
    "fasttext": "FastText con subpalabras",
    "beto": "BETO",
    "llm_llama3.1_k2": "llama3.1 (k=2)",
    "llm_qwen3.5_4b_k2": "qwen3.5:4b (k=2)",
}
LLM = [m for m in MODELS if m.startswith("llm_")]
LLM_COLORS = {"llm_llama3.1_k2": "#7a5cc7", "llm_qwen3.5_4b_k2": "#8a1f3d"}
COLUMNS = [
    "model",
    "split",
    "perturbation",
    "intensity",
    "seed",
    "idx",
    "label",
    "pred",
]


def load() -> pd.DataFrame:
    base = pd.read_parquet(RESULTS_DIR / "predictions.parquet", columns=COLUMNS)
    llm = pd.read_parquet(RESULTS_DIR / "llm_predictions.parquet", columns=COLUMNS)
    df = pd.concat([base, llm])
    df = df[(df["split"] == "validation") & df["model"].isin(MODELS)]
    df = df.assign(correct=(df["pred"] == df["label"]).astype(float))
    # promedio sobre semillas dentro de cada frase; el IC remuestrea frases
    by_sentence = df.groupby(["model", "perturbation", "intensity", "idx"])["correct"]
    return cast(pd.DataFrame, by_sentence.mean().reset_index())


def accuracy(acc: pd.DataFrame, model: str, pert: str, inten: float):
    """(accuracy, IC bajo, IC alto) de un modelo en una condición."""
    if pert == "clean":
        sel = acc[(acc["model"] == model) & (acc["perturbation"] == "clean")]
    else:
        sel = acc[
            (acc["model"] == model)
            & (acc["perturbation"] == pert)
            & (acc["intensity"] == inten)
        ]
    return paired_bootstrap(sel["correct"].to_numpy())


def costs() -> pd.DataFrame:
    lat = pd.read_csv(RESULTS_DIR / "latencia.csv").set_index("model")
    rows = {}
    for m in lat.index:
        rows[m] = {
            "p50_ms": lat.loc[m, "single_p50_ms"],
            "frases_por_s": lat.loc[m, "batch_texts_per_s"],
            # TF-IDF y FastText corren en CPU: la potencia de GPU medida es la de reposo
            "wh_por_1000": lat.loc[m, "batch_gpu_wh_per_1000"] if m == "beto" else None,
        }
    for line in (RESULTS_DIR / "llm_runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("tipo") == "evaluacion_validation" and r["model"] in LLM:
            rows[r["model"]] = {
                "p50_ms": r["latency_p50_ms"],
                "frases_por_s": r["frases_por_s"],
                "wh_por_1000": r["gpu_wh_por_1000"],
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def table(acc: pd.DataFrame) -> pd.DataFrame:
    cost = costs()
    conds = [
        ("clean", 0.0, "limpio"),
        ("typos", 0.25, "typos 0,25"),
        ("typos", 1.0, "typos 1,0"),
        ("mixed", 1.0, "combinado 1,0"),
    ]
    rows = []
    for m, label in MODELS.items():
        row: dict[str, object] = {"model": label}
        for pert, inten, name in conds:
            mean, lo, hi = accuracy(acc, m, pert, inten)
            row[name] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        row["caída typos 1,0"] = float(
            accuracy(acc, m, "clean", 0)[0] - accuracy(acc, m, "typos", 1.0)[0]
        )
        row.update({str(k): v for k, v in cost.loc[m].to_dict().items()})
        rows.append(row)
    return pd.DataFrame(rows)


def figure(acc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pert in zip(axes, ["typos", "mixed"], strict=True):
        for m, label in MODELS.items():
            llm = m in LLM
            color = LLM_COLORS[m] if llm else P.COLORS[m]
            # los LLM solo se evaluaron en 0,25 y 1,0
            levels = [0.25, 1.0] if llm else [0.25, 0.5, 0.75, 1.0]
            xs, ys, lo, hi = [0.0], [], [], []
            mean, a, b = accuracy(acc, m, "clean", 0)
            ys, lo, hi = [mean], [a], [b]
            for x in levels:
                mean, a, b = accuracy(acc, m, pert, x)
                xs.append(x)
                ys.append(mean)
                lo.append(a)
                hi.append(b)
            ax.errorbar(
                xs,
                ys,
                yerr=[
                    [y - a for y, a in zip(ys, lo, strict=True)],
                    [b - y for y, b in zip(ys, hi, strict=True)],
                ],
                color=color,
                lw=2.2 if m in ("beto", "llm_qwen3.5_4b_k2") else 1.5,
                ls="--" if llm else "-",
                marker="s" if llm else "o",
                ms=5,
                capsize=2,
                label=label if pert == "typos" else None,
            )
        ax.set_title(P.PERTURBATION_LABELS[pert])
        ax.set_xlabel("Intensidad del ruido")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(color=P.GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (validation, IC 95 %)")
    axes[0].set_ylim(0, 1)
    fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Modelos entrenados (líneas llenas) vs. LLM locales (punteadas)")
    P._save(fig, "6_cruce_llm")


def main():
    acc = load()
    tab = table(acc)
    RESULTS_DIR.mkdir(exist_ok=True)
    tab.to_csv(RESULTS_DIR / "comparacion.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(tab.round(3).to_string(index=False))
    figure(acc)
    print(f"figura: {FIGURES_DIR / '6_cruce_llm.png'}")


if __name__ == "__main__":
    main()
