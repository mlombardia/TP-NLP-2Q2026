import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tp.data import FIGURES_DIR
from tp.evaluation import average_seeds, per_class_f1
from tp.models import SUBWORD

MODEL_LABELS = {
    "tfidf_word": "TF-IDF palabras",
    "tfidf_char": "TF-IDF caracteres",
    "fasttext_word": "FastText sin subpalabras",
    "fasttext": "FastText con subpalabras",
    "beto": "BETO",
}
COLORS = {
    "tfidf_char": "#2a78d6",
    "tfidf_word": "#eb6834",
    "fasttext": "#1baf7a",
    "fasttext_word": "#eda100",
    "beto": "#e87ba4",
}
PERTURBATION_LABELS = {
    "no_accents": "Sin tildes",
    "lower_no_punct": "Minúsculas y sin puntuación",
    "typos": "Errores de tipeo",
    "abbreviations": "Abreviaturas",
    "voseo": "Voseo rioplatense",
    "colloquial": "Vocabulario coloquial",
    "mixed": "Combinado",
}
GROUP_COLORS = {"Subpalabras": "#2a78d6", "Palabras completas": "#eb6834"}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)

def _models(df):
    return [m for m in MODEL_LABELS if m in set(df["model"])]

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    return fig

def relative_drop(metrics: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Caída relativa (%) de cada conjunto con ruido respecto de validation limpio.

    Usa el promedio entre semillas de ruido y de entrenamiento."""
    metrics = average_seeds(metrics)
    val = metrics[metrics["split"] == "validation"]
    clean = val[val["perturbation"] == "clean"].set_index("model")[metric]
    noisy = val[val["perturbation"] != "clean"].copy()
    noisy["drop_pct"] = (100 * (1 - noisy[metric] / noisy["model"].map(clean))).round(
        1
    ) + 0.0
    return noisy

def plot_clean(metrics):
    """Rendimiento sin ruido sobre validation.

    No se usa test: con test se eligieron los hiperparámetros, así que su rendimiento
    está sesgado a favor de los modelos ajustados.
    """
    metrics = average_seeds(metrics)
    clean = metrics[
        (metrics["split"] == "validation") & (metrics["perturbation"] == "clean")
    ].set_index("model")
    models = _models(clean.reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (metric, title) in zip(
        axes, [("accuracy", "Accuracy"), ("f1_macro", "F1 macro")], strict=True
    ):
        values = clean.loc[models, metric]
        ax.barh(
            [MODEL_LABELS[m] for m in models],
            values,
            color=[COLORS[m] for m in models],
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=TEXT)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Rendimiento sin ruido (validation)", x=0.02, ha="left", fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, "1_rendimiento_limpio")

def plot_degradation(metrics, metric="f1_macro"):
    metrics = average_seeds(metrics)
    val = metrics[metrics["split"] == "validation"]
    clean = val[val["perturbation"] == "clean"]
    models = _models(val)
    perts = [p for p in PERTURBATION_LABELS if p in set(val["perturbation"])]
    ncols = 4
    nrows = -(-len(perts) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 4 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flat):
        if k >= len(perts):
            ax.set_visible(False)
        elif k + ncols >= len(perts):  # sin panel visible debajo: mostrar el eje x
            ax.set_xlabel("Intensidad del ruido")
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax, pert in zip(axes.flat, perts, strict=False):
        curve = pd.concat(
            [clean.assign(perturbation=pert), val[val["perturbation"] == pert]]
        )
        for m in models:
            c = curve[curve["model"] == m].sort_values("intensity")
            ax.plot(
                c["intensity"],
                c[metric],
                color=COLORS[m],
                linewidth=2,
                linestyle="-" if SUBWORD[m] else "--",
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=MODEL_LABELS[m],
            )
        ax.set_title(PERTURBATION_LABELS[pert], loc="left")
        ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 macro" if metric == "f1_macro" else metric)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.0)
    )
    fig.suptitle(
        "Degradación del rendimiento según tipo e intensidad del ruido "
        "(línea punteada = palabras completas)",
        y=1.04,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "2_curvas_degradacion")

def plot_drop_heatmap(metrics, intensity=1.0):
    drop = relative_drop(metrics)
    drop = drop[drop["intensity"] == intensity]
    table = drop.pivot(index="model", columns="perturbation", values="drop_pct")
    table = table.loc[
        _models(drop), [p for p in PERTURBATION_LABELS if p in table.columns]
    ]
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(table) + 1.5))
    vmax = max(table.values.max(), 1)
    im = ax.imshow(table.clip(lower=0), cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.1f}%",
                ha="center",
                va="center",
                color="white" if v > 0.55 * vmax else TEXT,
            )
    ax.set_xticks(
        range(table.shape[1]),
        [PERTURBATION_LABELS[p] for p in table.columns],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(table.shape[0]), [MODEL_LABELS[m] for m in table.index])
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, label="Caída relativa del F1 macro (%)", shrink=0.8)
    ax.set_title(
        f"Pérdida de F1 macro con ruido a intensidad {intensity:g}", loc="left"
    )
    fig.tight_layout()
    return _save(fig, "3_heatmap_caida")

def plot_word_vs_subword(metrics, intensity=1.0):
    drop = relative_drop(metrics)
    drop = drop[drop["intensity"] == intensity].copy()
    group_names = {True: "Subpalabras", False: "Palabras completas"}
    drop["group"] = drop["model"].map(SUBWORD).map(group_names)
    table = drop.groupby(["perturbation", "group"])["drop_pct"].mean().unstack()
    table = table.loc[[p for p in PERTURBATION_LABELS if p in table.index]]
    groups = [g for g in GROUP_COLORS if g in table.columns]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x, w = np.arange(len(table)), 0.38
    for k, g in enumerate(groups):
        offset = (k - (len(groups) - 1) / 2) * w
        bars = ax.bar(
            x + offset,
            table[g],
            w,
            color=GROUP_COLORS[g],
            label=g,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=2, color=TEXT, fontsize=9)
    ax.set_xticks(
        x, [PERTURBATION_LABELS[p] for p in table.index], rotation=15, ha="right"
    )
    ax.set_ylabel("Caída relativa promedio del F1 macro (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title(
        f"Palabras completas vs. subpalabras (intensidad {intensity:g})", loc="left"
    )
    fig.tight_layout()
    return _save(fig, "4_palabras_vs_subpalabras")

def plot_intent_drop(
    predictions,
    models=("tfidf_word", "tfidf_char"),
    perturbation="mixed",
    intensity=0.5,
    top=10,
):
    val = predictions[
        (predictions["split"] == "validation") & predictions["model"].isin(models)
    ]
    rows = []
    for m in models:
        for pert, p in (("clean", 0.0), (perturbation, intensity)):
            sub = val[
                (val["model"] == m)
                & (val["perturbation"] == pert)
                & (val["intensity"] == p)
            ]
            f1 = per_class_f1(sub["label"].tolist(), sub["pred"].tolist())
            rows += [
                {"model": m, "noisy": pert != "clean", "label": lab, "f1": v}
                for lab, v in f1.items()
            ]
    df = pd.DataFrame(rows).pivot_table(
        index=["model", "label"], columns="noisy", values="f1"
    )
    drop = (df[False] - df[True]).unstack("model")
    drop = drop.sort_values(models[0], ascending=False).head(top)

    fig, ax = plt.subplots(figsize=(10, 0.5 * top + 1.5))
    y, h = np.arange(len(drop)), 0.38
    for k, m in enumerate(models):
        ax.barh(
            y + (k - (len(models) - 1) / 2) * h,
            drop[m],
            h,
            color=COLORS[m],
            label=MODEL_LABELS[m],
            edgecolor=SURFACE,
            linewidth=2,
        )
    ax.set_yticks(y, drop.index)
    ax.invert_yaxis()
    ax.set_xlabel("Pérdida de F1 (F1 limpio − F1 con ruido)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title(
        f"Intenciones más afectadas: {PERTURBATION_LABELS[perturbation].lower()}, "
        f"intensidad {intensity:g}",
        loc="left",
    )
    fig.tight_layout()
    return _save(fig, "5_intenciones_afectadas")

def plot_all(metrics, predictions):
    return [
        plot_clean(metrics),
        plot_degradation(metrics),
        plot_drop_heatmap(metrics),
        plot_word_vs_subword(metrics),
        plot_intent_drop(predictions),
    ]
