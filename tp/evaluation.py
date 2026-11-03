from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from tp.data import SEED

# scikit-learn no declara tipos y pyright deduce `str` para `average` y
# `zero_division` a partir de sus valores por defecto; los `pyright: ignore` de abajo
# son falsos positivos.


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy y F1 macro.

    El F1 macro promedia sobre las intenciones presentes en `y_true`, que son fijas para
    un split. Por defecto, scikit-learn usa la unión de `y_true` e `y_pred`: una
    intención que el modelo predice pero que no está en el split (p. ej.
    `audio_volume_other` en validation) sumaría un F1 de 0 y el promedio dependería del
    modelo y del nivel de ruido.
    """
    labels = sorted(set(y_true))
    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": float(f1)}

def per_class_f1(y_true, y_pred) -> dict[str, float]:
    labels = sorted(set(y_true))
    scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,  # pyright: ignore[reportArgumentType]
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return dict(zip(labels, np.asarray(scores).tolist(), strict=True))

def changed_metrics(original, texts, labels, y_pred, y_pred_clean) -> dict[str, float]:
    """Cuánto cambió el conjunto y cómo rinde el modelo sobre las frases alteradas.

    Una perturbación puede no tocar una frase (p. ej. sin tildes en una frase que no
    tiene). Sobre el conjunto completo, esas frases diluyen la caída; por eso también se
    mide la accuracy solo sobre las frases alteradas, con ruido y en su versión limpia.
    Sobre ese subconjunto se usa accuracy y no F1 macro: la cantidad de ejemplos por
    intención queda muy desigual y algunas intenciones pueden no aparecer.
    """
    changed = np.array([o != t for o, t in zip(original, texts, strict=True)])
    y_true, y_pred, y_pred_clean = map(np.asarray, (labels, y_pred, y_pred_clean))
    out = {"changed_frac": float(changed.mean())}
    if not changed.any():
        return out | {"accuracy_changed": np.nan, "accuracy_changed_clean": np.nan}
    y = y_true[changed]
    return out | {
        "accuracy_changed": float((y_pred[changed] == y).mean()),
        "accuracy_changed_clean": float((y_pred_clean[changed] == y).mean()),
    }

def evaluate(
    model_name, model, sets, labels, split
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice cada conjunto {(perturbación, intensidad, semilla): textos}.

    `sets` tiene que incluir un conjunto "clean". Devuelve (métricas, predicciones).
    """
    names = list(sets)
    all_texts = [t for key in names for t in sets[key]]
    # Muchas frases se repiten entre conjuntos (una perturbación no siempre las toca):
    # en validation solo ~21 % de los textos son distintos. Se predice cada texto una
    # sola vez; como los modelos predicen siempre lo mismo para el mismo texto, el
    # resultado no cambia y la predicción tarda unas cinco veces menos.
    unique = list(dict.fromkeys(all_texts))
    pred_of = dict(zip(unique, model.predict(unique), strict=True))
    all_preds = [pred_of[t] for t in all_texts]

    n = len(labels)
    clean_key = next(key for key in names if key[0] == "clean")
    original = sets[clean_key]
    k_clean = names.index(clean_key)
    y_pred_clean = all_preds[k_clean * n : (k_clean + 1) * n]
    metrics, preds = [], []
    for k, (pert, p, seed) in enumerate(names):
        y_pred = all_preds[k * n : (k + 1) * n]
        texts = sets[(pert, p, seed)]
        info = {
            "model": model_name,
            "split": split,
            "perturbation": pert,
            "intensity": p,
            "seed": seed,
        }
        metrics.append(
            info
            | compute_metrics(labels, y_pred)
            | changed_metrics(original, texts, labels, y_pred, y_pred_clean)
        )
        preds.append(
            pd.DataFrame(
                info
                | {
                    "idx": range(n),
                    "original": original,
                    "text": texts,
                    "changed": [o != t for o, t in zip(original, texts, strict=True)],
                    "label": labels,
                    "pred": y_pred,
                }
            )
        )
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True)

METRIC_COLUMNS = [
    "accuracy",
    "f1_macro",
    "changed_frac",
    "accuracy_changed",
    "accuracy_changed_clean",
]

def average_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Una fila por modelo × split × perturbación × intensidad, promediando semillas.

    Promedia tanto las semillas de ruido como las de entrenamiento (`train_seed`).
    Agrega `<métrica>_std` (desvío entre todas esas corridas, NaN si hay una sola) y
    `n_seeds` (cantidad de corridas promediadas).
    """
    keys = ["model", "split", "perturbation", "intensity"]
    cols = [c for c in METRIC_COLUMNS if c in metrics.columns]
    groups = metrics.groupby(keys, sort=False)[cols]
    out = groups.mean().join(groups.std().add_suffix("_std"))
    return out.join(groups.size().rename("n_seeds")).reset_index()

def paired_bootstrap(
    values, n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float, float]:
    """Media de `values` (un valor por frase) e IC percentil remuestreando frases."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)

def _sentence_drops(predictions: pd.DataFrame) -> pd.Series:
    """Caída de accuracy de cada frase: acierto limpio (0 o 1) menos la fracción de
    semillas en que acierta con ruido. Índice: modelo, split, perturbación,
    intensidad, idx."""
    sentence = ["model", "split", "idx"]
    condition = ["perturbation", "intensity"]
    df = predictions.assign(
        correct=(predictions["pred"] == predictions["label"]).astype(float)
    )
    # Promediar sobre semillas: fracción de semillas en que la frase se acierta.
    acc = df.groupby(sentence + condition, as_index=False)["correct"].mean()
    clean = acc[acc["perturbation"] == "clean"][[*sentence, "correct"]]
    noisy = acc[acc["perturbation"] != "clean"]
    both = noisy.merge(clean, on=sentence, suffixes=("", "_clean"))
    both["drop"] = both["correct_clean"] - both["correct"]
    return both.set_index(["model", "split", *condition, "idx"])["drop"]

def _bootstrap_by_condition(drops: pd.Series, levels, n_boot) -> pd.DataFrame:
    rows = []
    for key, d in drops.groupby(level=levels):
        mean, low, high = paired_bootstrap(d.to_numpy(), n_boot)
        rows.append(
            dict(zip(levels, cast(tuple, key), strict=True))
            | {"drop": mean, "ci_low": low, "ci_high": high, "n": len(d)}
        )
    return pd.DataFrame(rows)

def accuracy_drops(predictions: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Caída de accuracy (limpio − con ruido) por condición, con IC 95 %.

    Bootstrap pareado: se remuestrean frases, y cada frase aporta su versión limpia y
    sus versiones con ruido juntas. El IC refleja la variabilidad por la muestra de
    frases; la variabilidad por semilla queda promediada dentro de cada frase.
    Se usa accuracy porque se descompone por frase; el F1 macro no.
    """
    levels = ["model", "split", "perturbation", "intensity"]
    return _bootstrap_by_condition(_sentence_drops(predictions), levels, n_boot)

def compare_drops(
    predictions: pd.DataFrame, model_a: str, model_b: str, n_boot: int = 1000
) -> pd.DataFrame:
    """Diferencia de caídas de accuracy (a − b) por condición, con IC 95 %.

    Positiva: `model_a` cae más que `model_b`. Si el IC no incluye 0, la diferencia
    no se explica por la muestra de frases. Pareado: las mismas frases para ambos.
    """
    drops = _sentence_drops(predictions)
    diff = drops.xs(model_a, level="model") - drops.xs(model_b, level="model")
    diff = diff.dropna()
    levels = ["split", "perturbation", "intensity"]
    out = _bootstrap_by_condition(diff, levels, n_boot)
    return out.rename(columns={"drop": "drop_diff"})
