import math

import pandas as pd
import pytest

from tp.evaluation import (
    accuracy_drops,
    average_seeds,
    compare_drops,
    compute_metrics,
    evaluate,
    paired_bootstrap,
    per_class_f1,
)

# Calculado a mano. "c" se predice pero no está en y_true.
#   a: P=1/1, R=1/2 → F1=2/3      b: P=1/2, R=1/2 → F1=1/2
# F1 macro sobre {a, b} = 7/12. La unión {a, b, c} daría (2/3 + 1/2 + 0)/3 = 7/18.
Y_TRUE = ["a", "a", "b", "b"]
Y_PRED = ["a", "b", "b", "c"]


def test_f1_macro_usa_solo_las_clases_de_y_true():
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["f1_macro"] == pytest.approx(7 / 12)


def test_per_class_f1_consistente_con_f1_macro():
    per_class = per_class_f1(Y_TRUE, Y_PRED)
    assert per_class == pytest.approx({"a": 2 / 3, "b": 1 / 2})
    mean = sum(per_class.values()) / len(per_class)
    assert mean == pytest.approx(compute_metrics(Y_TRUE, Y_PRED)["f1_macro"])


class LookupModel:
    """Modelo falso: predice según una tabla texto → etiqueta."""

    def __init__(self, table):
        self.table = table
        self.n_predicted = 0

    def predict(self, texts):
        self.n_predicted += len(texts)
        return [self.table[t] for t in texts]


def test_evaluate_mide_el_subconjunto_alterado():
    sets = {
        ("clean", 0.0, 42): ["uno", "dos", "tres", "cuatro"],
        ("ruido", 0.5, 42): ["uno", "dso", "tres", "cuatr"],
        ("nulo", 1.0, 42): ["uno", "dos", "tres", "cuatro"],
    }
    labels = ["x", "y", "x", "y"]
    model = LookupModel(
        {"uno": "x", "dos": "y", "tres": "y", "cuatro": "y", "dso": "x", "cuatr": "y"}
    )
    metrics, preds = evaluate("m", model, sets, labels, "validation")
    rows = {r["perturbation"]: r for r in metrics.to_dict("records")}

    # Alteradas: "dos" (falla con ruido, acierta limpia) y "cuatro" (acierta en ambas).
    ruido = rows["ruido"]
    assert ruido["changed_frac"] == pytest.approx(0.5)
    assert ruido["accuracy_changed"] == pytest.approx(0.5)
    assert ruido["accuracy_changed_clean"] == pytest.approx(1.0)
    assert ruido["accuracy"] == pytest.approx(0.5)

    nulo = rows["nulo"]
    assert nulo["changed_frac"] == 0.0
    assert math.isnan(nulo["accuracy_changed"])
    assert nulo["accuracy"] == rows["clean"]["accuracy"]

    # 12 textos en total, pero solo 6 distintos: cada uno se predice una vez.
    assert model.n_predicted == 6
    assert preds["pred"].tolist() == [model.table[t] for t in preds["text"]]

    assert preds.loc[preds["perturbation"] == "ruido", "changed"].tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_average_seeds_promedia_y_mide_desvio():
    metrics = pd.DataFrame(
        {
            "model": "m",
            "split": "validation",
            "perturbation": ["clean", "typos", "typos"],
            "intensity": [0.0, 0.5, 0.5],
            "seed": [42, 42, 43],
            "accuracy": [0.9, 0.6, 0.8],
        }
    )
    rows = average_seeds(metrics).set_index("perturbation").to_dict("index")
    assert rows["typos"]["accuracy"] == pytest.approx(0.7)
    assert rows["typos"]["accuracy_std"] == pytest.approx(0.1414, abs=1e-4)
    assert rows["typos"]["n_seeds"] == 2
    assert rows["clean"]["n_seeds"] == 1
    assert math.isnan(rows["clean"]["accuracy_std"])


def test_paired_bootstrap():
    mean, low, high = paired_bootstrap([0.3] * 50)
    assert (mean, low, high) == pytest.approx((0.3, 0.3, 0.3))
    mean, low, high = paired_bootstrap([0, 1] * 50)
    assert mean == pytest.approx(0.5)
    assert low < 0.5 < high
    assert paired_bootstrap([0, 1] * 50) == paired_bootstrap([0, 1] * 50)


def _predictions(model, clean_ok, noisy_ok_by_seed):
    """Predicciones falsas: 1 = acierta. Etiqueta real "x", error = "y"."""
    rows = [
        {
            "model": model,
            "split": "validation",
            "perturbation": "clean",
            "intensity": 0.0,
            "seed": 42,
            "idx": i,
            "label": "x",
            "pred": "x" if ok else "y",
        }
        for i, ok in enumerate(clean_ok)
    ]
    for seed, oks in noisy_ok_by_seed.items():
        rows += [
            {
                "model": model,
                "split": "validation",
                "perturbation": "typos",
                "intensity": 0.5,
                "seed": seed,
                "idx": i,
                "label": "x",
                "pred": "x" if ok else "y",
            }
            for i, ok in enumerate(oks)
        ]
    return pd.DataFrame(rows)


def test_accuracy_drops_promedia_semillas_por_frase():
    # Frase 0: acierta limpia y en 1 de 2 semillas → caída 0.5.
    # Frase 1: acierta siempre → 0.  Frase 2: falla limpia, acierta con ruido → -1.
    preds = _predictions("m", [1, 1, 0], {42: [1, 1, 1], 43: [0, 1, 1]})
    row = accuracy_drops(preds, n_boot=200).iloc[0]
    assert row["drop"] == pytest.approx((0.5 + 0 - 1) / 3)
    assert row["n"] == 3
    assert row["ci_low"] <= row["drop"] <= row["ci_high"]


def test_compare_drops_signo():
    fragil = _predictions("fragil", [1, 1, 1, 1], {42: [0, 0, 1, 1]})
    robusto = _predictions("robusto", [1, 1, 1, 1], {42: [1, 1, 1, 1]})
    preds = pd.concat([fragil, robusto], ignore_index=True)
    row = compare_drops(preds, "fragil", "robusto", n_boot=200).iloc[0]
    assert row["drop_diff"] == pytest.approx(0.5)
