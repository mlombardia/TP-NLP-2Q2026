import collections
import random
import string

import pytest

from tp.data import SEED
from tp.perturbations import (
    INTENSITIES,
    PERTURBATIONS,
    TYPO_OPS,
    _typo,
    make_perturbed_sets,
    perturb,
)

FRASES = [
    "despiértame a las nueve de la mañana el viernes",
    "por favor envía un correo electrónico a mi jefe",
    "qué tiempo hace mañana en córdoba",
    "pon la canción que más escucho",
    "dame la información del número de teléfono de mi trabajo",
    "¿Cuándo es la reunión? Avisame.",
]

# Perturbaciones que conservan la cantidad de palabras (para comparar palabra a
# palabra) y frases sin reemplazos de varias palabras como "por favor" → "xfa".
POR_PALABRA = ["typos", "abbreviations", "voseo", "colloquial"]
FRASES_POR_PALABRA = [FRASES[0], FRASES[2], FRASES[3], FRASES[4]]

NIVELES = list(zip(INTENSITIES[1:], INTENSITIES[2:], strict=False))
SEMILLAS = range(20)


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_deterministico(name):
    assert perturb(FRASES, name, 0.5) == perturb(FRASES, name, 0.5)


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_no_depende_del_orden_ni_del_lote(name):
    lote = perturb(FRASES, name, 0.5)
    assert perturb(FRASES[::-1], name, 0.5)[::-1] == lote
    assert [perturb([f], name, 0.5)[0] for f in FRASES] == lote


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_intensidad_cero_no_cambia(name):
    assert perturb(FRASES, name, 0.0) == FRASES


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_la_semilla_cambia_el_resultado(name):
    resultados = {tuple(perturb(FRASES, name, 0.5, seed=s)) for s in SEMILLAS}
    assert len(resultados) > 1


def _anidado(original, bajo, alto):
    """Toda unidad alterada en el nivel bajo está alterada igual en el alto."""
    return all(b == o or b == a for o, b, a in zip(original, bajo, alto, strict=True))


@pytest.mark.parametrize("name", POR_PALABRA)
def test_niveles_anidados_por_palabra(name):
    alteradas = 0
    for s in SEMILLAS:
        for p_bajo, p_alto in NIVELES:
            bajo = perturb(FRASES_POR_PALABRA, name, p_bajo, seed=s)
            alto = perturb(FRASES_POR_PALABRA, name, p_alto, seed=s)
            for o, b, a in zip(FRASES_POR_PALABRA, bajo, alto, strict=True):
                assert _anidado(o.split(), b.split(), a.split()), (o, b, a)
                alteradas += o != b
    assert alteradas > 0  # el test no es trivial


def test_niveles_anidados_tildes():
    for s in SEMILLAS:
        for p_bajo, p_alto in NIVELES:
            bajo = perturb(FRASES, "no_accents", p_bajo, seed=s)
            alto = perturb(FRASES, "no_accents", p_alto, seed=s)
            for o, b, a in zip(FRASES, bajo, alto, strict=True):
                assert _anidado(o, b, a), (o, b, a)


def test_sin_tildes_total_conserva_enie():
    out = perturb(FRASES, "no_accents", 1.0)
    assert not set("áéíóúüÁÉÍÓÚÜ") & set("".join(out))
    assert out[0].count("ñ") == FRASES[0].count("ñ")


def test_typos_solo_en_palabras_de_tres_letras_o_mas():
    for s in SEMILLAS:
        for o, t in zip(FRASES, perturb(FRASES, "typos", 1.0, seed=s), strict=True):
            for wo, wt in zip(o.split(), t.split(), strict=True):
                if len(wo) < 3 or not wo.isalpha():
                    assert wo == wt


def test_minusculas_sin_puntuacion_total():
    out = perturb(FRASES, "lower_no_punct", 1.0)
    assert out[-1] == "cuándo es la reunión avisame"
    assert all(t == t.lower() for t in out)
    assert not set(string.punctuation + "¿¡") & set("".join(out))


def test_make_perturbed_sets_limpio_una_vez_y_una_clave_por_semilla():
    sets = make_perturbed_sets(FRASES, n_seeds=3)
    assert [k for k in sets if k[0] == "clean"] == [("clean", 0.0, SEED)]
    assert len(sets) == 1 + 3 * len(PERTURBATIONS) * len(INTENSITIES[1:])
    assert sets[("typos", 0.5, SEED)] == perturb(FRASES, "typos", 0.5)
    assert sets[("typos", 0.5, SEED + 1)] == perturb(FRASES, "typos", 0.5, SEED + 1)


PALABRAS = ["alarma", "canción", "llamar", "mañana", "música", "aaa", "teléfono"]


def test_typo_siempre_cambia_la_palabra():
    rng = random.Random(0)
    for _ in range(2000):
        w = rng.choice(PALABRAS)
        assert _typo(w, rng) != w


def test_typo_operaciones_equiprobables():
    # Se clasifica el resultado: por largo (borrado/duplicado) o, con el mismo
    # largo, si son las mismas letras (intercambio) o no (tecla vecina).
    rng = random.Random(0)
    counts = collections.Counter()
    n = 8000
    for _ in range(n):
        w = rng.choice(PALABRAS[:5])
        t = _typo(w, rng)
        if len(t) < len(w):
            counts["delete"] += 1
        elif len(t) > len(w):
            counts["duplicate"] += 1
        else:
            counts["swap" if sorted(t) == sorted(w) else "neighbor"] += 1
    assert set(counts) == set(TYPO_OPS)
    for op in TYPO_OPS:
        assert counts[op] / n == pytest.approx(0.25, abs=0.03), counts


def test_typo_tecla_vecina_en_vocal_con_tilde():
    vecinas = {_typo("á", random.Random(s)) for s in range(200)} - {"", "áá"}
    assert vecinas <= set("qwszx")
