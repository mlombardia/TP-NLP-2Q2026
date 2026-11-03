import random

from tp.models import MODELS, FastTextModel, cache_path

VOCAB = {
    "alarm": "alarma despertar hora mañana",
    "music": "música canción poner tema",
    "weather": "tiempo lluvia clima frío",
}


def _toy_data(n=300):
    rng = random.Random(0)
    texts, labels = [], []
    for _ in range(n):
        label = rng.choice(list(VOCAB))
        words = VOCAB[label].split() + ["de", "la", "el"]
        texts.append(" ".join(rng.choices(words, k=5)))
        labels.append(label)
    return texts, labels


def test_fasttext_reproducible_en_el_mismo_proceso():
    # Entrenando en el proceso actual, el tercer modelo difiere de los dos primeros
    # aunque la semilla sea la misma (ver FastTextModel.fit).
    texts, labels = _toy_data()
    probe = ["alarma tema", "lluvia canción de", "hora clima"]
    probs = []
    for _ in range(3):
        model = FastTextModel(minn=3, maxn=6)
        model.params |= {"dim": 20, "epoch": 20}  # más rápido
        model.fit(texts, labels)
        probs.append([p.tolist() for p in model._trained().predict(probe, k=-1)[1]])
    assert probs[0] == probs[1] == probs[2]


def test_cache_path_depende_de_configuracion_y_semilla():
    paths = {name: cache_path(name, MODELS[name]()) for name in MODELS}
    # Estable: la misma configuración da la misma ruta.
    assert paths == {name: cache_path(name, MODELS[name]()) for name in MODELS}
    assert len(set(paths.values())) == len(MODELS)
    assert cache_path("fasttext", MODELS["fasttext"](43)) != paths["fasttext"]

    model = MODELS["fasttext"]()
    model.params["epoch"] = 50
    assert cache_path("fasttext", model) != paths["fasttext"]
