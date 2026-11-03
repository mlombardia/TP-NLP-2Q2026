import itertools
import random
import re
import string
import unicodedata

from tp.data import SEED

INTENSITIES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5

KEYBOARD_ROWS = ["qwertyuiop", "asdfghjklñ", "zxcvbnm"]

# Criterio de los diccionarios: cada reemplazo tiene que conservar el significado en
# los contextos en que la palabra aparece en train (revisado a mano sobre train, no
# sobre validation ni test). Se descartaron, por ejemplo, "vale"→"dale" (en MASSIVE es
# casi siempre el verbo valer: "cuánto vale") y "poner"→"mandar" ("poner una alarma").

# Para tfidf_word, casi ninguna abreviatura está en el vocabulario de train (salvo
# "cumple"): abreviar equivale a borrar la palabra.
ABBREVIATIONS = {
    "por favor": "xfa",
    "por qué": "xq",
    "porque": "xq",
    "que": "q",
    "qué": "q",
    "por": "x",
    "para": "pa",
    "también": "tmb",
    "de": "d",
    "mensaje": "msj",
    "mensajes": "msjs",
    "mañana": "mñn",
    "más": "+",
    "mas": "+",  # en train es siempre "más" sin tilde, no la conjunción
    "cuando": "cdo",
    "cuándo": "cdo",
    "información": "info",
    "cumpleaños": "cumple",
    "número": "nro",
    "teléfono": "tel",
    "minutos": "min",
    "horas": "hs",
    # Sin "uno": en train es casi siempre pronombre ("cada uno de los eventos").
    "dos": "2",
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "siete": "7",
    "ocho": "8",
    "nueve": "9",
    "diez": "10",
    "doce": "12",
}

# Verbos en imperativo o segunda persona → forma rioplatense con voseo. A veces cambia
# también el verbo (enciende → prendé). Las formas con voseo agregan tildes.
VOSEO = {
    "pon": "poné",
    "ponme": "poneme",
    "dime": "decime",
    "dame": "pasame",
    "apaga": "apagá",
    "enciende": "prendé",
    "sube": "subí",
    "baja": "bajá",  # en train aparece siempre como verbo ("baja el volumen")
    "busca": "buscá",
    "abre": "abrí",
    "borra": "borrá",
    "añade": "agregá",
    "reproduce": "poné",  # no "pasá": en música también significa saltear
    "envía": "mandá",
    "recuérdame": "acordame",
    "puedes": "podés",
}

# Resto del vocabulario → sinónimo regional o coloquial (sesgo rioplatense).
COLLOQUIAL = {
    "correo electrónico": "mail",
    "correos electrónicos": "mails",
    "correo": "mail",
    "correos": "mails",
    "reproducir": "poner",
    "enviar": "mandar",
    "necesito": "me hace falta",
    "canción": "tema",
    "canciones": "temas",
    "billete": "pasaje",
    "taxi": "remis",
    "reunión": "junta",
    "comida": "morfi",
    "noticias": "novedades",
    "coche": "auto",
    "ordenador": "compu",
    "trabajo": "laburo",
    "dinero": "guita",
}


def _neighbors() -> dict[str, str]:
    neigh = {}
    for r, row in enumerate(KEYBOARD_ROWS):
        for c, ch in enumerate(row):
            near = [row[i] for i in (c - 1, c + 1) if 0 <= i < len(row)]
            for other in (r - 1, r + 1):
                if 0 <= other < len(KEYBOARD_ROWS):
                    near += [
                        KEYBOARD_ROWS[other][i]
                        for i in (c - 1, c, c + 1)
                        if 0 <= i < len(KEYBOARD_ROWS[other])
                    ]
            neigh[ch] = "".join(near)
    return neigh


NEIGHBORS = _neighbors()


def _dict_regex(mapping: dict[str, str]) -> re.Pattern:
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.IGNORECASE)


ABBREV_RE = _dict_regex(ABBREVIATIONS)
VOSEO_RE = _dict_regex(VOSEO)
COLLOQUIAL_RE = _dict_regex(COLLOQUIAL)


def _unit_rng(key: str, i: int) -> random.Random:
    """Generador propio de la unidad i (palabra, carácter u ocurrencia) de una frase.

    `key` identifica la frase (semilla, perturbación y texto) pero no incluye `p` ni
    depende de las demás frases. Así cada unidad tira siempre el mismo dado: el
    resultado no depende del orden de los textos, y lo que se altera con p=0.1
    también se altera, de la misma forma, con p=0.25 (niveles anidados).
    Una semilla str se hashea con SHA-512, así que no depende de PYTHONHASHSEED.
    """
    return random.Random(f"{key}-{i}")


def _replace_from_dict(text, p, key, pattern, mapping):
    k = itertools.count()

    def repl(m):
        rng = _unit_rng(key, next(k))
        return mapping[m.group(0).lower()] if rng.random() < p else m.group(0)

    return pattern.sub(repl, text)


def _strip_accent(ch: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn"
    )


def remove_accents(text: str, p: float, key: str) -> str:
    out = []
    for i, ch in enumerate(text):
        plain = _strip_accent(ch)
        if ch.lower() != "ñ" and plain != ch and _unit_rng(key, i).random() < p:
            ch = plain
        out.append(ch)
    return "".join(out)


def lowercase_no_punct(text: str, p: float, key: str) -> str:
    if _unit_rng(key, 0).random() >= p:
        return text
    text = text.lower().translate(str.maketrans("", "", string.punctuation + "¿¡"))
    return " ".join(text.split())


def _key_neighbors(ch: str) -> str:
    # Las vocales con tilde están en la misma tecla que la vocal sin tilde.
    ch = ch.lower()
    return NEIGHBORS.get(ch) or NEIGHBORS.get(_strip_accent(ch), "")


TYPO_OPS = ["neighbor", "delete", "duplicate", "swap"]


def _typo(word: str, rng: random.Random) -> str:
    """Un error de tipeo. Primero se elige la operación (equiprobables) y después una
    posición donde esa operación cambie la palabra."""
    op = rng.choice(TYPO_OPS)
    if op == "neighbor":
        positions = [i for i, ch in enumerate(word) if _key_neighbors(ch)]
    elif op == "swap":
        positions = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
    else:
        positions = list(range(len(word)))
    if not positions:  # p. ej. "aaa" no admite swap
        op, positions = "delete", list(range(len(word)))
    i = rng.choice(positions)
    if op == "neighbor":
        return word[:i] + rng.choice(_key_neighbors(word[i])) + word[i + 1 :]
    if op == "duplicate":
        return word[:i] + word[i] + word[i:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word[:i] + word[i + 1 :]


def typos(text: str, p: float, key: str) -> str:
    out = []
    for i, w in enumerate(text.split()):
        rng = _unit_rng(key, i)
        if len(w) >= 3 and w.isalpha() and rng.random() < p:
            w = _typo(w, rng)
        out.append(w)
    return " ".join(out)


def abbreviations(text: str, p: float, key: str) -> str:
    return _replace_from_dict(text, p, key, ABBREV_RE, ABBREVIATIONS)


def voseo(text: str, p: float, key: str) -> str:
    return _replace_from_dict(text, p, key, VOSEO_RE, VOSEO)


def colloquial(text: str, p: float, key: str) -> str:
    return _replace_from_dict(text, p, key, COLLOQUIAL_RE, COLLOQUIAL)


def mixed(text: str, p: float, key: str) -> str:
    # Los diccionarios buscan palabras bien escritas: se aplican antes que el ruido.
    # Cada etapa usa su propia clave. El anidamiento es aproximado: una etapa que
    # cambia la frase corre las posiciones que ven las etapas siguientes.
    dictionaries = (colloquial, voseo, abbreviations)
    for f in (*dictionaries, lowercase_no_punct, remove_accents, typos):
        text = f(text, p, f"{key}-{f.__name__}")
    return text


PERTURBATIONS = {
    "no_accents": remove_accents,
    "lower_no_punct": lowercase_no_punct,
    "typos": typos,
    "abbreviations": abbreviations,
    "voseo": voseo,
    "colloquial": colloquial,
    "mixed": mixed,
}


def perturb(texts: list[str], name: str, p: float, seed: int = SEED) -> list[str]:
    # Frases idénticas reciben la misma perturbación (la clave es el texto).
    return [PERTURBATIONS[name](t, p, f"{seed}-{name}-{t}") for t in texts]


def make_perturbed_sets(
    texts: list[str], n_seeds: int = N_SEEDS
) -> dict[tuple[str, float, int], list[str]]:
    """Conjuntos {(perturbación, intensidad, semilla): textos}.

    Cada perturbación se genera con `n_seeds` semillas (SEED, SEED+1, …) para medir
    cuánto depende el resultado de qué palabras tocó el azar. El conjunto limpio no
    depende de la semilla: va una sola vez, con SEED.
    """
    sets = {("clean", 0.0, SEED): list(texts)}
    for seed in range(SEED, SEED + n_seeds):
        for name in PERTURBATIONS:
            for p in INTENSITIES[1:]:
                sets[(name, p, seed)] = perturb(texts, name, p, seed)
    return sets
