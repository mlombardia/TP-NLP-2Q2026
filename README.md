# Entender quien escribe mal

Clasificador de intenciones en español (MASSIVE es-ES, 60 intenciones) y estudio de su
robustez frente a ruido de escritura: sin tildes, errores de tipeo, abreviaturas,
minúsculas/sin puntuación y expresiones coloquiales.

**Grupo 1:** Benitez, Tomás · Lombardia, Maximiliano · Mónaco, Vianca · Vollert, Marcos

## Estructura

```
tp/
  data.py            carga de MASSIVE y rutas
  perturbations.py   generación de ruido
  models.py          TF-IDF (palabras y caracteres), FastText (con y sin subpalabras), BETO
  evaluation.py      métricas
  plots.py           gráficos
experimento.py       entrena y evalúa todos los modelos
resultados.ipynb     tablas, gráficos, análisis de errores y prueba de frases
```

## Instalación

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --extra cu130     # con GPU NVIDIA
uv sync --extra cpu       # sin GPU
```

## Uso

```bash
uv run python experimento.py                                # entrena y evalúa todo
uv run python experimento.py --modelos tfidf_word,fasttext  # solo algunos modelos
uv run python experimento.py --reentrenar                   # ignora los modelos guardados
uv run python experimento.py --modelos beto --semillas 2 --semillas-entrenamiento 1  # más rápido
uv run jupyter lab resultados.ipynb                         # ver resultados
uv run pytest                                               # tests
```

Los resultados quedan en `results/metrics.csv` y `results/predictions.parquet`, cada ejecución se
registra en `results/runs.jsonl` y los gráficos van a `results/figures/`.

## Dataset

[MASSIVE](https://github.com/alexa/massive) (Amazon Science), partición es-ES, desde
[`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent),
con la revisión fijada en `tp/data.py`. Licencia CC BY 4.0 (FitzGerald et al., 2022).
