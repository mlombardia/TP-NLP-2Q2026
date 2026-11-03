import hashlib
import json
import multiprocessing
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tp.data import DATA_REVISION, MODELS_DIR, SEED

NOT_TRAINED = "El modelo no está entrenado: llamar a fit() o load() primero."


class TfidfModel:
    def __init__(self, analyzer: str, ngram_range: tuple[int, int], seed: int = SEED):
        self.config = {
            "model": "tfidf_logreg",
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
            "sublinear_tf": True,
            "C": 10,
            "max_iter": 2000,
            "seed": seed,
        }
        c = self.config
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=analyzer,
                        ngram_range=ngram_range,
                        sublinear_tf=c["sublinear_tf"],
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c["C"], max_iter=c["max_iter"], random_state=seed
                    ),
                ),
            ]
        )

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return list(self.pipeline.predict(texts))

    def save(self, path: Path):
        joblib.dump(self.pipeline, path.with_suffix(".joblib"))

    def load(self, path: Path):
        self.pipeline = joblib.load(path.with_suffix(".joblib"))
        return self

    def exists(self, path: Path) -> bool:
        return path.with_suffix(".joblib").exists()


def _train_fasttext(data_path: str, model_path: str, params: dict):
    import fasttext

    model = fasttext.train_supervised(data_path, verbose=0, thread=1, **params)
    model.save_model(model_path)


class FastTextModel:
    def __init__(self, minn: int, maxn: int, seed: int = SEED):
        self.params = dict(
            minn=minn, maxn=maxn, wordNgrams=2, dim=100, epoch=100, lr=0.5, seed=seed
        )
        self.model = None

    @property
    def config(self) -> dict:
        return {"model": "fasttext", **self.params}

    def fit(self, texts, labels):
        import fasttext

        # FastText no es determinista dentro de un proceso: entrenar dos veces seguidas
        # con los mismos datos y la misma semilla puede dar modelos distintos. Entrenar
        # en un proceso nuevo hace que el resultado no dependa de qué se entrenó antes.
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "train.txt", Path(tmp) / "model.bin"
            data.write_text(
                "".join(
                    f"__label__{y} {x}\n" for x, y in zip(texts, labels, strict=True)
                )
            )
            proc = multiprocessing.get_context("spawn").Process(
                target=_train_fasttext, args=(str(data), str(out), self.params)
            )
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"FastText terminó con código {proc.exitcode}")
            self.model = fasttext.load_model(str(out))
        return self

    def _trained(self):
        if self.model is None:
            raise RuntimeError(NOT_TRAINED)
        return self.model

    def predict(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        labels, _ = self._trained().predict(texts)
        return [lab[0].removeprefix("__label__") for lab in labels]

    def save(self, path: Path):
        self._trained().save_model(str(path.with_suffix(".bin")))

    def load(self, path: Path):
        import fasttext

        self.model = fasttext.load_model(str(path.with_suffix(".bin")))
        return self

    def exists(self, path: Path) -> bool:
        return path.with_suffix(".bin").exists()


class TransformerModel:
    def __init__(
        self,
        name: str = "dccuchile/bert-base-spanish-wwm-cased",
        epochs: int = 3,
        seed: int = SEED,
    ):
        self.name = name
        self.epochs = epochs
        self.seed = seed
        self.model = self.tokenizer = None

    @property
    def config(self) -> dict:
        return {
            "model": "transformer",
            "name": self.name,
            "epochs": self.epochs,
            "seed": self.seed,
        }

    def fit(self, texts, labels):
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        # La capa de clasificación se inicializa al cargar el modelo, antes de que
        # Trainer fije su semilla: sin esto, todas las semillas compartirían la misma.
        set_seed(self.seed)
        classes = sorted(set(labels))
        label2id = {c: i for i, c in enumerate(classes)}
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.name,
            num_labels=len(classes),
            id2label=dict(enumerate(classes)),
            label2id=label2id,
        )
        ds = Dataset.from_dict({"text": texts, "label": [label2id[y] for y in labels]})
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=64),
            batched=True,
        )

        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=32,
            learning_rate=5e-5,
            warmup_steps=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            save_strategy="no",
            logging_steps=100,
            report_to=[],
            seed=self.seed,
        )
        Trainer(
            model=model,
            args=args,
            train_dataset=ds.remove_columns("text"),
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
        ).train()
        self.model, self.tokenizer = model, tokenizer
        return self

    def _trained(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(NOT_TRAINED)
        return self.model, self.tokenizer

    def predict(self, texts, batch_size: int = 128):
        import torch

        model, tokenizer = self._trained()
        # Mismo orden que usa Trainer al entrenar: CUDA, MPS (Apple) y después CPU.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model.to(device).eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(
                    texts[i : i + batch_size],
                    truncation=True,
                    max_length=64,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
        id2label = model.config.id2label
        return [id2label[int(i)] for i in np.concatenate(preds)]

    def save(self, path: Path):
        model, tokenizer = self._trained()
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    def load(self, path: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        return self

    def exists(self, path: Path) -> bool:
        return (path / "config.json").exists()


MODELS = {
    "tfidf_word": lambda seed=SEED: TfidfModel("word", (1, 2), seed),
    "tfidf_char": lambda seed=SEED: TfidfModel("char_wb", (2, 5), seed),
    "fasttext_word": lambda seed=SEED: FastTextModel(minn=0, maxn=0, seed=seed),
    "fasttext": lambda seed=SEED: FastTextModel(minn=3, maxn=6, seed=seed),
    "beto": lambda seed=SEED: TransformerModel(seed=seed),
}

# Modelos cuyo entrenamiento depende de la semilla y se entrenan con varias.
# TF-IDF + regresión logística (solver lbfgs) da el mismo modelo con cualquier semilla.
STOCHASTIC = {"fasttext_word", "fasttext", "beto"}


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(name: str, model) -> Path:
    """Ruta del modelo guardado: `models/<nombre>-<hash>`.

    El hash cubre la configuración del modelo (incluida la semilla) y la revisión de
    los datos. Si algo cambia, la ruta cambia y el modelo se reentrena, en vez de
    cargarse en silencio uno entrenado con otra configuración.
    """
    return MODELS_DIR / f"{name}-{config_hash(model.config | {'data': DATA_REVISION})}"


# Modelos que ven la palabra completa vs. los que ven fragmentos de palabra.
SUBWORD = {
    "tfidf_word": False,
    "fasttext_word": False,
    "tfidf_char": True,
    "fasttext": True,
    "beto": True,
}
