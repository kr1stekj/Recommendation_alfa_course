"""Единый интерфейс векторизации: локальная нейросеть либо лексический TF-IDF."""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from .missing import is_missing

MODEL = "BAAI/bge-m3"
LEGACY_MODEL = "cointegrated/rubert-tiny2"
MODEL_SPECS = {
    MODEL: {'revision': '5617a9f61b028005a4858fdac845db406aefb181', 'limit': 8192, 'dimension': 1024, 'batch_size': 4},
    LEGACY_MODEL: {'revision': 'e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae', 'limit': 512, 'dimension': 312, 'batch_size': 16},
}
STOP = set("и в во на с со для по к от до из а но или что как это не о об у при за есть вы мы он она они его ее их который чтобы все так то уже без ваш наши можно будет чем где также свой этой если ещё еще".split())


def normalize(x):
    """Приводит каждый вектор к длине 1; нулевой вектор остаётся нулевым.

    После этого скалярное произведение ненулевых векторов равно косинусной близости.
    """
    x = np.asarray(x, dtype=np.float32)
    # Пропуски в признаках не должны распространяться на все оценки косинуса.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lengths = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
    return (x / np.maximum(lengths, 1e-12)).astype(np.float32)


def tokens(text):
    """Выделяет слова для TF-IDF и удаляет частые служебные слова без полезной темы."""
    text = "" if is_missing(text) else str(text)
    return [w for w in re.findall(r"[а-яёa-z0-9]{2,}", text.casefold().replace("ё", "е")) if w not in STOP]


class Encoder:
    """Сохраняет одинаковый способ представления каталога и будущих запросов."""
    def __init__(self, config=None, device=None):
        self.config = dict(config) if config is not None else {"backend": "semantic", "model": MODEL}
        self.model = None
        self.local_only = False
        # Устройство — параметр исполнения, не пространство векторов. Один индекс
        # можно построить на Apple GPU, а использовать на CPU в контейнере жюри.
        self.device = device if device is not None else os.getenv('EMBEDDING_DEVICE', 'cpu')
        if self.config.get('backend') not in {'semantic', 'tfidf'}:
            raise ValueError('backend must be semantic or tfidf')
        if self.config['backend'] == 'semantic':
            if self.device not in ('cpu', 'mps'):
                raise ValueError('EMBEDDING_DEVICE must be cpu or mps')
            name = self.config.setdefault('model', MODEL)
            if name not in MODEL_SPECS:
                raise ValueError('Для модели нужен отдельный адаптер. Поддерживаются: ' + ', '.join(MODEL_SPECS))
            spec = MODEL_SPECS[name]
            self.config.setdefault('revision', spec['revision'])
            self.config.setdefault('pooling', 'cls')
            self.config.setdefault('max_length', 512)
            self.config.setdefault('batch_size', spec['batch_size'])
            if self.config['pooling'] != 'cls':
                raise ValueError('BGE-M3 и rubert-tiny2 используют CLS pooling')
            for field, limit in (('max_length', spec['limit']), ('batch_size', 64)):
                value = self.config[field]
                minimum = 16 if field == 'max_length' else 1
                if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= limit:
                    raise ValueError(field + ' is outside supported bounds')

    def fit(self, texts, progress=None):
        """Для TF-IDF строит словарь и веса слов; нейросеть дообучать не требуется."""
        if self.config["backend"] == "tfidf":
            # DF — в скольких документах встретилось слово, а не число повторов.
            # IDF уменьшает вклад слов, которые встречаются почти везде.
            df = Counter(t for text in texts for t in set(tokens(text)))
            minimum = self.config.get('min_df', 2)
            maximum = self.config.get('max_features', 8000)
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError('min_df must be a positive integer')
            if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
                raise ValueError('max_features must be a positive integer or None')
            words = [w for w, n in sorted(df.items(), key=lambda p: (-p[1], p[0])) if n >= minimum]
            if maximum is not None:
                words = words[:maximum]
            self.config.update(vocabulary=words, idf=[math.log((1 + len(texts)) / (1 + df[w])) + 1 for w in words])
        return self.encode(texts, progress=progress)

    def encode(self, texts, progress=None):
        """Возвращает матрицу: строка — текст, столбцы — признаки; ненулевые строки нормированы."""
        texts = ["" if is_missing(text) else str(text) for text in texts]
        if self.config["backend"] == "tfidf":
            vocab = {w: i for i, w in enumerate(self.config["vocabulary"])}
            x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
            for i, text in enumerate(texts):
                for word, n in Counter(tokens(text)).items():
                    if word in vocab: x[i, vocab[word]] = 1 + math.log(n)
            return normalize(x * np.asarray(self.config["idf"]))
        # Загружаем модель лениво: команды очистки профилей не требуют нейросети.
        if self.model is None:
            os.environ.setdefault("HF_HOME", str(Path("models/huggingface").resolve()))
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            torch.set_num_threads(2)
            if self.device == 'mps' and not torch.backends.mps.is_available():
                raise ValueError('Apple MPS недоступен; выберите EMBEDDING_DEVICE=cpu')
            name = self.config.get("model", MODEL)
            cache = str(Path(os.environ.get("HF_HOME", "models/huggingface")) / "hub")
            revision = self.config.get("revision")
            # Для сохранённой версии используем кеш: запросы работают без сети.
            self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache, revision=revision,
                local_files_only=self.local_only, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(name, cache_dir=cache, revision=revision,
                local_files_only=self.local_only, trust_remote_code=False,
                # У официальной BGE revision только pytorch_model.bin. PyTorch 2.6
                # загружает лишь tensor weights; accelerate не держит две копии модели.
                use_safetensors=name != MODEL, weights_only=True, low_cpu_mem_usage=True).eval()
            if self.model.config.hidden_size != MODEL_SPECS[name]['dimension']:
                raise ValueError('Неожиданная размерность выбранной модели')
            if self.device != 'cpu':
                self.model.to(self.device)
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        batches = []
        batch_size = self.config['batch_size']
        for offset in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[offset:offset + batch_size], padding=True, truncation=True,
                                   max_length=self.config['max_length'], return_tensors="pt")
            if self.device != 'cpu':
                batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                # Официальные конфигурации обеих поддерживаемых моделей задают CLS.
                v = self.model(**batch).last_hidden_state[:, 0, :].cpu().numpy()
            batches.append(normalize(v))
            if progress:
                progress(min(offset + batch_size, len(texts)), len(texts))
        return np.vstack(batches)

    def save(self, path):
        Path(path).write_text(json.dumps(self.config, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        encoder = cls(json.loads(Path(path).read_text(encoding="utf-8")))
        # Пользовательский запрос не должен скачивать/обновлять веса. Сеть — при сборке.
        encoder.local_only = True
        return encoder
