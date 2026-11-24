"""Разовая подготовка индекса: дамп → каталог → векторы → локальные артефакты."""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path

import numpy as np

from .data import load_backup, make_catalog, historical_interactions
from .embeddings import Encoder, MODEL, normalize
from .database import Database
from .index import fingerprint, write_manifest, publish_generation


def build(backup, output="artifacts", backend="semantic", model=MODEL, as_of=None, max_length=512, batch_size=None, device='cpu'):
    """Собирает отдельное поколение и публикует его только после успешного импорта.

    Читатели продолжают использовать предыдущее поколение; старые файлы не меняются.
    """
    return publish_generation(output, lambda staging: _build_generation(
        backup, staging, backend, model, as_of, max_length, batch_size, device))


def finalize_catalog(catalog, history, suggestions):
    """Одинаковый порядок производных полей для сборки и переноса готовых векторов."""
    for item, suggestion in zip(catalog, suggestions):
        item['topic_source'] = 'catalog' if item['topics'] else 'unknown'
        if suggestion is not None:
            item['suggested_topic'] = suggestion
    popularity = Counter(row['item_id'] for row in history)
    for item in catalog:
        item['history_count'] = popularity[item['id']]


def _build_generation(backup, output, backend, model, as_of, max_length, batch_size, device):
    output = Path(output)
    database = Database()
    database.initialize()
    output.mkdir(parents=True, exist_ok=True)
    tables = load_backup(backup)
    catalog, audit = make_catalog(tables, as_of or date.today().isoformat(), include_closed_history=True)
    if not catalog:
        raise ValueError("No eligible catalog items")
    history = historical_interactions(tables, catalog)
    config = {"backend": backend, "model": model}
    if backend == 'semantic':
        config['max_length'] = max_length
        if batch_size is not None:
            config['batch_size'] = batch_size
    encoder = Encoder(config, device=device)
    print("Embedding", len(catalog), "items with", backend, flush=True)
    def progress(done, total):
        if done % 32 == 0 or done == total:
            print(f'Embedding {done}/{total}', flush=True)
    vectors = encoder.fit([r["text"] for r in catalog], progress=progress)
    topics = audit["dictionaries"]["Topics"]
    topic_vectors = encoder.encode(topics)
    # Центроид — среднее направление векторов материалов одной темы.
    # Если примеров нет, вместо среднего используем вектор названия темы.
    topic_centroids = []
    for i, topic in enumerate(topics):
        indices = [j for j, item in enumerate(catalog) if topic in item["topics"]]
        topic_centroids.append(normalize(vectors[indices].mean(axis=0)) if indices else topic_vectors[i])
    suggestions = [None] * len(catalog)
    for i, item in enumerate(catalog):
        if not item["topics"] and len(topics):
            similarities = topic_vectors @ vectors[i]
            order = np.argsort(-similarities)
            best = int(order[0])
            margin = float(similarities[best] - similarities[order[1]]) if len(order) > 1 else 1.0
            suggestions[i] = dict(title=topics[best], cosine=round(float(similarities[best]), 4), margin=round(margin, 4))
            # Близость по вектору — предположение; исходную категорию им не подменяем.
    # Это число записей на курс, а не вероятность покупки и не число просмотров.
    finalize_catalog(catalog, history, suggestions)
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    encoder.save(output / "encoder.json")
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    np.save(output / "topic_vectors.npy", np.asarray(topic_centroids, dtype=np.float32), allow_pickle=False)
    (output / "topics.json").write_text(json.dumps(topics, ensure_ascii=False), encoding="utf-8")
    audit.update(history_rows=len(history), history_users=len({r["user_id"] for r in history}),
                 embedding_backend=backend, embedding_model=model if backend == "semantic" else "lexical TF-IDF",
                 embedding_revision=encoder.config.get('revision'),
                 embedding_max_length=encoder.config.get('max_length'),
                 embedding_pooling=encoder.config.get('pooling'),
                 embedding_device=encoder.device if backend == 'semantic' else None,
                 vector_shape=list(vectors.shape), topics_from_catalog=sum(bool(r["topics"]) for r in catalog))
    # Отпечаток позволяет сбросить профили, рассчитанные для другой версии индекса.
    audit["catalog_fingerprint"] = fingerprint(output)
    database.replace_history(audit["catalog_fingerprint"], history)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(output)
    print(json.dumps({k: audit[k] for k in ("catalog_kinds", "history_rows", "history_users", "vector_shape", "embedding_backend")}, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backup", required=True)
    p.add_argument("--output", default="artifacts")
    p.add_argument("--backend", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", choices=['cpu', 'mps'], default='cpu')
    args = p.parse_args()
    build(args.backup, args.output, args.backend, args.model, args.as_of, args.max_length, args.batch_size, args.device)
