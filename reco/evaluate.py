"""Сравнение методов на одинаковых тестовых профилях с заранее заданной разметкой."""
import argparse
import json
import math
from pathlib import Path
import time

from .engine import Recommender


def metrics(ranking, judgments, k):
    """Precision — доля подходящих в top-k, NDCG — качество с учётом порядка.

    judgments содержит оценки релевантности по ID. Неоценённые здесь считаются
    нулевыми, поэтому неполная разметка ограничивает смысл результата.
    """
    grades = [judgments.get(i, 0) for i in ranking[:k]]
    dcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    # Делим полезность полученного порядка на полезность идеального: NDCG от 0 до 1.
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return {"precision": sum(g > 0 for g in grades) / k, "ndcg": dcg / idcg if idcg else 0.0}


def evaluate(artifacts="artifacts", profiles="evaluation/profiles.json", k=5, engine=None):
    """Сравнивает тему, косинус и гибрид; не измеряет рост реальной конверсии."""
    # Benchmark может передать уже загруженный encoder, не создавая вторую копию BGE.
    engine = engine if engine is not None else Recommender(artifacts)
    fixture = json.loads(Path(profiles).read_text(encoding="utf-8"))
    known = {r["id"] for r in engine.catalog if r["eligible"]}
    rows, summary = [], {}
    # Прогреваем модель: загрузка весов не должна входить во время каждого запроса.
    engine.encoder.encode(["Проверка"])
    for algorithm in ("topic", "cosine", "hybrid"):
        all_items = set()
        for profile in fixture["profiles"]:
            unknown = set(profile["relevance"]) - known
            if unknown: raise ValueError("Judgments contain ineligible/missing items: " + str(unknown))
            start = time.perf_counter()
            result = engine.recommend(task=profile["task"], topics=profile.get("topics"), stage=profile.get("stage", ""), k=k, algorithm=algorithm)
            elapsed = 1000 * (time.perf_counter() - start)
            ids = [r["id"] for r in result["items"]]
            all_items.update(ids)
            values = metrics(ids, profile["relevance"], k)
            rows.append(dict(profile=profile["id"], algorithm=algorithm, latency_ms=round(elapsed, 2),
                             **{f"{name}@{k}": round(value, 4) for name, value in values.items()}, ids=ids))
        group = [r for r in rows if r["algorithm"] == algorithm]
        summary[algorithm] = {key: round(sum(r[key] for r in group) / len(group), 4) for key in (f"precision@{k}", f"ndcg@{k}")}
        summary[algorithm]["catalog_coverage"] = round(len(all_items) / len(known), 4)
    return {"backend": engine.encoder.config["backend"], "judgments_status": fixture["status"], "profiles": len(fixture["profiles"]), "summary": summary, "rows": rows,
            "limitations": "Tiny seed set; incomplete relevance judgments. No evidence of online conversion uplift. Do not tune and report on this same set."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--profiles", default="evaluation/profiles.json")
    p.add_argument("--output")
    a = p.parse_args()
    result = evaluate(a.artifacts, a.profiles)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output: Path(a.output).write_text(text, encoding="utf-8")
    print(text)
