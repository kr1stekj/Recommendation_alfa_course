"""Ограниченные признаки, веса и числовая диагностика ранжирования."""
import json
import logging
import math

import numpy as np

from .missing import integer_or_none

HYBRID_WEIGHTS = {'cosine': 0.75, 'topic': 0.16, 'format': 0.04,
                  'popularity': 0.05, 'dislike': -0.12, 'continuation': 0.03}
LOGGER = logging.getLogger(__name__)


def unit_interval(values):
    """NaN/Infinity не создают положительного сигнала; диапазон ограничен [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def normalized_popularity(catalog):
    """Счётчик отсутствует или некорректен → 0; максимум ненулевых логарифмов → 1."""
    counts = [max(integer_or_none(row.get('history_count')) or 0, 0) for row in catalog]
    # math.log умеет большие Python int без предварительного переполнения float.
    logs = np.asarray([math.log(count + 1) for count in counts], dtype=np.float64)
    scale = float(logs.max(initial=0.0)) or 1.0
    return unit_interval(logs / scale)


def score_components(algorithm, cosine, topic, format_match, popularity, dislike, continuation):
    """Разделяет ограниченные признаки и их взвешенные вклады.

    Только baseline cosine сохраняет исходную знаковую шкалу. Hybrid использует
    max(0, cosine), поэтому ортогональный вектор не получает искусственные 0.5.
    """
    factors = {name: unit_interval(values) for name, values in (
        ('cosine', cosine), ('topic', topic), ('format', format_match),
        ('popularity', popularity), ('dislike', dislike), ('continuation', continuation))}
    if algorithm == 'hybrid':
        weights = dict(HYBRID_WEIGHTS)
        contributions = {name: weights[name] * values for name, values in factors.items()}
    elif algorithm == 'topic':
        weights, contributions = {'topic': 1.0}, {'topic': factors['topic'].copy()}
    elif algorithm == 'cosine':
        weights, contributions = {'raw_cosine': 1.0}, {'raw_cosine': np.asarray(cosine, dtype=np.float64).copy()}
    else:
        raise ValueError('Unknown ranking algorithm')
    scores = np.sum(np.stack(list(contributions.values())), axis=0)
    return factors, contributions, weights, scores


def score_breakdown(algorithm, contributions, weights, index, penalties):
    """Описывает базовую оценку и штрафы разнообразия в момент выбора карточки."""
    base = sum(float(values[index]) for values in contributions.values())
    return {'formula': 'hybrid-v2' if algorithm == 'hybrid' else algorithm,
            'weights': dict(weights),
            'contributions': {name: round(float(values[index]), 8) for name, values in contributions.items()},
            'base_score': round(base, 8),
            'diversity_penalties': {name: round(value, 8) for name, value in penalties.items()},
            'selection_score': round(base + sum(penalties.values()), 8)}


def enable_ranking_logging():
    """Включает только числовой журнал ранжирования в stderr, не DEBUG сторонних API."""
    if not LOGGER.handlers:
        LOGGER.addHandler(logging.StreamHandler())
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def log_components(algorithm, rank, factors, breakdown):
    """Не принимает текст запроса, идентификаторы, названия или историю."""
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(json.dumps({'event': 'ranking_components', 'algorithm': algorithm, 'rank': rank,
                               'factors': factors, 'score_breakdown': breakdown}, allow_nan=False))
