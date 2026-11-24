"""Смешивание стартовой выдачи; допустимость материалов определяет движок."""
import hashlib
import math


def compose_feed(catalog, candidates, ranked, k, popularity, seen, seed, last_activity=None):
    count = min(k, len(candidates))
    popular_slots, discovery_slots = int(count >= 5), int(count >= 10)
    ordered = list(dict.fromkeys(ranked + candidates))
    personal_count = count - popular_slots - discovery_slots
    # Все метки равноправны: порядок в каталоге не задаёт главную тему.
    # Материал может подходить нескольким темам, но занимает только один слот.
    interests = {}
    for row in catalog:
        weight = seen.get(row['id'], 0)
        if weight > 0:
            for topic in set(row['topics']):
                interests[topic] = interests.get(topic, 0) + weight
    # Равные времена не получают искусственный порядок. Темы без live-даты
    # сохраняются, но следуют за датированными интересами.
    last_activity = last_activity or {}
    dates = sorted({last_activity[t] for t in interests if t in last_activity}, reverse=True)
    ranks = {at: rank for rank, at in enumerate(dates)}
    priorities = {t: math.sqrt(w) / (1 + ranks.get(last_activity.get(t), len(dates)))
                  for t, w in interests.items()}
    groups = {}
    for topic in sorted(interests, key=lambda t: (-priorities[t], t)):
        group = [i for i in ordered if topic in catalog[i]['topics']]
        if group:
            groups[topic] = group
        if len(groups) == 4:
            break
    selected = []
    allocated = {topic: 0 for topic in groups}
    while len(selected) < personal_count and any(groups.values()):
        for group in groups.values():
            while group and group[0] in selected:
                group.pop(0)
        available = [topic for topic, group in groups.items() if group]
        if not available:
            break
        # Сначала представляем разные интересы, затем выделяем больше мест свежим.
        unrepresented = [topic for topic in available if not allocated[topic]]
        topic = max(unrepresented or available,
                    key=lambda t: priorities[t] / (allocated[t] + 1))
        selected.append(groups[topic].pop(0))
        allocated[topic] += 1
    selected += [i for i in ordered if i not in selected][:personal_count - len(selected)]
    sources = {i: 'personal' for i in selected}

    def remaining():
        pool = [i for i in candidates if i not in selected]
        unseen = [i for i in pool if catalog[i]['id'] not in seen]
        return unseen or pool

    if popular_slots:
        chosen = min(remaining(), key=lambda i: (-float(popularity[i]), catalog[i]['id']))
        selected.append(chosen)
        sources[chosen] = 'popular' if popularity[chosen] > 0 else 'discovery'
    if discovery_slots:
        pool = remaining()
        known_topics = set(interests)
        known_topics.update(topic for i in selected if sources[i] == 'personal' for topic in catalog[i]['topics'])
        new_topics = [i for i in pool if catalog[i]['topics'] and
                      not known_topics.intersection(catalog[i]['topics'])]
        shown_topics = {topic for i in selected for topic in catalog[i]['topics']}
        unshown_topics = [i for i in new_topics if not shown_topics.intersection(catalog[i]['topics'])]
        pool = unshown_topics or new_topics or pool
        chosen = min(pool, key=lambda i: hashlib.sha256(
            (seed + ':' + catalog[i]['id']).encode()).digest())
        selected.append(chosen); sources[chosen] = 'discovery'
    return selected, sources
