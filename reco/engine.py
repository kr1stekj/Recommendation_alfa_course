"""Рекомендации: история и задача → профиль интересов → оценки → разнообразная выдача."""
import json
import re
import time

import numpy as np

from .data import canonical_label
from .embeddings import Encoder, normalize
from .llm import verify
from .state import ProfileStore
from .database import Database
from .index import resolve_index, validate_index
from .ranking import normalized_popularity, score_components, score_breakdown, log_components
from .feed import compose_feed

# Начальные эвристики: вовлечённое чтение сильнее клика, dislike — отрицательный сигнал.
EVENT_WEIGHTS = {"click": 1.0, "engaged": 3.0, "complete": 6.0, "dislike": -4.0}
DEFAULT_QUERY = 'Как открыть своё дело: первые шаги бизнеса и регистрация ИП'
SEARCH_HISTORY_WEIGHT = 0.10
POSITIVE_HALF_LIFE = 86400
NEGATIVE_HALF_LIFE = 7 * 86400


class Recommender:
    """Читает готовый индекс и объединяет его с ограниченными профилями пользователей."""
    def __init__(self, artifacts="artifacts", database=None):
        # Ссылку разрешаем один раз: переключение current не смешает файлы поколений.
        self.path = resolve_index(artifacts)
        self.audit = validate_index(self.path)
        self.catalog = json.loads((self.path / "catalog.json").read_text(encoding="utf-8"))
        self.vectors = np.load(self.path / "vectors.npy", allow_pickle=False)
        self.encoder = Encoder.load(self.path / "encoder.json")
        self.topic_names = json.loads((self.path / "topics.json").read_text(encoding="utf-8")) if (self.path / "topics.json").exists() else []
        self.topic_vectors = np.load(self.path / "topic_vectors.npy", allow_pickle=False) if self.topic_names else None
        if len(self.catalog) != len(self.vectors):
            raise ValueError("Catalog/vector mismatch: rebuild artifacts")
        self.index = {r["id"]: i for i, r in enumerate(self.catalog)}
        self.database = database or Database()
        self.database.initialize()
        self.dataset = self.audit["catalog_fingerprint"]
        if not self.database.has_dataset(self.dataset):
            raise ValueError("История индекса не импортирована: выполните python -m reco.migrate --backup <файл>")
        self.store = ProfileStore(self.database, self.dataset)
        self.popularity = normalized_popularity(self.catalog)

    def history(self, profile_id):
        """История из дампа доступна только для user:UUID; у гостя её нет."""
        if not profile_id.startswith("user:"):
            return []
        return self.database.history(self.dataset, profile_id[5:])

    def event(self, profile_id, item_id, event_type, event_id, now=None):
        """Обновляет агрегат интересов без сохранения отдельной строки на каждый клик.

        event_id защищает от повторной доставки в пределах последних 256 событий.
        now нужен для тестов; HTTP API всегда использует серверное время.
        """
        if item_id not in self.index or not self.catalog[self.index[item_id]]["eligible"]:
            raise ValueError("Unknown or ineligible catalog item")
        if event_type not in EVENT_WEIGHTS:
            raise ValueError("Unknown event type; use click, engaged, complete, dislike")
        self.validate_profile_id(profile_id)
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128:
            raise ValueError("event_id must contain 1..128 characters")
        now = time.time() if now is None else now
        duplicate = [False]
        def update(state):
            if event_id in state["event_ids"]:
                duplicate[0] = True
                return state
            old = state["items"].get(item_id, {})
            weight = EVENT_WEIGHTS[event_type]
            # Повторные клики не увеличивают вес бесконечно: сохраняем сильнейший сигнал.
            if weight > 0:
                weight = max(weight, max(0, old.get("weight", 0)))
            state["items"][item_id] = {"weight": weight, "at": now,
                                        "completed": event_type == "complete" or old.get("completed", False)}
            # Фиксированные пределы не дают профилю расти вместе со всей историей кликов.
            state["items"] = dict(sorted(state["items"].items(), key=lambda kv: kv[1]["at"])[-128:])
            state["event_ids"] = (state["event_ids"] + [event_id])[-256:]
            return state
        self.store.mutate(profile_id, "anonymous" if profile_id.startswith("anon:") else "registered", update, now)
        return {"accepted": True, "duplicate": duplicate[0], "stored_raw_event": False}

    @staticmethod
    def validate_profile_id(profile_id):
        if not isinstance(profile_id, str) or not re.fullmatch(r"(?:anon|user):[A-Za-z0-9_-]{1,100}", profile_id):
            raise ValueError("Expected anon:<random_token> or user:<pseudonymous_user_id>")

    def recommend(self, task="", profile_id=None, stage="", industry="", experience="", preferred_format="", topics=None, k=5, algorithm="hybrid", use_llm=False, include_continuations=False, query_first=False, required_format="", discovery=False):
        """Возвращает до k карточек с факторами оценки, основанием и следующим шагом.

        topic — тематический бейзлайн; cosine — только векторная близость;
        hybrid — близость плюс метки, формат, популярность и мягкое разнообразие.
        Итоговый score не является вероятностью покупки.
        """
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("k must be an integer in 1..20")
        if algorithm not in ("hybrid", "topic", "cosine"):
            raise ValueError("algorithm must be hybrid, topic or cosine")
        if not isinstance(use_llm, bool) or not isinstance(include_continuations, bool):
            raise ValueError("use_llm and include_continuations must be boolean")
        if not isinstance(query_first, bool):
            raise ValueError('query_first must be boolean')
        if not isinstance(required_format, str) or required_format not in {'', 'course', 'article', 'document', 'podcast'}:
            raise ValueError('Unknown required_format')
        if not all(isinstance(v, str) and len(v) <= 2000 for v in (task, stage, industry, experience, preferred_format)):
            raise ValueError("Onboarding fields must be strings of at most 2000 characters")
        if not isinstance(discovery, bool) or (discovery and (task.strip() or use_llm)):
            raise ValueError('discovery requires an empty task and disabled LLM')
        explicit_query = query_first and bool(task.strip())
        if topics is not None and (not isinstance(topics, list) or len(topics) > 20 or not all(isinstance(t, str) and len(t) <= 100 for t in topics)):
            raise ValueError("topics must be a list of up to 20 strings")
        if preferred_format and preferred_format not in {r["kind"] for r in self.catalog}:
            raise ValueError("Unknown preferred_format")
        if profile_id:
            self.validate_profile_id(profile_id)
        state = self.store.read(profile_id) if profile_id else self.store.empty()
        history = self.history(profile_id) if profile_id else []
        signals, excluded, unfinished, completed_ids = {}, set(), set(), set()
        last_activity = {}
        now = time.time()
        for row in history:
            # Один курс даёт один вклад, даже если в истории есть повторные записи.
            iid = row["item_id"]
            if iid in self.index:
                signals[iid] = max(signals.get(iid, 0), row["weight"])
                if row["completed"]:
                    completed_ids.add(iid)
                    excluded.add(iid)
                elif row["weight"] > 0:
                    unfinished.add(iid)
                if row["owned"] and not include_continuations: excluded.add(iid)
        for iid, r in state["items"].items():
            if iid not in self.index: continue
            # Положительный интерес обновляется быстрее; dislike не забываем за сутки.
            half_life = POSITIVE_HALF_LIFE if r['weight'] > 0 else NEGATIVE_HALF_LIFE
            weight = r["weight"] * 2 ** (-max(0, now - r["at"]) / half_life)
            signals[iid] = weight if weight < 0 else max(signals.get(iid, 0), weight)
            if weight > 0:
                for topic in self.catalog[self.index[iid]]['topics']:
                    last_activity[topic] = max(last_activity.get(topic, float('-inf')), r['at'])
            if r.get("completed"):
                completed_ids.add(iid)
            if not include_continuations or weight <= 0 or r.get("completed"):
                excluded.add(iid)
            elif weight > 0:
                unfinished.add(iid)
        unfinished -= completed_ids | excluded
        # Положительные и отрицательные интересы считаем отдельно: иначе они могут
        # взаимно погаситься и скрыть причину штрафа за нежелательные материалы.
        positive = np.zeros(self.vectors.shape[1], dtype=np.float32)
        negative = np.zeros_like(positive)
        inferred_topics = {}
        for iid, w in signals.items():
            i = self.index[iid]
            if w > 0:
                positive += w * self.vectors[i]
                for topic in self.catalog[i]["topics"]:
                    inferred_topics[topic] = inferred_topics.get(topic, 0) + w
            else:
                negative += abs(w) * self.vectors[i]
        selected_topics = topics or ([] if explicit_query else sorted(inferred_topics, key=inferred_topics.get, reverse=True)[:3])
        if not signals and not any((task, stage, industry, experience, topics)):
            selected_topics = [name for name in ("Первые шаги", "Регистрация бизнеса") if name in self.topic_names]
        selected_set = {canonical_label(t) for t in selected_topics}
        topic_score = np.asarray([len(selected_set & {canonical_label(t) for t in r["topics"]}) / max(len(selected_set), 1) for r in self.catalog], dtype=np.float32)
        input_text = " ".join(s for s in (task, stage, industry, experience, " ".join(topics or [])) if s)
        mode = "behavior" if signals else "onboarding" if input_text else "starter_collection"
        query_text = input_text or ("" if np.linalg.norm(positive) else DEFAULT_QUERY)
        if discovery and not input_text and not np.linalg.norm(positive):
            query_text = ''
        vector_source = ('query_and_history' if np.linalg.norm(positive) else 'query') if input_text else (
            'history' if np.linalg.norm(positive) else 'default')
        # /recommend сохраняет 75/25. Умный поиск сначала отбирает по текущему
        # запросу, затем добавляет только 10% истории. Без задачи берём историю.
        if query_text:
            context_text = " ".join(s for s in (stage, industry, experience, " ".join(topics or [])) if s)
            if task and context_text:
                # Длинный онбординг не должен вытеснять конкретную задачу из вектора.
                task_vector, context_vector = self.encoder.encode([task, context_text])
                query = normalize(0.85 * task_vector + 0.15 * context_vector)
            else:
                query = self.encoder.encode([query_text])[0]
            user_vector = (normalize(0.75 * query + 0.25 * normalize(positive))
                           if np.linalg.norm(positive) and not explicit_query else query)
        else:
            user_vector = normalize(positive)
        if not np.linalg.norm(user_vector):
            # Например, TF-IDF не знает ни одного слова запроса. Нулевая история
            # не должна приводить к делению на ноль или случайному NaN-ранжированию.
            topical = [i for i, row in enumerate(self.catalog) if row['eligible'] and
                       selected_set & {canonical_label(topic) for topic in row['topics']}]
            if topical:
                # Явная метка полезнее общего среднего, если encoder её не знает.
                user_vector = normalize(self.vectors[topical].mean(axis=0))
                vector_source = 'topic_centroid'
            if not np.linalg.norm(user_vector) and query_text != DEFAULT_QUERY and not discovery:
                user_vector = self.encoder.encode([DEFAULT_QUERY])[0]
                vector_source = 'default'
            if not np.linalg.norm(user_vector):
                available = [i for i, row in enumerate(self.catalog) if row['eligible']]
                user_vector = normalize(self.vectors[available].mean(axis=0)) if available else np.zeros_like(positive)
                vector_source = 'catalog_centroid' if np.linalg.norm(user_vector) else 'zero'
        # Для поиска сохраняем независимое от истории направление. История
        # уточняет порядок, но не определяет, какие кандидаты отвечают запросу.
        retrieval_cosine = np.clip(self.vectors @ user_vector, -1.0, 1.0)
        if explicit_query and np.linalg.norm(positive):
            user_vector = normalize((1 - SEARCH_HISTORY_WEIGHT) * user_vector +
                                    SEARCH_HISTORY_WEIGHT * normalize(positive))
        # Символ @ — умножение матрицы на вектор: сразу оцениваем весь каталог.
        cosine = np.clip(self.vectors @ user_vector, -1.0, 1.0)
        topic_recommendations = []
        if self.topic_names and mode != "starter_collection":
            topic_cosines = self.topic_vectors @ user_vector
            topic_recommendations = [{"topic": self.topic_names[int(i)], "cosine": round(float(topic_cosines[int(i)]), 4)} for i in np.argsort(-topic_cosines)[:5]]
        negative_score = np.maximum(0, self.vectors @ normalize(negative)) if np.linalg.norm(negative) else np.zeros(len(self.catalog))
        format_score = np.asarray([float(bool(preferred_format) and preferred_format == r["kind"]) for r in self.catalog])
        continuation = np.asarray([float(include_continuations and r['id'] in unfinished) for r in self.catalog])
        factors, contributions, weights, scores = score_components(
            algorithm, cosine, topic_score, format_score, self.popularity, negative_score, continuation)
        candidates = [i for i, r in enumerate(self.catalog) if r["eligible"] and r["id"] not in excluded
                      and (not required_format or r['kind'] == required_format)]
        if explicit_query and candidates:
            # Распознанная тема текущей задачи сильнее старых интересов. Если
            # есть её доступные материалы, не заполняем остаток выдачи чужими
            # темами ради достижения k. Без меток остаётся семантический отбор.
            topical_candidates = [i for i in candidates if topic_score[i] > 0]
            if topical_candidates:
                candidates = topical_candidates
            best = max(float(retrieval_cosine[i]) for i in candidates)
            threshold = best * 0.4 if self.encoder.config['backend'] == 'tfidf' else best - 0.2
            candidates = [i for i in candidates if float(retrieval_cosine[i]) >= threshold or topic_score[i] > 0]
            candidates.sort(key=lambda i: (-float(retrieval_cosine[i]), self.catalog[i]['id']))
            candidates = candidates[:max(20, 3 * k)]
        if algorithm == "topic":
            # При равенстве тем бейзлайн выбирает курсы с большей исторической активностью.
            candidates.sort(key=lambda i: (-float(scores[i]), -float(self.popularity[i]), self.catalog[i]["id"]))
        else:
            candidates.sort(key=lambda i: (-float(scores[i]), self.catalog[i]["id"]))
        selected, selection_penalties = [], {}
        # Разнообразие добавляем только в hybrid, чтобы бейзлайны оставались сравнимыми.
        if algorithm != "hybrid":
            selected = candidates[:k]
        else:
            if explicit_query:
                pool = candidates
            elif candidates:
                best_cosine = max(float(cosine[i]) for i in candidates)
                threshold = best_cosine * 0.4 if self.encoder.config['backend'] == 'tfidf' else best_cosine - 0.2
                pool = [i for i in candidates if float(cosine[i]) >= threshold or topic_score[i] > 0][:100]
            else:
                pool = []
            while pool and len(selected) < k:
                def diversity_penalties(i):
                    # Штраф за близость к уже выбранному и за повтор формата.
                    # Это мягкий баланс, а не требование выдать по карточке каждого типа.
                    redundancy = max((max(0.0, min(1.0, float(self.vectors[i] @ self.vectors[j])))
                                      for j in selected), default=0.0)
                    same_kind = sum(self.catalog[j]["kind"] == self.catalog[i]["kind"] for j in selected)
                    return {'similarity': -0.05 * redundancy, 'same_format': -0.01 * same_kind}
                best = max(pool, key=lambda i: float(scores[i]) + sum(diversity_penalties(i).values()))
                selection_penalties[best] = diversity_penalties(best)
                selected.append(best)
                pool.remove(best)
        feed_sources = {}
        if discovery:
            seed = str(profile_id or 'guest') + ':' + self.dataset + ':' + str(int(now // 86400))
            selected, feed_sources = compose_feed(self.catalog, candidates, selected, k,
                self.popularity, signals, seed, last_activity=last_activity)
        rows = []
        for i in selected:
            item = self.catalog[i]
            matching = sorted(selected_set & {canonical_label(t) for t in item["topics"]})
            reason = ("Совпадает с выбранными интересами: " + ", ".join(matching) + ".") if matching else ("Близок по тексту к вашей задаче и истории." if mode != "starter_collection" else "Материал из стартовой подборки; интересы ещё не определены.")
            # Основание берём из каталога, чтобы объяснение не приписывало материалу лишнее.
            rows.append({key: item[key] for key in ("id", "title", "description", "kind", "url", "price", "topics", "tags")})
            rows[-1].update(score=round(float(scores[i]), 5), cosine=round(float(cosine[i]), 5),
                            continuation=bool(include_continuations and item["id"] in unfinished),
                            factors={name: round(float(values[i]), 8) for name, values in factors.items()},
                            score_breakdown=score_breakdown(algorithm, contributions, weights, i, selection_penalties.get(i, {})),
                            explanation=reason, evidence=item["description"][:200] or item["title"],
                            next_step="Открыть страницу курса" if item["kind"] == "course" else "Открыть материал",
                            verification={"source": "catalog_rules", "llm_checked": False})
            rows[-1].update({key: item.get(key) for key in ("duration", "target", "lesson_quantity", "access")})
            if discovery:
                source = feed_sources[i]
                if source == 'personal' and not signals:
                    source = 'starter'
                rows[-1]['feed_source'] = source
                rows[-1]['score_breakdown'].update(diversity_penalties={},
                    selection_score=rows[-1]['score_breakdown']['base_score'],
                    selection_policy='home-recency-weighted')
                if source in {'popular', 'discovery'}:
                    rows[-1]['continuation'] = False
                    rows[-1]['explanation'] = ('Материал с высокой исторической активностью.'
                        if source == 'popular' else 'Дополнительный материал для знакомства и расширения интересов.')
            log_components(algorithm, len(rows), rows[-1]['factors'], rows[-1]['score_breakdown'])
        llm_status = "disabled"
        # LLM добавляет проверку, но не переставляет математическую выдачу.
        if use_llm:
            llm_items = []
            for row in rows:
                source = self.catalog[self.index[row['id']]]
                # Старый флаг, рассчитанный только по IsPublic, не подтверждает доступ.
                allowed = source.get('llm_public_metadata', False) and (source['kind'] == 'document' or
                           source.get('access') in {'public', 'paid', 'subscription'})
                llm_items.append(dict(row, llm_public_metadata=allowed))
            checked, llm_status = verify(input_text or " ".join(selected_topics), llm_items)
            for item in rows:
                if item["id"] in checked:
                    item["verification"] = dict(checked[item["id"]], llm_checked=True)
        return {"mode": mode, "backend": self.encoder.config["backend"], "algorithm": algorithm,
                "profile_topics": sorted(inferred_topics.items(), key=lambda kv: -kv[1])[:5],
                "topic_recommendations": topic_recommendations,
                "history_items": len(history), "live_items": len(state["items"]), "llm_status": llm_status,
                "score_is_probability": False, "query_vector_source": vector_source,
                "clarification": "Можно уточнить задачу, стадию бизнеса и удобный формат." if mode == "starter_collection" or (selected and float(cosine[selected[0]]) < 0.25) else None,
                "items": rows}
