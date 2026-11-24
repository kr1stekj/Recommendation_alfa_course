"""Короткое уточнение задачи и объяснимая выдача поверх локального ранжирования."""
import re
from urllib.parse import quote, urlparse

from .embeddings import Encoder, tokens
from .llm_privacy import business_summary
from .data import canonical_label
from .clarification import clarify
from .dialog import DialogCodec, MAX_ANSWERS, MAX_TOKEN_LENGTH


STAGES = {
    "beginner": ("Только начинаю", "Начинающий предприниматель, открытие бизнеса, вводные материалы", "Начальный опыт"),
    "owner": ("Уже есть бизнес", "Владелец малого бизнеса, решение конкретной практической задачи", "Практический опыт"),
    "growth": ("Развиваю и масштабирую", "Развивающийся бизнес, системное изучение и улучшение процессов", "Есть опыт, нужно углубить знания"),
}
GOALS = {
    "launch": ("Открыть своё дело", "Запуск бизнеса, регистрация ИП, первые шаги"),
    "marketing": ("Привлечь клиентов", "Маркетинг, продвижение, продажи и привлечение клиентов"),
    "finance": ("Разобраться с деньгами", "Финансы, бухгалтерский учет и налоги"),
    "team": ("Нанять и управлять командой", "Найм сотрудников и управление командой"),
    "operations": ("Наладить работу бизнеса", "Бизнес-процессы, автоматизация, операционная работа"),
}
FORMATS = {
    "overview": ("Коротко разобраться", "article"),
    "tool": ("Получить готовый инструмент", "document"),
    "course": ("Изучить системно", "course"),
}
GENERIC = {"бизнес", "бизнеса", "бизнесе", "дело", "дела", "помоги", "помощь", "хочу", "нужно", "найти", "материал", "материалы", "курсы", "курс", "обучение", "разобраться", "начать", "мне", "про", "пожалуйста"}


def requested_format(task):
    """Выделяет один явный формат; смешанные запросы/отрицания остаются семантическими."""
    if re.search(r'\b(?:не|кроме|без|вместо)\s+(?:стать|курс|документ|подкаст)', task, re.I):
        return ''
    patterns = {'article': r'\bстать(?:я|и|ю|е|ей|ями|ях)\b',
                'course': r'\bкурс(?:ы|а|ов|у|ом|е|ами|ах)?\b',
                'document': r'\b(?:документ(?:ы|а|ов|у|ом|е)?|шаблон(?:ы|а|ов|у|ом|е)?|чек.?лист(?:ы|а|ов)?)\b',
                'podcast': r'\bподкаст(?:ы|а|ов|у|ом|е)?\b'}
    found = [kind for kind, pattern in patterns.items() if re.search(pattern, task, re.I)]
    return found[0] if len(found) == 1 else ''


def material_url(item):
    """Сохраняет полный URL, а slug достраивает по маршрутам платформы.

    Не проверяет наличие страницы: удалённая с витрины ссылка может устареть.
    Неизвестные относительные пути остаются только как исходный URL и ID.
    """
    value = item.get("url", "")
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        return None
    try:
        parsed = urlparse(value)
        if parsed.scheme:
            return value if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password else None
        if parsed.netloc or ".." in value or "\\" in value:
            return None
        routes = {"course": "courses", "article": "articles", "document": "documents", "podcast": "podcasts"}
        if value.startswith("/"):
            return "https://kurs.alfabank.ru" + value if any(value.startswith("/" + r + "/") for r in routes.values()) else None
        if "/" in value or parsed.query or parsed.fragment or item.get("kind") not in routes:
            return None
        return "https://kurs.alfabank.ru/" + routes[item["kind"]] + "/" + quote(value, safe="-") + "/"
    except ValueError:
        return None


class SmartSearch:
    """Не хранит диалог: клиент возвращает только проверенные варианты ответов."""
    def __init__(self, engine):
        self.engine = engine
        self.dialog = DialogCodec()
        self.catalog_words = set(tokens(" ".join(r["title"] + " " + " ".join(r["topics"])
                                                 for r in engine.catalog if r["eligible"])))
        self.topics = {canonical_label(t): t for r in engine.catalog if r['eligible'] for t in r['topics']}
        # Обычный поиск должен находить даже термин единственной карточки. Лимиты
        # сохранённого рекомендательного TF-IDF индекса здесь не применяются.
        self.lexical = Encoder({'backend': 'tfidf', 'min_df': 1, 'max_features': None})
        self.lexical_vectors = self.lexical.fit([r['title'] + ' ' + r['title'] + ' ' +
            r['description'] + ' ' + ' '.join(r['topics']) for r in engine.catalog])

    def search(self, task="", profile_id=None, answers=None, skip_questions=False, use_llm=False,
               industry="", experience="", search_mode="smart", clarifications=None):
        if not isinstance(task, str) or len(task) > 2000:
            raise ValueError("task must be a string of at most 2000 characters")
        if not all(isinstance(v, str) and len(v) <= 200 for v in (industry, experience)):
            raise ValueError("industry and experience must be strings of at most 200 characters")
        if not isinstance(skip_questions, bool) or not isinstance(use_llm, bool):
            raise ValueError("skip_questions and use_llm must be boolean")
        if search_mode not in ('normal', 'smart'):
            raise ValueError("search_mode must be normal or smart")
        if profile_id is not None:
            self.engine.validate_profile_id(profile_id)
        answers = {} if answers is None else answers
        if not isinstance(answers, dict) or set(answers) - {"stage", "goal", "format"}:
            raise ValueError("Unknown answers")
        for key, choices in (("stage", STAGES), ("goal", GOALS), ("format", FORMATS)):
            if key in answers and (not isinstance(answers[key], str) or answers[key] not in choices):
                raise ValueError("Invalid answer: " + key)
        clarifications = [] if clarifications is None else clarifications
        if (not isinstance(clarifications, list) or len(clarifications) > MAX_ANSWERS or
                not all(isinstance(token, str) and 0 < len(token) <= MAX_TOKEN_LENGTH for token in clarifications)):
            raise ValueError('Expected at most two clarification tokens')
        if not task.strip() and not answers and not clarifications:
            return self._home_feed(profile_id, search_mode, industry, experience)
        if search_mode == 'normal':
            return self._normal_search(task)

        binding = self.dialog.binding({'task': task, 'profile_id': profile_id, 'industry': industry,
            'experience': experience, 'answers': answers, 'dataset': getattr(self.engine, 'dataset', '')})
        selected = self.dialog.read(clarifications, binding)
        dialogue = [{'question': row['question'], 'answer': row['answer']} for row in selected]
        # Только подтверждённые сервером варианты могут стать новым контекстом.
        dynamic_context = ' '.join(row['question'] + ' ' + row['answer'] for row in selected)

        meaningful = set(tokens(task)) - GENERIC
        task_features = set(business_summary(task).split('; ')) - {'', 'Обучение'}
        known_context = bool(task_features or meaningful & self.catalog_words)
        broad = not meaningful or not known_context
        stage = answers.get("stage")
        goal = answers.get("goal")
        fmt = answers.get("format")
        required = FORMATS[fmt][1] if fmt else requested_format(task)
        preferred = FORMATS[fmt][1] if fmt else {"beginner": "article", "owner": "document", "growth": "course"}.get(stage, "article" if not task.strip() else "")
        for row in selected:
            if row['preferred_format']:
                preferred = row['preferred_format']
                required = row['preferred_format']
        if preferred not in {r["kind"] for r in self.engine.catalog if r["eligible"]}:
            preferred = ""
        unknown = bool(task.strip() and meaningful) and not known_context
        context = ' '.join(s for s in (GOALS[goal][1] if goal else '', dynamic_context) if s)
        # Сохраняем место для выбранных уточнений даже при длинной исходной строке.
        task_budget = max(0, 2000 - len(context) - 1) if context else 2000
        original = '' if broad else task.strip()[:task_budget]
        # Encoder обрезает вход по токенам: новый ответ ставим в начало, чтобы
        # длинный расплывчатый запрос не вытеснил уточнение за окно модели.
        parts = (context, original) if selected else (original, context)
        query = ' '.join(s for s in parts if s)
        feature_topics = {"Старт бизнеса": ["Первые шаги", "Регистрация бизнеса"],
                          "Продажи": ["Продажи", "Продажи и переговоры"],
                          "Финансы и учет": ["Финансы и учет", "Финансы"],
                          "Автоматизация": ["ИИ и автоматизация", "Автоматизация"]}
        features = business_summary(query).split('; ') if query else []
        topical = []
        for feature in features:
            for name in feature_topics.get(feature, [feature]):
                topic = self.topics.get(canonical_label(name))
                if topic and topic not in topical:
                    topical.append(topic)
        result = self.engine.recommend(
            task=query, profile_id=profile_id, stage=STAGES[stage][1] if stage else "",
            industry=industry, experience=experience or (STAGES[stage][2] if stage else ""),
            preferred_format=preferred, topics=topical[:3] or None, k=6, use_llm=use_llm, include_continuations=True,
            query_first=bool(query.strip()), required_format=required)

        # Косинус разных моделей не калиброван. Дополнительно смотрим, известны ли
        # слова/бизнес-признаки запроса; отсутствие совпадения не выдаём за точный ответ.
        weak = unknown
        if result["items"] and not broad:
            weak = weak or result["items"][0]["cosine"] < (0.1 if result["backend"] == "tfidf" else 0.25)
        status = "no_match" if weak else "starter" if not task.strip() and not goal else "related"
        if task.strip() and broad and not weak and not goal:
            status = "broad"
        if not result["items"]:
            status = "empty"
        actionable = bool(re.search(r'выбрать|выбор|открыть|зарегистр|нанять|рассчит|состав|подать|\bусн\b|\bндс\b', task.casefold()))
        # Совпадение одного «выбрать» с заголовком не определяет задачу. Для
        # пропуска LLM нужна распознанная бизнес-тема и содержательный объект.
        explicit_task = bool(task_features) and (len(meaningful) >= 3 or
                                                (actionable and len(meaningful) >= 2))
        has_history = bool(result["history_items"] or result["live_items"])
        if status == "starter" and has_history:
            status = "personalized"
        context_sufficient = bool(goal or explicit_task or (has_history and not task.strip())) and not (weak and not goal)
        question, assistant_status = None, 'skipped' if skip_questions else 'idle'
        if not skip_questions and task.strip() and context_sufficient:
            # Внешняя модель видит только общие признаки и может потерять детали
            # конкретной задачи. Достаточный локальный контекст не уточняем повторно.
            assistant_status = 'not_needed'
        elif not skip_questions and task.strip():
            preferences = {key: table[answers[key]][0] for key, table in
                (('stage', STAGES), ('goal', GOALS), ('format', FORMATS)) if key in answers}
            prior = [dict(pair, preferred_format=row['preferred_format']) for pair, row in zip(dialogue, selected)]
            generated, assistant_status = clarify(query or task, prior, preferences,
                'unknown' if unknown else 'general', has_history,
                {row['kind'] for row in self.engine.catalog if row['eligible']},
                profile_topics=[row[0] for row in result['profile_topics'][:3]], request_task=task)
            if generated is not None:
                question = self.dialog.issue(generated, binding, len(dialogue) + len(answers) + 1, prior=clarifications)
            # При недостаточном локальном контексте решение о вопросе принимает LLM.
            context_sufficient = not bool(question) if assistant_status == 'ok' else None
        if question is not None and status in {'related', 'broad', 'starter'}:
            status = 'broad'
        elif (selected and status in {'broad', 'starter'} and
              assistant_status in {'ok', 'skipped', 'limit_reached', 'already_answered'}):
            status = 'related'
        summary = ("Близкого совпадения с запросом не нашли — предлагаем общую подборку доступных материалов." if weak else
                   "Запрос пока общий — ниже предварительная подборка доступных материалов." if status == "broad" else
                   "Подготовили стартовую подборку — опишите задачу для более точных рекомендаций." if status == "starter" else
                   "Подобрали материалы с учётом ваших интересов — можно продолжить знакомую тему или перейти к следующей." if status == "personalized" else
                   "Подобрали материалы под вашу задачу — начните с первого и переходите дальше по мере необходимости.")
        if status == "empty":
            summary = "Подходящих доступных материалов пока нет. Попробуйте другую задачу или формат."
        context = (["этап «" + STAGES[stage][0].lower() + "»"] if stage else [])
        if result["history_items"] or result["live_items"]:
            context.append("предыдущие взаимодействия с материалами")
        if context:
            summary += " Учли " + " и ".join(context) + "."
        for i, row in enumerate(result["items"]):
            row["material_url"] = material_url(row)
            context = ""
            if row.get("continuation"):
                context = "Вы уже открывали этот материал и не отмечали его завершённым. Можно продолжить. "
            elif fmt and row["kind"] == preferred:
                context = "Подходит выбранный формат: «" + FORMATS[fmt][0].lower() + "». "
            elif stage:
                context = "Учли выбранный этап бизнеса. "
            row["explanation"] = context + row["explanation"]
            if topical:
                row["explanation"] = row["explanation"].replace("выбранными интересами", "темами задачи")
            elif status == "starter" and not has_history:
                row["explanation"] = "Материал для первого знакомства с задачами бизнеса. " + row["evidence"]
            nxt = result["items"][i + 1] if i + 1 < len(result["items"]) else None
            row["next_material"] = ({k: nxt[k] for k in ("id", "title", "kind")} if nxt else None)
        result.update(summary=summary, question=question, match_status=status,
                      clarification=question["title"] if question else None,
                      question_source="cloud_llm" if question else None, assistant_status=assistant_status,
                      search_mode='smart', context_sufficient=context_sufficient, answers=answers, dialogue=dialogue)
        return result

    def _home_feed(self, profile_id, search_mode, industry, experience):
        result = self.engine.recommend(profile_id=profile_id, industry=industry,
            experience=experience, k=10, include_continuations=True, discovery=True)
        has_history = bool(result['live_items'] or result['history_items'])
        for i, row in enumerate(result['items']):
            row['material_url'] = material_url(row)
            nxt = result['items'][i + 1] if i + 1 < len(result['items']) else None
            row['next_material'] = {key: nxt[key] for key in ('id','title','kind')} if nxt else None
        summary = ('Подборка по разным темам ваших интересов.' if has_history else 'Стартовая подборка материалов.')
        sources = {row['feed_source'] for row in result['items']}
        extras = (["популярное"] if 'popular' in sources else []) + (["материалы для знакомства"] if 'discovery' in sources else [])
        if extras:
            summary += ' Добавили ' + ' и '.join(extras) + '.'
        if not result['items']:
            summary = 'Подходящих доступных материалов пока нет.'
        result.update(feed_mode=True, search_mode=search_mode, question=None,
            question_source=None, assistant_status='not_requested', answers={}, dialogue=[],
            context_sufficient=has_history, clarification=None,
            total_matches=len(result['items']),
            match_status=('personalized' if has_history else 'starter') if result['items'] else 'empty',
            summary=summary)
        return result

    def _normal_search(self, task):
        """Обычный поиск по словам: не читает профиль и не вызывает нейросеть/LLM."""
        query = self.lexical.encode([task])[0]
        scores = self.lexical_vectors @ query
        candidates = [i for i, r in enumerate(self.engine.catalog)
                      if r['eligible'] and (not task.strip() or float(scores[i]) > 0)]
        candidates.sort(key=lambda i: (-float(scores[i]), -self.engine.catalog[i].get('history_count', 0),
                                       self.engine.catalog[i]['id']))
        rows = []
        for i in candidates[:20]:
            source = self.engine.catalog[i]
            row = {key: source.get(key) for key in ('id', 'title', 'description', 'kind', 'url', 'price',
                                                   'topics', 'tags', 'duration', 'target', 'lesson_quantity', 'access')}
            row.update(score=round(float(scores[i]), 5), cosine=round(float(scores[i]), 5),
                       explanation='Совпадение слов запроса с названием, аннотацией или темами.' if task.strip() else 'Материал из каталога.',
                       evidence=source['description'][:200] or source['title'], continuation=False,
                       material_url=material_url(source), next_material=None,
                       verification={'source': 'lexical_search', 'llm_checked': False})
            rows.append(row)
        return {'search_mode': 'normal', 'mode': 'keyword_search', 'backend': 'tfidf', 'algorithm': 'cosine',
                'items': rows, 'total_matches': len(candidates), 'question': None, 'question_source': None,
                'assistant_status': 'not_requested', 'llm_status': 'disabled', 'answers': {}, 'dialogue': [],
                'history_items': 0, 'live_items': 0, 'profile_topics': [], 'topic_recommendations': [],
                'score_is_probability': False, 'context_sufficient': None, 'clarification': None,
                'match_status': 'related' if rows else 'empty',
                'summary': ('Поиск по словам в названиях, аннотациях и темах материалов.' if task.strip() else
                            'Каталог материалов. Введите запрос для поиска.') if rows else
                           'По запросу ничего не найдено. Попробуйте другие слова.'}
