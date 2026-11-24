"""Облачный экспорт только фиксированных бизнес-признаков, без свободного текста."""
import re
import unicodedata

# В сеть выходят только ключи этого словаря, никогда найденные подстроки.
# Например, имя/телефон внутри запроса не могут стать частью summary.
AI_PATTERN = r'\bии\b|\bai\b|нейросет|искус?ственн?.{0,10}интел+ект'
FEATURES = {
    'Налоги и законы': r'налог|закон|юрид|договор|правов|\bусн\b|\bндс\b|патент',
    'Финансы и учет': r'финанс|уч[её]т|бухгалтер|доход|расход|кассов|бюджет|плат[её]ж',
    'Маркетинг': r'маркетинг|реклам|продвиж|бренд|соцсет|социальн.{0,12}сет',
    'Продажи': r'продаж|прода[вт]|клиент|воронк',
    'Управление командой': r'команд|сотрудник|персонал|нан[яи]|найм|мотивац',
    'Старт бизнеса': r'старт|запуск|начать|начинающ|откры[тч]|регистрац|\bип\b|\bооо\b',
    'Маркетплейсы': r'маркетплейс|озон|wildberries|авито|flowwow',
    'Клиентский сервис': r'сервис|обслужив|удержан|лояльност',
    'Стратегия': r'стратег|бизнес.модел|масштабир|развити',
    'Автоматизация': r'автоматизац|\bcrm\b|' + AI_PATTERN,
    'Операционная работа': r'операцион|операционк|процесс|логистик|закупк',
    'Регистрация бизнеса': r'регистрац|открыть.{0,20}\b(?:ип|ооо|бизнес)\b',
    'Закрытие бизнеса': r'закры[тч]|ликвидац|прекращени.{0,15}деятельност',
    'Найм': r'найм|нан[яи]|подбор.{0,15}(?:команд|персонал|сотрудник)',
    'Режим работы': r'график|режим.{0,10}работ',
    'Обучение': r'обучени|изучить|разобраться|курс',
    'УСН': r'\bусн\b|упрощенн',
    'НДС': r'\bндс\b',
    'Самозанятость': r'самозанят',
    'Услуги': r'услуг',
    'Розница': r'розни[цч]|ретейл',
    'Общепит': r'общепит|кафе|ресторан|horeca',
    'Таблицы и аналитика': r'таблиц|аналитик|анализ.{0,10}данн',
}
KINDS = {'article', 'course', 'document', 'podcast'}
MAX_CANDIDATES = 5
FOCUS_PATTERNS = {'Искусственный интеллект': AI_PATTERN, 'CRM': r'\bcrm\b', **FEATURES}


def request_focus(text):
    """Сохраняет конкретное ИИ/CRM вместо широкой категории автоматизации."""
    text = unicodedata.normalize('NFKC', text).casefold()
    for name in ('Искусственный интеллект', 'CRM'):
        if re.search(FOCUS_PATTERNS[name], text):
            return [name]
    return [name for name in business_summary(text).split('; ')
            if name and name not in {'Обучение', 'Старт бизнеса'}][:3]


def profile_topic_summary(topics):
    """Экспортирует только константы, не текст даже ошибочной/личной метки."""
    topics = topics if isinstance(topics, (list, tuple)) else []
    text = ' '.join(value[:200] for value in topics[:3] if isinstance(value, str))
    return [name for name in business_summary(text).split('; ') if name][:3]


def business_summary(text):
    """Сводка из констант; неизвестный контекст остаётся пустым, а не уходит наружу."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text[:12000]).casefold().replace('ё', 'е')
    return '; '.join(label for label, pattern in FEATURES.items() if re.search(pattern, text))


# Эти признаки описывают бизнес-контекст, но не копируют слова из ответа модели.
DIALOG_FACETS = {
    'Цель бизнеса': r'цель|задач|что.{0,12}интерес',
    'Канал продвижения': r'канал|способ.{0,12}продвиг',
    'Бюджет продвижения': r'бюджет|расход.{0,12}реклам|вложен.{0,12}продвиж',
    'Уровень знаний': r'уровень|уровню|уровня|опыт|знани',
    'Формат материала': r'формат|способ.{0,12}обуч',
    'Аудитория бизнеса': r'аудитори|b2b|b2c',
    'Цифровой маркетинг': r'цифров|интернет.маркет',
    'Социальные сети': r'соцсет|социальн.{0,12}сет',
    'Поисковое продвижение': r'\bseo\b|поисков',
    'Платная реклама': r'платн.{0,12}реклам|таргет|контекстн',
    'Контент-маркетинг': r'контент|публикац|\bблог',
    'Партнёрства': r'партнер|партнёр',
    'Рассылки': r'рассыл|e-?mail|электронн.{0,12}почт',
    'Телефонные продажи': r'телефонн.{0,12}продаж|холодн.{0,12}звон',
    'Бренд': r'бренд|\bbrand\b',
    'Маркетинговая стратегия': r'стратег',
    'Исследования и аналитика': r'исследован|аналитик|маркетингов.{0,12}анализ',
    'Начальный уровень': r'начальн|основ|базов|нович|без.{0,5}опыта|с нуля',
    'Средний уровень': r'средн.{0,12}уров',
    'Продвинутый уровень': r'продвин|глубок|эксперт|профессиональн',
    'Системное обучение': r'системн|полное.{0,5}погружен',
    'Короткое знакомство': r'коротк|быстр|кратк|ознакомлен',
    'Практический инструмент': r'шаблон|чек.?лист|готов.{0,12}инструмент',
}


def dialog_facets(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return [label for label, pattern in DIALOG_FACETS.items() if re.search(pattern, text)]


def _budget_bucket(answer):
    """Укрупняет лишь однозначные целые суммы/пороги, сохраняя направление границы.

    Неизвестный диапазон, десятичная запись и лишние слова остаются локально.
    Верхняя граница сама по себе не означает, что бюджет выше меньшего порога.
    """
    text = ' '.join(unicodedata.normalize('NFKC', answer).casefold().split())
    if re.fullmatch(r'без\s+(?:рекламного\s+)?бюджета|бесплатно|без\s+затрат|нет\s+бюджета', text):
        return 'Без рекламного бюджета'
    amount = r'(?P<amount>\d+(?:\s\d{3})*)\s*(?P<unit>тыс\.?|тысяч[аи]?|млн\.?|миллион(?:а|ов)?)?\s*(?:₽|руб(?:\.|ль|ля|лей)?|р\.)?'
    for kind, prefix in (('upper', r'(?:до|не более|менее|меньше)\s+'),
                         ('lower', r'(?:более|свыше|больше)\s+'),
                         ('lower_inclusive', r'(?:от|не менее)\s+'), ('exact', '')):
        match = re.fullmatch(prefix + amount, text)
        if match is None:
            continue
        value = int(match['amount'].replace(' ', ''))
        unit = match['unit'] or ''
        value *= 1000 if unit.startswith('тыс') else 1000000 if unit.startswith('м') else 1
        if kind in {'upper', 'exact'}:
            if value == 0:
                return None if text.startswith(('менее', 'меньше')) else 'Без рекламного бюджета'
            if value <= 10000:
                return 'Бюджет до 10 тысяч рублей'
            if value <= 100000:
                return 'Бюджет до 100 тысяч рублей'
            return 'Бюджет свыше 100 тысяч рублей' if kind == 'exact' else None
        if (kind == 'lower' and value >= 100000) or (kind == 'lower_inclusive' and value > 100000):
            return 'Бюджет свыше 100 тысяч рублей'
        return None
    return None


def dialogue_projection(history):
    """Внешний контекст — только константы, даже если модель ошиблась и спросила имя.

    Подпись доказывает происхождение варианта, не его неперсональный смысл. Поэтому
    выбранные строки остаются локально; повторный API получает закрытый словарь.
    """
    projected = []
    for row in history[:2]:
        question, answer = row.get('question', ''), row.get('answer', '')
        qfacets, afacets = dialog_facets(question), dialog_facets(answer)
        if 'Бюджет продвижения' in qfacets:
            budget = _budget_bucket(answer)
            if budget is not None:
                afacets.append(budget)
        projected.append({'question_topics': business_summary(question).split('; ') if business_summary(question) else [],
                          'question_facets': qfacets,
                          'answer_topics': business_summary(answer).split('; ') if business_summary(answer) else [],
                          'answer_facets': afacets,
                          'preferred_format': row.get('preferred_format', '') if row.get('preferred_format') in KINDS else ''})
    return projected


def cloud_payload(task, items):
    """Отделяет локальные ID и тексты от экспортируемого ограниченного представления."""
    task_summary = business_summary(task)
    if not task_summary:
        return None, {}, 'no_safe_context'
    candidates, mapping = [], {}
    for item in items[:MAX_CANDIDATES]:
        if (not isinstance(item, dict) or item.get('llm_public_metadata') is not True
                or not isinstance(item.get('kind'), str) or item['kind'] not in KINDS):
            continue
        # Только название и исходные метки публичной карточки. Описание/тело
        # материала не читаем даже для локального вывода облачной сводки.
        topics = item.get('topics', [])
        if not isinstance(topics, list):
            continue
        text = ' '.join([str(item.get('title', ''))] + [t for t in topics if isinstance(t, str)])
        summary = business_summary(text)
        if not summary or not isinstance(item.get('id'), str):
            continue
        alias = 'c' + str(len(candidates) + 1)
        candidate = {'id': alias, 'kind': item['kind'], 'summary': summary}
        candidates.append(candidate)
        mapping[alias] = (item['id'], candidate)
    if not candidates:
        return None, {}, 'privacy_blocked'
    return {'task': task_summary, 'catalog': candidates}, mapping, 'ready'
