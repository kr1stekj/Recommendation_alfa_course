"""Чтение дампа и очистка каталога: SQL извлекается как текст и не исполняется."""
import collections
import html
import io
from html.parser import HTMLParser
import re
import subprocess
import unicodedata
from pathlib import Path
from .missing import is_missing, integer_or_none, clean_tables

# Идентификаторы взяты из дампа: одна и та же сущность имеет тип и собственный UUID.
PUBLISHED = "61f12dbc-3cd4-4c14-b64b-1709548b904e"
# Общий enum доступа из CourseTypes; Q&A-2 относит его также к Articles.TypeId.
PUBLIC_TYPE = "ed085df2-b9bf-43fc-9a29-e2140346e333"
ACCESS_TYPES = {PUBLIC_TYPE: "public", "1069fd62-8920-417b-b6c1-1bfb7c3f57fb": "subscription",
                "9cda1148-8db6-4d29-98fc-6d67ea538fff": "paid", "50c186e6-14c2-4446-81c5-5263fd5b3362": "private"}
ENTITY_TYPES = {
    "8b440b1a-00c3-423a-9bae-3376475eb203": "course",
    "8ede4179-0faf-407d-a4fc-1d94f9a63089": "article",
    "da1ad865-8605-4ab1-9f6c-4489a26fbc35": "document",
    "142c23e2-bf71-4a73-b12e-6b1b5e4d47cc": "podcast",
}
COPY = re.compile(r'COPY public\."?(\w+)"? \((.*?)\) FROM stdin;')
ESCAPE = re.compile(r"\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)")
ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}


def unescape(value):
    """Декодирует одно поле COPY, маркер NULL отличается от буквальной строки \\N."""
    if value == r"\N":
        return None
    def sub(m):
        token = m.group(1)
        if token[0] in "01234567":
            return chr(int(token, 8))
        if token.startswith("x") and len(token) > 1:
            return chr(int(token[1:], 16))
        return ESCAPES.get(token, token)
    return ESCAPE.sub(sub, value)


def parse_copy(lines):
    """Собирает таблицы вида {имя: [строки-словари]} из секций COPY дампа."""
    tables = collections.defaultdict(list)
    table, columns = None, []
    for line in lines:
        m = COPY.match(line)
        if m:
            table = m[1]
            columns = [s.strip(' "') for s in m[2].split(",")]
        elif line.rstrip("\r\n") == r"\.":
            table = None
        elif table:
            values = line.rstrip("\r\n").split("\t")
            if len(values) != len(columns):
                raise ValueError("Invalid COPY row in " + table)
            tables[table].append(dict(zip(columns, map(unescape, values))))
    return dict(tables)


def load_backup(path):
    """Извлекает данные через pg_restore без подключения к серверу PostgreSQL."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".sql":
        with path.open(encoding="utf-8") as stream:
            return parse_copy(stream)
    proc = subprocess.Popen(
        ["pg_restore", "--data-only", "--no-owner", "--no-privileges", "--file=-", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    # communicate читает stdout и stderr вместе, чтобы процесс не завис на полном буфере.
    # Весь дамп помещается в память: это допущение MVP для предоставленного файла.
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("pg_restore failed: " + err[:2000])
    return parse_copy(io.StringIO(out))


class TextParser(HTMLParser):
    """Оставляет текст HTML, пропуская код и стили, которые мешают поиску по смыслу."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def repair_homoglyphs(value):
    """Исправляет похожие латинские буквы в кириллических словах, сохраняя IT/CRM."""
    latin, cyrillic = "aceopxykmtbh", "асеорхукмтвн"
    trans = str.maketrans(latin + latin.upper(), cyrillic + cyrillic.upper())
    return re.sub(r"[\w]+", lambda m: m[0].translate(trans) if re.search("[а-яА-ЯёЁ]", m[0]) else m[0], value)


def clean_text(value):
    """Убирает HTML, повторные пробелы и различия Unicode перед векторизацией."""
    if is_missing(value):
        return ""
    p = TextParser()
    p.feed(str(value))
    return repair_homoglyphs(" ".join(unicodedata.normalize("NFKC", html.unescape(" ".join(p.parts))).split()))


def canonical_label(value):
    """Создаёт ключ сравнения меток: регистр и различие е/ё не влияют на совпадение."""
    value = clean_text(value).casefold().replace("ё", "е")
    return value


def is_live(row):
    return row.get("IsDeleted") != "t"


def make_catalog(tables, as_of, include_closed_history=False):
    """Возвращает очищенные карточки и отчёт об отсеянных данных.

    as_of — дата в формате YYYY-MM-DD. Закрытые карточки можно оставить для
    понимания истории; флаг eligible всё равно запрещает их рекомендовать.
    """
    tables = clean_tables(tables)
    dictionaries, alias_maps = {}, {}
    for table in ("Topics", "Tags", "Industries"):
        rows = [r for r in tables.get(table, []) if r.get("Id")]
        canonical = {canonical_label(r["Title"]): clean_text(r["Title"]) for r in rows if is_live(r) and clean_text(r.get("Title"))}
        alias_maps[table] = {r["Id"]: canonical.get(canonical_label(r["Title"])) for r in rows}
        dictionaries[table] = sorted(canonical.values())
    # Связь полиморфная: соединять только по UUID без типа материала нельзя.
    settings = {(r["EntityId"], ENTITY_TYPES.get(r["EntityTypeId"])): r for r in tables.get("ContentSettings", [])
                if is_live(r) and r.get("Id") and r.get("EntityId") and r.get("EntityTypeId") in ENTITY_TYPES}
    bindings = {}
    for table, suffix in (("Topics", "Topic"), ("Tags", "Tag"), ("Industries", "Industry")):
        by_setting = collections.defaultdict(set)
        for row in tables.get("ContentSettings" + table, []):
            if not row.get("ContentSettingsId") or not row.get(suffix + "Id"):
                continue
            title = alias_maps[table].get(row[suffix + "Id"])
            if title:
                by_setting[row["ContentSettingsId"]].add(title)
        bindings[table] = by_setting
    subjects = {r["Id"]: clean_text(r["Title"]) for r in tables.get("ArticleSubjects", []) if is_live(r) and r.get("Id")}
    article_blocks = collections.defaultdict(list)
    for r in tables.get("ArticleEmbedItems", []):
        article_blocks[r["ArticleId"]].append(r)
    catalog, rejected = [], collections.Counter()
    invalid_costs = 0
    for table, kind in (("Courses", "course"), ("Articles", "article"), ("Documents", "document"), ("Podcasts", "podcast")):
        for row in tables.get(table, []):
            cs = settings.get((row["Id"], kind))
            reason = None
            if not row.get("Id"): reason = "no_id"
            elif not is_live(row): reason = "deleted"
            elif row.get("StatusId") != PUBLISHED: reason = "unpublished"
            elif cs and cs.get("StatusId") != PUBLISHED: reason = "unpublished_settings"
            elif (row.get("IsClosed") == "t" or (kind != "document" and ACCESS_TYPES.get(row.get("TypeId")) == "private")) and not include_closed_history: reason = "closed"
            elif row.get("Published") and row["Published"][:10] > as_of: reason = "future"
            elif not clean_text(row.get("Title")): reason = "empty_title"
            elif not row.get("Url"): reason = "no_url"
            if reason:
                rejected[table + ":" + reason] += 1
                continue
            csid = cs["Id"] if cs else ""
            labels = {key.lower(): sorted(bindings[key].get(csid, [])) for key in bindings}
            if kind == "article" and not labels["topics"]:
                subject = subjects.get(row.get("SubjectId"))
                if subject: labels["topics"] = [subject]
            # По разъяснению организаторов NULL означает бесплатно только у Courses.Cost.
            cost = row.get("Cost")
            price = integer_or_none(cost)
            if price is not None and price < 0:
                price = None
            if cost is None and kind == "course":
                price = 0
            elif cost is not None and price is None:
                invalid_costs += 1
            title = clean_text(row["Title"])
            description = clean_text(row.get("Description"))[:4000]
            # Полный текст берём только у явно публичных статей; тексты уроков не читаем.
            public_body = ""
            if kind == "article" and row.get("TypeId") == PUBLIC_TYPE and row.get("IsClosed") != "t":
                blocks = sorted(article_blocks.get(row["Id"], []), key=lambda r: integer_or_none(r.get("Order")) or 0)
                public_body = clean_text(row.get("Content")) + " " + " ".join(clean_text(r.get("HtmlContent")) for r in blocks)
            # Двойное название усиливает его вклад в TF-IDF. Начало текста также
            # приоритетно для нейросети, которая обрезает длинный вход по токенам.
            text = " ".join([title, title, description, " ".join(labels["topics"]), " ".join(labels["tags"]), public_body[:10000]])
            catalog.append(dict(id=kind + ":" + row["Id"], entity_id=row["Id"], kind=kind,
                                title=title, description=description, text=text.strip(),
                                eligible=(row.get("IsClosed") != "t" and (kind == "document" or ACCESS_TYPES.get(row.get("TypeId")) != "private")),
                                url=row["Url"], price=price, published=row.get("Published"),
                                duration=clean_text(row.get("Duration")), target=clean_text(row.get("Target")),
                                lesson_quantity=integer_or_none(row.get("LessonQuantity")),
                                access=ACCESS_TYPES.get(row.get("TypeId"), "unknown") if kind != "document" else "unknown",
                                llm_public_metadata=(row.get("IsClosed") != "t" and
                                                     (kind == "document" or ACCESS_TYPES.get(row.get("TypeId")) in {"public", "paid", "subscription"})),
                                **labels))
    catalog.sort(key=lambda r: r["id"])
    audit = {
        "as_of": as_of, "table_rows": {k: len(v) for k, v in tables.items()},
        "indexed_items": len(catalog), "catalog_items": sum(r["eligible"] for r in catalog),
        "catalog_kinds": dict(collections.Counter(r["kind"] for r in catalog if r["eligible"])),
        "rejected": dict(rejected), "dictionaries": dictionaries,
        "invalid_costs": invalid_costs,
        "ratings_without_user_id": sum(r.get("UserId") is None for r in tables.get("UserRatings", [])),
        "course_buying_date_missing": sum(r.get("BuyingDate") is None for r in tables.get("CourseUserBindings", [])),
        "notes": ["Landings are not joined to Courses by title; direct Courses IDs preserve history.",
                  "Industries mixes sectors, experience labels and editorial collections.",
                  "Ratings without UserId cannot form a persistent anonymous profile.",
                  "No page-view or recommendation-impression log is present."]}
    return catalog, audit


def historical_interactions(tables, catalog):
    """Превращает записи на курсы и завершённые уроки в признаки интереса к курсам."""
    tables = clean_tables(tables)
    valid = {r["id"] for r in catalog}
    lessons = collections.defaultdict(list)
    for lesson in tables.get("CourseUserLessonBindings", []):
        if lesson.get("CourseUserId") and lesson.get("Completed"):
            lessons[lesson["CourseUserId"]].append(lesson["Completed"])
    interactions = []
    for row in tables.get("CourseUserBindings", []):
        if not row.get("Id") or not row.get("CourseId"):
            continue
        item = "course:" + row["CourseId"]
        if not row.get("UserId") or not is_live(row) or item not in valid:
            continue
        completed = row.get("IsCompleted") == "t"
        started = row.get("StudyingStartDate") is not None
        completed_lessons = lessons.get(row["Id"], [])
        # Завершённый курс — самый сильный сигнал. Число уроков усиливает интерес
        # с замедлением и ограничением, чтобы длинные курсы не подавляли остальные.
        weight = 6.0 if completed else (min(5.0, 2.0 + len(completed_lessons) ** 0.5) if completed_lessons else (3.0 if started else 1.0))
        timestamps = [t for t in [row.get("Completed"), row.get("StudyingStartDate"), row.get("Created")] + completed_lessons if t]
        interactions.append(dict(user_id=row["UserId"], item_id=item,
                                 weight=weight,
                                 completed=completed, owned=row.get("ContentAccess") in {"1", "2"},
                                 at=max(timestamps) if timestamps else None))
    # Оценки пока только аудируем: шкалы разных типов материалов нужно сопоставить отдельно.
    return interactions
