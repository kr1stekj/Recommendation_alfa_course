"""Публичные тестовые роли; стартовые события строятся без оценочной разметки."""
import secrets

from .data import canonical_label

DEMO_PREFIX = 'anon:demo_'
ACCOUNTS = (
    {'id': 'new', 'title': 'Новый пользователь', 'description': 'Без истории. Проверка холодного старта.', 'topics': []},
    {'id': 'start', 'title': 'Начинающий предприниматель', 'description': 'Первые шаги и регистрация своего дела.',
     'topics': ['Первые шаги', 'Регистрация бизнеса']},
    {'id': 'marketing', 'title': 'Маркетинг и продажи', 'description': 'Продвижение и привлечение клиентов.',
     'topics': ['Маркетинг', 'Продажи', 'Продажи и переговоры']},
    {'id': 'taxes', 'title': 'Налоги и маркетплейсы', 'description': 'Налоги, документы и торговля на площадках.',
     'topics': ['Налоги', 'Налоги и законы', 'Маркетплейсы']},
    {'id': 'team', 'title': 'Управление командой', 'description': 'Найм и организация работы сотрудников.',
     'topics': ['Управление командой', 'Команда']},
    {'id': 'finance', 'title': 'Финансы бизнеса', 'description': 'Учёт, доходы, расходы и финансовое планирование.',
     'topics': ['Финансы', 'Финансы и учет']},
)


class DemoProfiles:
    def __init__(self, engine):
        self.engine = engine
        self.accounts = {row['id']: row for row in ACCOUNTS}

    def seed_items(self, account_id):
        account = self.accounts[account_id]
        topics = {canonical_label(topic) for topic in account['topics']}
        candidates = sorted((row for row in self.engine.catalog if row['eligible'] and
            topics & {canonical_label(topic) for topic in row['topics']}), key=lambda row: row['id'])
        # По возможности показываем разные форматы, сохраняя детерминированность.
        selected, kinds = [], set()
        for row in candidates:
            if row['kind'] not in kinds:
                selected.append(row); kinds.add(row['kind'])
                if len(selected) == 3:
                    return selected
        return (selected + [row for row in candidates if row not in selected])[:3]

    def list_accounts(self):
        return [dict(row, seed_items=len(self.seed_items(row['id']))) for row in ACCOUNTS]

    def reset(self, profile_id, account_id, now):
        items = self.seed_items(account_id)
        state = {'items': {row['id']: {'weight': 6.0 if index == 2 else 3.0,
                 'completed': index == 2, 'at': now} for index, row in enumerate(items)},
                 'event_ids': []}
        self.engine.store.mutate(profile_id, 'anonymous', lambda previous: state, now=now)
        return len(items)

    def create(self, account_id, now):
        profile_id = DEMO_PREFIX + secrets.token_urlsafe(24)
        return profile_id, self.reset(profile_id, account_id, now)
