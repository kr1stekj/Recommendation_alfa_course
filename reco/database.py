"""Единственная БД приложения — PostgreSQL; исходные таблицы public не меняются."""
import os
import time

import psycopg
from psycopg.rows import dict_row


class Database:
    def __init__(self, dsn=None):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError("Задайте DATABASE_URL для подключения к PostgreSQL")

    def connect(self):
        # Контекст соединения фиксирует успешную транзакцию и откатывает ошибку.
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=10)

    def initialize(self):
        with self.connect() as db:
            # Сериализуем первый запуск нескольких экземпляров приложения.
            db.execute("SELECT pg_advisory_xact_lock(731950021)")
            db.execute("""
                CREATE SCHEMA IF NOT EXISTS reco;
                CREATE TABLE IF NOT EXISTS reco.datasets (
                    fingerprint text PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS reco.history (
                    dataset text NOT NULL, user_id text NOT NULL,
                    item_id text NOT NULL, weight double precision NOT NULL,
                    completed boolean NOT NULL, owned boolean NOT NULL, at text
                );
                CREATE INDEX IF NOT EXISTS history_user ON reco.history(dataset,user_id);
                CREATE TABLE IF NOT EXISTS reco.profiles (
                    dataset text NOT NULL, id text NOT NULL,
                    kind text NOT NULL CHECK (kind IN ('anonymous','registered')),
                    created double precision NOT NULL, updated double precision NOT NULL,
                    expires double precision, payload jsonb NOT NULL,
                    PRIMARY KEY (dataset,id)
                );
                CREATE INDEX IF NOT EXISTS profiles_expiry ON reco.profiles(dataset,expires);
                CREATE INDEX IF NOT EXISTS profiles_expiry_all ON reco.profiles(expires) WHERE expires IS NOT NULL;
                CREATE INDEX IF NOT EXISTS profiles_kind_updated ON reco.profiles(dataset,kind,updated);
            """)

    @staticmethod
    def lock(db, dataset):
        # В MVP изменения одного индекса сериализованы. Блокировка охватывает
        # и ещё не созданные профили, поэтому параллельные гости не обходят лимит.
        db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (dataset,))

    def has_dataset(self, dataset):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM reco.datasets WHERE fingerprint=%s", (dataset,)).fetchone() is not None

    def replace_history(self, dataset, history):
        """Атомарно заменяет историю одного индекса; другие индексы не затрагивает."""
        with self.connect() as db:
            self.lock(db, dataset)
            db.execute("DELETE FROM reco.history WHERE dataset=%s", (dataset,))
            with db.cursor() as cur:
                cur.executemany("INSERT INTO reco.history VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [(dataset, r['user_id'], r['item_id'], r['weight'], bool(r['completed']), bool(r['owned']), r['at']) for r in history])
            db.execute("INSERT INTO reco.datasets VALUES (%s) ON CONFLICT DO NOTHING", (dataset,))

    def history(self, dataset, user_id):
        with self.connect() as db:
            return db.execute("SELECT user_id,item_id,weight,completed,owned,at FROM reco.history WHERE dataset=%s AND user_id=%s", (dataset,user_id)).fetchall()

    def cleanup_expired(self):
        """Чистит также гостей старых версий индекса, которые уже не обслуживаются."""
        with self.connect() as db:
            # Граница времени — параметр, чтобы PostgreSQL мог использовать индекс expires.
            return db.execute("DELETE FROM reco.profiles WHERE expires IS NOT NULL AND expires <= %s", (time.time(),)).rowcount
