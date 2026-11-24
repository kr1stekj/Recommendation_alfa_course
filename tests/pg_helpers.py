"""Тесты используют только TEST_DATABASE_URL и собственные случайные пространства данных."""
import os
import unittest
import uuid

from reco.database import Database


class PostgresTestCase(unittest.TestCase):
    def setUp(self):
        dsn = os.getenv('TEST_DATABASE_URL')
        if not dsn:
            self.skipTest('Нужна отдельная тестовая PostgreSQL: TEST_DATABASE_URL')
        self.database = Database(dsn)
        self.database.initialize()
        self.dataset = 'test-' + uuid.uuid4().hex
        self.addCleanup(self.clean_database)

    def clean_database(self):
        # Удаляем только данные этого теста; никаких DROP/TRUNCATE рабочей схемы.
        with self.database.connect() as db:
            for table in ('profiles','history'):
                db.execute('DELETE FROM reco.'+table+' WHERE dataset LIKE %s',(self.dataset+'%',))
            db.execute('DELETE FROM reco.datasets WHERE fingerprint LIKE %s',(self.dataset+'%',))
