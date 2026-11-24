"""Обработка отказов сборки и проверка неизменяемых поколений индекса."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from reco.build import build
from reco.data import PUBLISHED
from reco.engine import Recommender
from reco.index import resolve_index, validate_index


class IndexPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'artifacts'
        self.database = MagicMock()
        self.database.has_dataset.return_value = True
        self.patch_db = patch('reco.build.Database', return_value=self.database)
        self.patch_db.start()
        self.addCleanup(self.patch_db.stop)
        self.tables = {'Courses': [dict(Id='one', Title='Старый бизнес курс',
                                      StatusId=PUBLISHED, Url='/course', IsDeleted='f')]}
        self.patch_data = patch('reco.build.load_backup', side_effect=lambda _: self.tables)
        self.patch_data.start()
        self.addCleanup(self.patch_data.stop)

    def build(self):
        # Мокируется источник и отказ БД; файловая сборка и потребитель настоящие.
        with contextlib.redirect_stdout(io.StringIO()):
            return build('synthetic.backup',self.root,backend='tfidf',as_of='2026-11-13')

    def test_failed_import_preserves_published_generation(self):
        self.build()
        previous = resolve_index(self.root)
        old_catalog = (previous/'catalog.json').read_bytes()
        self.tables['Courses'][0]['Title'] = 'Новый бизнес курс'
        self.database.replace_history.side_effect = RuntimeError('simulated PostgreSQL failure')
        with self.assertRaisesRegex(RuntimeError,'PostgreSQL failure'):
            self.build()
        self.assertEqual(resolve_index(self.root),previous)
        self.assertEqual((previous/'catalog.json').read_bytes(),old_catalog)
        self.assertEqual(len(list((self.root/'.generations').iterdir())),1)
        engine = Recommender(self.root,database=self.database)
        self.assertEqual(engine.catalog[0]['title'],'Старый бизнес курс')

    def test_failed_first_build_publishes_nothing(self):
        self.database.replace_history.side_effect = RuntimeError('simulated failure')
        with self.assertRaises(RuntimeError):
            self.build()
        self.assertFalse((self.root/'current').exists())
        self.assertFalse((self.root/'audit.json').exists())
        self.assertEqual(list((self.root/'.generations').iterdir()),[])

    def test_successful_rebuild_publishes_complete_new_generation(self):
        self.build()
        old = resolve_index(self.root)
        self.tables['Courses'][0]['Title'] = 'Новый бизнес курс'
        self.build()
        new = resolve_index(self.root)
        self.assertNotEqual(new,old)
        validate_index(new)
        self.assertEqual(json.loads((old/'catalog.json').read_text())[0]['title'],'Старый бизнес курс')
        self.assertEqual(Recommender(self.root,database=self.database).catalog[0]['title'],'Новый бизнес курс')

    def test_migration_reads_active_generation(self):
        from reco.migrate import import_history
        audit = self.build()
        with patch('reco.migrate.Database',return_value=self.database), \
             patch('reco.migrate.load_backup',return_value=self.tables), \
             contextlib.redirect_stdout(io.StringIO()):
            import_history('synthetic.backup',self.root)
        self.database.replace_history.assert_called_with(audit['catalog_fingerprint'],[])

    def test_startup_reuses_published_generation(self):
        self.build()
        script = Path(__file__).resolve().parents[1]/'scripts/start.py'
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root.parent)
            with patch.dict('os.environ',{'EMBEDDING_BACKEND':'tfidf'}), \
                 patch('reco.build.build',side_effect=AssertionError('Повторная сборка не нужна')), \
                 patch('reco.database.Database',return_value=self.database), \
                 patch('reco.server.serve') as serve:
                runpy.run_path(str(script),run_name='__main__')
            serve.assert_called_once_with(host='0.0.0.0')
        finally:
            os.chdir(previous_cwd)

    def test_changed_vectors_are_rejected_by_manifest(self):
        self.build()
        generation = resolve_index(self.root)
        (generation/'vectors.npy').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'vectors.npy'):
            Recommender(self.root,database=self.database)

    def test_startup_rejects_different_model_or_context_length(self):
        from types import SimpleNamespace
        from reco.embeddings import MODEL, LEGACY_MODEL
        self.build()
        script = Path(__file__).resolve().parents[1]/'scripts/start.py'
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root.parent)
            for model, length in ((LEGACY_MODEL, 512), (MODEL, 1024)):
                with self.subTest(model=model, length=length), \
                     patch.dict('os.environ', {'EMBEDDING_BACKEND': 'semantic', 'EMBEDDING_MODEL': MODEL, 'EMBEDDING_MAX_LENGTH': '512'}), \
                     patch('reco.index.validate_index', return_value={'embedding_backend': 'semantic'}), \
                     patch('reco.embeddings.Encoder.load', return_value=SimpleNamespace(config={'model': model, 'max_length': length})), \
                     patch('reco.server.serve') as serve, \
                     self.assertRaisesRegex(SystemExit, 'different embedding model/max_length'):
                    runpy.run_path(str(script), run_name='__main__')
                serve.assert_not_called()
        finally:
            os.chdir(previous_cwd)

    def test_missing_generation_manifest_is_rejected(self):
        self.build()
        (resolve_index(self.root)/'manifest.json').unlink()
        with self.assertRaisesRegex(ValueError,'manifest'):
            Recommender(self.root,database=self.database)

    def test_nonfinite_vectors_are_rejected_even_with_valid_checksum(self):
        import numpy as np
        from reco.index import write_manifest
        self.build()
        generation = resolve_index(self.root)
        for name in ('vectors.npy', 'topic_vectors.npy'):
            with self.subTest(name=name):
                original = (generation/name).read_bytes()
                np.save(generation/name, np.array([[float('nan')]], dtype=np.float32))
                write_manifest(generation)
                with self.assertRaisesRegex(ValueError, 'NaN/Infinity'):
                    validate_index(generation)
                (generation/name).write_bytes(original)
                write_manifest(generation)

    def test_legacy_index_without_topics_remains_supported(self):
        import shutil
        self.build()
        legacy = self.root.parent/'legacy'
        legacy.mkdir()
        for name in ('catalog.json', 'encoder.json', 'vectors.npy', 'audit.json'):
            shutil.copyfile(resolve_index(self.root)/name, legacy/name)
        self.assertEqual(validate_index(legacy)['catalog_fingerprint'],
                         validate_index(resolve_index(self.root))['catalog_fingerprint'])
        (legacy/'topics.json').write_text(json.dumps(['Налоги']))
        with self.assertRaisesRegex(ValueError, 'Отсутствуют векторы'):
            validate_index(legacy)

    def test_pointer_publication_failure_preserves_old_index(self):
        self.build()
        previous = resolve_index(self.root)
        self.tables['Courses'][0]['Title'] = 'Новый бизнес курс'
        with patch('reco.index.os.replace',side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError,'publication failed'):
                self.build()
        self.assertEqual(resolve_index(self.root),previous)
        validate_index(previous)
        self.assertEqual(len(list((self.root/'.generations').iterdir())),1)

    def test_mismatched_legacy_audit_is_rejected(self):
        self.build()
        generation = resolve_index(self.root)
        # Проверка плоского формата не зависит от наличия manifest.
        (generation/'manifest.json').unlink()
        catalog = json.loads((generation/'catalog.json').read_text())
        catalog[0]['title'] = 'Несогласованная карточка'
        (generation/'catalog.json').write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError,'Fingerprint'):
            Recommender(generation,database=self.database)
