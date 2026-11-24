"""Перенос числового индекса: модель и БД не вызываются в изолированных fixtures."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reco.build import finalize_catalog
from reco.data import make_catalog, historical_interactions, PUBLISHED
from reco.embeddings import Encoder, MODEL, LEGACY_MODEL, MODEL_SPECS
from reco.index import fingerprint, write_manifest, resolve_index, validate_index
from reco.precomputed import export_bundle, restore_bundle, load_bundle, sha256_file
from reco.model_cache import ensure_model_files


class PrecomputedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root/'source'; self.source.mkdir()
        self.bundle = self.root/'bundle'; self.output = self.root/'artifacts'
        self.backup = self.root/'fixture.backup'; self.backup.write_bytes(b'synthetic backup fixture')
        self.tables = {'Topics': [{'Id':'t','Title':'Налоги'}], 'Courses': [
            {'Id':str(i),'Title':'PRIVATE CATALOG TEXT '+str(i),'StatusId':PUBLISHED,'Url':'/course/'+str(i)}
            for i in range(3)], 'CourseUserBindings': [
                {'Id':'binding','CourseId':'0','UserId':'PRIVATE USER UUID','ContentAccess':'1'}]}
        self.config = Encoder({'backend':'semantic','model':MODEL}).config
        catalog, audit = make_catalog(self.tables, '2026-11-15', include_closed_history=True)
        history = historical_interactions(self.tables,catalog)
        finalize_catalog(catalog,history,[{'title':'Налоги','cosine':0.5,'margin':1.0}]*3)
        (self.source/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False))
        (self.source/'encoder.json').write_text(json.dumps(self.config,ensure_ascii=False))
        (self.source/'topics.json').write_text(json.dumps(['Налоги'],ensure_ascii=False))
        vectors = np.zeros((3,1024),dtype=np.float32); vectors[:,0]=1
        np.save(self.source/'vectors.npy',vectors); np.save(self.source/'topic_vectors.npy',vectors[:1])
        audit.update(catalog_fingerprint=fingerprint(self.source),embedding_device='mps',embedding_backend='semantic')
        (self.source/'audit.json').write_text(json.dumps(audit)); write_manifest(self.source)
        for name in ('fit','encode'):
            guard=patch.object(Encoder,name,side_effect=AssertionError('Embedding must not run'))
            guard.start(); self.addCleanup(guard.stop)
        source=patch('reco.precomputed.load_backup',side_effect=lambda _:copy.deepcopy(self.tables))
        source.start(); self.addCleanup(source.stop)
        self.database=MagicMock(); self.database.has_dataset.return_value=True
        db=patch('reco.precomputed.Database',return_value=self.database)
        db.start(); self.addCleanup(db.stop)

    def export(self): return export_bundle(self.backup,self.source,self.bundle)

    def restore(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return restore_bundle(self.backup,self.bundle,self.output)

    def update_manifest(self,callback):
        path=self.bundle/'manifest.json'; data=json.loads(path.read_text()); callback(data)
        path.write_text(json.dumps(data))

    def test_roundtrip_is_byte_identical_without_model_or_private_export(self):
        manifest=self.export(); self.restore(); active=resolve_index(self.output)
        self.assertEqual(set(p.name for p in self.bundle.iterdir()),
                         {'encoder.json','vectors.npy','topic_vectors.npy','manifest.json'})
        for name in ('catalog.json','encoder.json','topics.json','vectors.npy','topic_vectors.npy'):
            self.assertEqual((active/name).read_bytes(),(self.source/name).read_bytes())
        self.assertEqual(validate_index(active)['catalog_fingerprint'],manifest['catalog_fingerprint'])
        payload=(self.bundle/'manifest.json').read_text()+(self.bundle/'encoder.json').read_text()
        for private in ('PRIVATE CATALOG','PRIVATE USER','course:0','PRIVATE','Налоги'):
            self.assertNotIn(private,payload)
        self.assertEqual(self.database.replace_history.call_args.args[0],manifest['catalog_fingerprint'])
        self.assertEqual(len(self.database.replace_history.call_args.args[1]),1)

    def test_export_does_not_overwrite_existing_destination(self):
        self.export()
        with self.assertRaises(FileExistsError): self.export()

    def test_wrong_backup_and_changed_catalog_fail_before_database_or_publication(self):
        self.export(); self.backup.write_bytes(b'other source')
        with self.assertRaisesRegex(ValueError,'Backup SHA256'): self.restore()
        self.database.assert_not_called(); self.assertFalse(self.output.exists())
        self.backup.write_bytes(b'synthetic backup fixture')
        self.tables['Courses'][0]['Title']='Changed material'
        with self.assertRaisesRegex(ValueError,'Catalog or row order'): self.restore()
        self.database.assert_not_called()

    def test_checksum_and_nonfinite_matrix_fail_even_with_updated_digest(self):
        self.export(); path=self.bundle/'vectors.npy'; path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'): self.restore()
        vectors=np.zeros((3,1024),dtype=np.float32); vectors[0,0]=np.nan
        np.save(path,vectors)
        self.update_manifest(lambda m:m['files'].update({'vectors.npy':sha256_file(path)}))
        with self.assertRaisesRegex(ValueError,'NaN'): self.restore()
        self.database.assert_not_called()

    def test_model_revision_shape_and_annotation_mismatches_rejected(self):
        self.export()
        with self.assertRaisesRegex(ValueError,'model/max_length'): load_bundle(self.bundle,LEGACY_MODEL,512)
        with self.assertRaisesRegex(ValueError,'model/max_length'): load_bundle(self.bundle,MODEL,1024)
        self.update_manifest(lambda m:m.update(topic_annotations=[None]*3))
        with self.assertRaisesRegex(ValueError,'topic annotation'): self.restore()
        config_path=self.bundle/'encoder.json'; config=dict(self.config,revision='0'*40)
        config_path.write_text(json.dumps(config))
        self.update_manifest(lambda m:m['files'].update({'encoder.json':sha256_file(config_path)}))
        with self.assertRaisesRegex(ValueError,'revision'): self.restore()

    def test_matrix_rows_cannot_be_reassigned_to_another_catalog(self):
        self.export(); path=self.bundle/'vectors.npy'
        vectors=np.zeros((2,1024),dtype=np.float32); vectors[:,0]=1; np.save(path,vectors)
        self.update_manifest(lambda m:(m.update(vector_shape=[2,1024]),
            m['files'].update({'vectors.npy':sha256_file(path)})))
        with self.assertRaisesRegex(ValueError,'row count'): self.restore()
        self.database.assert_not_called()

    def test_database_failure_preserves_previous_generation(self):
        self.export(); self.restore(); previous=resolve_index(self.output)
        self.database.replace_history.side_effect=RuntimeError('fixture PostgreSQL failure')
        with self.assertRaisesRegex(RuntimeError,'PostgreSQL'): self.restore()
        self.assertEqual(resolve_index(self.output),previous)
        self.assertEqual(len(list((self.output/'.generations').iterdir())),1)
        validate_index(previous)

    def run_start(self,env):
        script=Path(__file__).resolve().parents[1]/'scripts/start.py'
        before=Path.cwd()
        try:
            os.chdir(self.root)
            with patch.dict(os.environ,{'EMBEDDING_BACKEND':'semantic','EMBEDDING_MODEL':MODEL,
                'EMBEDDING_MAX_LENGTH':'512','BACKUP_PATH':str(self.backup),**env}), patch('reco.database.Database',return_value=self.database), \
                patch('reco.model_cache.ensure_model_files') as cache, patch('reco.server.serve') as serve, \
                contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script),run_name='__main__')
                serve.assert_called_once_with(host='0.0.0.0'); cache.assert_called_once()
        finally: os.chdir(before)

    def test_startup_uses_shipped_vectors_instead_of_build(self):
        self.export(); shipped=self.root/'precomputed/bge-m3'; shipped.parent.mkdir()
        shutil.copytree(self.bundle,shipped)
        with patch('reco.build.build',side_effect=AssertionError('Rebuild forbidden')):
            self.run_start({'USE_PRECOMPUTED_EMBEDDINGS':'1'})
        self.assertEqual(validate_index(resolve_index(self.output))['embedding_origin'],'precomputed')

    def test_startup_opt_out_selects_explicit_build_and_missing_bundle_never_rebuilds(self):
        with patch('reco.build.build',side_effect=AssertionError('Rebuild forbidden')) as build:
            with self.assertRaises(FileNotFoundError): self.run_start({'USE_PRECOMPUTED_EMBEDDINGS':'1'})
            build.assert_not_called()
        with patch('reco.build.build',side_effect=lambda *a,**k:shutil.copytree(self.source,self.output)) as build:
            self.run_start({'USE_PRECOMPUTED_EMBEDDINGS':'0'})
            build.assert_called_once()


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name).resolve()
        env=patch.dict(os.environ,{'HF_HOME':str(self.home),'HF_HUB_OFFLINE':'1'})
        env.start(); self.addCleanup(env.stop)
        self.config=Encoder({'backend':'semantic','model':MODEL}).config
        self.snapshot=self.home/'hub/models--BAAI--bge-m3/snapshots'/MODEL_SPECS[MODEL]['revision']

    def populate(self):
        self.snapshot.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','tokenizer_config.json','tokenizer.json','pytorch_model.bin'):
            (self.snapshot/name).write_bytes(b'fixture')

    def test_ready_cache_never_uses_network_or_loads_model(self):
        self.populate()
        with patch('huggingface_hub.snapshot_download',side_effect=AssertionError('Network forbidden')), \
             patch.object(Encoder,'encode',side_effect=AssertionError('Model must not run')):
            self.assertEqual(ensure_model_files(self.config),self.snapshot)

    def test_download_is_pinned_public_and_never_infers_catalog(self):
        def fetched(**kwargs): self.populate(); return str(self.snapshot)
        with patch('huggingface_hub.snapshot_download',side_effect=fetched) as fetch, \
             patch.object(Encoder,'encode',side_effect=AssertionError('Model must not run')):
            ensure_model_files(self.config)
        self.assertIs(fetch.call_args.kwargs['token'],False)
        self.assertEqual(fetch.call_args.kwargs['revision'],MODEL_SPECS[MODEL]['revision'])
        self.assertTrue(fetch.call_args.kwargs['local_files_only'])
        self.assertIn('pytorch_model.bin',fetch.call_args.kwargs['allow_patterns'])

    def test_partial_offline_cache_fails_instead_of_serving_broken_semantic_search(self):
        self.snapshot.mkdir(parents=True)
        with patch('huggingface_hub.snapshot_download',return_value=str(self.snapshot)), \
             self.assertRaisesRegex(ValueError,'incomplete'):
            ensure_model_files(self.config)
