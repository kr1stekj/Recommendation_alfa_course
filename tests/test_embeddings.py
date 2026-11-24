"""Контракт BGE и legacy encoder без загрузки больших весов в unit-тестах."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from reco.embeddings import Encoder, MODEL, LEGACY_MODEL, MODEL_SPECS


class SemanticEncoderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'EMBEDDING_DEVICE': 'cpu'})
        env.start()
        self.addCleanup(env.stop)

    def test_defaults_and_legacy_configuration_are_explicit(self):
        bge = Encoder()
        self.assertEqual(bge.config['model'], MODEL)
        self.assertEqual(bge.config['revision'], MODEL_SPECS[MODEL]['revision'])
        self.assertEqual(bge.config['batch_size'], 4)
        self.assertEqual(bge.config['max_length'], 512)
        legacy = Encoder({'backend': 'semantic', 'model': LEGACY_MODEL})
        self.assertEqual(legacy.config['batch_size'], 16)
        self.assertEqual(legacy.config['pooling'], 'cls')

    def test_bad_model_pooling_and_resource_limits_rejected(self):
        for values in ({'model': 'intfloat/multilingual-e5-large'}, {'pooling': 'mean'},
                       {'max_length': 8193}, {'max_length': True}, {'max_length': 0},
                       {'batch_size': 0}, {'batch_size': 100}, {'batch_size': '4'}):
            with self.subTest(config=values), self.assertRaises(ValueError):
                Encoder({'backend': 'semantic', **values})
        with self.assertRaises(ValueError):
            Encoder({'backend': 'semantic', 'model': LEGACY_MODEL, 'max_length': 1024})
        self.assertEqual(Encoder({'backend': 'semantic', 'max_length': 8192}).config['max_length'], 8192)
        with self.assertRaises(ValueError):
            Encoder(device='unknown-gpu')

    def test_unavailable_mps_does_not_silently_change_device(self):
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('transformers.AutoModel.from_pretrained') as loader:
            with self.assertRaisesRegex(ValueError, 'MPS недоступен'):
                Encoder(device='mps').encode(['Проверка'])
            loader.assert_not_called()

    def test_bge_batch_cls_normalization_and_safe_loading_contract(self):
        import torch
        model = MagicMock()
        model.eval.return_value = model
        model.config.hidden_size = 1024
        def forward(**kwargs):
            n = len(kwargs['input_ids'])
            output = torch.zeros((n, 2, 1024))
            output[:, 0, 0] = 3
            output[:, 0, 1] = 4
            output[:, 1, 2] = 99  # Если взять mean/последний токен, проверка упадёт.
            return SimpleNamespace(last_hidden_state=output)
        model.side_effect = forward
        tokenizer = MagicMock(side_effect=lambda texts, **kwargs: {'input_ids': torch.zeros((len(texts), 2), dtype=torch.long)})
        with patch('transformers.AutoModel.from_pretrained', return_value=model) as loader, \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
             patch('torch.backends.mps.is_available', side_effect=AssertionError('CPU не должен требовать MPS')):
            encoder = Encoder({'backend': 'semantic', 'batch_size': 2, 'max_length': 1024})
            vectors = encoder.encode(['a', 'b', 'c'])
        self.assertEqual(vectors.shape, (3, 1024))
        np.testing.assert_allclose(vectors[:, :2], [[.6, .8]] * 3)
        np.testing.assert_allclose(vectors[:, 2:], 0)
        self.assertTrue(np.isfinite(vectors).all())
        self.assertEqual([len(call.args[0]) for call in tokenizer.call_args_list], [2, 1])
        self.assertEqual(tokenizer.call_args.kwargs['max_length'], 1024)
        self.assertTrue(loader.call_args.kwargs['weights_only'])
        self.assertTrue(loader.call_args.kwargs['low_cpu_mem_usage'])
        self.assertFalse(loader.call_args.kwargs['trust_remote_code'])
        self.assertFalse(loader.call_args.kwargs['use_safetensors'])
        self.assertEqual(loader.call_args.kwargs['revision'], MODEL_SPECS[MODEL]['revision'])

    def test_saved_configuration_pins_offline_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'encoder.json'
            original = Encoder({'backend': 'semantic', 'max_length': 1024, 'batch_size': 2}, device='mps')
            original.save(path)
            restored = Encoder.load(path)
            self.assertEqual(original.config, restored.config)
            self.assertTrue(restored.local_only)
            self.assertEqual(restored.device, 'cpu')
            self.assertNotIn('device', json.loads(path.read_text()))
            # Старые файлы не содержали pooling/max_length/batch_size.
            path.write_text(json.dumps({'backend': 'semantic', 'model': LEGACY_MODEL,
                                       'revision': MODEL_SPECS[LEGACY_MODEL]['revision']}))
            restored = Encoder.load(path)
            self.assertEqual(restored.config['model'], LEGACY_MODEL)
            self.assertEqual(restored.config['max_length'], 512)
            self.assertEqual(restored.config['batch_size'], 16)
