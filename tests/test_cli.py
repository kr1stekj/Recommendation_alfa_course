"""Обслуживание PostgreSQL не должно требовать файлов рекомендательного индекса."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from reco.cli import main


class CleanupCommandTests(unittest.TestCase):
    def test_demo_passes_explicit_llm_flag(self):
        with patch('sys.argv', ['reco.cli', 'demo', '--llm']), \
             patch('reco.cli.Recommender') as engine, patch('builtins.input', return_value='q'), \
             contextlib.redirect_stdout(io.StringIO()):
            engine.return_value.recommend.return_value = {'profile_topics': [], 'mode': 'onboarding',
                                                          'llm_status': 'disabled', 'items': []}
            main()
        self.assertTrue(engine.return_value.recommend.call_args.kwargs['use_llm'])

    def test_cleanup_does_not_construct_recommender(self):
        with patch('sys.argv',['reco.cli','cleanup','--artifacts','/nonexistent/index']), \
             patch('reco.cli.Database') as database, \
             patch('reco.cli.Recommender',side_effect=AssertionError('Индекс не нужен')) as recommender:
            database.return_value.cleanup_expired.return_value = 2
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main()
            self.assertEqual(json.loads(output.getvalue()),{'expired_profiles_deleted':2})
            database.return_value.initialize.assert_called_once()
            database.return_value.cleanup_expired.assert_called_once()
            recommender.assert_not_called()
