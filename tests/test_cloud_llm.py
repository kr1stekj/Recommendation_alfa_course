"""Облачные запросы перехватываются: внешних вызовов и расходов в тестах нет."""
import json
import os
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from reco.llm import verify
from reco.llm_cloud import NoRedirect
from reco.llm_privacy import business_summary, cloud_payload, FEATURES
from reco.data import make_catalog, PUBLISHED


class CloudLLMTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'LLM_PROVIDER': 'openrouter',
            'LLM_API_KEY': 'test-secret', 'LLM_MODEL': 'fixture-model'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.items = [{'id': 'course:private-uuid', 'kind': 'course',
                       'title': 'Как нанять сотрудника. Иван Иванов ivan@example.com',
                       'description': 'PRIVATE LESSON +7 999 123-45-67, паспорт 1234 567890',
                       'topics': ['Управление командой'], 'llm_public_metadata': True}]
        self.task = 'Я Иван Иванов, ivan@example.com, +7 999 123-45-67. Как нанять сотрудника?'
        self.opener = MagicMock()
        self.open_patch = patch('urllib.request.build_opener', return_value=self.opener)
        self.open_patch.start()
        self.addCleanup(self.open_patch.stop)
        self.respond([{'id': 'c1', 'relevant': True, 'evidence': 'Найм'}])

    def respond(self, rows):
        body = {'choices': [{'message': {'content': json.dumps({'items': rows})}}]}
        self.opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()

    def request_body(self):
        req = self.opener.open.call_args.args[0]
        return req, json.loads(req.data)

    def test_outgoing_payload_contains_only_controlled_features(self):
        result, status = verify(self.task, self.items)
        self.assertEqual(status, 'ok')
        self.assertTrue(result['course:private-uuid']['relevant'])
        req, body = self.request_body()
        payload = json.loads(body['messages'][1]['content'])
        # Проверяем декодированное содержимое, а не только ASCII-экранирование JSON.
        text = json.dumps(payload, ensure_ascii=False)
        for forbidden in ['Иван', 'ivan@example', '999', '1234', 'PRIVATE', 'private-uuid', 'description']:
            self.assertNotIn(forbidden, text)
        self.assertEqual(set(payload), {'task', 'catalog'})
        self.assertEqual(set(payload['catalog'][0]), {'id', 'kind', 'summary'})
        self.assertEqual(payload['task'], 'Управление командой; Найм')
        for summary in [payload['task']] + [r['summary'] for r in payload['catalog']]:
            self.assertTrue(set(summary.split('; ')) <= set(FEATURES))
        self.assertEqual(req.full_url, 'https://openrouter.ai/api/v1/chat/completions')
        self.assertEqual(req.get_header('Authorization'), 'Bearer test-secret')
        self.assertEqual(body['provider'], {'data_collection': 'deny', 'zdr': True, 'require_parameters': True})
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertEqual(body['max_tokens'], 1200)
        self.assertEqual(self.opener.open.call_args.kwargs['timeout'], 30)

    def test_private_and_legacy_cards_are_not_transmitted(self):
        for flag in (False, None):
            with self.subTest(flag=flag):
                card = dict(self.items[0], llm_public_metadata=flag)
                self.assertEqual(verify(self.task, [card])[1], 'privacy_blocked')
        self.opener.open.assert_not_called()

    def test_unknown_task_is_not_sent(self):
        self.assertEqual(verify('Иван Иванов ivan@example.com', self.items)[1], 'no_safe_context')
        self.opener.open.assert_not_called()

    def test_candidate_and_quote_caps(self):
        cards = [dict(self.items[0], id=str(i)) for i in range(20)]
        payload, mapping, _ = cloud_payload(self.task, cards)
        self.assertEqual(len(mapping), 5)
        self.assertEqual([r['id'] for r in payload['catalog']], ['c1', 'c2', 'c3', 'c4', 'c5'])
        self.respond([{'id': 'c1', 'relevant': True, 'evidence': 'Найм' * 100}])
        self.assertEqual(verify(self.task, self.items)[0], {})

    def test_bad_ids_quotes_types_and_duplicates_are_rejected(self):
        for rows in ([{'id': 'course:private-uuid', 'relevant': True, 'evidence': 'Найм'}],
                     [{'id': 'c1', 'relevant': True, 'evidence': 'PRIVATE LESSON'}],
                     [{'id': 'c1', 'relevant': 'true', 'evidence': 'Найм'}],
                     [{'id': 'c1', 'relevant': True, 'evidence': 'Найм'}] * 2):
            with self.subTest(rows=rows):
                self.respond(rows)
                self.assertEqual(verify(self.task, self.items), ({}, 'partial_invalid_output'))

    def test_missing_key_and_disabled_do_not_call_provider(self):
        with patch.dict(os.environ, {'LLM_API_KEY': ''}):
            self.assertEqual(verify(self.task, self.items)[1], 'disabled')
        with patch.dict(os.environ, {'LLM_PROVIDER': 'disabled'}):
            self.assertEqual(verify(self.task, self.items)[1], 'disabled')
        self.opener.open.assert_not_called()

    def test_http_errors_fall_back_without_details_or_retries(self):
        for code, expected in [(401, 'authentication_or_policy_error'), (429, 'rate_limited'),
                               (503, 'provider_error'), (302, 'provider_error')]:
            with self.subTest(code=code):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = urllib.error.HTTPError('https://service', code, 'secret error', {}, None)
                self.assertEqual(verify(self.task, self.items), ({}, expected))
                self.opener.open.assert_called_once()

    def test_timeout_and_malformed_response_fall_back(self):
        self.opener.open.side_effect = TimeoutError('secret error')
        self.assertEqual(verify(self.task, self.items), ({}, 'unavailable_or_invalid_output'))
        self.opener.open.side_effect = None
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'not-json'
        self.assertEqual(verify(self.task, self.items), ({}, 'unavailable_or_invalid_output'))

    def test_provider_cannot_exceed_output_count_limit(self):
        self.respond([{'id': 'c1', 'relevant': True, 'evidence': 'Найм'}] * 6)
        self.assertEqual(verify(self.task, self.items), ({}, 'invalid_output'))

    def test_compatible_endpoint_must_be_https_without_credentials(self):
        for url in ('http://api.example.com/v1', 'https://user:pass@api.example.com/v1',
                    'https://api.example.com/v1?key=secret', 'https://[bad'):
            with self.subTest(url=url), patch.dict(os.environ, {'LLM_PROVIDER': 'compatible', 'LLM_BASE_URL': url}):
                self.assertEqual(verify(self.task, self.items), ({}, 'invalid_endpoint'))
        self.opener.open.assert_not_called()

    def test_compatible_request_does_not_include_openrouter_parameters(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'compatible', 'LLM_BASE_URL': 'https://api.example.com/v1'}):
            self.assertEqual(verify(self.task, self.items)[1], 'ok')
        req, body = self.request_body()
        self.assertEqual(req.full_url, 'https://api.example.com/v1/chat/completions')
        self.assertNotIn('provider', body)

    def test_redirect_handler_refuses_redirect(self):
        request = urllib.request.Request('https://api.example.com', headers={'Authorization': 'Bearer secret'})
        self.assertIsNone(NoRedirect().redirect_request(request, None, 302, '', {}, 'https://other.example'))

    def test_business_summary_never_returns_matched_arbitrary_text(self):
        summary = business_summary('Как нанять секретного человека Иван Петров? паспорт 1111 222222')
        self.assertEqual(summary, 'Управление командой; Найм')


class PublicMetadataTests(unittest.TestCase):
    def test_only_public_article_and_open_cards_get_export_flag(self):
        base = dict(Id='one', Title='Налоги', StatusId=PUBLISHED, Url='/one', TypeId='ed085df2-b9bf-43fc-9a29-e2140346e333')
        tables = {'Articles': [dict(base, TypeId=None, IsPublic='t'), dict(base, Id='two', IsPublic='f')],
                  'Courses': [dict(base, IsClosed='t'), dict(base, Id='two')]}
        catalog, _ = make_catalog(tables, '2026-11-14', include_closed_history=True)
        flags = {r['id']: r['llm_public_metadata'] for r in catalog}
        self.assertEqual(flags, {'article:one': False, 'article:two': True,
                                 'course:one': False, 'course:two': True})
