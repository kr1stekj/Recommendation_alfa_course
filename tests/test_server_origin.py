"""Локальный API отклоняет чужие браузерные запросы до обращения к LLM и состоянию."""
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request

from reco.server import create_server


class BrowserOriginTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(store=MagicMock())
        self.search = MagicMock()
        self.search.search.return_value = {'items': [], 'assistant_status': 'ok'}
        with patch('reco.server.SmartSearch', return_value=self.search):
            self.server = create_server(self.engine, port=0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def request(self, headers=None, method='POST'):
        body = json.dumps({'task': 'бизнес', 'search_mode': 'smart'}).encode() if method == 'POST' else None
        request = urllib.request.Request(self.base + ('/search' if method == 'POST' else '/profiles/anon:fixture'),
                                         data=body, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_foreign_origins_never_reach_search_or_profile_store(self):
        for origin in ('https://attacker.example', 'null', self.base + '.attacker.example',
                       'http://localhost:1'):
            for content_type in ('text/plain', 'application/json'):
                with self.subTest(origin=origin, content_type=content_type):
                    self.assertEqual(self.request({'Origin': origin, 'Content-Type': content_type}), 403)
            self.assertEqual(self.request({'Origin': origin}, method='DELETE'), 403)
        # Чужое DNS-имя не становится доверенным при совпадении Origin с Host.
        self.assertEqual(self.request({'Origin': 'http://attacker.example', 'Host': 'attacker.example',
                                       'Content-Type': 'application/json'}), 403)
        self.search.search.assert_not_called()
        self.engine.store.delete.assert_not_called()

    def test_simple_post_and_missing_json_type_are_rejected(self):
        for content_type in ('text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data', ''):
            with self.subTest(content_type=content_type):
                self.assertEqual(self.request({'Content-Type': content_type}), 415)
        self.search.search.assert_not_called()

    def test_same_origin_and_cli_json_requests_work(self):
        self.assertEqual(self.request({'Origin': self.base, 'Content-Type': 'application/json; charset=utf-8'}), 200)
        self.assertEqual(self.request({'Content-Type': 'application/json'}), 200)
        self.assertEqual(self.search.search.call_count, 2)
        self.engine.validate_profile_id = MagicMock()
        self.assertEqual(self.request({'Origin': self.base}, method='DELETE'), 200)
        self.engine.store.delete.assert_called_once_with('anon:fixture')
