"""Синтетический вход: cookie/TTL/изоляция; без настоящих API и моделей."""
import copy
from http.cookies import SimpleCookie
from http.cookiejar import CookieJar
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np

from reco.demo_auth import DemoAuth, DemoAuthError, session_cookie, session_token
from reco.embeddings import Encoder
from reco.engine import Recommender
from reco.index import fingerprint
from reco.server import create_server
from tests.pg_helpers import PostgresTestCase


CATALOG = [
    ('article:start', 'Регистрация ИП и первые шаги бизнеса', 'Регистрация бизнеса', True),
    ('article:marketing', 'Маркетинг реклама и социальные сети', 'Маркетинг', True),
    ('course:marketing', 'Продвижение и маркетинг бизнеса', 'Маркетинг', True),
    ('document:marketing', 'План маркетинга и рекламы', 'Маркетинг', True),
    ('article:finance', 'Финансы и учет доходов расходов', 'Финансы', True),
    ('course:team', 'Управление командой и найм сотрудников', 'Управление командой', True),
    ('article:closed', 'Закрытый маркетинг', 'Маркетинг', False),
]


def catalog():
    return [dict(id=iid, entity_id=iid, kind=iid.split(':')[0], title=title, description=title,
        text=title, topics=[topic], tags=[], eligible=eligible, history_count=0, url='/', price=0)
        for iid, title, topic, eligible in CATALOG]


class MemoryStore:
    def __init__(self): self.rows = {}
    def mutate(self, profile_id, kind, callback, now=None):
        state = callback(copy.deepcopy(self.rows.get(profile_id, {'items': {}, 'event_ids': []})))
        self.rows[profile_id] = state
        return state
    def delete(self, profile_id): self.rows.pop(profile_id, None)


class DemoSessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.engine = SimpleNamespace(catalog=catalog(), store=MemoryStore())
        self.auth = DemoAuth(self.engine, ttl=60, max_sessions=2, clock=lambda: self.now)

    def test_cookie_is_opaque_http_only_and_has_strict_scope(self):
        token, result = self.auth.login('marketing')
        value = session_cookie(token)
        cookie = SimpleCookie(value)
        self.assertTrue(cookie['reco_demo_session']['httponly'])
        self.assertEqual(cookie['reco_demo_session']['samesite'], 'Strict')
        self.assertEqual(cookie['reco_demo_session']['path'], '/')
        self.assertEqual(session_token('other=value; ' + value), token)
        self.assertNotIn(token, json.dumps(result))
        self.assertNotIn(token, result['profile_id'])
        self.assertEqual(SimpleCookie(session_cookie())['reco_demo_session']['max-age'], '0')
        self.assertIsNone(session_token('reco_demo_session=invalid'))

    def test_limits_expiry_rotation_and_restart(self):
        first, one = self.auth.login('marketing')
        second, two = self.auth.login('finance')
        with self.assertRaises(DemoAuthError) as error: self.auth.login('new')
        self.assertEqual(error.exception.status, 429)
        self.assertEqual(len(self.engine.store.rows), 2)
        rotated, fresh = self.auth.login('new', first)
        self.assertNotEqual(fresh['profile_id'], one['profile_id'])
        self.assertFalse(self.auth.snapshot(first)['authenticated'])
        self.assertNotIn(one['profile_id'], self.engine.store.rows)
        self.assertFalse(DemoAuth(self.engine).snapshot(rotated)['authenticated'])
        self.now += 61; self.auth.cleanup()
        self.assertEqual(self.auth.sessions, {})
        self.assertEqual(self.engine.store.rows, {})

    def test_new_has_no_events_missing_topics_do_not_invent_history(self):
        token, result = self.auth.login('new')
        self.assertEqual(result['account']['seed_items'], 0)
        self.assertEqual(self.engine.store.rows[result['profile_id']]['items'], {})
        token, result = self.auth.login('taxes', token)
        self.assertEqual(result['account']['seed_items'], 0)
        self.engine.catalog = []
        self.assertTrue(all(row['seed_items'] == 0 for row in self.auth.profiles.list_accounts()))

    def test_unknown_account_and_forged_session_cannot_create_or_use_profile(self):
        for account_id in ('real-user', None, [], {'id': 'marketing'}):
            with self.subTest(value=account_id), self.assertRaises(ValueError): self.auth.login(account_id)
        self.assertEqual(self.engine.store.rows, {})
        token, result = self.auth.login('marketing')
        for supplied in (None, 'x' * 43):
            with self.assertRaises(DemoAuthError): self.auth.bind_profile(result['profile_id'], supplied)
        with self.assertRaises(DemoAuthError) as error: self.auth.bind_profile('anon:other', token)
        self.assertEqual(error.exception.status, 403)

    def test_logout_revokes_cookie_even_if_database_cleanup_fails(self):
        token, result = self.auth.login('marketing')
        with patch.object(self.engine.store, 'delete', side_effect=OSError('fixture database unavailable')), \
             self.assertLogs(level='ERROR'):
            self.assertFalse(self.auth.logout(token)['authenticated'])
        self.assertFalse(self.auth.snapshot(token)['authenticated'])


class DemoHttpTests(PostgresTestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rows = catalog()
        encoder = Encoder({'backend': 'tfidf', 'min_df': 1, 'test_nonce': self.dataset})
        vectors = encoder.fit([row['text'] for row in rows])
        (root/'catalog.json').write_text(json.dumps(rows))
        encoder.save(root/'encoder.json'); np.save(root/'vectors.npy', vectors)
        self.dataset = fingerprint(root)
        (root/'audit.json').write_text(json.dumps({'catalog_fingerprint': self.dataset, 'catalog_items': 6}))
        self.source = [{'user_id': 'fixture-real', 'item_id': 'article:start', 'weight': 6,
                        'completed': True, 'owned': True, 'at': '2026-01-01'}]
        self.database.replace_history(self.dataset, self.source)
        self.engine = Recommender(root, database=self.database)
        self.server = create_server(self.engine, port=0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(thread.join, 2); self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)
        self.a, self.b = self.client(), self.client()
        for name in ('reco.clarification.structured_call', 'reco.engine.verify'):
            guard = patch(name, side_effect=AssertionError('Real LLM is forbidden in demo auth tests'))
            guard.start(); self.addCleanup(guard.stop)

    @staticmethod
    def client(): return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def request(self, client, path, data=None, method=None, headers=None):
        request = urllib.request.Request(self.base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type': 'application/json', **(headers or {})}, method=method)
        with client.open(request, timeout=10) as response:
            return json.load(response), response.headers

    def assert_status(self, status, *args, **kwargs):
        with self.assertRaises(urllib.error.HTTPError) as error: self.request(*args, **kwargs)
        self.assertEqual(error.exception.code, status)

    def test_sign_in_restores_cookie_and_personalizes_without_source_writes(self):
        accounts, _ = self.request(self.a, '/demo/accounts')
        self.assertEqual(len(accounts['accounts']), 6)
        self.assertFalse(self.request(self.a, '/demo/session')[0]['authenticated'])
        session, headers = self.request(self.a, '/demo/login', {'account_id': 'marketing'})
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertEqual(session['account']['seed_items'], 3)
        self.assertEqual(self.request(self.a, '/demo/session')[0], session)
        profile = self.engine.store.read(session['profile_id'])
        self.assertEqual(len(profile['items']), 3)
        self.assertNotIn('article:closed', profile['items'])
        self.assertEqual(sum(row['completed'] for row in profile['items'].values()), 1)
        result, _ = self.request(self.a, '/smart-search', {'skip_questions': True})
        self.assertEqual(result['mode'], 'behavior')
        self.assertEqual(result['live_items'], 3)
        self.assertEqual(result['history_items'], 0)
        self.assertEqual(result['profile_topics'][0][0], 'Маркетинг')
        self.assertEqual(self.engine.history('user:fixture-real'), self.source)
        with self.database.connect() as db:
            row = db.execute('SELECT kind,expires FROM reco.profiles WHERE dataset=%s AND id=%s',
                             (self.dataset,session['profile_id'])).fetchone()
        self.assertEqual(row['kind'], 'anonymous'); self.assertIsNotNone(row['expires'])

    def test_two_browsers_and_guest_are_isolated_and_logout_restores_guest(self):
        guest, _ = self.request(self.a, '/profiles', {})
        self.request(self.a, '/events', {'profile_id': guest['profile_id'], 'item_id': 'course:team',
                                       'event_type': 'engaged', 'event_id': 'guest-one'})
        before = self.engine.store.read(guest['profile_id'])
        one, _ = self.request(self.a, '/demo/login', {'account_id': 'marketing'})
        two, _ = self.request(self.b, '/demo/login', {'account_id': 'marketing'})
        self.assertNotEqual(one['profile_id'], two['profile_id'])
        untouched = self.engine.store.read(two['profile_id'])
        self.request(self.a, '/events', {'item_id': 'article:finance', 'event_type': 'dislike', 'event_id': 'only-a'})
        self.assertEqual(self.engine.store.read(two['profile_id']), untouched)
        self.assertNotIn('article:finance', untouched['items'])
        self.assertFalse(self.request(self.a, '/demo/logout', {})[0]['authenticated'])
        self.assertEqual(self.engine.store.read(one['profile_id'])['items'], {})
        self.assertTrue(self.request(self.b, '/demo/session')[0]['authenticated'])
        self.assertEqual(self.engine.store.read(guest['profile_id']), before)
        result, _ = self.request(self.a, '/recommend', {'profile_id': guest['profile_id']})
        self.assertEqual(result['live_items'], 1)
        self.assertFalse(self.request(self.a, '/demo/logout', {})[0]['authenticated'])

    def test_profile_forgery_delete_and_merge_are_rejected_before_mutation(self):
        one, _ = self.request(self.a, '/demo/login', {'account_id': 'marketing'})
        two, _ = self.request(self.b, '/demo/login', {'account_id': 'finance'})
        before = self.engine.store.read(two['profile_id'])
        for path in ('/recommend','/search','/smart-search'):
            self.assert_status(403,self.a,path,{'profile_id':two['profile_id']})
        self.assert_status(403,self.a,'/events',{'profile_id':two['profile_id'], 'item_id':'course:team',
                                              'event_type':'complete','event_id':'forged'})
        self.assert_status(403,self.a,'/profiles/'+two['profile_id'],method='DELETE')
        self.assert_status(401,self.client(),'/recommend',{'profile_id':one['profile_id']})
        self.assert_status(401,self.client(),'/profiles/'+one['profile_id'],method='DELETE')
        self.assert_status(403,self.client(),'/merge',{'guest_id':one['profile_id'], 'registered_id':'user:fake'})
        self.assertEqual(self.engine.store.read(two['profile_id']), before)

    def test_reset_rotation_expiry_and_cookie_clear(self):
        one, _ = self.request(self.a, '/demo/login', {'account_id':'marketing'})
        original = self.engine.store.read(one['profile_id'])
        self.request(self.a,'/events',{'item_id':'article:finance','event_type':'dislike','event_id':'change'})
        reset, _ = self.request(self.a,'/demo/reset',{})
        self.assertEqual(reset['profile_id'],one['profile_id'])
        state = self.engine.store.read(one['profile_id'])
        self.assertEqual(set(state['items']),set(original['items']))
        self.assertEqual(state['event_ids'],[])
        rotated, _ = self.request(self.a,'/demo/login',{'account_id':'new'})
        self.assertNotEqual(rotated['profile_id'],one['profile_id'])
        self.assertEqual(self.engine.store.read(one['profile_id'])['items'],{})
        self.server.demo_auth.clock = lambda: time.time()+3601
        self.assert_status(401,self.a,'/events',{'item_id':'course:team','event_type':'click','event_id':'late'})
        session, _ = self.request(self.a,'/demo/session')
        self.assertFalse(session['authenticated'])
        self.assertEqual(self.engine.store.read(rotated['profile_id'])['items'],{})

    def test_new_profile_is_cold_start_and_auth_routes_preserve_csrf_guards(self):
        self.assert_status(403,self.a,'/demo/login',{'account_id':'new'},headers={'Origin':'https://foreign.example'})
        self.assert_status(415,self.a,'/demo/login',{'account_id':'new'},headers={'Content-Type':'text/plain'})
        self.assert_status(400,self.a,'/demo/login',{'account_id':'new','register':True})
        self.assert_status(400,self.a,'/demo/login',{'account_id':'unknown'})
        self.assert_status(401,self.a,'/demo/reset',{})
        self.request(self.a,'/demo/login',{'account_id':'new'})
        result, _ = self.request(self.a,'/smart-search',{'skip_questions':True})
        self.assertEqual(result['mode'],'starter_collection')
        self.assertEqual(result['live_items'],0)
        self.assertEqual(self.engine.history('user:fixture-real'),self.source)

    def test_expiry_cleanup_waits_for_authorized_event_and_does_not_leave_orphan(self):
        session, _ = self.request(self.a,'/demo/login',{'account_id':'new'})
        entered, release, cleanup_started, cleaned = (threading.Event() for _ in range(4))
        original = self.engine.event
        response, errors = [], []
        def delayed_event(**data):
            entered.set()
            if not release.wait(3): raise AssertionError('Fixture did not release event')
            return original(**data)
        def send_event():
            try:
                response.append(self.request(self.a,'/events',{'item_id':'course:team',
                    'event_type':'click','event_id':'in-flight'})[0])
            except Exception as error: errors.append(error)
        def cleanup():
            cleanup_started.set(); self.server.demo_auth.cleanup(); cleaned.set()
        with patch.object(self.engine,'event',side_effect=delayed_event):
            event_thread = threading.Thread(target=send_event)
            event_thread.start()
            cleanup_thread = None
            try:
                self.assertTrue(entered.wait(2))
                self.server.demo_auth.clock = lambda: time.time()+3601
                cleanup_thread = threading.Thread(target=cleanup); cleanup_thread.start()
                self.assertTrue(cleanup_started.wait(1))
                self.assertFalse(cleaned.wait(0.1))
            finally:
                release.set(); event_thread.join(3)
                if cleanup_thread: cleanup_thread.join(3)
        self.assertFalse(errors)
        self.assertTrue(response[0]['accepted'])
        self.assertTrue(cleaned.is_set())
        self.assertFalse(self.server.demo_auth.sessions)
        self.assertEqual(self.engine.store.read(session['profile_id'])['items'],{})
