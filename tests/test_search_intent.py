"""Запрос ИИ сохраняет приоритет на всех ролях; векторы fixtures, PostgreSQL настоящая."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from reco.demo_profiles import ACCOUNTS, DemoProfiles
from reco.embeddings import Encoder
from reco.engine import Recommender
from reco.index import fingerprint
from reco.search import SmartSearch, requested_format
from tests.pg_helpers import PostgresTestCase


class SearchIntentTests(PostgresTestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        catalog = []
        for account in ACCOUNTS:
            if not account['topics']:
                continue
            for kind in ('article', 'course', 'document'):
                catalog.append(dict(id=kind+':'+account['id'], entity_id=account['id'], kind=kind,
                    title=account['title'], description=account['title'], topics=account['topics'],
                    text=account['title'], tags=[], industries=[], eligible=True, history_count=0,
                    price=0, url='/fixture'))
        for number, kind in enumerate(('article', 'article', 'article', 'course')):
            catalog.append(dict(id=kind+':ai'+str(number), entity_id='ai'+str(number), kind=kind,
                title='ИИ для бизнеса '+str(number), description='Применение искусственного интеллекта',
                topics=['ИИ и автоматизация'], text='ИИ', tags=[], industries=[], eligible=True,
                history_count=0, price=0, url='/fixture'))
        # Старые темы семантически близки, но не равны явной теме ИИ. Раньше их
        # исторический бонус вместе с примесью истории вытеснял нужные материалы.
        vectors = np.asarray([[1., 0.] if row['topics'] == ['ИИ и автоматизация'] else
                              [0.85, np.sqrt(1-0.85**2)] for row in catalog], dtype=np.float32)
        encoder = Encoder({'backend':'tfidf', 'vocabulary':['ии','сторонняя'], 'idf':[1.,1.],
                           'test_nonce':self.dataset})
        (root/'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False))
        encoder.save(root/'encoder.json')
        np.save(root/'vectors.npy', vectors)
        self.dataset = fingerprint(root)
        (root/'audit.json').write_text(json.dumps({'catalog_fingerprint':self.dataset, 'catalog_items':len(catalog)}))
        self.database.replace_history(self.dataset, [])
        self.engine = Recommender(root, database=self.database)
        self.search = SmartSearch(self.engine)
        self.demo = DemoProfiles(self.engine)

    def test_ai_articles_on_every_demo_profile_match_new_guest(self):
        fresh = self.search.search(task='статьи про ИИ', skip_questions=True)
        expected = {row['id'] for row in fresh['items']}
        self.assertEqual(len(expected), 3)
        for account in ACCOUNTS:
            profile = 'anon:role_'+account['id']
            self.demo.reset(profile, account['id'], now=time.time())
            before = self.engine.store.read(profile)
            result = self.search.search(task='статьи про ИИ', profile_id=profile, skip_questions=True)
            with self.subTest(role=account['id']):
                self.assertEqual({row['id'] for row in result['items']}, expected)
                self.assertTrue(all(row['kind']=='article' for row in result['items']))
                self.assertTrue(all(row['factors']['topic']==1 for row in result['items']))
                self.assertEqual(self.engine.store.read(profile), before)

    def test_without_task_history_still_personalizes(self):
        self.demo.reset('anon:old', 'marketing', now=time.time())
        result = self.search.search(profile_id='anon:old', skip_questions=True)
        self.assertTrue(result['profile_topics'])
        self.assertGreater(result['live_items'], 0)
        self.assertTrue(any('Маркетинг' in row['topics'] for row in result['items']))

    def test_unmapped_new_query_does_not_inherit_historical_topic_bonus(self):
        self.demo.reset('anon:old', 'finance', now=time.time())
        result = self.engine.recommend(task='ИИ', profile_id='anon:old', query_first=True)
        self.assertTrue(result['items'])
        self.assertTrue(all(row['factors']['topic']==0 for row in result['items']))

    def test_completed_and_disliked_materials_stay_excluded(self):
        self.engine.event('anon:signals', 'article:ai0', 'complete', 'one')
        self.engine.event('anon:signals', 'article:ai1', 'dislike', 'two')
        result = self.search.search(task='статьи про ИИ', profile_id='anon:signals', skip_questions=True)
        self.assertEqual([row['id'] for row in result['items']], ['article:ai2'])

    def test_query_candidates_do_not_follow_history_only_direction(self):
        # Новый запрос ортогонален старым материалам. Их история не должна
        # возвращать их в поисковый пул через старую метку/нулевой вектор запроса.
        self.engine.vectors[:15] = [0.,1.]
        self.demo.reset('anon:old', 'finance', now=time.time())
        result = self.engine.recommend(task='ИИ', profile_id='anon:old', query_first=True)
        self.assertTrue(result['items'])
        self.assertTrue(all('ИИ и автоматизация' in row['topics'] for row in result['items']))

    def test_format_choices_override_task_and_unavailable_format_is_empty(self):
        result = self.search.search(task='статьи про ИИ', answers={'format':'course'}, skip_questions=True)
        self.assertEqual({row['kind'] for row in result['items']}, {'course'})
        result = self.search.search(task='подкасты про ИИ', skip_questions=True)
        self.assertEqual(result['items'], [])
        self.assertEqual(result['match_status'], 'empty')

    def test_ai_aliases_resolve_to_catalog_topic(self):
        for task in ('ИИ', 'AI', 'нейросети', 'искусственный интеллект'):
            with self.subTest(task=task), patch.object(self.engine, 'recommend', wraps=self.engine.recommend) as recommend:
                self.search.search(task=task, skip_questions=True)
                self.assertIn('ИИ и автоматизация', recommend.call_args.kwargs['topics'])

    def test_invalid_internal_options_rejected(self):
        for options in ({'query_first':'true'}, {'required_format':None}, {'required_format':'video'},
                        {'task':None, 'query_first':True}, {'task':5, 'query_first':True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.engine.recommend(**options)

    def test_misspelled_ai_question_gets_current_focus_and_safe_profile_topics(self):
        question = {'title':'Как вы хотите использовать ИИ в своей задаче?', 'options':[
            {'label':'Для создания контента','preferred_format':''},
            {'label':'Для анализа данных','preferred_format':''}]}
        with patch('reco.clarification.structured_call', return_value=(
                {'needs_clarification':True,'question':question},'ok')) as cloud:
            for account in ACCOUNTS:
                profile='anon:question_'+account['id']
                self.demo.reset(profile,account['id'],now=time.time())
                before=self.engine.store.read(profile)
                result=self.search.search(task='искуственный интеллект',profile_id=profile)
                with self.subTest(role=account['id']):
                    self.assertEqual(result['question']['title'],question['title'])
                    self.assertEqual(cloud.call_args.args[0]['request_focus'],['Искусственный интеллект'])
                    self.assertEqual(bool(cloud.call_args.args[0]['profile_topics']),account['id']!='new')
                    self.assertEqual(self.engine.store.read(profile),before)

    def test_search_uses_ten_percent_history_and_recommend_keeps_legacy_weight(self):
        self.demo.reset('anon:weights', 'finance', now=time.time())
        history = self.engine.vectors[0]
        expected = 0.90 * np.asarray([1.,0.]) + 0.10 * history
        expected /= np.linalg.norm(expected)
        result = self.engine.recommend(task='ИИ', profile_id='anon:weights', query_first=True)
        ai = next(row for row in result['items'] if row['id']=='article:ai0')
        self.assertAlmostEqual(ai['cosine'], expected[0], places=5)
        legacy = 0.75 * np.asarray([1.,0.]) + 0.25 * history
        legacy /= np.linalg.norm(legacy)
        result = self.engine.recommend(task='ИИ', profile_id='anon:weights', k=20)
        ai = next(row for row in result['items'] if row['id']=='article:ai0')
        self.assertAlmostEqual(ai['cosine'], legacy[0], places=5)

    def test_http_login_then_ai_search_retains_requested_topic_and_format(self):
        import http.cookiejar
        import threading
        import urllib.request
        from reco.server import create_server
        server = create_server(self.engine, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def post(path, data):
            request = urllib.request.Request('http://127.0.0.1:'+str(server.server_port)+path,
                data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})
            with client.open(request, timeout=10) as response: return json.load(response)
        for account in ACCOUNTS:
            session = post('/demo/login', {'account_id':account['id']})
            result = post('/smart-search', {'task':'статьи про ИИ', 'skip_questions':True})
            with self.subTest(account=account['id']):
                self.assertEqual(result['live_items'], session['account']['seed_items'])
                self.assertEqual(len(result['items']), 3)
                self.assertTrue(all(row['kind']=='article' and 'ИИ и автоматизация' in row['topics']
                                    for row in result['items']))


class FormatIntentTests(unittest.TestCase):
    def test_only_unambiguous_formats_become_filters(self):
        for text, kind in [('статьи про ИИ','article'), ('Как стать ИП',''),
                           ('курсы по ИИ','course'), ('статьи и курсы про ИИ',''),
                           ('не статьи про ИИ',''), ('кроме курсов',''),
                           ('шаблон договора','document'), ('подкаст про бизнес','podcast')]:
            with self.subTest(text=text): self.assertEqual(requested_format(text), kind)
