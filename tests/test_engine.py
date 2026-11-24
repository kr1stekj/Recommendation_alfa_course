import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from reco.embeddings import Encoder, normalize
from reco.engine import Recommender
from reco.llm import verify
from reco.evaluate import metrics
from reco.index import fingerprint
from tests.pg_helpers import PostgresTestCase


class EngineTests(PostgresTestCase):
    def setUp(self):
        super().setUp()
        self.llm_env = patch.dict('os.environ', {'LLM_PROVIDER': 'disabled'})
        self.llm_env.start()
        self.addCleanup(self.llm_env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        catalog = []
        for iid, title, topic, eligible in [('a', 'налоги бизнес учет', 'Налоги', True), ('b', 'налоги учет курс', 'Налоги', True), ('c', 'маркетинг продажи бизнес', 'Маркетинг', True), ('closed', 'налоги бизнес', 'Налоги', False)]:
            catalog.append(dict(id='course:' + iid, entity_id=iid, kind='course', title=title, description=title,
                                text=title, topics=[topic], tags=[], industries=[], eligible=eligible,
                                history_count=0, url='/' + iid, price=0))
        encoder = Encoder({'backend': 'tfidf'})
        vectors = encoder.fit([r['text'] for r in catalog])
        (root/'catalog.json').write_text(json.dumps(catalog))
        np.save(root/'vectors.npy', vectors)
        # У каждого теста свой настоящий fingerprint, чтобы параллельные fixtures
        # не делили один dataset при одинаковых карточках.
        encoder.config['test_nonce'] = self.dataset
        encoder.save(root/'encoder.json')
        self.dataset = fingerprint(root)
        (root/'audit.json').write_text(json.dumps({'catalog_fingerprint': self.dataset, 'catalog_items': 3}))
        self.database.replace_history(self.dataset,[dict(user_id='u',item_id='course:closed',weight=6,completed=True,owned=True,at='2026-01-01')])
        self.engine = Recommender(root,database=self.database)

    def test_closed_history_informs_interests_without_being_recommended(self):
        result = self.engine.recommend(profile_id='user:u')
        self.assertEqual(result['mode'], 'behavior')
        self.assertEqual(result['profile_topics'][0][0], 'Налоги')
        self.assertNotIn('course:closed', [r['id'] for r in result['items']])

    def test_guest_isolation_deduplication_and_repeated_click_cap(self):
        e = self.engine
        e.event('anon:a', 'course:a', 'engaged', 'one')
        self.assertTrue(e.event('anon:a', 'course:a', 'engaged', 'one')['duplicate'])
        for i in range(20): e.event('anon:a', 'course:a', 'click', str(i))
        state = e.store.read('anon:a')
        self.assertEqual(state['items']['course:a']['weight'], 3)
        self.assertEqual(e.recommend(profile_id='anon:b')['mode'], 'starter_collection')
        after = e.recommend(profile_id='anon:a')
        self.assertEqual(after['mode'], 'behavior')
        self.assertNotIn('course:a', [r['id'] for r in after['items']])

    def test_expired_guest_behaves_as_new_user(self):
        self.engine.event('anon:a', 'course:a', 'engaged', 'one', now=time.time()-2*86400)
        self.assertEqual(self.engine.recommend(profile_id='anon:a')['mode'], 'starter_collection')

    def test_reject_ineligible_events_and_invalid_requests(self):
        with self.assertRaises(ValueError): self.engine.event('anon:a', 'course:closed', 'click', 'one')
        with self.assertRaises(ValueError): self.engine.recommend(profile_id='fake')
        with self.assertRaises(ValueError): self.engine.recommend(k=100000)
        with self.assertRaises(ValueError): self.engine.recommend(topics='Налоги')

    def test_each_profile_is_bounded_under_many_events(self):
        # Защита от повторов хранит ограниченное число ID, а не вечный журнал событий.
        for i in range(270): self.engine.event('anon:a', 'course:a', 'click', str(i))
        self.assertEqual(len(self.engine.store.read('anon:a')['event_ids']), 256)
        self.assertEqual(len(self.engine.store.read('anon:a')['items']), 1)

    def test_cloud_check_preserves_ranking_and_profile(self):
        import os
        for item in self.engine.catalog:
            item['llm_public_metadata'] = item['eligible']
            item['access'] = 'public'
        profile = 'anon:cloud'
        expected = self.engine.recommend(task='Налоги', profile_id=profile, k=2)
        before = self.engine.store.read(profile)
        opener = MagicMock()
        def answer(request, **kwargs):
            body = json.loads(request.data)
            payload = json.loads(body['messages'][1]['content'])
            self.assertNotIn(profile, request.data.decode())
            rows = [{'id': r['id'], 'relevant': False, 'evidence': r['summary'][:100]}
                    for r in payload['catalog']]
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                'choices': [{'message': {'content': json.dumps({'items': rows})}}]}).encode()
            return response
        opener.open.side_effect = answer
        with patch.dict(os.environ, {'LLM_PROVIDER': 'openrouter', 'LLM_API_KEY': 'fixture',
                                     'LLM_MODEL': 'fixture'}, clear=True), \
             patch('urllib.request.build_opener', return_value=opener):
            actual = self.engine.recommend(task='Налоги', profile_id=profile, k=2, use_llm=True)
        self.assertEqual(actual['llm_status'], 'ok')
        self.assertEqual([r['id'] for r in actual['items']], [r['id'] for r in expected['items']])
        self.assertEqual(self.engine.store.read(profile), before)
        self.assertTrue(all(r['verification']['llm_checked'] for r in actual['items']))

    def test_smart_search_llm_question_and_answer_context(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        self.assertIsNone(search.search()['question'])
        question = {'title': 'В какой области вам сейчас нужна поддержка?', 'options': [
            {'label': 'Налоги и учет', 'preferred_format': ''},
            {'label': 'Маркетинг и продажи', 'preferred_format': ''}]}
        with patch('reco.search.clarify', return_value=(question, 'ok')) as model:
            vague = search.search(task='бизнес')
        self.assertEqual(vague['question']['title'], question['title'])
        self.assertEqual(vague['question']['options'][0]['label'], 'Налоги и учет')
        self.assertEqual(vague['question_source'], 'cloud_llm')
        self.assertFalse(vague['context_sufficient'])
        model.assert_called_once()
        token = vague['question']['options'][0]['value']
        with patch('reco.search.clarify', return_value=(None, 'ok')) as model:
            done = search.search(task='бизнес', clarifications=[token])
        self.assertIsNone(done['question'])
        self.assertTrue(done['context_sufficient'])
        self.assertEqual(done['dialogue'], [{'question': question['title'], 'answer': 'Налоги и учет'}])
        self.assertEqual([{key: row[key] for key in ('question', 'answer')} for row in model.call_args.args[1]], done['dialogue'])
        self.assertEqual(done['match_status'], 'related')
        self.assertNotIn('Запрос пока общий', done['summary'])
        self.assertTrue(done['items'])
        with patch('reco.search.clarify', side_effect=AssertionError('Достаточный контекст не уточняется')):
            legacy = search.search(task='бизнес', answers={'goal': 'finance'})
            specific = search.search(task='Как вести учет бизнеса и платить налоги')
        self.assertEqual(legacy['assistant_status'], 'not_needed')
        self.assertEqual(specific['assistant_status'], 'not_needed')
        with patch('reco.search.clarify', side_effect=AssertionError('Пропуск не вызывает LLM')):
            self.assertIsNone(search.search(task='бизнес', skip_questions=True)['question'])
        self.assertEqual(search.search(task='бизнес')['match_status'], 'broad')
        with patch.object(self.engine, 'recommend', wraps=self.engine.recommend) as recommend:
            search.search(task='налоги', answers={'stage': 'beginner', 'format': 'course'})
        self.assertEqual(recommend.call_args.kwargs['preferred_format'], 'course')
        self.assertIn('Начинающий', recommend.call_args.kwargs['stage'])

    def test_normal_search_never_calls_llm_or_semantic_encoder(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        self.engine.event('anon:ordinary', 'course:a', 'complete', 'done')
        with patch('reco.search.clarify', side_effect=AssertionError('LLM запрещена')), \
             patch.object(self.engine, 'recommend', side_effect=AssertionError('Персонализация запрещена')):
            a = search.search(task='налоги', profile_id='anon:ordinary', search_mode='normal', use_llm=True)
            b = search.search(task='налоги', search_mode='normal')
            empty = search.search(task='qzfoobarzz', search_mode='normal')
        self.assertEqual(a, b)
        self.assertIsNone(a['question'])
        self.assertEqual(a['assistant_status'], 'not_requested')
        self.assertIn('course:a', [r['id'] for r in a['items']])
        self.assertNotIn('course:closed', [r['id'] for r in a['items']])
        self.assertEqual(empty['items'], [])

    def test_missing_llm_does_not_create_static_question(self):
        from reco.search import SmartSearch
        result = SmartSearch(self.engine).search(task='бизнес')
        self.assertIsNone(result['question'])
        self.assertIsNone(result['question_source'])
        self.assertEqual(result['assistant_status'], 'disabled')
        self.assertTrue(result['items'])

    def test_format_alone_does_not_replace_business_goal(self):
        from reco.search import SmartSearch
        question = {'title': 'Какую задачу бизнеса хотите решить?', 'options': [
            {'label': 'Разобраться с налогами', 'preferred_format': ''},
            {'label': 'Привлечь новых клиентов', 'preferred_format': ''}]}
        with patch('reco.search.clarify', return_value=(question, 'ok')) as model:
            result = SmartSearch(self.engine).search(task='бизнес', answers={'format': 'course'})
        self.assertEqual(result['question']['title'], question['title'])
        self.assertFalse(result['context_sufficient'])
        model.assert_called_once()

    def test_generic_action_does_not_skip_clarification(self):
        from reco.search import SmartSearch
        self.engine.catalog[0]['title'] = 'Как выбрать налоговый режим'
        sample = self.engine.recommend(task='налоги')
        # Изолируем gate от encoder: похожая карточка не задаёт объект запроса.
        sample['backend'] = 'semantic'
        sample['items'][0]['cosine'] = .8
        question = {'title': 'Какую задачу бизнеса хотите решить?', 'options': [
            {'label': 'Налоги и учет', 'preferred_format': ''},
            {'label': 'Маркетинг и продажи', 'preferred_format': ''}]}
        search = SmartSearch(self.engine)
        for task in ('как выбрать', 'Помоги выбрать курс для бизнеса', 'нанять'):
            with self.subTest(task=task), patch.object(self.engine, 'recommend', return_value=sample), \
                 patch('reco.search.clarify', return_value=(question, 'ok')) as model:
                result = search.search(task=task)
                self.assertEqual(result['question']['title'], question['title'])
                self.assertFalse(result['context_sufficient'])
                model.assert_called_once()

    def test_smart_search_full_cloud_contract_and_refined_ranking(self):
        from reco.search import SmartSearch
        opener = MagicMock()
        histories = []
        question = {'title': 'Что сейчас важнее для вашего бизнеса?', 'options': [
            {'label': 'Маркетинг и продажи', 'preferred_format': ''},
            {'label': 'Налоги и учет', 'preferred_format': ''}]}
        def respond(request, **kwargs):
            body = json.loads(request.data)
            self.assertEqual(body['response_format']['json_schema']['name'], 'search_clarification')
            payload = json.loads(body['messages'][1]['content'])
            histories.append(payload['answers'])
            self.assertNotIn('available_questions', payload)
            answer = {'needs_clarification': not bool(payload['answers']),
                      'question': question if not payload['answers'] else None}
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({'choices': [
                {'message': {'content': json.dumps(answer)}}]}).encode()
            return response
        opener.open.side_effect = respond
        search = SmartSearch(self.engine)
        before = self.engine.store.read('anon:llm-dialog')
        with patch.dict('os.environ', {'LLM_PROVIDER': 'openrouter', 'LLM_API_KEY': 'fixture', 'LLM_MODEL': 'fixture'}), \
             patch('urllib.request.build_opener', return_value=opener):
            first = search.search(task='бизнес', profile_id='anon:llm-dialog')
            token = first['question']['options'][0]['value']
            second = search.search(task='бизнес', profile_id='anon:llm-dialog', clarifications=[token])
        self.assertEqual(first['question']['options'][0]['label'], 'Маркетинг и продажи')
        self.assertIsNone(second['question'])
        self.assertEqual(second['assistant_status'], 'ok')
        self.assertEqual(histories[0], [])
        self.assertIn('Маркетинг', histories[1][0]['answer_topics'])
        self.assertIn('Продажи', histories[1][0]['answer_topics'])
        self.assertNotIn(question['title'], json.dumps(histories, ensure_ascii=False))
        self.assertEqual(second['match_status'], 'related')
        self.assertNotEqual([row['id'] for row in first['items']], [row['id'] for row in second['items']])
        self.assertEqual(second['items'][0]['id'], 'course:c')
        self.assertEqual(self.engine.store.read('anon:llm-dialog'), before)

    def test_rephrased_goal_is_suppressed_after_selected_answer(self):
        from reco.search import SmartSearch
        question = {'title': 'Какая цель продвижения вашего бизнеса?', 'options': [
            {'label': 'Маркетинг и продажи', 'preferred_format': ''},
            {'label': 'Налоги и учет', 'preferred_format': ''}]}
        repeated = dict(question, title='Какая основная цель маркетинга вашего бизнеса?')
        search = SmartSearch(self.engine)
        with patch('reco.clarification.structured_call', side_effect=[
                ({'needs_clarification': True, 'question': question}, 'ok'),
                ({'needs_clarification': True, 'question': repeated}, 'ok')]) as cloud:
            first = search.search(task='бизнес')
            token = first['question']['options'][0]['value']
            second = search.search(task='бизнес', clarifications=[token])
        self.assertEqual(cloud.call_count, 2)
        self.assertIsNone(second['question'])
        self.assertEqual(second['assistant_status'], 'already_answered')
        self.assertEqual(second['match_status'], 'related')
        self.assertNotIn('Запрос пока общий', second['summary'])
        self.assertEqual(second['items'][0]['id'], 'course:c')
        self.assertEqual(second['dialogue'][0]['answer'], 'Маркетинг и продажи')

    def test_dynamic_dialogue_stops_after_two_answers(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        questions = [
            {'title': 'Что вы хотите улучшить в бизнесе?', 'options': [
                {'label': 'Налоги и учет', 'preferred_format': ''},
                {'label': 'Маркетинг и продажи', 'preferred_format': ''}]},
            {'title': 'Какой формат обучения вам удобнее?', 'options': [
                {'label': 'Подробный курс', 'preferred_format': 'course'},
                {'label': 'Короткий курс', 'preferred_format': 'course'}]}]
        # Модель и транспорт подставлены; codec, ranking и PostgreSQL настоящие.
        from reco.clarification import clarify as real_clarify
        with patch('reco.search.clarify', side_effect=[(questions[0], 'ok'), (questions[1], 'ok')]):
            first = search.search(task='бизнес')
            one = first['question']['options'][0]['value']
            second = search.search(task='бизнес', clarifications=[one])
        two = second['question']['options'][0]['value']
        with patch('reco.search.clarify', wraps=real_clarify), \
             patch('reco.clarification.structured_call', side_effect=AssertionError('Третий вызов запрещён')):
            last = search.search(task='бизнес', clarifications=[one, two])
        self.assertEqual(last['assistant_status'], 'limit_reached')
        self.assertIsNone(last['question'])
        self.assertEqual(len(last['dialogue']), 2)
        self.assertEqual(last['match_status'], 'related')
        self.assertNotIn('Запрос пока общий', last['summary'])
        self.assertTrue(last['items'])

    def test_dynamic_answer_preserves_unknown_query_warning(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        task = 'Квантовые телепортаторы Юпитера'
        question = {'title': 'Какую задачу бизнеса хотите решить?', 'options': [
            {'label': 'Налоги и учет', 'preferred_format': ''},
            {'label': 'Маркетинг и продажи', 'preferred_format': ''}]}
        with patch('reco.search.clarify', return_value=(question, 'ok')):
            first = search.search(task=task)
        token = first['question']['options'][0]['value']
        with patch('reco.search.clarify', return_value=(None, 'ok')):
            second = search.search(task=task, clarifications=[token])
        self.assertEqual(second['match_status'], 'no_match')
        self.assertIn('не нашли', second['summary'])

    def test_forged_or_transferred_answer_rejected_before_cloud_and_ranking(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        question = {'title': 'Как вы хотите улучшить продажи?', 'options': [
            {'label': 'Через партнёрства', 'preferred_format': ''},
            {'label': 'Через рекламу', 'preferred_format': ''}]}
        with patch('reco.search.clarify', return_value=(question, 'ok')):
            first = search.search(task='бизнес', profile_id='anon:one')
        token = first['question']['options'][0]['value']
        with patch('reco.search.clarify', side_effect=AssertionError('Нельзя экспортировать подменённый ответ')), \
             patch.object(self.engine, 'recommend', side_effect=AssertionError('Нельзя ранжировать подменённый ответ')):
            for data in ({'task': 'бизнес', 'profile_id': 'anon:one', 'clarifications': ['Ivan ivan@example.com']},
                         {'task': 'бизнес', 'profile_id': 'anon:two', 'clarifications': [token]},
                         {'task': 'налоги', 'profile_id': 'anon:one', 'clarifications': [token]}):
                with self.subTest(task=data['task']), self.assertRaises(ValueError): search.search(**data)
        # Обычный поиск не использует диалог даже при наличии старых токенов.
        self.assertTrue(search.search(task='налоги', search_mode='normal', clarifications=['ignored'])['items'])

    def test_long_query_keeps_selected_answer_in_ranking_text(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        task = 'маркетинг ' * 200
        question = {'title': 'Какая задача сейчас наиболее важна?', 'options': [
            {'label': 'Налоги и учет', 'preferred_format': ''},
            {'label': 'Маркетинг и продажи', 'preferred_format': ''}]}
        with patch('reco.search.clarify', return_value=(question, 'ok')):
            first = search.search(task=task)
        token = first['question']['options'][0]['value']
        with patch('reco.search.clarify', return_value=(None, 'ok')), \
             patch.object(self.engine, 'recommend', wraps=self.engine.recommend) as rank:
            search.search(task=task, clarifications=[token])
        self.assertTrue(rank.call_args.kwargs['task'].startswith(question['title'] + ' Налоги и учет'))
        self.assertLessEqual(len(rank.call_args.kwargs['task']), 2000)

    def test_smart_search_unknown_query_and_no_persistent_query(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        before = self.engine.store.read('anon:search')
        result = search.search(task='квантовые телепортаторы на Юпитере', profile_id='anon:search', answers={'stage': 'owner', 'goal': 'finance'})
        self.assertEqual(result['match_status'], 'no_match')
        self.assertIn('не нашли', result['summary'])
        self.assertEqual(self.engine.store.read('anon:search'), before)
        for row in result['items']:
            self.assertNotIn(row['id'], ['course:closed'])
            if row['next_material']:
                self.assertIn(row['next_material']['id'], [r['id'] for r in result['items']])

    def test_smart_search_continuations_completed_and_negative(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        self.engine.event('anon:continue', 'course:a', 'click', 'one')
        self.assertIsNone(search.search(profile_id='anon:continue')['question'])
        rows = search.search(task='налоги', profile_id='anon:continue')['items']
        self.assertTrue(next(r for r in rows if r['id'] == 'course:a')['continuation'])
        self.engine.event('anon:continue', 'course:a', 'complete', 'two')
        self.assertNotIn('course:a', [r['id'] for r in search.search(task='налоги', profile_id='anon:continue')['items']])
        self.engine.event('anon:continue', 'course:b', 'dislike', 'three')
        self.assertNotIn('course:b', [r['id'] for r in search.search(task='налоги', profile_id='anon:continue')['items']])

    def test_smart_search_rejects_invalid_answers_flags_and_fields(self):
        from reco.search import SmartSearch
        search = SmartSearch(self.engine)
        for data in ({'answers': []}, {'answers': {'stage': ['owner']}}, {'answers': {'goal': 'unknown'}},
                     {'answers': {'secret': 'yes'}}, {'skip_questions': 'false'}, {'use_llm': 1},
                     {'task': None}, {'industry': 8}, {'task': 'x' * 2001}, {'search_mode': 'invalid'}):
            with self.subTest(data=data), self.assertRaises(ValueError): search.search(**data)

    def test_task_vector_has_priority_over_long_generic_context(self):
        result = self.engine.recommend(task='налоги учет', stage='маркетинг продажи бизнес ' * 30)
        self.assertEqual(result['items'][0]['topics'], ['Налоги'])

    def test_missing_history_format_dislikes_and_popularity_have_safe_fallbacks(self):
        result = self.engine.recommend()
        self.assertEqual(result['query_vector_source'], 'catalog_centroid')
        self.assertTrue(result['items'])
        for row in result['items']:
            self.assertGreater(row['cosine'], 0)
            self.assertEqual(row['factors']['dislike'], 0)
            self.assertEqual(row['factors']['format'], 0)
            self.assertEqual(row['factors']['popularity'], 0)
            self.assertAlmostEqual(sum(row['score_breakdown']['contributions'].values()), row['score'], places=4)
        json.dumps(result, allow_nan=False)

    def test_no_history_preserves_explicit_query_vector(self):
        direction = normalize(np.ones(self.engine.vectors.shape[1]))
        with patch.object(self.engine.encoder, 'encode', return_value=np.asarray([direction])) as encode:
            result = self.engine.recommend(task='моя задача')
        encode.assert_called_once_with(['моя задача'])
        self.assertEqual(result['query_vector_source'], 'query')

    def test_zero_query_uses_selected_topic_before_general_fallback(self):
        # В маленьком TF-IDF fixture слова одной карточки не проходят min_df=2.
        result = self.engine.recommend(task='маркетинг', topics=['Маркетинг'])
        self.assertEqual(result['query_vector_source'], 'topic_centroid')
        self.assertEqual(result['items'][0]['id'], 'course:c')

    def test_signed_cosine_stays_raw_in_response_and_nonnegative_in_hybrid(self):
        direction = np.zeros(self.engine.vectors.shape[1], dtype=np.float32); direction[0] = 1
        self.engine.vectors[0] = direction
        self.engine.vectors[1] = np.zeros_like(direction)
        self.engine.vectors[1, :2] = [-0.5, np.sqrt(0.75)]
        self.engine.vectors[2] = np.zeros_like(direction); self.engine.vectors[2, 1] = 1
        with patch.object(self.engine.encoder, 'encode', side_effect=lambda texts: np.tile(direction, (len(texts), 1))):
            result = self.engine.recommend(task='задача', topics=['Налоги', 'Маркетинг'])
        row = next(row for row in result['items'] if row['id'] == 'course:b')
        self.assertAlmostEqual(row['cosine'], -0.5)
        self.assertEqual(row['factors']['cosine'], 0)
        self.assertEqual(row['score_breakdown']['contributions']['cosine'], 0)
        self.assertLessEqual(row['score_breakdown']['diversity_penalties']['similarity'], 0)

    def test_topic_match_is_metadata_even_with_identical_embeddings(self):
        direction = normalize(np.ones(self.engine.vectors.shape[1]))
        self.engine.vectors[:] = direction
        with patch.object(self.engine.encoder, 'encode', side_effect=lambda texts: np.tile(direction, (len(texts), 1))):
            result = self.engine.recommend(task='задача', topics=['Маркетинг'])
        rows = {row['id']: row for row in result['items']}
        self.assertEqual(result['items'][0]['id'], 'course:c')
        self.assertEqual(rows['course:c']['factors']['topic'], 1)
        self.assertEqual(rows['course:a']['factors']['topic'], 0)
        self.assertEqual(rows['course:b']['factors']['topic'], 0)
        self.assertTrue(all(abs(row['cosine'] - 1) < 1e-6 for row in rows.values()))

    def test_dislike_similarity_penalty_is_bounded_and_absent_without_dislikes(self):
        direction = np.zeros(self.engine.vectors.shape[1], dtype=np.float32); direction[0] = 1
        other = np.zeros_like(direction); other[1] = 1
        self.engine.vectors[0] = direction
        self.engine.vectors[1] = direction
        self.engine.vectors[2] = other
        with patch.object(self.engine.encoder, 'encode', side_effect=lambda texts: np.tile(direction, (len(texts), 1))):
            before = self.engine.recommend(task='задача', topics=['Налоги', 'Маркетинг'])
            self.engine.event('anon:penalty', 'course:a', 'dislike', 'one')
            after = self.engine.recommend(task='задача', topics=['Налоги', 'Маркетинг'], profile_id='anon:penalty')
        first = {row['id']: row for row in before['items']}
        last = {row['id']: row for row in after['items']}
        self.assertNotIn('course:a', last)
        self.assertAlmostEqual(first['course:b']['score'] - last['course:b']['score'], 0.12)
        self.assertEqual(last['course:b']['factors']['dislike'], 1)
        self.assertEqual(last['course:b']['score_breakdown']['contributions']['dislike'], -0.12)
        self.assertEqual(last['course:c']['factors']['dislike'], 0)

    def test_ranking_log_contains_numeric_contributions_without_task_or_ids(self):
        with self.assertLogs('reco.ranking', level='INFO') as log:
            result = self.engine.recommend(task='налоги секретный текст', profile_id='anon:private-profile')
        self.assertEqual(len(log.records), len(result['items']))
        for rank, record in enumerate(log.records, 1):
            message = record.getMessage()
            payload = json.loads(message)
            self.assertEqual(set(payload), {'event','algorithm','rank','factors','score_breakdown'})
            self.assertEqual(payload['rank'], rank)
            self.assertAlmostEqual(sum(payload['score_breakdown']['contributions'].values()),
                                   result['items'][rank-1]['score'], places=4)
            for forbidden in ('секретный', 'private-profile', 'course:', 'profile_id', 'title', 'task'):
                self.assertNotIn(forbidden, message)

    def test_empty_catalog_has_finite_zero_fallback(self):
        self.engine.catalog = []
        self.engine.vectors = self.engine.vectors[:0]
        self.engine.index = {}
        self.engine.popularity = np.zeros(0)
        result = self.engine.recommend()
        self.assertEqual(result['items'], [])
        self.assertEqual(result['query_vector_source'], 'zero')
        json.dumps(result, allow_nan=False)

    def test_search_empty_catalog_and_old_export_flag(self):
        from reco.search import SmartSearch
        for row in self.engine.catalog:
            row['llm_public_metadata'] = True
        with patch('reco.engine.verify', return_value=({}, 'privacy_blocked')) as verify_call:
            self.engine.recommend(task='налоги', use_llm=True)
        self.assertFalse(any(r['llm_public_metadata'] for r in verify_call.call_args.args[1]))
        for row in self.engine.catalog: row['eligible'] = False
        result = SmartSearch(self.engine).search(skip_questions=True)
        self.assertEqual(result['match_status'], 'empty')
        self.assertEqual(result['items'], [])

    def test_web_http_contract_and_encoded_delete(self):
        import threading
        import urllib.request
        import urllib.error
        from reco.server import create_server
        server = create_server(self.engine, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base = 'http://127.0.0.1:' + str(server.server_port)
        with urllib.request.urlopen(base + '/') as response:
            self.assertIn('text/html', response.headers['Content-Type'])
            self.assertIn('frame-ancestors', response.headers['Content-Security-Policy'])
            self.assertIn('Умный поиск', response.read().decode())
        with urllib.request.urlopen(base + '/health') as response:
            self.assertEqual(json.load(response)['status'], 'ok')
        body = json.dumps({'task': 'налоги', 'skip_questions': True}).encode()
        req = urllib.request.Request(base + '/smart-search', body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            self.assertTrue(result['items'])
            self.assertIn('query_vector_source', result)
            for row in result['items']:
                self.assertTrue(all(0 <= value <= 1 for value in row['factors'].values()))
                trace = row['score_breakdown']
                self.assertEqual(trace['formula'], 'hybrid-v2')
                self.assertAlmostEqual(sum(trace['contributions'].values()), row['score'], places=4)
                self.assertAlmostEqual(trace['base_score'] + sum(trace['diversity_penalties'].values()),
                                       trace['selection_score'])
        req = urllib.request.Request(base + '/search', body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            self.assertEqual(result['search_mode'], 'normal')
            self.assertIsNone(result['question'])
        dynamic = {'title': 'Как вы хотите развивать продажи?', 'options': [
            {'label': 'Маркетинг и продажи', 'preferred_format': ''},
            {'label': 'Налоги и учет', 'preferred_format': ''}]}
        payload = {'task': 'бизнес', 'search_mode': 'smart'}
        with patch('reco.search.clarify', return_value=(dynamic, 'ok')):
            req = urllib.request.Request(base + '/search', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                token = json.load(response)['question']['options'][0]['value']
        payload['clarifications'] = [token]
        with patch('reco.search.clarify', return_value=(None, 'ok')):
            req = urllib.request.Request(base + '/search', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                self.assertEqual(json.load(response)['dialogue'][0]['answer'], 'Маркетинг и продажи')
        payload['clarifications'] = [token + 'x']
        with patch('reco.search.clarify', side_effect=AssertionError('Подмена не должна дойти до LLM')):
            req = urllib.request.Request(base + '/search', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code, 400)
        req = urllib.request.Request(base + '/profiles/anon%3Aweb', method='DELETE')
        with urllib.request.urlopen(req) as response:
            self.assertTrue(json.load(response)['deleted'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + '/../../env.example')
        self.assertEqual(error.exception.code, 404)


class VerificationTests(unittest.TestCase):
    def test_llm_cannot_invent_ids_or_evidence(self):
        output = {'items': [{'id': 'other', 'relevant': True, 'evidence': 'источник'},
                            {'id': 'one', 'relevant': True, 'evidence': 'выдуманная цитата'}]}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'message': {'content': json.dumps(output)}}).encode()
        with patch.dict('os.environ', {'LLM_PROVIDER': 'ollama', 'OLLAMA_URL': 'http://localhost:11434', 'OLLAMA_MODEL': 'fixture'}), patch('urllib.request.urlopen', return_value=response):
            result, status = verify('task', [{'id': 'one', 'title': 'источник', 'description': 'описание', 'topics': []}])
        self.assertEqual(result, {})
        self.assertEqual(status, 'partial_invalid_output')

    def test_ndcg_has_known_perfect_and_reversed_values(self):
        self.assertEqual(metrics(['a', 'b'], {'a': 3, 'b': 1}, 2)['ndcg'], 1)
        self.assertLess(metrics(['b', 'a'], {'a': 3, 'b': 1}, 2)['ndcg'], 1)
        self.assertEqual(metrics(['x', 'y'], {'a': 3}, 2)['precision'], 0)


if __name__ == '__main__':
    unittest.main()
