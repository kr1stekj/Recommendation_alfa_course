"""Модель генерирует вопрос и произвольные варианты; транспорт в unit-тестах подставлен."""
import copy
import json
import os
from pathlib import Path
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from reco.clarification import clarify, SCHEMA


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'LLM_PROVIDER': 'openrouter', 'LLM_API_KEY': 'fixture', 'LLM_MODEL': 'fixture'}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.opener = MagicMock()
        self.transport = patch('urllib.request.build_opener', return_value=self.opener)
        self.transport.start(); self.addCleanup(self.transport.stop)
        self.question = {'title': 'Как вы хотите привлекать клиентов?', 'options': [
            {'label': 'Через партнёрства', 'preferred_format': ''},
            {'label': 'Через платную рекламу', 'preferred_format': ''}]}
        self.respond({'needs_clarification': True, 'question': self.question})

    def respond(self, answer):
        envelope = {'choices': [{'message': {'content': json.dumps(answer)}}]}
        self.opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(envelope).encode()

    def call(self, task='бизнес', history=None, preferences=None, hint='general', formats=('course', 'article')):
        return clarify(task, history or [], preferences or {}, hint, False, formats)

    def test_question_and_answers_are_authored_by_model(self):
        question, status = self.call()
        self.assertEqual(status, 'ok')
        self.assertEqual(question, self.question)
        self.assertEqual(question['options'][0]['label'], 'Через партнёрства')
        self.respond({'needs_clarification': False, 'question': None})
        self.assertEqual(self.call(), (None, 'ok'))

    def test_numeric_budget_and_new_business_options_not_limited_to_old_choices(self):
        question = {'title': 'Какой бюджет вы выделяете на продвижение?', 'options': [
            {'label': '0 ₽', 'preferred_format': ''},
            {'label': 'До 50 000 ₽', 'preferred_format': ''},
            {'label': 'Более 50 000 ₽', 'preferred_format': ''}]}
        self.respond({'needs_clarification': True, 'question': question})
        self.assertEqual(self.call()[0], question)

    def test_business_channels_are_not_personal_contact_requests(self):
        question = {'title': 'Какой канал продвижения вам интересен?', 'options': [
            {'label': 'Email-маркетинг', 'preferred_format': ''},
            {'label': 'Телефонные продажи', 'preferred_format': ''}]}
        self.respond({'needs_clarification': True, 'question': question})
        self.assertEqual(self.call()[0], question)

    def test_projection_has_no_raw_query_identity_catalog_or_tokens(self):
        self.call('Я Иван Петров, ivan@example.com, паспорт 1111 222222. Как привлечь клиентов?')
        req = self.opener.open.call_args.args[0]
        body = json.loads(req.data); payload = json.loads(body['messages'][1]['content'])
        text = json.dumps(payload, ensure_ascii=False)
        for forbidden in ('Иван', 'Петров', 'ivan@', '1111', '222222', 'catalog', 'profile_id', 'task', 'available_questions'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(payload['topics'], ['Продажи'])
        self.assertEqual(payload['actions'], ['Привлечь клиентов'])
        self.assertEqual(payload['answers'], [])
        self.assertEqual(body['max_tokens'], 900)
        self.assertEqual(body['provider']['data_collection'], 'deny')
        self.assertTrue(body['provider']['zdr'])

    def test_verified_prior_question_and_answer_inform_next_question(self):
        history = [{'question': 'Как вы хотите продвигаться?', 'answer': 'Через партнёрства'}]
        self.call(history=history)
        body = json.loads(self.opener.open.call_args.args[0].data)
        projected = json.loads(body['messages'][1]['content'])['answers']
        self.assertIn('Партнёрства', projected[0]['answer_facets'])
        self.assertNotIn('Через партнёрства', json.dumps(projected, ensure_ascii=False))

    def test_prior_personal_attributes_cannot_escape_even_if_generated_by_model(self):
        for question, answer in [('Как к вам обращаться?', 'Иван Иванов'),
                                 ('Как вас называть?', 'Пётр Петров'),
                                 ('Сколько вам лет?', '18–25'),
                                 ('Какой у вас домашний адрес?', 'Улица Ленина, дом 12')]:
            with self.subTest(question=question):
                self.call(history=[{'question': question, 'answer': answer}])
                body = json.loads(self.opener.open.call_args.args[0].data)
                exported = json.dumps(json.loads(body['messages'][1]['content']), ensure_ascii=False)
                self.assertNotIn(question, exported)
                self.assertNotIn(answer, exported)
                self.assertNotIn('Иван', exported)
                self.assertNotIn('Петров', exported)

    def test_budget_is_projected_to_constant_range_not_original_number(self):
        self.call(history=[{'question': 'Какой бюджет на рекламу?', 'answer': 'До 54 321 ₽'}])
        body = json.loads(self.opener.open.call_args.args[0].data)
        exported = json.loads(body['messages'][1]['content'])
        self.assertIn('Бюджет до 100 тысяч рублей', exported['answers'][0]['answer_facets'])
        self.assertNotIn('54 321', json.dumps(exported, ensure_ascii=False))

    def test_budget_projection_keeps_bound_direction_and_omits_ambiguous_numbers(self):
        from reco.llm_privacy import dialogue_projection
        cases = [('Более 100 000 ₽', 'Бюджет свыше 100 тысяч рублей'),
                 ('Свыше 10 000 ₽', None), ('До 0,01 млн рублей', None),
                 ('До 1 млн рублей', None), ('От 100 000 ₽', None),
                 ('От 100 001 ₽', 'Бюджет свыше 100 тысяч рублей'),
                 ('50 000–100 000 ₽', None), ('Не более 5 тыс. рублей', 'Бюджет до 10 тысяч рублей'),
                 ('До 50 000 ₽', 'Бюджет до 100 тысяч рублей'), ('0 ₽', 'Без рекламного бюджета'),
                 ('Более 0 ₽', None), ('Менее 0 ₽', None)]
        for label, expected in cases:
            with self.subTest(label=label):
                row = dialogue_projection([{'question': 'Какой бюджет на рекламу?', 'answer': label}])[0]
                buckets = [value for value in row['answer_facets'] if value.startswith('Бюджет ') or value == 'Без рекламного бюджета']
                self.assertEqual(buckets, [] if expected is None else [expected])

    def test_invalid_questions_options_and_personal_requests_rejected(self):
        questions = [None, {}, dict(self.question, title='Какой у вас телефон?'),
                     dict(self.question, title='Какой вариант подходит\ud800?'),
                     *[dict(self.question, title=title) for title in ('Как к вам обращаться?', 'Как вас называть?',
                         'Сколько вам лет?', 'Какой у вас домашний адрес?')],
                     dict(self.question, title='Укажите email?'), dict(self.question, title='<script>Задача?</script>'),
                     dict(self.question, title='Какая ваша задача? https://example.com'),
                     dict(self.question, options=[]), dict(self.question, options='bad'),
                     dict(self.question, options=[self.question['options'][0]] * 2)]
        for label, fmt in [('Введите свой пароль', ''), ('<script>Да</script>', ''),
                           ('Ваш контакт ivan@example.com', ''), ('', ''), ('Ответ ' * 30, ''),
                           ('Готовый шаблон', 'document'), ('Курс', ['course']), ('Соцсети\ud800', '')]:
            question = copy.deepcopy(self.question)
            question['options'][0] = {'label': label, 'preferred_format': fmt}
            questions.append(question)
        for question in questions:
            with self.subTest(question=question):
                self.respond({'needs_clarification': True, 'question': question})
                self.assertEqual(self.call(), (None, 'invalid_output'))
        for answer in ({'needs_clarification': 'true', 'question': self.question},
                       {'needs_clarification': False, 'question': self.question},
                       {'needs_clarification': False, 'question': None, 'secret': True}):
            self.respond(answer)
            self.assertEqual(self.call(), (None, 'invalid_output'))

    def test_repeated_question_and_two_question_cap(self):
        history = [{'question': self.question['title'], 'answer': 'Через партнёрства'}]
        self.assertEqual(self.call(history=history), (None, 'already_answered'))
        self.opener.open.reset_mock()
        self.assertEqual(self.call(history=history * 2), (None, 'limit_reached'))
        self.opener.open.assert_not_called()

    def test_paraphrased_known_parameter_is_not_asked_again_but_new_parameter_is_allowed(self):
        history = [{'question': 'Какая цель продвижения вашего бизнеса?',
                    'answer': 'Увеличение узнаваемости бренда'}]
        question = dict(self.question, title='Какая основная цель маркетинга вашего бизнеса?')
        self.respond({'needs_clarification': True, 'question': question})
        self.assertEqual(self.call(history=history), (None, 'already_answered'))
        question['title'] = 'Какой канал продвижения вам интересен?'
        self.respond({'needs_clarification': True, 'question': question})
        self.assertEqual(self.call(history=history), (question, 'ok'))

    def test_missing_key_and_provider_errors_have_no_static_fallback(self):
        with patch.dict(os.environ, {'LLM_API_KEY': ''}): self.assertEqual(self.call(), (None, 'disabled'))
        self.opener.open.assert_not_called()
        self.opener.open.side_effect = TimeoutError('PRIVATE ERROR')
        self.assertEqual(self.call(), (None, 'unavailable_or_invalid_output'))
        self.opener.open.side_effect = urllib.error.HTTPError('https://example', 429, 'PRIVATE ERROR', {}, None)
        self.assertEqual(self.call(), (None, 'rate_limited'))
        self.opener.open.side_effect = None
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'not-json'
        self.assertEqual(self.call(), (None, 'unavailable_or_invalid_output'))

    def test_schema_file_matches_runtime_contract(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / 'specs/llm-clarification.schema.json').read_text())
        schema.pop('$schema')
        self.assertEqual(schema, SCHEMA)
