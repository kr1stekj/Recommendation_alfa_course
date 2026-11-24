"""Фокус уточнения и безопасные признаки профиля; ответы LLM в тестах подставлены."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from reco.clarification import clarify
from reco.llm_privacy import business_summary, request_focus


class ClarificationContextTests(unittest.TestCase):
    def setUp(self):
        self.generic = {'title':'Какую бизнес-задачу вы хотите решить?', 'options':[
            {'label':'Продвижение', 'preferred_format':''},
            {'label':'Учёт', 'preferred_format':''}]}
        self.ai = {'title':'Для какой задачи вы хотите использовать искусственный интеллект?', 'options':[
            {'label':'Создавать контент', 'preferred_format':''},
            {'label':'Анализировать данные', 'preferred_format':''}]}

    def call(self, task='искуственный интеллект', profile_topics=None, history=None):
        return clarify(task, history or [], {}, 'general', bool(profile_topics),
                       {'article','course'}, profile_topics=profile_topics)

    def test_common_ai_spelling_errors_share_safe_focus(self):
        for task in ('искуственный интеллект','искусственный интеллект','искуственый интелект',
                     'искусственного интеллекта','ИИ','AI','нейросети'):
            with self.subTest(task=task):
                self.assertIn('Автоматизация', business_summary(task))
                self.assertEqual(request_focus(task), ['Искусственный интеллект'])
        self.assertEqual(request_focus('CRM'), ['CRM'])
        self.assertNotIn('Автоматизация', business_summary('искусство продаж'))

    def test_generic_question_repaired_once_without_transmitting_rejected_text(self):
        responses = [({'needs_clarification':True, 'question':self.generic}, 'ok'),
                     ({'needs_clarification':True, 'question':self.ai}, 'ok')]
        with patch('reco.clarification.structured_call', side_effect=responses) as cloud:
            question, status = self.call(profile_topics=['Маркетинг','Продажи'])
        self.assertEqual((question,status), (self.ai,'ok'))
        self.assertEqual(cloud.call_count, 2)
        first = cloud.call_args_list[0].args[0]
        retry = cloud.call_args_list[1].args[0]
        self.assertEqual(first['request_focus'], ['Искусственный интеллект'])
        self.assertEqual(first['profile_topics'], ['Маркетинг','Продажи'])
        self.assertNotIn('validation_feedback', first)
        self.assertIn('validation_feedback', retry)
        self.assertNotIn(self.generic['title'], json.dumps(retry,ensure_ascii=False))
        self.assertEqual({k:v for k,v in retry.items() if k!='validation_feedback'}, first)

    def test_persistent_generic_or_wrong_topic_is_never_shown(self):
        for question in (self.generic, dict(self.generic,title='Какой канал продвижения вам интересен?')):
            with self.subTest(question=question), patch('reco.clarification.structured_call',
                    return_value=({'needs_clarification':True,'question':question},'ok')) as cloud:
                self.assertEqual(self.call(), (None,'topic_mismatch'))
                self.assertEqual(cloud.call_count, 2)

    def test_relevant_options_do_not_rescue_generic_title(self):
        generic = dict(self.generic,options=self.ai['options'])
        with patch('reco.clarification.structured_call', return_value=(
                {'needs_clarification':True,'question':generic},'ok')):
            self.assertEqual(self.call(), (None,'topic_mismatch'))

    def test_question_can_be_unneeded_and_unrecognized_task_can_be_general(self):
        with patch('reco.clarification.structured_call', return_value=(
                {'needs_clarification':False,'question':None},'ok')) as cloud:
            self.assertEqual(self.call(), (None,'ok'))
            self.assertEqual(cloud.call_count, 1)
        with patch('reco.clarification.structured_call', return_value=(
                {'needs_clarification':True,'question':self.generic},'ok')):
            self.assertEqual(self.call(task='бизнес'), (self.generic,'ok'))

    def test_original_request_focus_wins_over_enriched_context(self):
        with patch('reco.clarification.structured_call', return_value=(
                {'needs_clarification':True,'question':self.ai},'ok')) as cloud:
            clarify('Маркетинг и продвижение ИИ для контента',[],{},'general',True,{'article'},
                    profile_topics=['Финансы и учет'],request_task='искуственный интеллект')
        self.assertEqual(cloud.call_args.args[0]['request_focus'], ['Искусственный интеллект'])

    def test_wire_contains_only_constant_topics_not_query_or_raw_profile_labels(self):
        opener = MagicMock()
        envelope = {'choices':[{'message':{'content':json.dumps({'needs_clarification':True,'question':self.ai})}}]}
        opener.open.return_value.__enter__.return_value.read.return_value=json.dumps(envelope).encode()
        with patch.dict(os.environ,{'LLM_PROVIDER':'openrouter','LLM_API_KEY':'fixture','LLM_MODEL':'fixture'},clear=True), \
             patch('urllib.request.build_opener',return_value=opener):
            self.call(task='искуственный интеллект Иван Петров ivan@example.com',
                      profile_topics=['Маркетинг Иван Петров','Продажи +7 999 1234567','СЕКРЕТНАЯ МЕТКА'])
        body = json.loads(opener.open.call_args.args[0].data)
        payload = json.loads(body['messages'][1]['content'])
        self.assertEqual(payload['request_focus'], ['Искусственный интеллект'])
        self.assertEqual(payload['profile_topics'], ['Маркетинг','Продажи'])
        serialized=json.dumps(payload,ensure_ascii=False)
        for forbidden in ('искуственный','Иван','Петров','ivan@','999','1234567','СЕКРЕТНАЯ','profile_id'):
            self.assertNotIn(forbidden,serialized)
        self.assertEqual(body['max_tokens'],900)
        self.assertTrue(body['provider']['zdr'])

    def test_errors_and_two_question_cap_do_not_retry(self):
        for response in (({},'rate_limited'), ({'needs_clarification':True,'question':{}},'ok')):
            with self.subTest(response=response), patch('reco.clarification.structured_call',return_value=response) as cloud:
                self.assertNotEqual(self.call()[1], 'ok')
                self.assertEqual(cloud.call_count,1)
        with patch('reco.clarification.structured_call',side_effect=AssertionError('Лимит')):
            self.assertEqual(self.call(history=[{'question':'Как использовать ИИ?','answer':'Контент'}]*2),
                             (None,'limit_reached'))
