"""Подпись связывает выбранный ответ с запросом и предшествующим диалогом."""
import base64
import json
import unittest

from reco.dialog import DialogCodec


class DialogTokenTests(unittest.TestCase):
    def setUp(self):
        self.codec = DialogCodec(ttl=60)
        self.binding = self.codec.binding({'task': 'бизнес', 'profile': 'anon:fixture', 'dataset': 'one'})
        self.question = {'title': 'Как вы хотите привлекать клиентов?', 'options': [
            {'label': 'Через партнёрства', 'preferred_format': ''},
            {'label': 'Через платную рекламу', 'preferred_format': ''}]}

    def issue(self, prior=None):
        return self.codec.issue(self.question, self.binding, 1 if not prior else 2, now=1000, prior=prior)

    def test_round_trip_two_selected_answers(self):
        first = self.issue()['options'][0]['value']
        second = self.issue([first])['options'][1]['value']
        rows = self.codec.read([first, second], self.binding, now=1010)
        self.assertEqual([row['answer'] for row in rows], ['Через партнёрства', 'Через платную рекламу'])
        self.assertEqual(rows[0]['question'], self.question['title'])

    def test_client_cannot_replace_generated_text(self):
        token = self.issue()['options'][0]['value']
        encoded, signature = token.split('.')
        claim = json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)))
        claim['answer'] = 'Иван Петров ivan@example.com'
        changed = base64.urlsafe_b64encode(json.dumps(claim).encode()).decode().rstrip('=') + '.' + signature
        with self.assertRaises(ValueError): self.codec.read([changed], self.binding, now=1010)

    def test_token_cannot_move_to_other_query_profile_or_catalog(self):
        token = self.issue()['options'][0]['value']
        for context in ({'task': 'налоги', 'profile': 'anon:fixture', 'dataset': 'one'},
                        {'task': 'бизнес', 'profile': 'anon:other', 'dataset': 'one'},
                        {'task': 'бизнес', 'profile': 'anon:fixture', 'dataset': 'two'}):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.codec.read([token], self.codec.binding(context), now=1010)

    def test_expired_tokens_and_server_restart_require_new_dialogue(self):
        token = self.issue()['options'][0]['value']
        with self.assertRaises(ValueError): self.codec.read([token], self.binding, now=1060)
        with self.assertRaises(ValueError): DialogCodec().read([token], self.binding, now=1010)

    def test_two_options_of_same_question_and_forked_dialogue_rejected(self):
        question = self.issue()
        first, alternative = [row['value'] for row in question['options']]
        child = self.issue([first])['options'][0]['value']
        for tokens in ([first, alternative], [alternative, child], [first, self.issue()['options'][0]['value']]):
            with self.subTest(tokens_count=len(tokens)), self.assertRaises(ValueError):
                self.codec.read(tokens, self.binding, now=1010)

    def test_malformed_or_excessive_answers_rejected(self):
        token = self.issue()['options'][0]['value']
        for tokens in ({}, [None], ['.'], ['x' * 5000], [token, token, token]):
            with self.subTest(kind=type(tokens).__name__), self.assertRaises(ValueError):
                self.codec.read(tokens, self.binding, now=1010)
