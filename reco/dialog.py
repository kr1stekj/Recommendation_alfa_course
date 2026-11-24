"""Подписанные варианты временного диалога без хранения вопросов в БД."""
import base64
import hashlib
import hmac
import json
import re
import secrets
import time

MAX_ANSWERS = 2
MAX_TOKEN_LENGTH = 4096


def _encode(value):
    return base64.urlsafe_b64encode(value).decode().rstrip('=')


def _decode(value):
    return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)


class DialogCodec:
    """Подтверждает происхождение ответа от модели для данного запроса.

    Ключ живёт в одном экземпляре API. Перезапуск сбрасывает незавершённые диалоги;
    токены не шифруются и не содержат пользовательскую строку или секреты.
    """
    def __init__(self, ttl=3600):
        self._key = secrets.token_bytes(32)
        self.ttl = ttl

    def binding(self, context):
        raw = json.dumps(context, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        return hmac.new(self._key, b'context:' + raw, hashlib.sha256).hexdigest()

    def _sign(self, encoded):
        return _encode(hmac.new(self._key, b'choice:' + encoded.encode(), hashlib.sha256).digest())

    @staticmethod
    def _parent(tokens):
        return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()

    def issue(self, question, binding, step, now=None, prior=None):
        now = time.time() if now is None else now
        qid = secrets.token_urlsafe(12)
        options = []
        for option in question['options']:
            claim = {'v': 1, 'qid': qid, 'question': question['title'], 'answer': option['label'],
                     'preferred_format': option['preferred_format'], 'binding': binding,
                     'expires': int(now + self.ttl), 'parent': self._parent(prior or [])}
            raw = json.dumps(claim, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
            encoded = _encode(raw)
            options.append({'label': option['label'], 'value': encoded + '.' + self._sign(encoded)})
        return {'id': qid, 'title': question['title'], 'step': step, 'total': MAX_ANSWERS, 'options': options}

    def read(self, tokens, binding, now=None):
        now = time.time() if now is None else now
        if not isinstance(tokens, list) or len(tokens) > MAX_ANSWERS:
            raise ValueError('Expected at most two clarification tokens')
        selected, seen = [], set()
        for offset, token in enumerate(tokens):
            try:
                if not isinstance(token, str) or len(token) > MAX_TOKEN_LENGTH or not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', token):
                    raise ValueError()
                encoded, signature = token.split('.')
                if not hmac.compare_digest(signature, self._sign(encoded)):
                    raise ValueError()
                claim = json.loads(_decode(encoded))
                if (claim.get('v') != 1 or not hmac.compare_digest(claim['binding'], binding) or
                        not hmac.compare_digest(claim['parent'], self._parent(tokens[:offset])) or
                        type(claim['expires']) is not int or claim['expires'] <= now or claim['qid'] in seen):
                    raise ValueError()
                seen.add(claim['qid'])
                selected.append(claim)
            except (ValueError, TypeError, KeyError, AttributeError):
                raise ValueError('Clarification is invalid or expired; start the search again') from None
        return selected
