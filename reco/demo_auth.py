"""Ограниченные серверные сессии для общедоступных синтетических ролей."""
from http.cookies import SimpleCookie, CookieError
import logging
import re
import secrets
import threading
import time

from .demo_profiles import DemoProfiles, DEMO_PREFIX

COOKIE_NAME = 'reco_demo_session'


class DemoAuthError(ValueError):
    def __init__(self, message, status=401):
        super().__init__(message)
        self.status = status


def session_token(header):
    try:
        cookie = SimpleCookie(); cookie.load(header or '')
        token = cookie[COOKIE_NAME].value if COOKIE_NAME in cookie else None
        return token if token and re.fullmatch(r'[A-Za-z0-9_-]{43}', token) else None
    except CookieError:
        return None


def session_cookie(token=None):
    cookie = SimpleCookie(); cookie[COOKIE_NAME] = token or ''
    cookie[COOKIE_NAME]['httponly'] = True
    cookie[COOKIE_NAME]['samesite'] = 'Strict'
    cookie[COOKIE_NAME]['path'] = '/'
    cookie[COOKIE_NAME]['max-age'] = 3600 if token else 0
    return cookie.output(header='').strip()


class DemoAuth:
    def __init__(self, engine, ttl=3600, max_sessions=512, clock=None):
        self.profiles = DemoProfiles(engine)
        self.sessions = {}
        self.ttl, self.max_sessions = ttl, max_sessions
        self.clock = clock or time.time
        self.lock = threading.RLock()

    def _discard(self, token):
        session = self.sessions.pop(token, None)
        if session:
            try:
                self.profiles.engine.store.delete(session['profile_id'])
            except Exception:
                # Сессия отзывается и при отказе БД; агрегат остаётся под гостевым TTL.
                logging.exception('Demo profile cleanup failed')

    def cleanup(self):
        with self.lock:
            for token in list(self.sessions):
                if self.sessions[token]['expires'] <= self.clock():
                    self._discard(token)

    def get(self, token):
        with self.lock:
            session = self.sessions.get(token)
            if session and session['expires'] <= self.clock():
                self._discard(token)
                return None
            return session

    def snapshot(self, token):
        with self.lock:
            session = self.get(token)
            account = (dict(self.profiles.accounts[session['account_id']], seed_items=session['seed_items'])
                       if session else None)
            return {'authenticated': bool(session), 'account': account,
                    'profile_id': session['profile_id'] if session else None, 'synthetic': True}

    def login(self, account_id, previous=None):
        with self.lock:
            if not isinstance(account_id, str) or account_id not in self.profiles.accounts:
                raise ValueError('Unknown demo account')
            self.cleanup()
            if len(self.sessions) >= self.max_sessions and previous not in self.sessions:
                raise DemoAuthError('Too many demo sessions; try later', 429)
            profile_id, count = self.profiles.create(account_id, self.clock())
            token = secrets.token_urlsafe(32)
            self._discard(previous)
            self.sessions[token] = {'profile_id': profile_id, 'account_id': account_id,
                                    'seed_items': count, 'expires': self.clock() + self.ttl}
            return token, self.snapshot(token)

    def logout(self, token):
        with self.lock:
            self._discard(token)
            return self.snapshot(None)

    def reset(self, token):
        with self.lock:
            session = self.get(token)
            if not session:
                raise DemoAuthError('Demo session expired; sign in again')
            session['seed_items'] = self.profiles.reset(session['profile_id'], session['account_id'], self.clock())
            return self.snapshot(token)

    def bind_profile(self, profile_id, token):
        with self.lock:
            session = self.get(token)
            if token and not session:
                raise DemoAuthError('Demo session expired; sign in again')
            if session:
                if profile_id is not None and profile_id != session['profile_id']:
                    raise DemoAuthError('This profile belongs to another session', 403)
                return session['profile_id']
            if isinstance(profile_id, str) and profile_id.startswith(DEMO_PREFIX):
                raise DemoAuthError('Demo profile requires its session')
            return profile_id

    def forbid_merge(self, token, *profile_ids):
        if token or any(isinstance(value, str) and value.startswith(DEMO_PREFIX) for value in profile_ids):
            raise DemoAuthError('Demo profiles cannot be merged', 403)
