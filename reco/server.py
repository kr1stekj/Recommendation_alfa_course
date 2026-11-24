"""Локальный API с демо-входом; проверку реальных пользователей выполняет платформа."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import secrets
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from .engine import Recommender
from .search import SmartSearch
from .ranking import enable_ranking_logging
from .demo_auth import DemoAuth, DemoAuthError, session_token, session_cookie


def create_server(engine, host="127.0.0.1", port=8000):
    """Собирает HTTP-сервис с интерфейсом; жизненным циклом управляет serve."""
    search = SmartSearch(engine)
    demo = DemoAuth(engine)
    web = Path(__file__).with_name("web")
    assets = {"/": ("index.html", "text/html; charset=utf-8"),
              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
              "/style.css": ("style.css", "text/css; charset=utf-8")}
    # Обработчики используют общий экземпляр модели; в MVP операции выполняются по очереди.
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def security_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")

        def send(self, value, status=200, cookie=None):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            if cookie:
                self.send_header('Set-Cookie', cookie)
            self.security_headers()
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            # Не записываем задачи пользователя и идентификаторы в журнал HTTP-запросов.
            pass

        def allow_origin(self):
            """Браузер может изменять состояние только со страницы локального сервиса."""
            origin = self.headers.get('Origin')
            if origin is None:
                return True  # CLI и серверные клиенты не обязаны присылать Origin.
            host = self.headers.get('Host', '')
            try:
                parsed = urlparse('http://' + host)
                local = (parsed.hostname in {'localhost', '127.0.0.1', '::1'} and
                         not parsed.username and not parsed.password and not parsed.path and
                         not parsed.query and not parsed.fragment)
            except ValueError:
                local = False
            return local and origin.casefold() == ('http://' + host).casefold()

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/demo/accounts':
                self.send({'accounts': demo.profiles.list_accounts(), 'synthetic': True})
            elif parsed.path == '/demo/session':
                token = session_token(self.headers.get('Cookie'))
                result = demo.snapshot(token)
                self.send(result, cookie=session_cookie() if token and not result['authenticated'] else None)
            elif parsed.path in assets:
                filename, content_type = assets[parsed.path]
                data = (web / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.security_headers()
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/health":
                self.send({"status": "ok", "backend": engine.encoder.config["backend"], "items": engine.audit["catalog_items"],
                           "embedding_model": engine.encoder.config.get('model') if engine.encoder.config['backend'] == 'semantic' else None,
                           "embedding_revision": engine.encoder.config.get('revision'),
                           "vector_dimension": int(engine.vectors.shape[1]),
                           "embedding_device": engine.encoder.device,
                           "embedding_max_length": engine.encoder.config.get('max_length'),
                           "demo_only": True, "endpoints": ["POST /profiles", "POST /search", "POST /smart-search", "POST /recommend", "POST /events", "POST /merge", "DELETE /profiles/<id>", "GET /catalog?limit=20", "GET /demo/accounts", "GET /demo/session", "POST /demo/login", "POST /demo/logout", "POST /demo/reset"]})
            elif parsed.path == "/catalog":
                try: limit = min(max(int(parse_qs(parsed.query).get("limit", [20])[0]), 1), 100)
                except ValueError: return self.send({"error": "Invalid limit"}, 400)
                self.send([{k: r[k] for k in ("id", "title", "kind", "topics", "url")} for r in engine.catalog if r["eligible"]][:limit])
            else: self.send({"error": "Unknown endpoint"}, 404)

        def do_POST(self):
            # text/plain допускается браузером без CORS preflight: чужая страница
            # не должна запускать платный LLM-вызов по ключу локального сервера.
            if not self.allow_origin():
                return self.send({'error': 'Cross-origin requests are forbidden'}, 403)
            if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
                return self.send({'error': 'Content-Type must be application/json'}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 20000: return self.send({"error": "Request too large"}, 413)
                data = json.loads(self.rfile.read(size) or "{}")
                if not isinstance(data, dict): raise ValueError("JSON object required")
                token = session_token(self.headers.get('Cookie'))
                # Проверка демосессии и действие атомарны относительно её cleanup.
                with lock, demo.lock:
                    if self.path == '/demo/login':
                        if set(data) != {'account_id'}: raise ValueError('Expected account_id')
                        new_token, result = demo.login(data['account_id'], token)
                        return self.send(result, cookie=session_cookie(new_token))
                    elif self.path in ('/demo/logout', '/demo/reset'):
                        if data: raise ValueError('Expected empty object')
                        result = demo.logout(token) if self.path.endswith('logout') else demo.reset(token)
                        return self.send(result, cookie=session_cookie() if self.path.endswith('logout') else None)
                    elif self.path == "/profiles":
                        result = {"profile_id": "anon:" + secrets.token_urlsafe(16)}
                    elif self.path == "/recommend":
                        data['profile_id'] = demo.bind_profile(data.get('profile_id'), token)
                        result = engine.recommend(**data)
                    elif self.path in ("/search", "/smart-search"):
                        data['profile_id'] = demo.bind_profile(data.get('profile_id'), token)
                        data.setdefault('search_mode', 'normal' if self.path == '/search' else 'smart')
                        result = search.search(**data)
                    elif self.path == "/events":
                        # Клиент не может подставить будущее время и обойти срок жизни гостя.
                        if set(data) - {"profile_id", "item_id", "event_type", "event_id"}:
                            raise ValueError("Event timestamp is set by the server; unknown event fields")
                        data['profile_id'] = demo.bind_profile(data.get('profile_id'), token)
                        result = engine.event(**data)
                    elif self.path == "/merge":
                        demo.forbid_merge(token, data.get('guest_id'), data.get('registered_id'))
                        # Это только демонстрация привязки, не проверка личности пользователя.
                        engine.validate_profile_id(data["guest_id"])
                        engine.validate_profile_id(data["registered_id"])
                        engine.store.merge(**data)
                        result = {"merged": True}
                    else: return self.send({"error": "Unknown endpoint"}, 404)
                self.send(result)
            except DemoAuthError as e:
                self.send({'error': str(e)}, e.status, cookie=session_cookie() if e.status == 401 else None)
            except (ValueError, TypeError, KeyError) as e:
                self.send({"error": str(e)}, 400)
            except Exception:
                logging.exception("API request failed")
                self.send({"error": "Internal error; see server output"}, 500)

        def do_DELETE(self):
            if not self.allow_origin():
                return self.send({'error': 'Cross-origin requests are forbidden'}, 403)
            prefix = "/profiles/"
            if not self.path.startswith(prefix): return self.send({"error": "Unknown endpoint"}, 404)
            profile_id = unquote(self.path[len(prefix):])
            with lock, demo.lock:
                try:
                    demo.bind_profile(profile_id, session_token(self.headers.get('Cookie')))
                except DemoAuthError as e:
                    return self.send({'error': str(e)}, e.status, cookie=session_cookie() if e.status == 401 else None)
                try: engine.validate_profile_id(profile_id)
                except ValueError as e: return self.send({"error": str(e)}, 400)
                engine.store.delete(profile_id)
            self.send({"deleted": True})

    server = ThreadingHTTPServer((host, port), Handler)
    server.demo_auth = demo
    return server


def serve(artifacts="artifacts", host="127.0.0.1", port=8000, ranking_log=False):
    """Запускает интерфейс/API и очистку гостей. Внешнему доступу нужна авторизация."""
    if ranking_log:
        enable_ranking_logging()
    engine = Recommender(artifacts)
    server = create_server(engine, host, port)
    stop = threading.Event()

    def sweep():
        while not stop.wait(60):
            try:
                server.demo_auth.cleanup()
                engine.database.cleanup_expired()
            except Exception: logging.exception("Profile cleanup failed")
    threading.Thread(target=sweep, daemon=True).start()
    print(f"Local demo API: http://{host}:{port} | {engine.encoder.config['backend']}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ranking-log", action="store_true", help="Числовые вклады ранжирования в stderr")
    a = p.parse_args()
    serve(a.artifacts, a.host, a.port, ranking_log=a.ranking_log)
