"""Docker: готовые векторы → локальный каталог/история → кеш модели → API."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reco.build import build
from reco.server import serve
from reco.database import Database
from reco.migrate import import_history
from reco.index import resolve_index, validate_index
from reco.embeddings import MODEL, Encoder
from reco.precomputed import restore_bundle
from reco.model_cache import ensure_model_files

backend = os.getenv("EMBEDDING_BACKEND", "semantic")
model = os.getenv('EMBEDDING_MODEL', MODEL)
max_length = int(os.getenv('EMBEDDING_MAX_LENGTH', '512'))
batch_size = int(os.environ['EMBEDDING_BATCH_SIZE']) if os.getenv('EMBEDDING_BATCH_SIZE') else None
device = os.getenv('EMBEDDING_DEVICE', 'cpu')
backup = os.getenv("BACKUP_PATH", "data/alfa_course_hackathon.backup")
audit = resolve_index('artifacts')/'audit.json'
if not audit.exists():
    reuse = os.getenv('USE_PRECOMPUTED_EMBEDDINGS', '1')
    if reuse not in {'0', '1'}:
        raise SystemExit('USE_PRECOMPUTED_EMBEDDINGS must be 0 or 1')
    if backend == 'semantic' and reuse == '1':
        restore_bundle(backup, model=model, max_length=max_length)
    else:
        build(backup, backend=backend, model=model, max_length=max_length, batch_size=batch_size, device=device)
    audit = resolve_index('artifacts')/'audit.json'
existing = validate_index(audit.parent)
if existing["embedding_backend"] != backend:
    raise SystemExit("Existing artifacts use a different backend. Use a separate artifacts directory or rebuild explicitly.")
saved = Encoder.load(audit.parent/'encoder.json').config
if backend == 'semantic' and (saved['model'] != model or saved['max_length'] != max_length):
    raise SystemExit('Existing artifacts use a different embedding model/max_length. Rebuild the index or select its original configuration.')
database = Database()
database.initialize()
if not database.has_dataset(existing['catalog_fingerprint']):
    import_history(backup)
ensure_model_files(saved)
serve(host="0.0.0.0")
