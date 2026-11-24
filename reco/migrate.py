"""Импорт истории из исходного дампа для уже рассчитанного индекса, без пересчёта модели."""
import argparse
import json

from .database import Database
from .data import load_backup, historical_interactions
from .index import resolve_index, validate_index


def import_history(backup, artifacts='artifacts'):
    root = resolve_index(artifacts)
    audit = validate_index(root)
    catalog = json.loads((root/'catalog.json').read_text())
    rows = historical_interactions(load_backup(backup), catalog)
    database = Database()
    database.initialize()
    database.replace_history(audit['catalog_fingerprint'], rows)
    print('История импортирована в PostgreSQL:', len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backup', required=True)
    p.add_argument('--artifacts', default='artifacts')
    args = p.parse_args()
    import_history(args.backup,args.artifacts)
