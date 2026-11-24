"""Перенос числовых эмбеддингов без публикации каталога и истории пользователей."""
import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from .build import finalize_catalog
from .data import load_backup, make_catalog, historical_interactions
from .database import Database
from .embeddings import Encoder, MODEL_SPECS
from .index import resolve_index, validate_index, fingerprint, write_manifest, publish_generation

BUNDLE_FILES = ('encoder.json', 'vectors.npy', 'topic_vectors.npy')
MANIFEST_KEYS = {'format_version', 'backup_sha256', 'as_of', 'catalog_fingerprint',
    'base_catalog_sha256', 'catalog_sha256', 'topics_sha256', 'vector_shape', 'topic_shape',
    'embedding_device', 'topic_annotations', 'files'}
CONFIG_KEYS = {'backend', 'model', 'revision', 'pooling', 'max_length', 'batch_size'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != CONFIG_KEYS or
            config.get('backend') != 'semantic' or not isinstance(config.get('model'), str) or
            config['model'] not in MODEL_SPECS):
        raise ValueError('Bundle requires supported semantic encoder configuration')
    spec = MODEL_SPECS[config['model']]
    if config['revision'] != spec['revision']:
        raise ValueError('Bundle model revision differs from the pinned model')
    Encoder(config)  # Проверка pooling/длин/батча без загрузки модели.
    return spec['dimension']


def validate_matrix(path, shape):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if (values.shape != tuple(shape) or values.ndim != 2 or values.dtype.kind != 'f' or
            values.dtype.itemsize != 4 or not np.isfinite(values).all()):
        raise ValueError('Invalid precomputed matrix shape/dtype/NaN/Infinity: ' + Path(path).name)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.all((norms == 0) | np.isclose(norms, 1.0, atol=1e-4)):
        raise ValueError('Precomputed vectors must be normalized: ' + Path(path).name)


def _annotations(catalog, topics, encoded):
    if not isinstance(encoded, list) or len(encoded) != len(catalog):
        raise ValueError('Invalid topic annotation row count')
    suggestions = []
    for item, row in zip(catalog, encoded):
        needed = not item['topics'] and bool(topics)
        if row is None and not needed:
            suggestions.append(None)
            continue
        if (not needed or not isinstance(row, list) or len(row) != 3 or type(row[0]) is not int or
                not 0 <= row[0] < len(topics) or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in row[1:]) or
                not -1 <= row[1] <= 1 or not 0 <= row[2] <= 2):
            raise ValueError('Invalid numeric topic annotation')
        suggestions.append(dict(title=topics[row[0]], cosine=row[1], margin=row[2]))
    return suggestions


def _materialize(backup, manifest):
    if sha256_file(backup) != manifest['backup_sha256']:
        raise ValueError('Backup SHA256 differs from precomputed bundle; use the original backup or rebuild explicitly')
    tables = load_backup(backup)
    catalog, audit = make_catalog(tables, manifest['as_of'], include_closed_history=True)
    history = historical_interactions(tables, catalog)
    topics = audit['dictionaries']['Topics']
    if sha256_bytes(json_bytes(catalog)) != manifest['base_catalog_sha256']:
        raise ValueError('Catalog or row order differs from precomputed bundle')
    if sha256_bytes(json_bytes(topics)) != manifest['topics_sha256']:
        raise ValueError('Topic order differs from precomputed bundle')
    finalize_catalog(catalog, history, _annotations(catalog, topics, manifest['topic_annotations']))
    if sha256_bytes(json_bytes(catalog)) != manifest['catalog_sha256']:
        raise ValueError('Reconstructed catalog differs from precomputed bundle')
    if sha256_file(backup) != manifest['backup_sha256']:
        raise ValueError('Backup changed while preparing the index')
    return catalog, audit, history, topics


def load_bundle(bundle, model=None, max_length=None):
    root = Path(bundle)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS or
            type(manifest['format_version']) is not int or manifest['format_version'] != 1):
        raise ValueError('Unsupported precomputed manifest')
    for key in ('backup_sha256', 'base_catalog_sha256', 'catalog_sha256', 'topics_sha256', 'catalog_fingerprint'):
        if not isinstance(manifest[key], str) or not re.fullmatch('[0-9a-f]{64}', manifest[key]):
            raise ValueError('Invalid precomputed digest: ' + key)
    if not isinstance(manifest['as_of'], str) or date.fromisoformat(manifest['as_of']).isoformat() != manifest['as_of']:
        raise ValueError('Invalid precomputed snapshot date')
    if manifest['embedding_device'] not in ('cpu', 'mps'):
        raise ValueError('Invalid precomputed source device')
    files = manifest['files']
    if not isinstance(files, dict) or set(files) != set(BUNDLE_FILES):
        raise ValueError('Invalid precomputed file manifest')
    for name in BUNDLE_FILES:
        if not isinstance(files[name], str) or not re.fullmatch('[0-9a-f]{64}', files[name]) or sha256_file(root/name) != files[name]:
            raise ValueError('Precomputed file checksum mismatch: ' + name)
    config = json.loads((root / 'encoder.json').read_text(encoding='utf-8'))
    dimension = validate_config(config)
    if ((model is not None and config['model'] != model) or
            (max_length is not None and config['max_length'] != max_length)):
        raise ValueError('Precomputed model/max_length differs from configuration; disable reuse to rebuild explicitly')
    for key in ('vector_shape', 'topic_shape'):
        shape = manifest[key]
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 0 for n in shape) or
                shape[1] != dimension):
            raise ValueError('Invalid precomputed matrix dimensions')
    validate_matrix(root/'vectors.npy', manifest['vector_shape'])
    validate_matrix(root/'topic_vectors.npy', manifest['topic_shape'])
    return manifest, config


def export_bundle(backup, artifacts='artifacts', bundle='precomputed/bge-m3'):
    destination = Path(bundle)
    if destination.exists():
        raise FileExistsError('Export destination already exists; use a new directory')
    source = resolve_index(artifacts)
    audit = validate_index(source)
    config = json.loads((source/'encoder.json').read_text(encoding='utf-8'))
    dimension = validate_config(config)
    catalog = json.loads((source/'catalog.json').read_text(encoding='utf-8'))
    topics = json.loads((source/'topics.json').read_text(encoding='utf-8'))
    # Никаких ID/названий: только позиция темы и числовые оценки по строкам матрицы.
    annotations = [[topics.index(row['suggested_topic']['title']), row['suggested_topic']['cosine'],
                    row['suggested_topic']['margin']] if 'suggested_topic' in row else None for row in catalog]
    base_catalog = [{key: value for key, value in row.items()
                     if key not in {'topic_source', 'suggested_topic', 'history_count'}} for row in catalog]
    manifest = dict(format_version=1, backup_sha256=sha256_file(backup), as_of=audit['as_of'],
        catalog_fingerprint=audit['catalog_fingerprint'], base_catalog_sha256=sha256_bytes(json_bytes(base_catalog)),
        catalog_sha256=sha256_bytes((source/'catalog.json').read_bytes()),
        topics_sha256=sha256_bytes((source/'topics.json').read_bytes()),
        vector_shape=[len(catalog), dimension], topic_shape=[len(topics), dimension],
        embedding_device=audit['embedding_device'], topic_annotations=annotations,
        files={name: sha256_file(source/name) for name in BUNDLE_FILES})
    _materialize(backup, manifest)  # Проверка происхождения без БД и нейросети.
    for name, shape in (('vectors.npy',manifest['vector_shape']),('topic_vectors.npy',manifest['topic_shape'])):
        validate_matrix(source/name, shape)
    destination.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copyfile(source/name, destination/name)
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    load_bundle(destination)
    return manifest


def restore_bundle(backup, bundle='precomputed/bge-m3', artifacts='artifacts', model=None, max_length=None):
    source = Path(bundle)
    manifest, config = load_bundle(source, model, max_length)
    catalog, audit, history, topics = _materialize(backup, manifest)
    if manifest['vector_shape'][0] != len(catalog) or manifest['topic_shape'][0] != len(topics):
        raise ValueError('Precomputed matrix row count differs from reconstructed metadata')

    def write(staging):
        for name in BUNDLE_FILES:
            shutil.copyfile(source/name, staging/name)
            if sha256_file(staging/name) != manifest['files'][name]:
                raise ValueError('Precomputed files changed while restoring')
        (staging/'catalog.json').write_bytes(json_bytes(catalog))
        (staging/'topics.json').write_bytes(json_bytes(topics))
        if fingerprint(staging) != manifest['catalog_fingerprint']:
            raise ValueError('Precomputed fingerprint differs from reconstructed index')
        audit.update(history_rows=len(history), history_users=len({row['user_id'] for row in history}),
            embedding_backend='semantic', embedding_model=config['model'], embedding_revision=config['revision'],
            embedding_max_length=config['max_length'], embedding_pooling=config['pooling'],
            embedding_device=manifest['embedding_device'], vector_shape=manifest['vector_shape'],
            topics_from_catalog=sum(bool(row['topics']) for row in catalog),
            catalog_fingerprint=manifest['catalog_fingerprint'], embedding_origin='precomputed')
        (staging/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        write_manifest(staging)
        validate_index(staging)
        database = Database(); database.initialize()
        database.replace_history(audit['catalog_fingerprint'], history)
        return audit

    result = publish_generation(artifacts, write)
    print('Готовые эмбеддинги восстановлены без пересчёта:', manifest['vector_shape'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Перенос готовых векторов без каталога и истории')
    parser.add_argument('operation', choices=['export', 'restore'])
    parser.add_argument('--backup', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--bundle', default='precomputed/bge-m3')
    args = parser.parse_args()
    if args.operation == 'export':
        result = export_bundle(args.backup, args.artifacts, args.bundle)
        print('Числовой bundle сохранён:', args.bundle, result['vector_shape'])
    else:
        restore_bundle(args.backup, args.bundle, args.artifacts)
