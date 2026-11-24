"""Версии файлового индекса и проверка целостности перед использованием."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np

INDEX_FILES = ('catalog.json', 'encoder.json', 'vectors.npy', 'topics.json',
               'topic_vectors.npy', 'audit.json')


def publish_generation(root, writer):
    """Готовит поколение отдельно; отказ оставляет прежний current действующим."""
    root = Path(root).resolve()
    generations = root / '.generations'
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='index-', dir=generations))
    pointer = root / ('.current-' + uuid.uuid4().hex)
    published = False
    try:
        audit = writer(staging)
        validate_index(staging)
        pointer.symlink_to(staging.relative_to(root), target_is_directory=True)
        os.replace(pointer, root / 'current')
        published = True
        return audit
    finally:
        if pointer.is_symlink(): pointer.unlink()
        active = (root / 'current').is_symlink() and (root / 'current').resolve() == staging
        if not published and not active:
            shutil.rmtree(staging)


def resolve_index(root):
    """Фиксирует одно поколение; без current поддерживает прежний плоский индекс."""
    root = Path(root)
    current = root / 'current'
    if current.is_symlink():
        target = current.resolve()
        if not target.is_dir():
            raise ValueError('Указатель current ведёт в отсутствующее поколение индекса')
        return target
    if current.exists():
        raise ValueError('current должен быть символической ссылкой на поколение индекса')
    return root


def fingerprint(root):
    root = Path(root)
    return hashlib.sha256((root/'catalog.json').read_bytes() + (root/'encoder.json').read_bytes()).hexdigest()


def write_manifest(root):
    """Контрольные суммы связывают все файлы одного законченного поколения."""
    root = Path(root)
    manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in INDEX_FILES}
    (root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')


def validate_index(root):
    """Проверяет legacy fingerprint, чтобы не принять старый audit с новым каталогом."""
    root = Path(root)
    audit = json.loads((root/'audit.json').read_text(encoding='utf-8'))
    if audit.get('catalog_fingerprint') != fingerprint(root):
        raise ValueError('Fingerprint индекса не совпадает с каталогом и encoder; восстановите или пересоберите индекс')
    manifest_path = root/'manifest.json'
    if root.parent.name == '.generations' and not manifest_path.is_file():
        raise ValueError('У опубликованного поколения отсутствует manifest')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or set(manifest) != set(INDEX_FILES):
            raise ValueError('Некорректный manifest индекса')
        for name in INDEX_FILES:
            if manifest[name] != hashlib.sha256((root/name).read_bytes()).hexdigest():
                raise ValueError('Нарушена целостность файла индекса: '+name)
    # Контрольная сумма подтверждает целостность, но не корректность чисел.
    for name in ('vectors.npy', 'topic_vectors.npy'):
        # В legacy-индексе без справочника тем центроидов может не быть.
        # Поколения с manifest обязаны содержать все перечисленные файлы.
        if name == 'topic_vectors.npy' and not (root/name).exists():
            topics = json.loads((root/'topics.json').read_text()) if (root/'topics.json').exists() else []
            if topics:
                raise ValueError('Отсутствуют векторы непустого справочника тем; пересоберите индекс')
            continue
        values = np.load(root/name, allow_pickle=False)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError('Индекс содержит NaN/Infinity: '+name+'; пересоберите индекс')
    return audit
