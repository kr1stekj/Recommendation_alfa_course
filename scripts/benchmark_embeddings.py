"""Воспроизводимые метрики и warm latency одного индекса без облачных вызовов."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
from reco.engine import Recommender
from reco.evaluate import evaluate


def benchmark(artifacts, profiles='evaluation/profiles.json', repeats=3):
    """Каждую модель запускают отдельным процессом: нет двух больших копий в RAM."""
    start = time.perf_counter()
    engine = Recommender(artifacts)
    engine.encoder.encode(['Проверка готовности encoder'])
    cold_seconds = time.perf_counter() - start
    fixture = json.loads(Path(profiles).read_text())
    elapsed, token_lengths = [], []
    for _ in range(repeats):
        for profile in fixture['profiles']:
            task = profile['task']
            start = time.perf_counter()
            engine.encoder.encode([task])
            elapsed.append(1000 * (time.perf_counter() - start))
    if engine.encoder.config['backend'] == 'semantic':
        token_lengths = [len(engine.encoder.tokenizer(p['task'], truncation=True,
            max_length=engine.encoder.config['max_length'])['input_ids']) for p in fixture['profiles']]
    quality = evaluate(artifacts, profiles, engine=engine)
    fields = ('id', 'eligible', 'title', 'description', 'text', 'topics', 'tags', 'industries', 'price')
    original = [{key: row.get(key) for key in fields} for row in engine.catalog]
    projection = json.dumps(original, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    with engine.database.connect() as db:
        history = db.execute('SELECT user_id,item_id,weight,completed,owned,at FROM reco.history '
            'WHERE dataset=%s ORDER BY user_id,item_id,at,weight,completed,owned', (engine.dataset,)).fetchall()
    history_sha = hashlib.sha256(json.dumps(history, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        'encoder': engine.encoder.config, 'catalog_fingerprint': engine.dataset,
        'parameter_device': str(next(engine.encoder.model.parameters()).device) if engine.encoder.model is not None else None,
        'parameter_dtype': str(next(engine.encoder.model.parameters()).dtype) if engine.encoder.model is not None else None,
        'source_projection_sha256': hashlib.sha256(projection).hexdigest(),
        'history_projection_sha256': history_sha, 'history_rows': len(history),
        'catalog_items': engine.audit['catalog_items'], 'vector_shape': list(engine.vectors.shape),
        'platform': {'machine': platform.machine(), 'cpu_count': psutil.cpu_count(),
                     'torch_threads': 2, 'device': engine.encoder.device,
                     'ram_gib': round(psutil.virtual_memory().total / 1024**3, 2)},
        'runtime_rss_mib': round(psutil.Process().memory_info().rss / 1024**2, 2),
        'cold_start_seconds': round(cold_seconds, 3),
        'warm_single_query': {'samples': len(elapsed), 'token_lengths': token_lengths,
            'p50_ms': round(float(np.percentile(elapsed, 50)), 2),
            'p95_ms': round(float(np.percentile(elapsed, 95)), 2)},
        'quality': quality, 'cloud_called': False,
        'limitations': 'Five known seed profiles with incomplete author labels; repeated measurement, not independent test. '
                      'Latency is for these short queries, not full 512/8192-token documents or production load.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--profiles', default='evaluation/profiles.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    os.environ['EMBEDDING_DEVICE'] = args.device
    result = benchmark(args.artifacts, args.profiles)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('encoder', 'warm_single_query', 'runtime_rss_mib')}, ensure_ascii=False))
    print(json.dumps(result['quality']['summary'], ensure_ascii=False))
