"""Сквозная проверка локального API; тестовый гость удаляется даже при ошибке."""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def request(path, body=None, method=None):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


health = request("/health")
profile = request("/profiles", {})["profile_id"]
try:
    first = request("/recommend", {"profile_id": profile, "k": 5})
    assert first["mode"] == "starter_collection"
    assert first["items"]
    item = first["items"][0]["id"]
    body = {"profile_id": profile, "item_id": item, "event_type": "engaged", "event_id": "smoke-one"}
    assert request("/events", body)["accepted"]
    # Повторная доставка события не должна повторно усиливать интерес.
    assert request("/events", body)["duplicate"]
    second = request("/recommend", {"profile_id": profile, "k": 5})
    assert second["mode"] == "behavior"
    assert item not in [r["id"] for r in second["items"]]
    assert second["live_items"] == 1
    try:
        request("/recommend", {"k": 100})
        raise AssertionError("Invalid k accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 400
    print(json.dumps({"api": "passed", "backend": health["backend"], "catalog_items": health["items"],
                      "embedding_model": health.get('embedding_model'), "embedding_device": health.get('embedding_device'),
                      "vector_dimension": health.get('vector_dimension'),
                      "starter": "passed", "event_dedup": "passed", "behavior_update": "passed", "invalid_request": "passed"}))
finally:
    request("/profiles/" + profile, method="DELETE")
