from ttl_cache import TTLCache


def make(ttl=60):
    now = [0]
    return TTLCache(ttl, clock=lambda: now[0]), now


def test_get_before_expiry():
    cache, now = make()
    cache.set("k", "v")
    now[0] = 10
    assert cache.get("k") == "v"


def test_reading_does_not_extend_ttl():
    cache, now = make()
    cache.set("k", "v")
    now[0] = 30
    assert cache.get("k") == "v"
    now[0] = 61
    assert cache.get("k") is None
