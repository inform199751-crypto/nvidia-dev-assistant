from ttl_cache import TTLCache


def make(ttl=60):
    now = [0]
    return TTLCache(ttl, clock=lambda: now[0]), now


def test_expires_exactly_at_ttl():
    cache, now = make()
    cache.set("k", "v")
    now[0] = 59
    assert cache.get("k") == "v"
    now[0] = 60
    assert cache.get("k", "none") == "none"


def test_frequent_reads_still_expire():
    cache, now = make()
    cache.set("k", "v")
    for t in range(0, 60, 5):
        now[0] = t
        assert cache.get("k") == "v"
    now[0] = 61
    assert cache.get("k") is None


def test_set_again_restarts_ttl():
    cache, now = make()
    cache.set("k", "v1")
    now[0] = 50
    cache.set("k", "v2")
    now[0] = 100
    assert cache.get("k") == "v2"
    now[0] = 110
    assert cache.get("k") is None
