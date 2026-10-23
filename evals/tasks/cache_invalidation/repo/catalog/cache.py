"""商品頁的快取(正式環境是 Redis):資料庫撐不住每次都查,所以一定要有。"""
import time


class Cache:
    def __init__(self, ttl=600, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._data = {}

    def get(self, key):
        hit = self._data.get(key)
        if hit is None or self.clock() >= hit[1]:
            return None
        return hit[0]

    def set(self, key, value):
        self._data[key] = (value, self.clock() + self.ttl)

    def delete(self, key):
        self._data.pop(key, None)
