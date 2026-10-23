"""有過期時間的快取:每筆資料在 set 之後 ttl 秒過期(讀取不會延長)。clock 可以換掉(測試用),預設是 time.monotonic。"""
import time


class TTLCache:
    def __init__(self, ttl, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._data = {}  # key -> (value, 到期時間)

    def set(self, key, value):
        self._data[key] = (value, self.clock() + self.ttl)

    def get(self, key, default=None):
        """拿值;過期了(現在時間 >= 到期時間)就當作沒有,並且刪掉。"""
        if key not in self._data:
            return default
        value, expires = self._data[key]
        if self.clock() >= expires:
            del self._data[key]
            return default
        return value  # 讀取不延長:常被讀的資料也要在 ttl 秒後重新抓
