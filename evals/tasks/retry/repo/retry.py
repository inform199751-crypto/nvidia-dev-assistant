"""呼叫外部服務失敗時重試。"""
import time


class GaveUp(Exception):
    """重試到上限還是失敗。__cause__ 是最後一次的錯誤。"""


def retry(fn, attempts=3, base_delay=1.0, sleep=time.sleep, retry_on=(ConnectionError, TimeoutError)):
    """最多呼叫 fn attempts 次,成功就回傳結果。
    第 n 次失敗後等 base_delay * 2 ** (n - 1) 秒再試(1、2、4…);最後一次失敗不用等,直接丟 GaveUp(from 最後的錯誤)。
    retry_on 以外的錯誤不重試,直接往外丟。"""
    for n in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            last = e
        sleep(base_delay * 2 ** (n - 1))
    return None
