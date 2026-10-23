"""分頁游標:給 App 的不透明字串,裡面記著上一頁最後一筆的位置。"""
import base64
import json


def encode(position):
    return base64.urlsafe_b64encode(json.dumps(position).encode()).decode()


def decode(cursor):
    if not cursor:
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except ValueError:
        raise ValueError("游標格式不對") from None
