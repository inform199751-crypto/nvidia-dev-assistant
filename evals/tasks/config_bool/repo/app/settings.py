"""從環境變數讀設定。"""
from dataclasses import dataclass

from app.utils import parse_bool, parse_int


@dataclass
class Settings:
    debug: bool = False
    workers: int = 4
    log_sql: bool = False


def load(env):
    """env 是環境變數的 dict(正式環境傳 os.environ)。"""
    return Settings(
        debug=parse_bool(env.get("APP_DEBUG", "false")),
        workers=parse_int(env.get("APP_WORKERS", ""), 4),
        log_sql=parse_bool(env.get("APP_LOG_SQL", "no")),
    )
