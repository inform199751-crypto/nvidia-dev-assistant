"""依設定決定要印哪些紀錄。"""
import logging

from app.settings import Settings


def configure(settings: Settings) -> list[str]:
    """設定 logging,回傳開了哪些(給啟動畫面顯示)。"""
    enabled = []
    logging.getLogger("app").setLevel(logging.DEBUG if settings.debug else logging.INFO)
    if settings.debug:
        enabled.append("debug")
    if settings.log_sql:
        logging.getLogger("app.sql").setLevel(logging.DEBUG)
        enabled.append("sql")
    return enabled
