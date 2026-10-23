from app.logging_setup import configure
from app.settings import load


def test_workers():
    assert load({"APP_WORKERS": "8"}).workers == 8


def test_debug_on():
    assert configure(load({"APP_DEBUG": "true"})) == ["debug"]


def test_production_flags_off():
    assert configure(load({"APP_DEBUG": "false", "APP_LOG_SQL": "0"})) == []
