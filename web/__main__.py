"""啟動網頁版:python -m web(或直接執行這個檔)。只聽 127.0.0.1 —— 沒有登入,不能讓同網路的其他電腦連進來。"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 不管從哪個目錄啟動,都找得到 agent / tools / web

import uvicorn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


def build_app():
    load_dotenv(ROOT / ".env")  # 要在 import agent / tools 之前,工具才讀得到 SANDBOX_IMAGE
    from llm_registry import ModelRegistry
    from token_guard import TokenQuota
    from web.app import create_app
    from web.conversations import ConversationManager
    from web.workspaces import WorkspaceRegistry

    registry = ModelRegistry.from_yaml()
    quota = TokenQuota(registry.limits["daily_tokens_per_user"], registry.limits["max_concurrent_per_user"])
    workspaces = WorkspaceRegistry(ROOT / "workspaces.yaml", base_dir=ROOT, uploads_dir=ROOT / "uploads")
    log = logging.getLogger("dev_assistant.web")

    def safe_cleanup() -> None:
        try:
            workspaces.cleanup()
        except Exception:  # 清理失敗不能讓伺服器起不來,也不能讓每小時的清理就此停掉
            log.exception("清理過期上傳失敗")

    safe_cleanup()

    def janitor() -> None:
        while True:
            time.sleep(3600)
            safe_cleanup()

    threading.Thread(target=janitor, daemon=True).start()
    manager = ConversationManager(ROOT / "conversations", workspaces, registry, quota)
    return create_app(manager, workspaces, registry)


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=int(os.getenv("WEB_PORT", "8765")))
