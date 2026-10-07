"""FastAPI 路由。只聽本機(見 __main__.py),沒有登入。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent import SYSTEM_PROMPT, build_user_prompt
from token_guard import ContextTooLarge, message_tokens
from web.conversations import Busy, ConversationManager
from web.workspaces import UploadRejected, WorkspaceRegistry

STATIC = Path(__file__).with_name("static")


class TurnIn(BaseModel):
    question: str
    mode: str = "fix"
    role: str | None = None
    thinking: bool | None = None
    files: str | None = None

    def options(self) -> dict:
        return {"role": self.role, "thinking": self.thinking, "files": self.files}


class ConversationIn(TurnIn):
    workspace: str


def create_app(manager: ConversationManager, workspaces: WorkspaceRegistry, registry,
               poll_seconds: float = 0.3) -> FastAPI:
    app = FastAPI(title="研發助理")
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")

    def conversation_or_404(cid: str):
        try:
            return manager.get(cid)
        except KeyError:
            raise HTTPException(404, "沒有這個對話") from None

    def workspace_or_404(wid: str):
        try:
            return workspaces.get(wid)
        except KeyError:
            raise HTTPException(404, "沒有這個資料夾") from None

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/workspaces")
    def list_workspaces() -> list[dict]:
        return [w.to_dict() for w in workspaces.list()]

    @app.post("/api/uploads")
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        blobs = [(f.filename or "file", await f.read()) for f in files]
        try:
            ws, skipped = workspaces.add_upload(blobs)
        except UploadRejected as e:
            raise HTTPException(400, str(e)) from None
        return {"workspace": ws.to_dict(), "skipped": skipped}

    @app.get("/api/estimate")
    def estimate(workspace: str) -> dict:
        ws = workspace_or_404(workspace)
        files = workspaces.whole_files(ws)
        prompt = build_user_prompt("", files, registry.limits["inline_file_max_tokens"], True, root=ws.path)
        tokens = message_tokens([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        limit = registry.budget(registry.routing["long_context_role"])
        try:
            role = registry.pick_role(prompt_tokens=tokens, needs_tools=False)
        except ContextTooLarge:
            return {"files": len(files), "tokens": tokens, "limit": limit, "ok": False, "role": None, "model": None}
        return {"files": len(files), "tokens": tokens, "limit": limit, "ok": True,
                "role": role, "model": registry.spec(role).model}

    @app.get("/api/conversations")
    def list_conversations() -> list[dict]:
        return manager.list()

    @app.post("/api/conversations")
    def new_conversation(body: ConversationIn) -> dict:
        try:
            cid, tid = manager.start(body.workspace, body.question, body.mode, body.options())
        except KeyError:
            raise HTTPException(400, "沒有這個資料夾") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"conversation_id": cid, "turn_id": tid}

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str) -> dict:
        return conversation_or_404(cid).detail()

    @app.post("/api/conversations/{cid}/turns")
    def add_turn(cid: str, body: TurnIn) -> dict:
        conversation_or_404(cid)
        try:
            tid = manager.add_turn(cid, body.question, body.mode, body.options())
        except Busy as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"turn_id": tid}

    @app.post("/api/conversations/{cid}/turns/{tid}/cancel")
    def cancel(cid: str, tid: str) -> dict:
        try:
            manager.cancel(cid, tid)
        except KeyError:
            raise HTTPException(404, "沒有這一輪") from None
        return {"ok": True}

    @app.get("/api/conversations/{cid}/turns/{tid}/events")
    async def events(cid: str, tid: str, request: Request, after: int = -1) -> Response:
        conv = conversation_or_404(cid)
        try:
            turn = conv.turn(tid)
        except KeyError:
            raise HTTPException(404, "沒有這一輪") from None
        last = request.headers.get("last-event-id", "")
        start = int(last) + 1 if last.isdigit() else after + 1
        if turn.events and turn.events[-1]["type"] == "end" and start > turn.events[-1]["seq"]:
            return Response(status_code=204)  # 已經全部送完;回 204 瀏覽器的 EventSource 才不會一直自動重連

        async def stream():
            i = start
            while True:
                evs = turn.events
                while i < len(evs):
                    event = evs[i]
                    i += 1
                    yield f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event["type"] == "end":
                        return
                if evs and evs[-1]["type"] == "end":  # 斷線重連時結束事件已經送過,不要空等
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(poll_seconds)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
