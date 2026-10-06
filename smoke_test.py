"""Catalog 冒煙測試:每個角色打幾個最小請求,確認模型還活著、思考開關與 tool calling 有效。

只送固定的測試字串,不送任何程式碼。直接呼叫單一模型(不經備援鏈),失敗才看得出是哪一顆。
用法:python smoke_test.py [--env-file 其他專案的 .env] [--roles ...] [--checks ...] [--full]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.tools import tool

from llm_registry import ModelRegistry


@tool
def get_weather(city: str) -> str:
    """查詢城市目前的天氣。"""
    return "晴"


def run_check(registry: ModelRegistry, role: str, check: str) -> tuple[bool, str, object]:
    if check == "tool":
        ai = registry.chat(role).bind_tools([get_weather]).invoke("台北現在天氣如何?請用工具查。")
        names = [c["name"] for c in ai.tool_calls]
        return names[:1] == ["get_weather"], f"tool_calls={names}", ai
    if check == "think_on_raw":  # 診斷用:同 think_on,但不送 extra_body(force_nonempty_content)
        original = registry.specs[role]
        registry.specs[role] = replace(original, extra_body=None)
        try:
            ai = registry.chat(role, thinking=True).invoke("只回答兩個字:收到")
        finally:
            registry.specs[role] = original
    else:
        thinking = {"plain": None, "think_on": True, "think_off": False}[check]
        ai = registry.chat(role, thinking=thinking).invoke("只回答兩個字:收到")
    content = str(ai.content).strip()
    reasoning = ai.additional_kwargs.get("reasoning_content") or ""
    # 回答必須乾淨:有「收到」而且很短,推理文字混進回答也算失敗
    clean = "收到" in content and len(content) <= 20
    ok = clean and (not reasoning if check == "think_off" else bool(reasoning) or check == "plain")
    return ok, f"回答={content[:24]!r} 思考={len(reasoning)}字", ai


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    ap.add_argument("--pause", type=float, default=2.0, help="每次呼叫間隔秒數(免費層約 40 RPM)")
    ap.add_argument("--roles", nargs="+", help="只測這些角色,例如 --roles code long_context")
    ap.add_argument("--out", default=str(Path(__file__).with_name("smoke_results.log")),
                    help="結果同步附加到這個檔案(終端機關掉也查得到)")
    ap.add_argument("--checks", nargs="+", choices=["plain", "think_on", "think_off", "tool", "think_on_raw"],
                    help="只跑這些檢查;think_on_raw 是診斷用(不送 force_nonempty_content)")
    ap.add_argument("--full", action="store_true", help="印出完整回答與推理內容")
    args = ap.parse_args()

    log_file = open(args.out, "a", encoding="utf-8")

    def emit(line: str = "") -> None:
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    key = dotenv_values(args.env_file).get("NVIDIA_API_KEY")
    if not key:
        emit(f"{args.env_file} 裡沒有 NVIDIA_API_KEY")
        return 2
    os.environ["NVIDIA_API_KEY"] = key
    warnings.simplefilter("ignore")  # 套件對不在內建表的模型會一直警告

    registry = ModelRegistry.from_yaml(backend="catalog")
    failures = 0
    emit(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    emit(f"{'角色':<13}{'模型':<38}{'檢查':<10}{'結果':<6}{'秒':>6}{'tokens':>8}  備註")
    tested_models: set[str] = set()
    for role, spec in registry.specs.items():
        if args.roles and role not in args.roles:
            continue
        if "catalog" not in spec.available_on:
            emit(f"{role:<13}{spec.model:<38}{'-':<10}{'略過':<6}{'':>6}{'':>8}  只提供地端")
            continue
        if spec.model in tested_models and not args.roles:
            emit(f"{role:<13}{spec.model:<38}{'-':<10}{'略過':<6}{'':>6}{'':>8}  同一顆模型前面測過了")
            continue
        tested_models.add(spec.model)
        checks = ["plain"]
        if spec.thinking_on:
            checks += ["think_on", "think_off"]
        if spec.supports_tools:
            checks.append("tool")
        if args.checks:
            checks = [c for c in args.checks if c in checks or (c == "think_on_raw" and spec.thinking_on)]
        for check in checks:
            t0 = time.perf_counter()
            try:
                ok, note, ai = run_check(registry, role, check)
                tokens = (ai.usage_metadata or {}).get("total_tokens", 0)
            except Exception as e:  # 404 / 410 / 429 / 逾時都記下來,繼續測下一個
                ok, note, tokens, ai = False, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}", 0, None
            failures += not ok
            emit(f"{role:<13}{spec.model:<38}{check:<10}{'通過' if ok else '失敗':<6}"
                 f"{time.perf_counter() - t0:>6.1f}{tokens:>8}  {note}")
            if args.full and ai is not None:
                emit(f"    [回答] {str(ai.content)[:800]!r}")
                emit(f"    [推理] {(ai.additional_kwargs.get('reasoning_content') or '')[:800]!r}")
            time.sleep(args.pause)
    emit(f"\n{failures} 項失敗")
    log_file.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
