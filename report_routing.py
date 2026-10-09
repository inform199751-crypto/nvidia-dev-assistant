"""依難度選模型的報表:分類器判斷的分布、多花了多少、各模型實際表現,以及給人工抽查的題目。

用法:python report_routing.py [--days 7] [--examples 10] [--log 路徑]
紀錄由 agent 每輪寫一行(difficulty.record)。影子模式時 pick 是「本來會選的」、used 是實際用的,
所以「判斷為難題、實際還是用一般模型」那組如果明顯比較吃力(步數多、tokens 多、碰到步數上限),
就是分類器分得出難易的證據;再抽查題目確認,才把 mode 改成 on。
但這只證明分類器分得出難易,不證明難題模型答得比較好:那要看「做到上限後換模型接著做」那組救回多少,
或挑幾題讓兩顆模型各跑一次比較。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

import difficulty
from token_guard import TAIPEI

HERE = Path(__file__).parent


def load(path: Path, days: int | None = None) -> list[dict]:
    """讀紀錄(含輪替下來的 .1);壞掉的行略過。"""
    since = datetime.now(TAIPEI) - timedelta(days=days) if days else None
    entries = []
    for p in (path.with_name(path.name + ".1"), path):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                if since is None or datetime.fromisoformat(e["ts"]) >= since:
                    entries.append(e)
            except (ValueError, KeyError, TypeError):
                continue
    return entries


def _avg(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _share(n: int, total: int) -> str:
    return f"{n}({n / total:.0%})" if total else str(n)


def _stats(rows: list[dict]) -> str:
    finished = [r for r in rows if r["outcome"] in ("done", "max_steps")]
    errors = len(rows) - len(finished)
    if not finished:
        return f"{len(rows)} 輪,全部出錯"
    return (f"{len(rows)} 輪 · 平均 {_avg(r['steps'] for r in finished):.1f} 步 · "
            f"{_avg(r['tokens'] for r in finished):,.0f} tokens · {_avg(r['ms'] for r in finished) / 1000:.1f} 秒 · "
            f"碰到步數上限 {sum(r['outcome'] == 'max_steps' for r in rows)} · 出錯 {errors}")


def _when(e: dict) -> str:
    return datetime.fromisoformat(e["ts"]).strftime("%m-%d %H:%M")


def report(entries: list[dict], hard_role: str = "code_deep", examples: int = 10) -> list[str]:
    if not entries:
        return ["還沒有紀錄。models.yaml 的 routing.difficulty.mode 設成 shadow 或 on,用一陣子再來看。"]
    by = Counter(e["by"] for e in entries)
    classified = [e for e in entries if e["by"] == "classifier"]
    lines = ["模式:" + " · ".join(f"{m} {n} 輪" for m, n in Counter(e["mode"] for e in entries).items())]

    picks = Counter(e["pick"] or "分類失敗" for e in classified)
    lines.append("分類器判斷:" + " · ".join(f"{k} {_share(n, len(classified))}" for k, n in picks.most_common())
                 + f"(另有手動指定 {by['manual']} 輪、照計畫執行沿用 {by['plan']} 輪、"
                   f"做到上限後換模型 {by['escalate']} 輪)")
    if classified:
        spent = sum(e["classify_tokens"] for e in classified)
        total = sum(e["tokens"] or 0 for e in entries)
        lines.append(f"分類器成本:每次平均 {_avg(e['classify_ms'] for e in classified) / 1000:.1f} 秒、"
                     f"{_avg(e['classify_tokens'] for e in classified):,.0f} tokens"
                     + (f",占全部 tokens 的 {spent / total:.1%}" if total else ""))

    lines.append("實際使用的模型:")
    used = defaultdict(list)
    for e in entries:
        used[e["used"] or "(出錯,未記)"].append(e)
    lines += [f"  {role:<16}{_stats(rows)}" for role, rows in used.items()]

    shadow = defaultdict(list)
    for e in classified:
        if e["mode"] == "shadow" and e["pick"]:
            shadow[e["pick"]].append(e)
    if shadow:
        lines.append("影子模式依判斷分組(實際都用預設模型;判斷為難題的那組應該明顯比較吃力):")
        lines += [f"  判斷 {pick:<11}{_stats(rows)}" for pick, rows in sorted(shadow.items())]
    if escalated := [e for e in entries if e["by"] == "escalate"]:
        lines.append("一般模型做到步數上限、使用者選擇換模型接著做(沒再碰到上限的,就是難題模型救回來的):")
        lines.append(f"  {_stats(escalated)}")

    hard = [e for e in classified if e["pick"] == hard_role][-examples:]
    if hard:
        lines.append(f"最近判斷為難題({hard_role})的 {len(hard)} 題,請人工抽查準不準:")
        lines += [f"  {_when(e)}  [{e['reason'] or '-'}] {e['task']}" for e in hard]
    failed = [e for e in classified if e["error"]][-5:]
    if failed:
        lines.append("最近分類失敗(照預設模型回答):")
        lines += [f"  {_when(e)}  {e['error']}" for e in failed]
    return lines


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")  # 跟 agent 一樣用 AGENT_OUTPUT 找紀錄
    out = Path(os.getenv("AGENT_OUTPUT", "output"))
    default_log = (out if out.is_absolute() else HERE / out) / difficulty.LOG_NAME
    ap = argparse.ArgumentParser(description="依難度選模型的紀錄報表")
    ap.add_argument("--days", type=int, default=7, help="只看最近幾天;0 = 全部")
    ap.add_argument("--examples", type=int, default=10, help="列出幾題判斷為難題的題目給人工抽查")
    ap.add_argument("--log", type=Path, default=default_log)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "models.yaml").read_text(encoding="utf-8"))["routing"].get("difficulty") or {}
    entries = load(args.log, args.days or None)
    print(f"{args.log} · {'最近 %d 天' % args.days if args.days else '全部'} · {len(entries)} 輪")
    print("\n".join(report(entries, cfg.get("hard_role", "code_deep"), args.examples)))


if __name__ == "__main__":
    main()
