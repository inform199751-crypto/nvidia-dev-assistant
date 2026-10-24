"""端到端評測:讓 HSUAN 自己修 evals/tasks/ 裡埋好 bug 的小專案,看「改完驗收通過」的比例、tokens 和秒數。

每一題:把 repo/ 複製到暫存工作區 → 照題目問一輪(跟網頁 / CLI 同一套 AgentSession 和工具)→ 拿 HSUAN 草稿裡的修改驗收:
原本的測試(HSUAN 改過的測試檔換回原本的,改測試讓它過不算)加上 HSUAN 看不到的 hidden/ 測試,在 Docker 沙箱裡全部通過才算。
--configs 比較不同設定:super(預設)、deepseek(指定難題模型)、routed(依難度選模型 on)、split(分工)、
escalate(Super 做到步數上限就換 DeepSeek 接著做,等於使用者每次都按「換模型」)。

會呼叫模型(.env 的 NVIDIA_API_KEY);題目都是自己寫的,不含公司程式碼。需要 Docker Desktop 執行中。
用法:python run_evals.py [--configs super deepseek] [--tasks paging retry] [--repeat 2] [--budget 3000000]
     python run_evals.py --out output/evals/<之前那次>    # 接著跑沒跑完的(已經有結果的跳過,出錯的重跑)
     python run_evals.py --report output/evals/<那次>     # 只重新產生報表
結果:output/evals/<時間>/results.jsonl(每次一行)和 report.md。
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

import agent
import guard
import tools as toolbox
from token_guard import TAIPEI, QuotaExceeded, TokenQuota

EVALS = Path(__file__).with_name("evals")
TOOLS = toolbox.ALL_TOOLS + [agent.delegate]  # 跟 agent.prepare() 一樣(不開上網:題目用不到,也不用每次問)
USER = "eval"
# 驗收時不採用 HSUAN 新增的這些檔案:能改變 pytest 收集測試的方式(例如跳過隱藏測試)
PROTECTED = {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    role: str | None = None   # 指定模型(像 --role);None = 照路由
    routing: str = "off"      # DIFFICULTY_ROUTING
    split: bool = False       # 「分工」開關
    escalate: bool = False    # 做到步數上限就換 hard_role 接著做


CONFIGS = {c.name: c for c in [
    Config("super", "Super(預設)"),
    Config("deepseek", "DeepSeek(指定難題模型)", role="code_deep"),
    Config("routed", "依難度選模型", routing="on"),
    Config("split", "分工(子 agent)", split=True),
    Config("escalate", "Super,做到上限換 DeepSeek", escalate=True),
]}


@dataclass
class Task:
    id: str
    prompt: str
    root: Path
    kind: str = ""
    level: str = "normal"
    files: list[str] = field(default_factory=list)
    fails_visible: bool = True  # 原本的測試會不會失敗;False = 只有描述,要 HSUAN 自己重現
    history: list[dict] = field(default_factory=list)  # git 歷史:照順序提交的快照 {dir, message, author, date},最後一個是 repo


def load_tasks(root: Path = EVALS) -> list[Task]:
    items = yaml.safe_load((root / "tasks.yaml").read_text(encoding="utf-8"))
    return [Task(root=root / "tasks" / it["id"], **it) for it in items]


def reference_fix(task: Task) -> dict[str, str]:
    """參考解答:fixed/ 裡的檔案(相對路徑 → 內容),格式跟草稿一樣。"""
    fixed = task.root / "fixed"
    return {p.relative_to(fixed).as_posix(): p.read_text(encoding="utf-8") for p in sorted(fixed.rglob("*")) if p.is_file()}


def make_workspace(task: Task, dest: Path) -> Path:
    """題目的 repo/ 複製一份給 HSUAN,建成 git repo(git_log / git_blame 才有東西看)。
    有 history 的題目照順序提交每個快照,HSUAN 可以從歷史查出是哪次改壞的;沒有就只有一個「初始版本」。"""
    dest.mkdir(parents=True)
    has_git = shutil.which("git") is not None
    git = ["git", "-c", "core.autocrlf=false", "-c", "init.defaultBranch=main"]
    if has_git:
        subprocess.run(git + ["init", "-q"], cwd=dest, capture_output=True)
    for step in task.history or [{"dir": "repo", "message": "初始版本"}]:
        for p in dest.iterdir():  # 換成這個快照:上一個快照有、這個沒有的檔案要刪掉
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        shutil.copytree(task.root / step["dir"], dest, dirs_exist_ok=True)
        if has_git:
            date = {"GIT_AUTHOR_DATE": step["date"], "GIT_COMMITTER_DATE": step["date"]} if step.get("date") else {}
            who = ["-c", f"user.name={step.get('author', 'eval')}", "-c", "user.email=eval@example.invalid"]
            subprocess.run(git + ["add", "-A"], cwd=dest, capture_output=True)
            subprocess.run(git + who + ["commit", "-q", "-m", step["message"]], cwd=dest, capture_output=True,
                           env={**os.environ, **date})
    return dest


def docker_tests(root: Path) -> str:
    """驗收:跟 run_tests 同一個沙箱(無網路、唯讀、限資源)—— HSUAN 寫的程式不在主機上直接跑。"""
    return toolbox._docker(root, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"])


def verify(task: Task, draft: dict[str, str], run: Callable[[Path], str] = docker_tests, hidden: bool = True) -> dict:
    """原始專案 + HSUAN 的修改,但 tests/ 裡原本就有的檔案換回原本的,再放進隱藏測試,全部通過才算。"""
    repo = task.root / "repo"
    ignored = []
    with tempfile.TemporaryDirectory(prefix="hsuan_eval_") as tmp:
        root = Path(tmp) / "ws"
        shutil.copytree(repo, root)
        for rel, text in draft.items():
            original_test = rel.startswith("tests/") and (repo / rel).is_file()
            new_non_test = rel.startswith("tests/") and not Path(rel).name.startswith("test_")
            if original_test or new_non_test or Path(rel).name in PROTECTED:
                ignored.append(rel)
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        if hidden:
            for f in sorted((task.root / "hidden").glob("*.py")):
                shutil.copy(f, root / "tests" / f.name)
        out = run(root)
    return {"passed": out.startswith("exit code 0"), "output": out, "ignored": ignored}


def changed_lines(draft: dict[str, str]) -> int:
    """草稿跟原始檔案差幾行(+ 和 - 都算);要在 toolbox.using(這一題的 Workbench) 裡呼叫。"""
    n = 0
    for rel, text in draft.items():
        before = toolbox._original(rel) or ""
        n += sum(1 for line in difflib.unified_diff(before.splitlines(), text.splitlines(), lineterm="")
                 if line[:1] in "+-" and not line.startswith(("+++", "---")))
    return n


@contextmanager
def routing(mode: str):
    """這一題的依難度選模型(difficulty.settings 每次都重讀環境變數)。"""
    old = os.environ.get("DIFFICULTY_ROUTING")
    os.environ["DIFFICULTY_ROUTING"] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("DIFFICULTY_ROUTING", None)
        else:
            os.environ["DIFFICULTY_ROUTING"] = old


def run_one(task: Task, cfg: Config, registry, quota, work: Path, run: Callable[[Path], str] = docker_tests) -> dict:
    """跑一題、驗收,回傳一行結果。額度用完(QuotaExceeded)往外丟,其他錯誤記在 error。"""
    ws = make_workspace(task, work / "ws")
    wb = toolbox.Workbench(workspace=ws, patch_dir=work / "patches", output_dir=work / "docs")
    row = {"task": task.id, "config": cfg.name, "kind": task.kind, "level": task.level, "passed": False,
           "self_passed": None, "tokens": 0, "seconds": 0.0, "steps": 0, "roles": [], "hit_limit": False,
           "error": None, "judged": None, "files_changed": 0, "lines_changed": 0, "ignored": [], "answer": "", "tests": ""}
    began, session = time.monotonic(), None
    with toolbox.using(wb), routing(cfg.routing):
        try:
            guard.check_folder(ws, public=True)  # 交給模型的資料夾一律過這一關(題目是自己寫的)
            session = agent.AgentSession(registry, quota, user=USER, role=cfg.role, tools=TOOLS)
            r = session.ask(task.prompt, task.files, split=cfg.split)
            row.update(steps=r.steps, roles=[r.role], hit_limit=r.hit_limit, answer=r.answer[:500], served=list(r.models))
            if cfg.escalate and (hard := session.escalation(r)):
                r = session.ask(agent.CONTINUE_TASK, role=hard)
                row.update(steps=row["steps"] + r.steps, roles=row["roles"] + [r.role], hit_limit=r.hit_limit,
                           answer=r.answer[:500], served=row["served"] + [m for m in r.models if m not in row["served"]])
        except QuotaExceeded:
            raise
        except Exception as e:  # 伺服器忙線、放不下…:記下來,換下一題
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        if session and (d := session.judging) and d.done() and d.result().by == "classifier" and d.result().pick:
            hard = (registry.routing.get("difficulty") or {}).get("hard_role")  # 只算分類器判斷的(不算手動指定)
            row["judged"] = "hard" if d.result().pick == hard else "normal"
        row["seconds"] = round(time.monotonic() - began, 1)
        row["tokens"] = session.tokens_used if session else 0
        draft = dict(wb.draft)
        row.update(files_changed=len(draft), lines_changed=changed_lines(draft),
                   self_passed=wb.patches[-1][1] if wb.patches else None)
    if draft:
        result = verify(task, draft, run)
        row.update(passed=result["passed"], ignored=result["ignored"], tests=result["output"][-1500:])
    else:
        row["tests"] = "(HSUAN 沒有提出修改)"
    return row


def docker_problem() -> str | None:
    if not shutil.which("docker"):
        return "這台電腦沒有 Docker"
    r = subprocess.run(["docker", "image", "inspect", toolbox.SANDBOX_IMAGE], capture_output=True)
    if r.returncode != 0:
        return (f"Docker 沒有在執行,或還沒建沙箱映像:先打開 Docker Desktop,"
                f"再跑 docker build -t {toolbox.SANDBOX_IMAGE} sandbox")
    return None


def read_rows(out: Path) -> list[dict]:
    path = out / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest(rows: list[dict]) -> list[dict]:
    """同一題、同一個設定、同一次重複跑了好幾遍(接著跑時重跑出錯的),只留最後一筆。"""
    keep = {}
    for r in rows:
        keep[(r["task"], r["config"], r.get("rep", 1))] = r
    return list(keep.values())


def _k(tokens: float) -> str:
    return f"{tokens / 1000:.1f}k"


def _cell(rs: list[dict]) -> str:
    """每題結果表的一格。"""
    if not rs:
        return ""
    ok = sum(r["passed"] for r in rs)
    mark = ("✓" if ok else "✗") if len(rs) == 1 else f"{ok}/{len(rs)}"
    why = "" if ok else " 出錯" if any(r["error"] for r in rs) else " 上限" if any(r["hit_limit"] for r in rs) else ""
    return f"{mark}{why} · {_k(statistics.mean(r['tokens'] for r in rs))} · {statistics.mean(r['seconds'] for r in rs):.0f} 秒"


def summarize(rows: list[dict], when: str = "") -> str:
    rows = latest(rows)
    configs = list(dict.fromkeys(r["config"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    info = {r["task"]: r for r in rows}
    reps = max((r.get("rep", 1) for r in rows), default=1)
    out = [f"# HSUAN 端到端評測{(' · ' + when) if when else ''}", "",
           f"{len(tasks)} 題 × {len(configs)} 種設定" + (f" × 每題 {reps} 次" if reps > 1 else "") + f",共 {len(rows)} 次。"
           "通過 = 套用 HSUAN 的修改後,原本的測試(HSUAN 改過的測試檔換回原本的)加上它看不到的隱藏測試,"
           "在 Docker 沙箱裡全部通過。模型的 temperature 是 1.0,同一題每次結果可能不同。", "",
           "| 設定 | 通過 | 一般題 | 難題 | 平均 tokens | 每解一題 tokens | 秒數中位數 | 平均步數 | 做到上限 | 出錯 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        ok = [r for r in rs if r["passed"]]

        def rate(level):
            sub = [r for r in rs if r["level"] == level]
            return f"{sum(r['passed'] for r in sub)}/{len(sub)}" if sub else "-"

        total = sum(r["tokens"] for r in rs)
        label = CONFIGS[c].label if c in CONFIGS else c
        out.append(f"| {label} | **{len(ok)}/{len(rs)}** | {rate('normal')} | {rate('hard')} | "
                   f"{_k(total / len(rs))} | {_k(total / len(ok)) if ok else '-'} | "
                   f"{statistics.median(r['seconds'] for r in rs):.0f} | {statistics.mean(r['steps'] for r in rs):.1f} | "
                   f"{sum(r['hit_limit'] for r in rs)} | {sum(bool(r['error']) for r in rs)} |")
    out += ["", "## 每一題", "", "| 題目 | 類型 | 難度 | " + " | ".join(CONFIGS[c].label if c in CONFIGS else c for c in configs) + " |",
            "|---|---|---|" + "---|" * len(configs)]
    for t in tasks:
        cells = [_cell([r for r in rows if r["task"] == t and r["config"] == c]) for c in configs]
        out.append(f"| {t} | {info[t]['kind']} | {'難題' if info[t]['level'] == 'hard' else '一般'} | " + " | ".join(cells) + " |")

    claimed = [r for r in rows if r["self_passed"] and not r["passed"]]
    if claimed:
        out += ["", "## HSUAN 自己跑測試通過、驗收沒過", "",
                "通常是只修到題目舉的例子(隱藏測試抓到),或是改了測試檔讓它通過(驗收換回原本的測試)。", ""]
        for r in claimed:
            note = f"(驗收沒採用:{', '.join(r['ignored'])})" if r["ignored"] else ""
            out.append(f"- {r['task']} · {r['config']}{note}")

    judged = [r for r in rows if r.get("judged")]
    if judged:
        agree = sum(r["judged"] == r["level"] for r in judged)
        pairs = Counter((r["level"], r["judged"]) for r in judged)
        out += ["", "## 難度分類器", "", f"分類器的判斷跟出題時標的難度一致 {agree}/{len(judged)}"
                f"(標成難題的 {sum(n for (lv, _), n in pairs.items() if lv == 'hard')} 題,"
                f"判成難題 {pairs[('hard', 'hard')]};標成一般的判成難題 {pairs[('normal', 'hard')]})。"]
    errors = [r for r in rows if r["error"]]
    if errors:
        out += ["", "## 出錯", ""] + [f"- {r['task']} · {r['config']}:{r['error']}" for r in errors]
    return "\n".join(out) + "\n"


def write_report(out: Path) -> Path:
    path = out / "report.md"
    path.write_text(summarize(read_rows(out), out.name), encoding="utf-8")
    return path


def main() -> None:
    warnings.filterwarnings("ignore")  # 套件對不在內建表的模型一直警告
    sys.stdout.reconfigure(errors="replace")  # 輸出轉向到檔案時是系統編碼(cp950),印不出 ✓ 也不要中斷
    ap =argparse.ArgumentParser(description="HSUAN 端到端評測(會呼叫模型)")
    ap.add_argument("--configs", nargs="+", default=["super"], choices=list(CONFIGS))
    ap.add_argument("--tasks", nargs="+", help="只跑這幾題(題目 id,見 evals/tasks.yaml)")
    ap.add_argument("--repeat", type=int, default=1, help="每題跑幾次(temperature 1.0,多跑幾次比較準)")
    ap.add_argument("--budget", type=int, default=3_000_000, help="這次最多用掉多少 tokens,用完就停")
    ap.add_argument("--out", help="結果資料夾;給之前那次的就接著跑")
    ap.add_argument("--report", metavar="資料夾", help="只重新產生那次的報表")
    args = ap.parse_args()
    if args.report:
        print(write_report(Path(args.report)).read_text(encoding="utf-8"))
        return

    tasks = load_tasks()
    if args.tasks:
        unknown = set(args.tasks) - {t.id for t in tasks}
        if unknown:
            raise SystemExit(f"沒有這些題目:{', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t.id in args.tasks]
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    registry, _, _, _ = agent.prepare()
    if problem := docker_problem():
        raise SystemExit(problem)
    out = Path(args.out) if args.out else toolbox.OUTPUT_DIR / "evals" / datetime.now(TAIPEI).strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    # 這次的路由紀錄、成品都放在結果資料夾:不混進平常使用的 routing_log.jsonl(report_routing.py 看的是真實使用)
    toolbox.configure(output_dir=out)
    spent = 0  # 這次用掉的 tokens(--budget 是整次的上限)
    done = {(r["task"], r["config"], r.get("rep", 1)) for r in latest(read_rows(out)) if not r["error"]}
    todo = [(t, CONFIGS[c], rep) for rep in range(1, args.repeat + 1) for c in args.configs for t in tasks
            if (t.id, c, rep) not in done]
    print(f"{len(todo)} 次要跑(跳過已經有結果的 {len(done)} 次),結果在 {out}")
    with (out / "results.jsonl").open("a", encoding="utf-8") as f:
        for n, (task, cfg, rep) in enumerate(todo, 1):
            # 額度只記在記憶體,不算進平常的用量;每題用剩下的預算重建 —— TokenQuota 是「每日」額度,
            # 10/25 跑到一半過了午夜,用量歸零,預算和最後印的總用量都不對
            quota = TokenQuota(max(args.budget - spent, 0), registry.limits["max_concurrent_per_user"])
            with tempfile.TemporaryDirectory(prefix=f"hsuan_eval_{task.id}_") as tmp:
                try:
                    row = run_one(task, cfg, registry, quota, Path(tmp))
                except QuotaExceeded as e:
                    print(f"這次的 token 預算用完了,停止:{e}")
                    break
            row.update(rep=rep, at=datetime.now(TAIPEI).isoformat(timespec="seconds"),
                       # 實際回答的模型(伺服器回報的):10/25 Super 下架,備援 DeepSeek 默默接手,以前照角色記會記錯
                       models=[m.split("/")[-1] for m in row.pop("served", None) or
                               [registry.spec(r).model for r in row["roles"]]])
            spent += row["tokens"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            mark = "✓ 通過" if row["passed"] else "✗ 出錯" if row["error"] else "✗ 上限" if row["hit_limit"] else "✗ 沒過"
            print(f"[{n:>2}/{len(todo)}] {cfg.name:<8} {task.id:<18} {mark}  {row['steps']} 步  "
                  f"{row['tokens']:,} tokens  {row['seconds']:.0f} 秒" + (f"  {row['error']}" if row["error"] else ""),
                  flush=True)
    print(f"\n這次用掉 {spent:,} tokens。報表:{write_report(out)}")


if __name__ == "__main__":
    main()
