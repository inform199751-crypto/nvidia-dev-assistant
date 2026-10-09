"""知識庫:把一個資料夾的文件(手冊、規格書、會議紀錄…)切成小段、算向量存起來,問問題時照意思找出最相關的幾段。

- 切段:依標題切(Markdown 的 #、Word 的標題,documents.extract_text 會轉成 #),每段帶著標題路徑
  (員工手冊.md › 一、請假 › 申請流程);PDF 記頁碼;太長的段落在空行再切。參考 rag-chatbot 依標題切 Markdown 的做法。
- 增量更新:記每個檔案的大小和修改時間,改過的才重新切、重新算向量;刪掉的拿掉;換了 embedding 模型就全部重算。
- 搜尋:問題和每段的向量算餘弦相似度(models.yaml 的 embedding)。算不了(沒網路、服務忙線)就退回關鍵字(BM25,中文兩字一組)。
  10/9 實測 12 題換句話說的中文問題:關鍵字第 1 名命中 7/12,Nemotron 3 Embed 10/12(前 3 名 12/12);
  兩種排名混在一起反而變差(7–8/12),專有名詞和數字向量也找得到(6 題 5 對,錯的那題兩段都有提到),所以平常只用向量。
- 問題改寫成好搜尋的句子:交給主模型 —— search_knowledge 的 query 由它依前後文寫成完整的句子,不另外多呼叫一次模型。
- 執行時只用標準函式庫:向量正規化後存成 float32 二進位(array),相似度用純 Python 算(幾千段以內夠快)。
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import zlib
from array import array
from collections import Counter
from operator import mul
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool

import documents
import tools as toolbox
from token_guard import estimate_tokens

# embed(文字們, "passage" | "query") → (向量們, 用掉的 tokens);llm_registry.ModelRegistry.embedder() 建的,測試換成假的
Embedder = Callable[[list[str], str], tuple[list[list[float]], int]]

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".csv"}
SUFFIXES = TEXT_SUFFIXES | documents.DOC_SUFFIXES
MAX_FILE_BYTES = 20_000_000
MAX_FILES = 2000
CHUNK_TOKENS = 400  # 一段多長:太長找不準(一段講好幾件事),太短缺上下文
EMBED_BATCH = 16    # 一次送幾段去算向量
TOP_K = 5

HEADING = re.compile(r"^(#{1,6})\s+(.+)")
PAGE = re.compile(r"^--- 第 (\d+) 頁 ---$")  # documents._pdf 的分頁標記


def split(rel: str, text: str) -> list[dict]:
    """依標題切段:[{file, where(標題路徑), page, text}]。只有標題沒有內容的不算一段;PDF 的段落不跨頁。"""
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []
    page: int | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            where = " › ".join(title for _, title in stack)
            chunks.extend({"file": rel, "where": where, "page": page, "text": part} for part in _pieces(body))

    for line in text.splitlines():
        if m := PAGE.match(line.strip()):
            flush()
            page = int(m[1])
        elif m := HEADING.match(line):
            flush()
            level = len(m[1])
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m[2].strip())]
        else:
            buf.append(line)
    flush()
    return chunks


def _pieces(body: str) -> list[str]:
    """太長的段落在空行切開,單一段還是太長就照行、再照字數切。"""
    if estimate_tokens(body) <= CHUNK_TOKENS:
        return [body]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        if estimate_tokens(para) <= CHUNK_TOKENS:
            units.append(para)
            continue
        for line in para.splitlines():
            n = estimate_tokens(line)
            step = max(1, int(len(line) * CHUNK_TOKENS / n)) if n > CHUNK_TOKENS else len(line) or 1
            units += [line[i:i + step] for i in range(0, len(line), step)] or [line]
    out, cur, used = [], [], 0
    for u in units:
        n = estimate_tokens(u)
        if cur and used + n > CHUNK_TOKENS:
            out.append("\n\n".join(cur))
            cur, used = [], 0
        cur.append(u)
        used += n
    if cur:
        out.append("\n\n".join(cur))
    return [p for p in out if p.strip()]


def label(c: dict) -> str:
    """一段的出處:員工手冊.md › 一、請假 › 申請流程(第 3 頁)。"""
    where = f"{c['file']} › {c['where']}" if c["where"] else c["file"]
    return where + (f"(第 {c['page']} 頁)" if c.get("page") else "")


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def terms(text: str) -> list[str]:
    """關鍵字搜尋用的詞:英數字一個字一詞,中文兩個字一組(不用斷詞套件)。"""
    text = text.lower()
    out = re.findall(r"[a-z0-9][a-z0-9._-]*", text)
    for run in re.findall(r"[㐀-鿿]+", text):
        out += [run[i:i + 2] for i in range(len(run) - 1)] or [run]
    return out


class KnowledgeBase:
    """一個資料夾的知識庫。索引存在 store/(index.json + vectors.bin),重開程式不用重算。

    update() 可以在背景執行緒跑;跑的時候 search() 還是用舊的索引,做完才一次換掉。"""

    def __init__(self, folder: Path, store: Path, embed: Embedder | None = None, name: str = ""):
        self.folder, self.store, self.embed = Path(folder).resolve(), Path(store), embed
        self.name = name or self.folder.name
        self.lock = threading.Lock()  # 同時只跑一個 update
        self.state, self.error, self.progress = "idle", "", (0, 0)  # 畫面上顯示的狀態
        self.files: dict[str, dict] = {}  # 相對路徑 → {size, mtime, n(幾段), error?}
        self.index: tuple[list[dict], array, int] = ([], array("f"), 0)  # (段落, 向量, 維度);一起換掉
        self.model = ""   # 算向量用的模型;換了就要全部重算
        self.tokens = 0   # 建索引累計用掉的 tokens
        self.updated = 0.0
        self._keywords: tuple[list[Counter], Counter, float] | None = None
        self._load()

    @property
    def chunks(self) -> list[dict]:
        return self.index[0]

    # ---------- 存檔 ----------

    def _load(self) -> None:
        try:
            meta = json.loads((self.store / "index.json").read_text(encoding="utf-8"))
            vectors = array("f")
            if meta["dim"]:
                vectors.frombytes((self.store / "vectors.bin").read_bytes())
            chunks, dim = meta["chunks"], meta["dim"]
        except (OSError, ValueError, KeyError, TypeError):
            return  # 還沒建過、或壞掉了:當作空的,下次 update 全部重建
        if dim and len(vectors) != len(chunks) * dim:  # 寫到一半當掉之類的:向量不能用,下次全部重算
            vectors, dim, meta["model"] = array("f"), 0, ""
        self.files, self.model = meta["files"], meta.get("model", "")
        self.tokens, self.updated = meta.get("tokens", 0), meta.get("updated", 0.0)
        self.index = (chunks, vectors, dim)

    def _save(self) -> None:
        chunks, vectors, dim = self.index
        self.store.mkdir(parents=True, exist_ok=True)
        for name, data in (("vectors.bin", vectors.tobytes()),
                           ("index.json", json.dumps({"folder": str(self.folder), "model": self.model, "dim": dim,
                                                      "files": self.files, "chunks": chunks, "tokens": self.tokens,
                                                      "updated": self.updated}, ensure_ascii=False).encode())):
            tmp = self.store / (name + ".tmp")  # 先寫暫存檔再改名:寫到一半當掉不會留下壞檔
            tmp.write_bytes(data)
            tmp.replace(self.store / name)

    # ---------- 建索引 ----------

    def scan(self) -> dict[str, tuple[int, float]]:
        """資料夾裡要建索引的檔案:相對路徑 → (大小, 修改時間)。跟列檔工具一樣跳過隱藏路徑、大型目錄、機密檔、
        指到資料夾外面的連結;太大的檔案不收。"""
        out: dict[str, tuple[int, float]] = {}
        if not self.folder.is_dir():
            return out
        for f in sorted(self.folder.rglob("*")):
            rel = f.relative_to(self.folder)
            if f.suffix.lower() not in SUFFIXES or toolbox._is_secret(f) or \
                    any(part.startswith(".") or part in toolbox._SKIP_DIRS for part in rel.parts):
                continue
            real = f.resolve()
            if not (real.is_file() and real.is_relative_to(self.folder)) or toolbox._is_secret(real):
                continue
            st = real.stat()
            if st.st_size <= MAX_FILE_BYTES:
                out[rel.as_posix()] = (st.st_size, st.st_mtime)
                if len(out) >= MAX_FILES:
                    break
        return out

    def changes(self) -> dict[str, int]:
        """跟上次建索引比:幾個新檔、幾個改過、幾個刪掉(只看大小和時間,不讀內容)。"""
        found = self.scan()
        changed = sum(1 for rel, (size, mtime) in found.items()
                      if rel in self.files and (self.files[rel]["size"], self.files[rel]["mtime"]) != (size, mtime))
        return {"new": sum(rel not in self.files for rel in found), "changed": changed,
                "removed": sum(rel not in found for rel in self.files)}

    def _read(self, rel: str) -> str:
        path = self.folder / rel
        if path.suffix.lower() in documents.DOC_SUFFIXES:
            text = documents.extract_text(path)
            if text.startswith("(這份 PDF"):  # 掃描檔:沒有文字層,也沒有看圖模型可以讀
                raise ValueError("PDF 沒有文字層(掃描檔),讀不到內容")
            if "這次讀不出來" in text:  # 有掃描頁這次沒讀到:記成錯誤,下次更新再試(不要把錯誤訊息收進索引)
                raise ValueError("有掃描頁這次讀不出來(看圖模型忙線?),下次更新會再試")
            return text
        data = path.read_bytes()
        for enc in ("utf-8-sig", "cp950"):  # 台灣的舊文字檔、Excel 存的 CSV 常是 Big5
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def update(self) -> dict:
        """增量更新:新增 / 改過的檔案重新切段、算向量,刪掉的拿掉,沒變的沿用。
        回傳 {added, updated, removed, unchanged, chunks, tokens};出錯時 error 有值,舊的索引照用。"""
        with self.lock:
            self.state, self.error = "indexing", ""
            try:
                return self._update()
            except Exception as e:  # 算向量的服務忙線、資料夾不見了…:舊的索引不動
                self.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
                return {"error": self.error}
            finally:
                self.state, self.progress = ("error" if self.error else "idle"), (0, 0)

    def _update(self) -> dict:
        found = self.scan()
        model = getattr(self.embed, "model", "") if self.embed else ""
        old_chunks, old_vectors, old_dim = self.index
        same = model == self.model
        keep = {rel for rel, (size, mtime) in found.items()  # 上次讀不了的(看圖模型忙線…)檔案沒改也要再試
                if same and rel in self.files and (self.files[rel]["size"], self.files[rel]["mtime"]) == (size, mtime)
                and not self.files[rel].get("error")}
        todo = [rel for rel in found if rel not in keep]
        chunks, vectors = [], array("f")
        for i, c in enumerate(old_chunks):
            if c["file"] in keep:
                chunks.append(c)
                if old_dim:
                    vectors.extend(old_vectors[i * old_dim:(i + 1) * old_dim])
        files = {rel: self.files[rel] for rel in keep}
        fresh: list[dict] = []
        self.progress = (0, len(todo))
        for n, rel in enumerate(todo, 1):
            size, mtime = found[rel]
            try:
                parts = split(rel, self._read(rel))
            except Exception as e:  # 一份讀不了(密碼保護、掃描檔…)不擋其他的
                files[rel] = {"size": size, "mtime": mtime, "n": 0, "error": str(e)[:200]}
                continue
            files[rel] = {"size": size, "mtime": mtime, "n": len(parts)}
            fresh += parts
            self.progress = (n, len(todo))
        tokens, dim = 0, old_dim if keep else 0
        if fresh and self.embed:
            self.progress = (0, len(fresh))
            for i in range(0, len(fresh), EMBED_BATCH):
                batch = fresh[i:i + EMBED_BATCH]
                vecs, used = self.embed([f"{label(c)}\n{c['text']}" for c in batch], "passage")
                tokens += used
                dim = dim or len(vecs[0])
                for v in vecs:
                    vectors.extend(_unit(v))
                self.progress = (min(i + EMBED_BATCH, len(fresh)), len(fresh))
        elif not self.embed:
            vectors, dim = array("f"), 0  # 沒有 embedding:只能用關鍵字搜尋
        before = set(self.files)
        stats = {"added": sum(rel not in before for rel in todo), "updated": sum(rel in before for rel in todo),
                 "removed": len(before - set(found)), "unchanged": len(keep), "chunks": len(chunks) + len(fresh),
                 "tokens": tokens}
        self.files, self.model = files, model
        self.tokens += tokens
        self.updated = time.time()
        self.index = (chunks + fresh, vectors, dim)
        self._keywords = None
        self._save()
        return stats

    # ---------- 搜尋 ----------

    def search(self, query: str, k: int = TOP_K) -> tuple[list[dict], str]:
        """最相關的 k 段 [{file, where, page, text, score}],以及這次用哪種搜尋(「照意思」或「關鍵字」)。"""
        chunks, vectors, dim = self.index
        if not chunks:
            return [], ""
        scores, how = None, "關鍵字"
        if dim and self.embed:
            try:
                (qv,), _ = self.embed([query], "query")
                q = _unit(qv)
                scores = [sum(map(mul, q, vectors[i * dim:(i + 1) * dim])) for i in range(len(chunks))]
                how = "照意思"
            except Exception as e:  # 服務忙線:這次先用關鍵字,不要整個失敗
                how = f"關鍵字(算向量的服務暫時不能用:{str(e).splitlines()[0][:80] if str(e) else type(e).__name__})"
        if scores is None:
            scores = self._keyword_scores(query)
        top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
        return [{**chunks[i], "score": round(scores[i], 3), "kb": self.name} for i in top if scores[i] > 0], how

    def _keyword_scores(self, query: str) -> list[float]:
        """BM25:詞在這段出現越多、在全部段落越少見,分數越高。"""
        if self._keywords is None:
            docs = [Counter(terms(f"{label(c)}\n{c['text']}")) for c in self.chunks]
            df = Counter(t for d in docs for t in d)
            self._keywords = (docs, df, sum(sum(d.values()) for d in docs) / max(len(docs), 1))
        docs, df, avg = self._keywords
        q = set(terms(query))
        out = []
        for d in docs:
            n, score = sum(d.values()), 0.0
            for t in q & d.keys():
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * n / (avg or 1)))
            out.append(score)
        return out

    def summary(self) -> dict:
        """畫面上列的狀態。"""
        return {"name": self.name, "files": len(self.files), "chunks": len(self.chunks), "updated": self.updated,
                "state": self.state, "error": self.error, "progress": list(self.progress), "tokens": self.tokens,
                "semantic": bool(self.index[2]), "unreadable": sorted(r for r, f in self.files.items() if f.get("error"))}


def search_all(kbs: list[KnowledgeBase], query: str, k: int = TOP_K) -> tuple[list[dict], str]:
    """在好幾個知識庫一起找,依分數合併。"""
    hits, hows = [], set()
    for kb in kbs:
        found, how = kb.search(query, k)
        hits += found
        if how:
            hows.add(how)
    hits.sort(key=lambda h: -h["score"])
    return hits[:k], "、".join(sorted(hows))


def folder_id(folder: Path) -> str:
    """CLI 的 --kb 用:同一個資料夾每次用同一份索引。"""
    return f"cli-{zlib.crc32(str(Path(folder).resolve()).lower().encode()):08x}"


@tool
def search_knowledge(query: str, top_k: int = TOP_K) -> str:
    """在使用者的知識庫(他們自己的文件:手冊、規章、規格書、會議紀錄…)裡照意思找相關段落,不只比對關鍵字。
    query 寫成完整、不靠上下文的句子(把「它」「那個」換成實際的名稱)。回答時註明用到的來源(檔名和段落)。"""
    kbs = toolbox.bench().knowledge
    if not kbs:
        return "目前沒有知識庫。"
    hits, how = search_all(kbs, query, max(1, min(top_k, 10)))
    if not hits:
        return f"知識庫({'、'.join(kb.name for kb in kbs)})裡找不到相關的段落。請跟使用者說文件裡沒有寫到,不要自己編。"
    rows = [f"[{i}] {label(h)}(相關度 {h['score']:.2f})\n{h['text']}" for i, h in enumerate(hits, 1)]
    return (f"用「{how}」搜尋了知識庫({'、'.join(kb.name for kb in kbs)}),最相關的 {len(hits)} 段:\n\n"
            + "\n\n".join(rows) + "\n\n回答時請註明用到的來源(檔名 › 段落),文件裡沒寫到的不要自己補。")
