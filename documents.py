"""辦公檔案轉文字(PDF、Word、Excel、PowerPoint),給 read_document 工具、@附檔、拖進來的檔案和知識庫用。"""
from __future__ import annotations

import io
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable
from xml.etree import ElementTree

from token_guard import estimate_tokens

DOC_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}
MAX_SHEET_ROWS = 500  # 每個工作表最多轉幾列,大表格請使用者先篩選

# 掃描檔:沒有文字層的頁,把頁面上的掃描圖交給看圖模型讀(agent.prepare 設定成 agent.page_reader)。
# ocr(圖片, mime) → 文字;None = 不讀,照舊說是掃描檔。10/10 實測 Nano Omni 一頁約 4 秒、2,200 tokens,逐字全對;
# Nemotron Parse 2.0 在 catalog 的 API 格式不相容(400 / 500)
OCR: Callable[[bytes, str], str] | None = None
MAX_OCR_PAGES = 20   # 一份最多讀幾頁掃描頁(每頁約 2,200 tokens)
OCR_SIDE = 1600      # 掃描圖縮到這個邊長再送(字還看得清楚,tokens 少很多)
_ocr_cache: dict[tuple, str] = {}  # (檔案, 大小, 修改時間, 頁) → 文字:同一份檔案讀第二次不用再花 token


def extract_text(path: Path) -> str:
    """辦公檔案轉成文字;其他檔案當純文字讀。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf(path)
    if suffix == ".docx":
        return _docx(path)
    if suffix in (".xlsx", ".xlsm"):
        return _xlsx(path)
    if suffix == ".pptx":
        return _pptx(path)
    if suffix in (".doc", ".xls", ".ppt"):
        raise ValueError(f"{suffix} 是舊版 Office 格式,請先另存成 {suffix}x 再讀")
    return path.read_text(encoding="utf-8", errors="replace")


# 網頁拖進來 / 上傳的單一檔案(11/5 使用者要「每個檔案都可以丟進去」;以前只收圖片,其他的默默丟掉):
# 辦公檔案轉文字、其他文字檔照讀,讀不了的先說清楚原因
OLD_OFFICE = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}
ARCHIVES = {".7z", ".rar", ".gz", ".tgz", ".tar", ".bz2", ".xz"}  # 只解得開 .zip(Python 內建)
BINARY_SUFFIXES = {".exe", ".dll", ".msi", ".so", ".dylib", ".bin", ".iso", ".img", ".dmg", ".apk", ".jar", ".class", ".pyc",
                   ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg",
                   ".psd", ".ai", ".sketch", ".fig", ".sqlite", ".db", ".mdb", ".accdb", ".ttf", ".otf", ".woff", ".woff2"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".tif", ".tiff", ".ico", ".svg"}


def decode_text(data: bytes) -> str | None:
    """文字檔的內容;前面就有 NUL 的是二進位檔,回 None。台灣的舊檔案常是 Big5(cp950),UTF-8 讀不了再試它。"""
    if b"\x00" in data[:8192]:
        return None
    for encoding in ("utf-8-sig", "cp950"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def check_upload(name: str, data: bytes) -> str | None:
    """拖進來的檔案能不能讀:辦公檔案回 None(送出後才轉文字,掃描頁要看圖模型讀,不能卡住上傳),
    文字檔回內容;讀不了丟 ValueError,訊息直接給使用者看。"""
    suffix = Path(name).suffix.lower()
    if suffix in DOC_SUFFIXES:
        return None
    if why := _unreadable(suffix):
        raise ValueError(f"{name}:{why}")
    text = decode_text(data)
    if text is None:
        raise ValueError(f"{name}:不是文字檔,JUSTIN 讀不了(可以傳 PDF、Word、Excel、PowerPoint、圖片、程式碼、"
                         ".zip 或其他文字檔)")
    return text


def _unreadable(suffix: str) -> str:
    """看副檔名就知道讀不了的,回原因;其他回空字串。"""
    if suffix in OLD_OFFICE:
        return f"{suffix} 是舊版 Office 格式,請先另存成 {OLD_OFFICE[suffix]} 再傳"
    if suffix in ARCHIVES:
        return "只解得開 .zip,請改壓成 .zip 再傳"
    if suffix in BINARY_SUFFIXES:
        return "JUSTIN 讀不了影音、執行檔和這類二進位檔"
    return ""


# 拖進來的 .zip(11/5):解開讀裡面的文字檔和文件,不寫到硬碟。不相信標頭寫的大小(壓縮炸彈),每個檔案實際只解到上限
ZIP_MAX_MEMBERS = 2000       # 項目多於這個就不看(多半是整個 node_modules)
ZIP_MAX_FILE = 8_000_000     # 裡面一個檔案最多解多少
ZIP_MAX_TOTAL = 40_000_000   # 全部解開加起來
ZIP_SKIP_DIRS = {"node_modules", ".git", ".svn", ".venv", "venv", "__pycache__", ".idea", ".vs", "__MACOSX",
                 "dist", "build", ".next", "target", ".gradle"}


def read_zip(name: str, data: bytes, secret: Callable[[str], bool]) -> tuple[list[tuple[str, str, object]], list[str]]:
    """壓縮檔 → ([(壓縮檔名/裡面的路徑, "doc" / "text", 內容)], [略過的檔案(原因)])。
    相依套件、版本控制的資料夾整個跳過不列;機密檔(secret 判斷)、圖片、讀不了的列在略過裡。全部都讀不了丟 ValueError。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError):
        raise ValueError(f"{name}:壓縮檔壞掉了,或不是 .zip") from None
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if len(infos) > ZIP_MAX_MEMBERS:
            raise ValueError(f"{name}:壓縮檔裡有 {len(infos):,} 個檔案(最多 {ZIP_MAX_MEMBERS:,} 個);請只壓要看的資料夾")
        files, skipped, total = [], [], 0
        for info in infos:
            path = re.sub(r'[<>"\r\n\t]', "_", info.filename.replace("\\", "/")).strip("/")
            parts = path.split("/")
            if any(p in ZIP_SKIP_DIRS or (p.startswith(".") and p not in (".", "..")) for p in parts[:-1]) \
                    or parts[-1].startswith("._"):  # 相依套件、.git、macOS 的附加檔:不看也不列
                continue
            suffix = PurePosixPath(path).suffix.lower()
            why = ("機密檔" if secret(path) else "有密碼" if info.flag_bits & 0x1 else
                   "圖片請直接傳" if suffix in IMAGE_SUFFIXES else
                   "壓縮檔裡的壓縮檔不解開" if suffix == ".zip" or suffix in ARCHIVES else
                   "讀不了" if suffix in BINARY_SUFFIXES or suffix in OLD_OFFICE else
                   "太大" if info.file_size > ZIP_MAX_FILE or total + info.file_size > ZIP_MAX_TOTAL else "")
            if not why:
                try:
                    with archive.open(info) as f:
                        raw = f.read(ZIP_MAX_FILE + 1)
                except Exception:  # 不支援的壓縮方式、資料壞掉
                    raw, why = b"", "讀不出來"
                if len(raw) > ZIP_MAX_FILE:
                    why = "太大"
                elif not why:
                    total += len(raw)
                    if suffix in DOC_SUFFIXES:
                        files.append((f"{name}/{path}", "doc", raw))
                        continue
                    text = decode_text(raw)
                    if text is not None:
                        files.append((f"{name}/{path}", "text", text))
                        continue
                    why = "不是文字檔"
            skipped.append(f"{path}({why})")
    if not files:
        detail = "、".join(skipped[:5]) + ("…" if len(skipped) > 5 else "")
        raise ValueError(f"{name}:壓縮檔裡沒有 JUSTIN 讀得了的檔案" + (f"({detail})" if detail else ""))
    return files, skipped


def extract_bytes(name: str, data: bytes) -> str:
    """辦公檔案的內容(不是電腦上的路徑)轉文字:寫進暫存檔再交給 extract_text。"""
    suffix = Path(name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="hsuan-upload-", ignore_cleanup_errors=True) as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(data)
        return extract_text(path)


# PowerPoint(11/5):.pptx 是裝 XML 的 zip,直接讀投影片和備忘稿的文字(不另外裝 python-pptx)。
# 順序照 presentation.xml 的投影片清單(檔名的編號在搬動投影片後會跟順序對不上)
_PML = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_DML = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
PPTX_PART_BYTES = 5_000_000  # 一張投影片的 XML 最多多大


def _pptx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        def xml(part: str):
            if z.getinfo(part).file_size > PPTX_PART_BYTES:
                raise ValueError("簡報裡有一張投影片的內容太大")
            with z.open(part) as f:
                return ElementTree.fromstring(f.read(PPTX_PART_BYTES + 1))

        def rels(part: str) -> dict[str, tuple[str, str]]:  # 關聯 id → (類型, 目標的完整路徑)
            folder, file = posixpath.split(part)
            name = posixpath.join(folder, "_rels", file + ".rels")
            if name not in z.namelist():
                return {}
            out = {}
            for r in xml(name).iter(f"{_PKG}Relationship"):
                target = r.get("Target", "")
                full = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
                out[r.get("Id")] = (r.get("Type", ""), full)
            return out

        def paragraphs(part: str) -> list[str]:
            lines = []
            for p in xml(part).iter(f"{_DML}p"):
                line = "".join(t.text or "" for t in p.iter(f"{_DML}t")).strip()
                if line:
                    lines.append(line)
            return lines

        try:
            deck = rels("ppt/presentation.xml")
            order = [deck[s.get(f"{_REL}id")][1] for s in xml("ppt/presentation.xml").iter(f"{_PML}sldId")]
        except KeyError:
            raise ValueError("這份簡報的格式讀不懂(可能壞掉了),請另存成 PDF 再傳") from None
        out = []
        for n, slide in enumerate(order, 1):
            text = paragraphs(slide)
            notes = [t for kind, t in rels(slide).values() if kind.endswith("/notesSlide")]
            # 備忘稿的頁碼欄位(a:fld)只是一個數字,拿掉
            said = [line for line in (paragraphs(notes[0]) if notes else []) if not line.isdigit()]
            out.append(f"## 第 {n} 張投影片\n" + ("\n".join(text) or "(沒有文字)")
                       + ("\n備忘稿:" + "\n".join(said) if said else ""))
    return "\n\n".join(out) or "(這份簡報沒有投影片)"


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("這份 PDF 有密碼保護,請先解除密碼再讀")
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    note = ""
    if OCR is not None and not all(pages):  # 有掃描頁:交給看圖模型讀(只讀沒有文字的頁)
        pages, note = _ocr_pages(path, reader, pages)
    if not any(pages):
        return f"(這份 PDF 共 {len(pages)} 頁,但沒有文字層,可能是掃描檔,要先做 OCR 才讀得到內容)"
    return "\n\n".join(f"--- 第 {i} 頁 ---\n{text}" for i, text in enumerate(pages, 1)) + note


def _ocr_pages(path: Path, reader, pages: list[str]) -> tuple[list[str], str]:
    """沒有文字層的頁:取頁面上最大的那張圖(掃描檔一頁就是一張圖),縮小後交給 OCR。讀不了的頁寫原因,不擋其他頁。"""
    st = path.stat()
    out, done = list(pages), 0
    for i, text in enumerate(pages):
        if text:
            continue
        if done >= MAX_OCR_PAGES:
            return out, f"\n\n[掃描頁太多,只讀了前 {MAX_OCR_PAGES} 頁]"
        key = (str(path.resolve()), st.st_size, st.st_mtime, i)
        if key not in _ocr_cache:
            try:
                images = sorted(reader.pages[i].images, key=lambda im: len(im.data), reverse=True)
                if not images:
                    continue  # 真的是空白頁
                img = images[0].image.convert("RGB")
                img.thumbnail((OCR_SIDE, OCR_SIDE))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                _ocr_cache[key] = "(掃描頁,由看圖模型讀出,可能有錯字)\n" + OCR(buf.getvalue(), "image/jpeg").strip()
            except Exception as e:  # 看圖模型忙線、圖片格式特殊:這頁先跳過,下次再讀
                out[i] = f"(第 {i + 1} 頁是掃描頁,這次讀不出來:{str(e).splitlines()[0][:100] if str(e) else type(e).__name__})"
                continue
        out[i] = _ocr_cache[key]
        done += 1
    return out, ""


def _docx(path: Path) -> str:
    from docx import Document
    from docx.table import Table

    out = []
    for block in Document(path).iter_inner_content():  # 段落和表格照原本的順序
        if isinstance(block, Table):
            out += [" | ".join(cell.text.strip() for cell in row.cells) for row in block.rows]
            out.append("")
        elif text := block.text.strip():
            style = block.style.name if block.style is not None else ""
            level = style.removeprefix("Heading ").strip()
            out.append(f"{'#' * int(level)} {text}" if style.startswith("Heading ") and level.isdigit() else text)
    return "\n".join(out).strip()


def _xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)  # data_only:公式給計算結果,不給公式本身
    out = []
    try:
        for ws in wb.worksheets:
            out.append(f"## 工作表:{ws.title}")
            shown = 0
            for row in ws.iter_rows(values_only=True):
                cells = ["" if v is None else str(v) for v in row]
                if not any(cells):
                    continue
                if shown == MAX_SHEET_ROWS:
                    out.append(f"[只顯示前 {MAX_SHEET_ROWS} 列;表格更大,請先篩選或分成小檔]")
                    break
                out.append(" | ".join(cells).rstrip(" |"))
                shown += 1
            out.append("")
    finally:
        wb.close()
    return "\n".join(out).strip()


def split_parts(text: str, max_tokens: int) -> list[str]:
    """照行切成每段不超過 max_tokens 的幾段(單行太長就自己成一段)。"""
    parts, current, used = [], [], 0
    for line in text.splitlines():
        cost = estimate_tokens(line) + 1
        if current and used + cost > max_tokens:
            parts.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        parts.append("\n".join(current))
    return parts or [""]
