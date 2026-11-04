"""辦公檔案轉文字(PDF、Word、Excel),給 read_document 工具、@附檔和知識庫用。"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Callable

from token_guard import estimate_tokens

DOC_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm"}
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
    if suffix in (".doc", ".xls", ".ppt"):
        raise ValueError(f"{suffix} 是舊版 Office 格式,請先另存成 {suffix}x 再讀")
    return path.read_text(encoding="utf-8", errors="replace")


# 網頁拖進來 / 上傳的單一檔案(11/5 使用者要「每個檔案都可以丟進去」;以前只收圖片,其他的默默丟掉):
# 辦公檔案轉文字、其他文字檔照讀,讀不了的先說清楚原因
OLD_OFFICE = {".doc": ".docx", ".xls": ".xlsx", ".ppt": "PDF", ".pptx": "PDF"}  # 沒裝 python-pptx,簡報請存成 PDF
BINARY_SUFFIXES = {".zip", ".7z", ".rar", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".exe", ".dll", ".msi", ".so", ".dylib",
                   ".bin", ".iso", ".img", ".dmg", ".apk", ".jar", ".class", ".pyc", ".mp4", ".mov", ".avi", ".mkv",
                   ".webm", ".wmv", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".psd", ".ai", ".sketch", ".fig",
                   ".sqlite", ".db", ".mdb", ".accdb", ".ttf", ".otf", ".woff", ".woff2"}


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
    if suffix in OLD_OFFICE:
        raise ValueError(f"{name}:{suffix} 請先另存成 {OLD_OFFICE[suffix]} 再傳")
    if suffix in BINARY_SUFFIXES:
        raise ValueError(f"{name}:JUSTIN 讀不了壓縮檔、影音和執行檔")
    text = decode_text(data)
    if text is None:
        raise ValueError(f"{name}:不是文字檔,JUSTIN 讀不了(可以傳 PDF、Word、Excel、圖片、程式碼或其他文字檔)")
    return text


def extract_bytes(name: str, data: bytes) -> str:
    """辦公檔案的內容(不是電腦上的路徑)轉文字:寫進暫存檔再交給 extract_text。"""
    suffix = Path(name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="hsuan-upload-", ignore_cleanup_errors=True) as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(data)
        return extract_text(path)


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
