"""從 Markdown 標題產生目錄(錨點規則跟 GitHub 一樣)。"""
import re


def slugify(title):
    """GitHub 的錨點:轉小寫、去掉標點符號(中日韓文字、英數字、-、_ 都保留)、空白換成 -。"""
    return re.sub(r"[^a-z0-9\- ]", "", title.lower()).replace(" ", "-")


def toc(markdown):
    """每個 ## / ### 標題一行「- [標題](#錨點)」,### 前面多縮排兩格。
    同名標題的錨點依序加 -1、-2(跟 GitHub 一樣)。"""
    lines = []
    for line in markdown.splitlines():
        m = re.match(r"(#{2,3}) (.+)", line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2).strip()
        indent = "  " * (level - 2)
        lines.append(f"{indent}- [{title}](#{slugify(title)})")
    return "\n".join(lines)
