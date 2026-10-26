"""畫圖(10/26,像 ChatGPT 畫圖):generate_image 工具請 NVIDIA 的 FLUX 畫一張圖,存在輸出資料夾,對話裡直接顯示。

畫圖的模型在 models.yaml 的 image(registry.image_generator());agent.prepare 把它放在 GENERATE,沒有就不綁這個工具。
描述交給主模型寫成英文(FLUX 看英文比較準)。訪客不能畫(用的是主人的金鑰)。畫圖不算 token,不扣每日額度。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Literal

from langchain_core.tools import tool

import tools as toolbox

GENERATE: Callable[[str, int, int], tuple[bytes, str]] | None = None  # agent.prepare 設定;None = 不能畫
SIZES = {"square": (1024, 1024), "landscape": (1344, 768), "portrait": (768, 1344)}
MAX_PROMPT_CHARS = 1000


def _save(data: bytes, prompt: str) -> Path:
    """存成「圖片-描述開頭.jpg」,同名自動加編號(跟 save_document 一樣存在這個人的輸出資料夾)。"""
    out_dir = toolbox.bench().output_dir or toolbox.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "圖片-" + (re.sub(r"[^\w-]+", "-", prompt).strip("-")[:30] or "未命名")
    target, n = out_dir / f"{stem}.jpg", 2
    while target.exists():
        target, n = out_dir / f"{stem}_{n}.jpg", n + 1
    target.write_bytes(data)
    return target


@tool
def generate_image(prompt: str, size: Literal["square", "landscape", "portrait"] = "square") -> str:
    """畫一張圖。prompt 用英文寫,描述主體、風格、構圖、顏色、光線;size 是 square / landscape(橫)/ portrait(直)。
    畫好存成檔案,使用者在對話裡直接看得到。不要畫真實人物的肖像、商標或不當內容。"""
    if GENERATE is None:
        return "錯誤:這裡沒有畫圖的模型(models.yaml 的 image)"
    prompt = " ".join(str(prompt).split())[:MAX_PROMPT_CHARS]
    if not prompt:
        return "錯誤:prompt 是空的,請描述要畫什麼"
    try:
        data, model = GENERATE(prompt, *SIZES.get(size, SIZES["square"]))
    except ValueError as e:  # 被內容過濾擋下:換描述
        return f"錯誤:{e}"
    except Exception as e:  # 忙線、逾時:回報,讓模型跟使用者說
        return f"錯誤:畫圖失敗({type(e).__name__}: {str(e)[:200]})"
    target = _save(data, prompt)
    return (f"已畫好 {target.name}\n(用 {model.split('/')[-1]} 畫的;使用者在對話裡看得到這張圖,也能從側邊欄「成品」下載。"
            "你看不到圖片本身,不要描述你沒看到的細節)")
