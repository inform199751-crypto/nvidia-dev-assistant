"""畫圖(10/26,像 ChatGPT 畫圖):generate_image 工具請 NVIDIA 的 FLUX 畫一張圖,存在輸出資料夾,對話裡直接顯示。

畫圖的模型在 models.yaml 的 image(registry.image_generator());agent.prepare 把它放在 GENERATE,沒有就不綁這個工具。
描述交給主模型寫成英文(FLUX 看英文比較準)。訪客不能畫(用的是主人的金鑰)。畫圖不算 token,不扣每日額度。

改圖(10/31,edit_image):catalog 上能改圖的 FLUX.1 Kontext 和 klein 的改圖模式,試用 API 都只收網頁上的範例圖
(自己的圖用 base64 或先上傳成 asset 都回 422「Expected: example_id」)。改成記下每張圖的描述、尺寸、種子和模型
(輸出資料夾的 .images.json),改圖時同一顆模型、同一個種子、改過的描述重畫:描述一樣畫出來一模一樣,
加一句「站著一隻粉紅色獨角獸」彩虹、雲、兩邊的花都沒動。構圖大致保留,但不是在原圖上直接改。
"""
from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Callable, Literal

from langchain_core.tools import tool

import tools as toolbox

GENERATE: Callable[..., tuple[bytes, str]] | None = None  # agent.prepare 設定;None = 不能畫
SIZES = {"square": (1024, 1024), "landscape": (1344, 768), "portrait": (768, 1344)}
MAX_PROMPT_CHARS = 1000
INDEX = ".images.json"  # 畫過的圖:檔名 → 描述、尺寸、種子、模型(點開頭,「成品」不會列、也下載不到)
MAX_REMEMBERED = 300


def _out_dir() -> Path:
    return toolbox.bench().output_dir or toolbox.OUTPUT_DIR


def _save(data: bytes, prompt: str) -> Path:
    """存成「圖片-描述開頭.jpg」,同名自動加編號(跟 save_document 一樣存在這個人的輸出資料夾)。"""
    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "圖片-" + (re.sub(r"[^\w-]+", "-", prompt).strip("-")[:30] or "未命名")
    target, n = out_dir / f"{stem}.jpg", 2
    while target.exists():
        target, n = out_dir / f"{stem}_{n}.jpg", n + 1
    target.write_bytes(data)
    return target


def _drawings() -> dict:
    try:
        data = json.loads((_out_dir() / INDEX).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember(name: str, prompt: str, size: tuple[int, int], seed: int, model: str) -> None:
    """記下這張圖怎麼畫的(改圖要用);只留最近 MAX_REMEMBERED 張。記不下來不影響畫好的圖。"""
    data = _drawings()
    data[name] = {"prompt": prompt, "size": list(size), "seed": seed, "model": model}
    data = dict(list(data.items())[-MAX_REMEMBERED:])
    try:
        (_out_dir() / INDEX).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _drawn(target: Path, model: str, note: str = "") -> str:
    return (f"已畫好 {target.name}\n(用 {model.split('/')[-1]} 畫的{note};使用者在對話裡看得到這張圖,也能從側邊欄「成品」下載。"
            "你看不到圖片本身,不要描述你沒看到的細節)")


@tool
def generate_image(prompt: str, size: Literal["square", "landscape", "portrait"] = "square") -> str:
    """畫一張圖。prompt 用英文寫,描述主體、風格、構圖、顏色、光線;size 是 square / landscape(橫)/ portrait(直)。
    有名的建築、地標,畫圖模型常常只看名字畫不出來:名字之外,把外型寫清楚(輪廓、分幾段、每段的形狀、材質和顏色)。
    畫好存成檔案,使用者在對話裡直接看得到。不要畫真實人物的肖像、商標或不當內容。"""
    if GENERATE is None:
        return "錯誤:這裡沒有畫圖的模型(models.yaml 的 image)"
    prompt = " ".join(str(prompt).split())[:MAX_PROMPT_CHARS]
    if not prompt:
        return "錯誤:prompt 是空的,請描述要畫什麼"
    dims, seed = SIZES.get(size, SIZES["square"]), secrets.randbelow(2**31)
    try:
        data, model = GENERATE(prompt, *dims, seed=seed)
    except ValueError as e:  # 被內容過濾擋下:換描述
        return f"錯誤:{e}"
    except Exception as e:  # 忙線、逾時:回報,讓模型跟使用者說
        return f"錯誤:畫圖失敗({type(e).__name__}: {str(e)[:200]})"
    target = _save(data, prompt)
    _remember(target.name, prompt, dims, seed, model)
    return _drawn(target, model)


@tool
def edit_image(image: str, prompt: str) -> str:
    """改你畫過的圖(加東西、拿掉、換顏色或風格):image 是「已畫好」後面的檔名;prompt 是改好之後完整的英文描述
    (拿原本的描述改,沒要改的照原本寫)。用同一個隨機種子重畫,構圖大致保留,但不是在原圖上直接改。"""
    if GENERATE is None:
        return "錯誤:這裡沒有畫圖的模型(models.yaml 的 image)"
    name = Path(str(image).strip()).name
    meta = _drawings().get(name)
    if not meta:
        return f"錯誤:找不到 {name} 的畫圖紀錄,只能改 HSUAN 畫的圖;可以用 generate_image 照使用者的意思重新畫一張"
    prompt = " ".join(str(prompt).split())[:MAX_PROMPT_CHARS]
    if not prompt:
        return "錯誤:prompt 是空的,請寫改好之後完整的描述"
    try:
        data, model = GENERATE(prompt, *meta["size"], seed=meta["seed"], prefer=meta["model"])
    except ValueError as e:
        return f"錯誤:{e}"
    except Exception as e:
        return f"錯誤:改圖失敗({type(e).__name__}: {str(e)[:200]})"
    target = _save(data, prompt)
    _remember(target.name, prompt, tuple(meta["size"]), meta["seed"], model)
    note = (f",照 {name} 的種子重畫,沒改的地方大致一樣" if model == meta["model"] else
            f";原本那顆模型沒回應,換了一顆,構圖會跟 {name} 不一樣")
    return _drawn(target, model, note)
