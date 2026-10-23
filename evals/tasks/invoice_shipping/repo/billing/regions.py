"""從地址判斷本島 / 離島。"""

ISLANDS = ("澎湖", "金門", "連江", "馬祖", "綠島", "蘭嶼", "小琉球")


def region_of(address):
    text = address.replace("台", "臺").replace("臺灣", "").strip()
    return "離島" if any(text.startswith(name) or f"縣{name}" in text for name in ISLANDS) else "本島"
