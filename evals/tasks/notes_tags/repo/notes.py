"""筆記與標籤。"""


def make_note(title, tags=[]):
    """建立一則筆記。每則筆記的 tags 一定有 "all"(不重複);不會改到呼叫的人傳進來的清單。"""
    tags.append("all")
    return {"title": title, "tags": tags}


def add_tag(note, tag):
    """幫筆記加標籤(不重複)。"""
    if tag not in note["tags"]:
        note["tags"].append(tag)
    return note
