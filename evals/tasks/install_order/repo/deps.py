"""套件安裝順序:每個套件都要等它依賴的套件裝好才能裝。"""


class CycleError(Exception):
    """依賴有循環,排不出順序。"""


def install_order(deps):
    """deps:套件 → 它依賴的套件清單(只出現在依賴清單裡的套件代表它自己沒有依賴)。
    回傳安裝順序:每個套件都排在它依賴的套件後面,每個套件只出現一次;依賴有循環就丟 CycleError。"""
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for dep in sorted(deps.get(name, [])):
            visit(dep)
        order.append(name)

    for name in sorted(deps):
        visit(name)
    return order
