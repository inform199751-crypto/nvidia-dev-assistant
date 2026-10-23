"""套件安裝順序:每個套件都要等它依賴的套件裝好才能裝。"""


class CycleError(Exception):
    """依賴有循環,排不出順序。"""


def install_order(deps):
    """deps:套件 → 它依賴的套件清單(只出現在依賴清單裡的套件代表它自己沒有依賴)。
    回傳安裝順序:每個套件都排在它依賴的套件後面,每個套件只出現一次;依賴有循環就丟 CycleError。"""
    # 以前只有一個 seen:走回正在處理的套件(循環)也當成「處理過」直接跳過。
    # 要分開「處理中」和「處理完」:碰到處理中的才是循環,碰到處理完的(共用的依賴)是正常的
    order, done, visiting = [], set(), []

    def visit(name):
        if name in done:
            return
        if name in visiting:
            cycle = visiting[visiting.index(name):] + [name]
            raise CycleError(" → ".join(cycle))
        visiting.append(name)
        for dep in sorted(deps.get(name, [])):
            visit(dep)
        visiting.pop()
        done.add(name)
        order.append(name)

    for name in sorted(deps):
        visit(name)
    return order
