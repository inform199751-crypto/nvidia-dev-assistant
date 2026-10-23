"""清單分頁。頁碼從 1 開始。"""


def total_pages(count, per_page):
    """總共幾頁;沒有資料也算 1 頁(畫面顯示「沒有資料」)。"""
    return max(1, count // per_page)


def page_items(items, page, per_page):
    """第 page 頁的資料;頁碼超出範圍回傳空清單。"""
    start = page * per_page
    return items[start:start + per_page]
