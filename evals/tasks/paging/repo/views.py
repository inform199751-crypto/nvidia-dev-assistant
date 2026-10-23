"""訂單列表頁。"""
from paging import page_items, total_pages


def order_page(orders, page, per_page=10):
    return {
        "page": page,
        "pages": total_pages(len(orders), per_page),
        "orders": page_items(orders, page, per_page),
    }
