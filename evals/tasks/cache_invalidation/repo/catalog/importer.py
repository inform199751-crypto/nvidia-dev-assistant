"""從 Excel 匯出的 CSV 批次改價格。"""
import csv
import io


def parse(text):
    """CSV 欄位:sku,price;回傳 [(sku, 價格)]。"""
    rows = list(csv.DictReader(io.StringIO(text)))
    return [(r["sku"].strip(), int(r["price"])) for r in rows if r.get("sku")]


def import_prices(db, text):
    """把 CSV 的價格寫進資料庫;不存在的商品跳過。回傳改了幾筆。"""
    changed = []
    for sku, price in parse(text):
        product = db.get(sku)
        if product is not None:
            product["price"] = price
            changed.append(product)
    if changed:
        db.save_many(changed)
    return len(changed)
