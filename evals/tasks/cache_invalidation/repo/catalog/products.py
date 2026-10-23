"""商品服務:前台讀商品都走這裡(先看快取,沒有才查資料庫)。"""


def _key(sku):
    return f"product:{sku}"


class ProductService:
    def __init__(self, db, cache):
        self.db = db
        self.cache = cache

    def get(self, sku):
        product = self.cache.get(_key(sku))
        if product is None:
            product = self.db.get(sku)
            if product is not None:
                self.cache.set(_key(sku), product)
        return product

    def update_price(self, sku, price):
        product = self.db.get(sku)
        if product is None:
            raise KeyError(f"沒有這個商品:{sku}")
        product["price"] = price
        self.db.save(product)
        return product
