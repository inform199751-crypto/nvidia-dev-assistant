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

    def forget(self, skus):
        """這些商品在資料庫改過了:快取裡的舊資料丟掉,下次讀重新查(其他商品的快取留著)。"""
        for sku in skus:
            self.cache.delete(_key(sku))

    def update_price(self, sku, price):
        product = self.db.get(sku)
        if product is None:
            raise KeyError(f"沒有這個商品:{sku}")
        product["price"] = price
        self.db.save(product)
        self.forget([sku])  # 以前沒有清快取:前台要等快取過期(10 分鐘)才看到新價格
        return product
