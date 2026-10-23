"""商品價格(元)。"""

PRICES = {"tea": 120, "mug": 450, "beans": 680, "grinder": 1500, "filter": 90, "kettle": 980}


def price(sku):
    try:
        return PRICES[sku]
    except KeyError:
        raise KeyError(f"沒有這個商品:{sku}") from None
