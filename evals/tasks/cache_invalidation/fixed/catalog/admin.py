"""後台:改單一商品價格、上傳 CSV 批次改價。"""
from catalog import importer


class AdminPanel:
    def __init__(self, service):
        self.service = service

    def change_price(self, sku, price):
        if price <= 0:
            raise ValueError("價格要大於 0")
        return self.service.update_price(sku, price)

    def upload_prices(self, csv_text):
        """回傳改了幾筆。"""
        changed = importer.import_prices(self.service.db, csv_text)
        # 匯入直接寫資料庫、不經過 ProductService:這裡要自己把改到的商品從快取拿掉
        self.service.forget(sku for sku, _ in importer.parse(csv_text))
        return changed
