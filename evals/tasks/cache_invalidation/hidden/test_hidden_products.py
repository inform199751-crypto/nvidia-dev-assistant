from catalog.admin import AdminPanel
from catalog.cache import Cache
from catalog.db import Database
from catalog.products import ProductService


def setup():
    now = [0]
    db = Database([{"sku": "A", "name": "茶包", "price": 120}, {"sku": "B", "name": "馬克杯", "price": 450},
                   {"sku": "C", "name": "濾紙", "price": 90}])
    service = ProductService(db, Cache(ttl=600, clock=lambda: now[0]))
    return db, service, AdminPanel(service), now


def test_price_change_visible_right_away():
    _, service, admin, now = setup()
    service.get("A")
    now[0] = 30
    admin.change_price("A", 99)
    assert service.get("A")["price"] == 99


def test_upload_visible_right_away():
    _, service, admin, _ = setup()
    service.get("A"), service.get("B")
    assert admin.upload_prices("sku,price\nA,100\nB,400\nZ,1\n") == 2
    assert (service.get("A")["price"], service.get("B")["price"]) == (100, 400)


def test_cache_still_used_after_change():
    db, service, admin, _ = setup()
    service.get("A"), service.get("B")
    admin.change_price("A", 99)
    reads = db.reads
    service.get("A"), service.get("A"), service.get("B")
    assert db.reads == reads + 1  # 只有改過的 A 重查一次,B 還在快取裡


def test_cache_still_used_after_upload():
    db, service, admin, _ = setup()
    admin.upload_prices("sku,price\nC,80\n")
    service.get("C")
    reads = db.reads
    service.get("C"), service.get("C")
    assert db.reads == reads and service.get("C")["price"] == 80
