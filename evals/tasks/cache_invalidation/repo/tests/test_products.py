from catalog.admin import AdminPanel
from catalog.cache import Cache
from catalog.db import Database
from catalog.products import ProductService


def setup():
    now = [0]
    db = Database([{"sku": "A", "name": "茶包", "price": 120}, {"sku": "B", "name": "馬克杯", "price": 450}])
    service = ProductService(db, Cache(ttl=600, clock=lambda: now[0]))
    return db, service, AdminPanel(service), now


def test_reads_are_cached():
    db, service, _, _ = setup()
    assert service.get("A")["price"] == 120
    assert service.get("A")["price"] == 120
    assert db.reads == 1


def test_change_price_saved():
    db, _, admin, _ = setup()
    admin.change_price("A", 99)
    assert db.rows["A"]["price"] == 99


def test_upload_saved():
    db, _, admin, _ = setup()
    assert admin.upload_prices("sku,price\nA,100\nB,400\nZ,1\n") == 2
    assert (db.rows["A"]["price"], db.rows["B"]["price"]) == (100, 400)
