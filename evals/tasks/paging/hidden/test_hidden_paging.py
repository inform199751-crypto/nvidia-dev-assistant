from paging import page_items, total_pages

ITEMS = list(range(23))


def test_last_partial_page():
    assert page_items(ITEMS, 3, 10) == [20, 21, 22]


def test_beyond_last_page_is_empty():
    assert page_items(ITEMS, 4, 10) == []


def test_page_counts():
    assert total_pages(0, 10) == 1
    assert total_pages(20, 10) == 2
    assert total_pages(21, 10) == 3
    assert total_pages(5, 10) == 1
