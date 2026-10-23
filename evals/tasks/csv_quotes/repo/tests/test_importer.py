from importer import load_customers


def test_simple_rows():
    rows = load_customers("name,phone\n王小明,0912345678\n\n李小華,0922333444\n")
    assert rows == [{"name": "王小明", "phone": "0912345678"}, {"name": "李小華", "phone": "0922333444"}]


def test_quoted_name():
    rows = load_customers('name,phone\n"陳大文",0933111222\n')
    assert rows[0]["name"] == "陳大文"
