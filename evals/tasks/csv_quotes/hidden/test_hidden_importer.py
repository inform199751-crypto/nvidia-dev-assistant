from importer import load_customers, parse_line


def test_address_with_comma():
    text = 'name,address,phone\n王小明,"台北市中正區,忠孝東路 1 號",0912345678\n'
    row = load_customers(text)[0]
    assert row["address"] == "台北市中正區,忠孝東路 1 號"
    assert row["phone"] == "0912345678"


def test_escaped_quotes():
    assert parse_line('a,"他說 ""你好""",c') == ["a", '他說 "你好"', "c"]


def test_empty_fields():
    assert parse_line("a,,c") == ["a", "", "c"]
    assert parse_line('"",x') == ["", "x"]


def test_quoted_comma_at_end():
    assert parse_line('1,"x, y"') == ["1", "x, y"]
