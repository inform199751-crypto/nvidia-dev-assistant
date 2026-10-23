from toc import slugify, toc


def test_punctuation_removed():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("API(v2)說明") == "apiv2說明"


def test_mixed_chinese_english():
    assert slugify("設定 Config 檔") == "設定-config-檔"


def test_duplicate_headings_get_suffix():
    md = "## 範例\n### 範例\n## 範例\n## Other\n"
    assert toc(md) == ("- [範例](#範例)\n  - [範例](#範例-1)\n- [範例](#範例-2)\n- [Other](#other)")
