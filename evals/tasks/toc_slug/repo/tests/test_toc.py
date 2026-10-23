from toc import slugify, toc


def test_english_slug():
    assert slugify("Getting Started") == "getting-started"


def test_nested_headings():
    assert toc("# Title\n## Install\n### On Windows\n") == "- [Install](#install)\n  - [On Windows](#on-windows)"


def test_chinese_heading():
    assert toc("## 安裝步驟\n") == "- [安裝步驟](#安裝步驟)"
