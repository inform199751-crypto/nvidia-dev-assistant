from notes import add_tag, make_note


def test_add_tag_once():
    note = make_note("週會")
    add_tag(note, "work")
    add_tag(note, "work")
    assert note["tags"].count("work") == 1


def test_notes_do_not_share_tags():
    a = make_note("a")
    add_tag(a, "work")
    b = make_note("b")
    assert b["tags"] == ["all"]
