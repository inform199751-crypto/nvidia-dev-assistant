from notes import add_tag, make_note


def test_given_tags_are_copied():
    mine = ["work"]
    note = make_note("a", mine)
    assert note["tags"] == ["work", "all"]
    add_tag(note, "urgent")
    assert mine == ["work"]


def test_all_not_duplicated():
    assert make_note("a", ["all"])["tags"] == ["all"]


def test_many_notes_independent():
    notes = [make_note(str(n)) for n in range(3)]
    add_tag(notes[0], "x")
    assert [n["tags"] for n in notes] == [["all", "x"], ["all"], ["all"]]
