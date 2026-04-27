from codesign_optimizer.io.jsonc import strip_jsonc


def test_strip_jsonc_removes_comments_and_trailing_commas() -> None:
    raw = """
    {
      "a": 1, // comment
      "b": [1,2,3,],
    }
    """
    cleaned = strip_jsonc(raw)
    assert "//" not in cleaned
    assert ",]" not in cleaned
    assert ",}" not in cleaned
