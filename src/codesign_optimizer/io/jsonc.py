from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_LINE_COMMENT_RE = re.compile(r"(^|[^:])//.*?$", flags=re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_jsonc(raw_text: str) -> str:
    """
    Remove // comments and trailing commas from JSONC-like text.

    This lightweight parser intentionally targets the simulator exchange format.
    """

    def _remove_comment(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return prefix

    without_comments = _LINE_COMMENT_RE.sub(_remove_comment, raw_text)
    return _TRAILING_COMMA_RE.sub(r"\1", without_comments)


def load_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(strip_jsonc(text))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
