#!/usr/bin/env python3
"""Check local Markdown links and fenced code blocks.

The script intentionally ignores external URLs. It is designed for this repo's
plain Markdown structure and uses only the Python standard library.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
)


def iter_markdown_files() -> list[Path]:
    ignored_parts = {".git"}
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not ignored_parts.intersection(path.relative_to(ROOT).parts)
    )


def strip_link_target(raw: str) -> str:
    target = raw.strip()
    if " " in target and not target.startswith("<"):
        target = target.split(" ", 1)[0]
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target


def is_external(target: str) -> bool:
    return target.startswith(EXTERNAL_PREFIXES) or target.startswith("#")


def remove_fenced_blocks(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            lines.append("")
        elif in_fence:
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def check_links(path: Path, errors: list[str]) -> None:
    text = remove_fenced_blocks(path.read_text(encoding="utf-8"))
    for match in LINK_RE.finditer(text):
        target = strip_link_target(match.group(1))
        if is_external(target):
            continue

        local = target.split("#", 1)[0]
        if not local:
            continue

        local_path = (path.parent / unquote(local)).resolve()
        try:
            local_path.relative_to(ROOT)
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)}: link escapes repo: {target}")
            continue

        if not local_path.exists():
            errors.append(f"{path.relative_to(ROOT)}: missing link target: {target}")


def check_fences(path: Path, errors: list[str]) -> None:
    fence_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fence_count += 1
    if fence_count % 2 != 0:
        errors.append(f"{path.relative_to(ROOT)}: unbalanced fenced code block")


def main() -> int:
    errors: list[str] = []
    for path in iter_markdown_files():
        check_links(path, errors)
        check_fences(path, errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Markdown links and code fences OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
