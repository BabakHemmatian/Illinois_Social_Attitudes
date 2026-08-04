#!/usr/bin/env python3
"""Sync the Data Use Agreement into a HuggingFace dataset card.

The DUA is maintained in this repo and served live by the ISAAC website at /dua,
so the website and the Python package always show the current text. A HuggingFace
dataset card is a static Markdown file, so without something like this it drifts
the first time Legal edits the agreement.

This script keeps two parts of a card current:

  * ``extra_gated_prompt`` in the YAML frontmatter — the text a requester
    actually reads before ticking the consent box. This is the one that matters
    for consent, so the full agreement goes here rather than a link to it.
  * optionally, a block in the body between ``<!-- DUA:BEGIN -->`` and
    ``<!-- DUA:END -->``, if those markers are present.

Everything else in the card is preserved byte-for-byte (the frontmatter is
edited surgically rather than round-tripped through a YAML parser, which would
reorder keys and reformat the config blocks).

Usage
-----
    python code/sync_hf_dua.py path/to/dataset_card.md
    python code/sync_hf_dua.py path/to/dataset_card.md --check   # CI: fail if stale

``--check`` writes nothing and exits 1 when the card is out of date, so it can
gate the publish step in the runbook.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DUA_ENDPOINT = "https://isaac.psychology.illinois.edu/dua"
BEGIN = "<!-- DUA:BEGIN -->"
END = "<!-- DUA:END -->"
GATED_KEY = "extra_gated_prompt"
INDENT = "  "


def fetch_dua(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "isaac-sync-hf-dua"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if not data.get("markdown") or not data.get("sha256"):
        raise SystemExit(f"{url} returned no agreement text")
    return data


def stamp(dua: dict) -> str:
    return (
        f"Version {dua.get('version') or 'unknown'} (SHA-256 {dua['sha256'][:16]}…). "
        f"The authoritative current text is served at {DUA_ENDPOINT}."
    )


def split_frontmatter(card: str) -> tuple[list[str], list[str]]:
    """Return (frontmatter_lines, rest_lines). Frontmatter includes its '---' fences."""
    lines = card.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SystemExit("Card does not start with a YAML frontmatter block ('---').")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[: i + 1], lines[i + 1:]
    raise SystemExit("Unterminated YAML frontmatter (no closing '---').")


def set_gated_prompt(fm: list[str], body_text: str) -> list[str]:
    """Replace the extra_gated_prompt key's value with a literal block scalar."""
    start = next((i for i, ln in enumerate(fm) if ln.startswith(GATED_KEY + ":")), None)
    if start is None:
        raise SystemExit(
            f"Card frontmatter has no `{GATED_KEY}:` key. Add one (it is what "
            "enables HuggingFace gating) and re-run."
        )
    # Consume the existing value: any following indented or blank lines.
    end = start + 1
    while end < len(fm) and (fm[end].startswith((" ", "\t")) or not fm[end].strip()):
        end += 1
    # Trailing blanks belong to whatever follows, not to this value.
    while end - 1 > start and not fm[end - 1].strip():
        end -= 1

    block = [f"{GATED_KEY}: |"]
    for ln in body_text.splitlines():
        block.append(f"{INDENT}{ln}".rstrip())
    return fm[:start] + block + fm[end:]


def set_body_block(rest: list[str], body_text: str) -> list[str]:
    """Update the marker block in the card body, if the markers are present."""
    text = "\n".join(rest)
    s, e = text.find(BEGIN), text.find(END)
    if s == -1 or e == -1:
        return rest  # markers are optional
    if e < s:
        raise SystemExit(f"{END} appears before {BEGIN} in the card body.")
    block = f"{BEGIN}\n\n{body_text}\n\n{END}"
    return (text[:s] + block + text[e + len(END):]).splitlines()


def render(card: str, dua: dict) -> str:
    body_text = f"{dua['markdown'].strip()}\n\n{stamp(dua)}"
    fm, rest = split_frontmatter(card)
    fm = set_gated_prompt(fm, body_text)
    rest = set_body_block(rest, body_text)
    return "\n".join(fm + rest) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("card", type=Path, help="path to the HuggingFace dataset card")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the card is out of date; write nothing")
    ap.add_argument("--url", default=DUA_ENDPOINT,
                    help=f"DUA endpoint (default: {DUA_ENDPOINT})")
    args = ap.parse_args(argv)

    if not args.card.is_file():
        raise SystemExit(f"No such file: {args.card}")

    dua = fetch_dua(args.url)
    current = args.card.read_text(encoding="utf-8")
    updated = render(current, dua)

    version = dua.get("version")
    short = dua["sha256"][:16]
    if updated == current:
        print(f"Card is up to date (version {version}, sha256 {short}…).")
        return 0
    if args.check:
        print(f"STALE: {args.card} does not match the current Data Use Agreement "
              f"(version {version}, sha256 {short}…).", file=sys.stderr)
        return 1
    args.card.write_text(updated, encoding="utf-8")
    print(f"Updated {args.card} to version {version} (sha256 {short}…).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
