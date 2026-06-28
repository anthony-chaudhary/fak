#!/usr/bin/env python3
"""Tests for the broken-link / dead-reference gate (`check_links.py`).

Covers the markdown-link logic (`_dead_links`), the inline-code reference logic
(`_dead_inline_refs`, added for issue #288 — a doc citing a non-existent file as
authority via inline code like `` `POSITION-….md §5` ``, invisible to the
markdown-link regex), the scrub-private reference logic (`_scrub_private_refs`,
added for issue #258 — a public front-door doc citing a file the scrubber deletes
from every public copy, e.g. `` `CLAUDE.md` ``, invisible because the target
exists in canonical), and the `fak/X` -> repo-root resolution convention. Closes
with LIVE regression assertions that the real front-door set ships no dead or
scrub-private reference.

Run: `python tools/check_links_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_links_test.py -q`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_links as cl  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)


def _doc(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# --- markdown links --------------------------------------------------------

def test_dead_markdown_link_detected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "a.md", "See [the guide](missing.md) for details.")
        assert cl._dead_links(str(root), "a.md") == [("missing.md", "missing.md")]


def test_live_markdown_link_is_clean() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "exists.md", "x")
        _doc(root, "a.md", "See [the guide](exists.md) and [home](#top) and "
                           "[web](https://example.com).")
        assert cl._dead_links(str(root), "a.md") == []


# --- inline-code references (issue #288) -----------------------------------

def test_dead_inline_ref_detected() -> None:
    # The exact #288 shape: an inline-code citation with a trailing section ref.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "a.md", "licensing is in `POSITION-tool-call-2026-06-18.md §5`.")
        dead = cl._dead_inline_refs(str(root), "a.md")
        assert dead == [("POSITION-tool-call-2026-06-18.md §5",
                         "POSITION-tool-call-2026-06-18.md")], dead


def test_inline_ref_resolves_via_fak_root_convention() -> None:
    # `fak/X.md` denotes X.md at the repo root (no fak/ subdir) — not a dead ref.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "ARCHITECTURE.md", "x")
        _doc(root, "a.md", "the guide is `fak/ARCHITECTURE.md` (the model).")
        assert cl._dead_inline_refs(str(root), "a.md") == []


def test_inline_ref_ignores_non_md_code() -> None:
    # Commands, flags, non-.md paths inside inline code must not be flagged.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "a.md", "Run `git commit -s -m msg`, `make ci`, `tools/x.py`.")
        assert cl._dead_inline_refs(str(root), "a.md") == []


def test_inline_ref_present_md_is_clean() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "LICENSING.md", "x")
        _doc(root, "a.md", "see `LICENSING.md` for the strategy.")
        assert cl._dead_inline_refs(str(root), "a.md") == []


# --- scrub-private references (issue #258) ---------------------------------

def test_scrub_private_inline_ref_detected() -> None:
    # The exact #258 shape: a public front-door doc cites CLAUDE.md (scrubbed
    # from every public copy) via inline code. The file EXISTS in canonical, so
    # _dead_inline_refs passes it — only the scrub-private check catches it.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "CLAUDE.md", "x")  # present in canonical -> not a "dead" ref
        _doc(root, "CONTRIBUTING.md", "the deep guide is `CLAUDE.md`.")
        assert cl._dead_inline_refs(str(root), "CONTRIBUTING.md") == []
        bad = cl._scrub_private_refs(str(root), "CONTRIBUTING.md")
        assert bad == [("`CLAUDE.md`", "CLAUDE.md")], bad


def test_scrub_private_markdown_link_detected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "CLAUDE.md", "x")
        _doc(root, "CONTRIBUTING.md", "See [why](CLAUDE.md) for the WSL note.")
        assert cl._dead_links(str(root), "CONTRIBUTING.md") == []
        bad = cl._scrub_private_refs(str(root), "CONTRIBUTING.md")
        assert bad == [("](CLAUDE.md)", "CLAUDE.md")], bad


def test_scrub_private_fak_prefix_detected() -> None:
    # The `fak/X` -> repo-root convention must not hide a scrub-private target.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "CLAUDE.md", "x")
        _doc(root, "CONTRIBUTING.md", "the pointer is `fak/CLAUDE.md`.")
        bad = cl._scrub_private_refs(str(root), "CONTRIBUTING.md")
        assert bad and bad[0][1] == "fak/CLAUDE.md", bad


def test_scrub_private_file_itself_exempt() -> None:
    # A scrub-private file citing another scrub-private file is not a public
    # leak (both die together) — must not be flagged.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "CLAUDE.md", "see PUBLIC-SCRUB-POLICY.md for the map.")
        assert cl._scrub_private_refs(str(root), "CLAUDE.md") == []


# --- live regression guard (the #288 + #258 invariants on the real tree) ---

def test_issue_288_front_door_has_no_dead_reference() -> None:
    """No front-door doc may cite a non-existent file — by markdown link OR by
    inline-code reference. This is the gate that would have caught #288."""
    findings = []
    for f in cl.FRONT_DOOR:
        if not (Path(ROOT) / f).exists():
            continue
        for link, tgt in cl._dead_links(ROOT, f):
            findings.append(f"{f}: ]({link}) -> {tgt}")
        for span, ref in cl._dead_inline_refs(ROOT, f):
            findings.append(f"{f}: `{span}` -> {ref}")
    assert not findings, "dead references in front-door docs:\n" + "\n".join(findings)


def test_issue_258_front_door_has_no_scrub_private_reference() -> None:
    """No public front-door doc may cite a scrub-private file (CLAUDE.md /
    PUBLIC-SCRUB-POLICY.md) — present in canonical but deleted from every public
    copy, so the reference is dead in the export. This is the gate that would
    have caught #258 (CONTRIBUTING.md -> CLAUDE.md)."""
    findings = []
    for f in cl.FRONT_DOOR:
        if not (Path(ROOT) / f).exists():
            continue
        for cite, tgt in cl._scrub_private_refs(ROOT, f):
            findings.append(f"{f}: {cite} -> scrub-private {tgt}")
    assert not findings, "scrub-private references in front-door docs:\n" + "\n".join(findings)


def test_issue_288_no_position_doc_cited_and_authorities_exist() -> None:
    """The specific #288 defect stays fixed: no front-door doc cites a `POSITION*`
    doc, and the licensing authorities those docs DO cite all exist on disk."""
    for f in ("CONTRIBUTING.md", "CLA.md"):
        txt = (Path(ROOT) / f).read_text(encoding="utf-8")
        assert "POSITION-tool-call" not in txt, f"{f} still cites the missing POSITION doc"
    for auth in ("LICENSE", "CLA.md", "LICENSING.md", "GOVERNANCE.md"):
        assert (Path(ROOT) / auth).exists(), f"cited licensing authority {auth} is missing"


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
