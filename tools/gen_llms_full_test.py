#!/usr/bin/env python3
"""Tests for the llms-full.txt generator and its --check drift gate (`gen_llms_full.py`).

Covers the local-.md link collector (`collect_targets`), the build (`build_corpus`),
and the --check mode added for #511 (exit 0 when the committed corpus is in sync,
exit 1 when it is missing or stale). Closes with a LIVE regression: the real repo's
llms-full.txt is in sync with its llms.txt, which is exactly the gate #511 asks for.

Run: `python tools/gen_llms_full_test.py`  (exit 0 = all pass),
or `python -m pytest tools/gen_llms_full_test.py -q`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_llms_full as g  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


# --- link collection -------------------------------------------------------

def test_collect_targets_skips_external_and_anchors() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        orig = g.ROOT
        g.ROOT = td
        try:
            Path(td, "a.md").write_text("x", encoding="utf-8")
            txt = ("[a](a.md) [web](https://x/y.md) [anchor](#top) [mail](mailto:x@y) "
                   "[dup](a.md) [pic](pic.png)")
            targets = g.collect_targets(txt)
            assert [t for _, t in targets] == ["a.md"]
        finally:
            g.ROOT = orig


def test_collect_targets_dedups_and_orders() -> None:
    # Monkeypatch ROOT to a temp dir so isfile resolves local fake docs.
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        orig = g.ROOT
        g.ROOT = td
        try:
            for name in ("a.md", "b.md", "c.md"):
                Path(td, name).write_text("x", encoding="utf-8")
            txt = "[c](c.md) [a](a.md) [c-again](c.md) [b](b.md) [missing](z.md)"
            targets = g.collect_targets(txt)
            assert [t for _, t in targets] == ["c.md", "a.md", "b.md"]  # missing skipped, dedup'd
        finally:
            g.ROOT = orig


def test_absolutize_local_links_resolves_from_source_doc() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        orig = g.ROOT
        g.ROOT = td
        try:
            docs = Path(td, "docs", "guide")
            docs.mkdir(parents=True)
            Path(td, "README.md").write_text("root", encoding="utf-8")
            (docs / "next.md").write_text("next", encoding="utf-8")
            (docs / "assets").mkdir()
            text = ("[next](next.md#install) [root](../../README.md) "
                    "[asset](assets) [web](https://example.com/x) [same](#local)")
            out = g.absolutize_local_links(text, "docs/guide/start.md")
            assert "[next](https://github.com/anthony-chaudhary/fak/blob/main/docs/guide/next.md#install)" in out
            assert "[root](https://github.com/anthony-chaudhary/fak/blob/main/README.md)" in out
            assert "[asset](https://github.com/anthony-chaudhary/fak/tree/main/docs/guide/assets)" in out
            assert "[web](https://example.com/x)" in out
            assert "[same](#local)" in out
        finally:
            g.ROOT = orig


# --- build_corpus ----------------------------------------------------------

def test_build_corpus_structure_and_version() -> None:
    text, targets = g.build_corpus()
    assert text.startswith("# fak — the agent kernel: full documentation corpus")
    assert "## Index (start with the curated map)" in text
    # Every inlined doc carries a Source marker pointing at a tracked .md.
    import re
    sources = re.findall(r"^> Source: `([^`]+)`", text, re.M)
    # One Source marker per inlined target; the exact count tracks llms.txt as
    # docs are added, so assert the alignment + a sane floor, not a frozen number.
    assert len(sources) == len(targets) >= 50, (len(sources), len(targets))
    assert text.endswith("\n") and "\r" not in text  # LF, single trailing newline


# --- --check drift gate (#511) --------------------------------------------

def test_check_passes_on_working_tree() -> None:
    """LIVE: the current llms-full.txt is in sync with the current generator,
    llms.txt, and linked docs -- the property the #511 CI gate enforces."""
    r = subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--check", "--root", str(ROOT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, "llms-full.txt is stale vs the working tree:\n" + r.stdout


def test_check_fails_when_stale() -> None:
    """A stale llms-full.txt (missing, or tampered) must trip --check. Uses an
    isolated temp tree via --root so the real repo corpus is never touched."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, "VERSION").write_text("9.9.9-test\n", encoding="utf-8")
        Path(td, "llms.txt").write_text("- [a](a.md)\n", encoding="utf-8")
        Path(td, "a.md").write_text("body\n", encoding="utf-8")
        # missing llms-full.txt -> exit 1
        r = subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--check", "--root", td],
                           capture_output=True, text=True)
        assert r.returncode == 1, "missing llms-full.txt did not trip --check"
        # generate it, then tamper -> exit 1
        subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--root", td],
                       capture_output=True, text=True, check=True)
        Path(td, "llms-full.txt").write_text(
            Path(td, "llms-full.txt").read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8")
        r = subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--check", "--root", td],
                           capture_output=True, text=True)
        assert r.returncode == 1, "stale llms-full.txt did not trip --check"
        # regenerate -> exit 0
        subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--root", td],
                       capture_output=True, text=True, check=True)
        r = subprocess.run([PY, str(ROOT / "tools/gen_llms_full.py"), "--check", "--root", td],
                           capture_output=True, text=True)
        assert r.returncode == 0, "fresh llms-full.txt did not pass --check"


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
