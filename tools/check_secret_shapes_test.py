#!/usr/bin/env python3
"""Tests for the secret-shape leak gate (tools/check_secret_shapes.py).

Drives the pure scanner `_scan_text` — the part that decides whether a blob
carries an operator-leak SHAPE (a real home path or an internal lab host) the
literal needle list would miss — across every class it claims to catch and every
placeholder it claims to exempt. Read-only; no git, no disk.

NOTE: the shape fixtures are ASSEMBLED FROM FRAGMENTS at runtime (see the
builders below) so this test file does not itself contain a contiguous leak-shape
literal — otherwise the very gate under test would flag it under `--audit-tree`.
The runtime strings ARE the full shapes, so `_scan_text` sees exactly what it
would in real content; only the on-disk source is fragmented.

Run: `python tools/check_secret_shapes_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_secret_shapes_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_secret_shapes as css  # noqa: E402

_BS = chr(92)  # backslash, kept out of literal form


def _win(user: str, sep: str = _BS) -> str:
    return f"C:{sep}Users{sep}{user}"


def _mac(user: str) -> str:
    return f"/Users/{user}"


def _msl(name: str) -> str:
    return "msl" + "-" + name


def _lab(name: str) -> str:
    return name + "." + "lab"


def _codename(suffix: str) -> str:
    # assembled so this test file carries no contiguous codename literal.
    return "rtp" + "-" + suffix


def _shapes(text: str) -> list[tuple[str, str]]:
    return list(css._scan_text(text))


# --- operator home paths ---------------------------------------------------

def test_windows_backslash_operator_path_is_caught() -> None:
    p = _win("bob")
    assert ("operator-path", p) in _shapes(f"see {p}{_BS}work for the log")


def test_windows_forwardslash_operator_path_is_caught() -> None:
    # forward-slash form must not slip past (the exact motivating miss).
    p = _win("bob", "/")
    assert {s for s, _ in _shapes(f"path {p}/work")} == {"operator-path"}


def test_mac_operator_path_is_caught() -> None:
    p = _mac("bob")
    assert ("operator-path", p) in _shapes(f"cache at {p}/Library")


def test_mac_operator_path_is_caught_any_case() -> None:
    # A lowercase `/users/<operator>` literal is the SAME home path. The motivating
    # miss: one such literal inside a _test.go fixture (which the whole-tree export
    # sweep exempts) was published while this gate reported the tree clean.
    for form in (_mac("bob").lower(), _mac("bob").upper()):
        assert ("operator-path", form) in _shapes(f"cache at {form}/Library"), form


def test_mac_path_after_a_url_or_relative_prefix_is_not_a_home_path() -> None:
    # The path-segment boundary: a URL path and a relative path are not home paths.
    # Without it, widening the case would newly flag third-party URLs and prose.
    assert _shapes("source: https://example.com/Users/alice/fak") == []
    assert _shapes("source: ../Users/alice/fak") == []
    assert _shapes("source: docs/Users/alice/fak") == []


def test_mac_path_after_a_separator_is_still_a_home_path() -> None:
    # ...but a real home path is preceded by whitespace, a quote, or a line start.
    p = _mac("bob")
    for prefix in ("", " ", '"', "(", "=", ":", "\t"):
        assert ("operator-path", p) in _shapes(f"{prefix}{p}/Library"), prefix


def test_placeholder_users_are_exempt() -> None:
    # the redaction placeholders must NOT be flagged or the gate cries wolf.
    for blob in (_win("USER"), _mac("runner"), _win("Public"), _mac("you")):
        assert _shapes(blob) == [], f"placeholder wrongly flagged: {blob}"


def test_placeholder_match_is_case_insensitive() -> None:
    # group is "User" -> lower "user" -> in PLACEHOLDER_USERS -> exempt.
    assert _shapes(_win("User")) == []


# --- internal lab hosts ----------------------------------------------------

def test_msl_host_is_caught_any_case() -> None:
    assert ("internal-host", _msl("gpu01")) in _shapes(f"ssh {_msl('gpu01')}")
    assert any(s == "internal-host" for s, _ in _shapes(_msl("box").upper()))


def test_lab_host_is_caught() -> None:
    host = _lab("testbox")
    assert ("internal-host", host) in _shapes(f"host {host} down")


def test_example_lab_placeholder_is_exempt() -> None:
    # the documented placeholder host must pass clean.
    assert _shapes(f"connect to {_lab('example')}") == []


def test_short_first_label_is_not_a_lab_host() -> None:
    # A 1-2 character DNS label is not a hostname convention, and it is exactly the shape
    # a Go field selector takes: the tree carries `tt.lab` (a struct field in
    # internal/devindex), which a 1+ label rule reports as an internal host on every audit.
    assert _shapes("problems := validate(tt.portable, tt.maintainer, tt.lab)") == []
    assert _shapes(f"host {_lab('ab')} down") == []
    # ...but every real lab host in the tree has a 3+ character first label. The names
    # here are the GENERIC machine classes only; a site-specific alias would itself be
    # the leak this gate exists to catch, so it must not be spelled in a fixture.
    for name in ("gpu", "gpu1", "gpu-server", "node7", "worker-12"):
        assert ("internal-host", _lab(name)) in _shapes(f"host {_lab(name)} down"), name


# --- site/cluster codenames ------------------------------------------------

def test_codename_forms_are_caught() -> None:
    for suffix in ("test-h200", "h200", "test"):
        c = _codename(suffix)
        assert ("site-codename", c) in _shapes(f"measured on {c} yesterday"), c


def test_codename_is_caught_any_case() -> None:
    assert any(s == "site-codename" for s, _ in _shapes(_codename("test-h200").upper()))


def test_public_rtp_llm_repo_is_exempt() -> None:
    # `alibaba/rtp-llm` is a PUBLIC third-party repo cited in docs/research; a bare
    # `rtp-` prefix rule would cry wolf on it. The guard is load-bearing.
    assert _shapes("see https://github.com/alibaba/rtp-llm/blob/main/x.patch") == []
    # the guard is a post-match predicate, not a lookahead (RE2 has none), so it must
    # behave like `(?!llm\b)` at every edge: case, end-of-text, and a following word char.
    assert _shapes("see " + _codename("LLM") + " upstream") == []
    assert _shapes("see " + _codename("llm")) == []
    assert _shapes("see " + _codename("llm-x") + " upstream") == []
    # ...but `llm` followed by a word character is NOT the public repo -> still a leak.
    assert ("site-codename", _codename("llmx")) in _shapes("see " + _codename("llmx"))
    # `_` and a non-ASCII byte both stop the regex body, so the MATCH is the bare
    # `rtp-llm`; the guard then sees the following word char and flags it. A non-ASCII
    # byte counts as a word character, so the guard FAILS SAFE rather than exempting.
    assert ("site-codename", _codename("llm")) in _shapes("see " + _codename("llm_1"))
    assert ("site-codename", _codename("llm")) in _shapes("see " + _codename("llmé"))


def test_bare_rtp_is_not_a_codename() -> None:
    # ERCOT "RTP" and a scraped `--served-model-name rtp` are public; only the
    # QUALIFIED lab forms are the leak.
    assert _shapes("the RTP settlement interval and --served-model-name rtp") == []


# --- clean / negative ------------------------------------------------------

def test_clean_text_yields_nothing() -> None:
    assert _shapes("a perfectly ordinary sentence with no secrets") == []


def test_relative_users_path_without_drive_is_not_an_operator_path() -> None:
    # a bare "Users<sep>alice" with no drive letter and no leading slash is not
    # the shape (no C: prefix, no leading /).
    assert _shapes(f"the Users{_BS}alice folder") == []


# --- the rule's self-reference allowlist ------------------------------------

def test_self_ref_includes_this_gate_and_policy() -> None:
    # the files that DEFINE the rule must be self-exempt or the tree audit
    # would forever flag its own regexes.
    assert "tools/check_secret_shapes.py" in css.SELF_REF
    assert "PUBLIC-SCRUB-POLICY.md" in css.SELF_REF


def test_generated_corpus_is_exempt_from_the_shape_heuristic() -> None:
    # docs/research/inventory/*.jsonl are verbatim scrapes of UPSTREAM projects, so a home
    # path inside one is a THIRD-PARTY developer's, not the operator's. The literal-needle
    # leak gate still scans them; only the shape heuristic stands down.
    assert css._is_generated_corpus("docs/research/inventory/llm-d-llm-d-corpus.jsonl")
    assert css._is_generated_corpus("docs\\research\\inventory\\x.jsonl")
    # a sibling directory must NOT match (the glob is spelled out, not a leading `*`).
    assert not css._is_generated_corpus("docs/research/inventory-notes/x.jsonl")
    assert not css._is_generated_corpus("docs/research/inventory/README.md")


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
