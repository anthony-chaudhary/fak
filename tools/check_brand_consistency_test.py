"""Tests for check_brand_consistency.py — the primary-descriptor drift guard.

Proves the guard (a) FLAGS a retired descriptor used as fak's primary noun,
(b) does NOT flag the legitimate secondary uses (keyword lists, analogies, the
named video asset), and (c) the live tracked tree is currently clean.
"""
import check_brand_consistency as bc


def _flagged(line: str) -> bool:
    """Replicates the per-line decision the audit makes."""
    return bool(bc.PRIMARY_RE.search(line)) and not bool(bc.ALLOW_MARKERS.search(line))


# Primary-descriptor DRIFT — fak declared to BE a retired descriptor. Must flag.
DRIFT = [
    "`fak` is an agent tool firewall: a single Go binary that sits between an agent and its tools",
    "fak — agent tool firewall",
    'fmt.Fprintf(os.Stderr, "fak - Agent Tool Firewall (Fused Agent Kernel, v%s)")',
    "`fak` is the tool-call policy gateway for your fleet",
    "fak is a tool call policy gateway",
]

# Legitimate secondary uses. Must NOT flag.
ALLOWED = [
    "`fak` is an **agent kernel** (also described as an *agent tool firewall*): an in-process gate",
    "<sub>Topics: agent kernel · agent tool firewall · AI agent security · prompt injection</sub>",
    "  - agent tool firewall",
    '"alternateName": ["the agent kernel", "agent tool firewall"],',
    'aria-label="fak — the agent tool firewall: a ~44 second explainer reveal">',
    "`fak` is the Fused Agent Kernel: a single Go binary that sits between an agent and its tools",
    "It is also described as an agent tool firewall.",
]


def test_drift_is_flagged():
    for s in DRIFT:
        assert _flagged(s), f"should flag primary-descriptor drift: {s!r}"


def test_allowed_forms_not_flagged():
    for s in ALLOWED:
        assert not _flagged(s), f"should NOT flag legitimate secondary use: {s!r}"


def test_live_tree_is_clean():
    debt = bc.audit()
    assert debt == [], f"primary-descriptor drift on the tracked tree: {debt}"
