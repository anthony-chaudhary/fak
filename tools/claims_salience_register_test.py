#!/usr/bin/env python3
"""Tests for claims-salience-register — the first wired consumer of `dos salience`.

Two tiers, mirroring the CI contract (the dos-touching tool tests are HERMETIC):

  (1) HERMETIC — no `dos` dependency. The pure ledger plumbing: CLAIMS.md parsing
      (legend lines excluded, tags + sections extracted, labels unique + stable),
      tag counts, the slug, and the invariant verifier driven by FABRICATED verdict
      data — so the no-loss / cross-check / recoverability gates are exercised without
      importing the kernel. These run in hermetic CI.

  (2) DOS-GATED live smoke — imports the real `dos.salience` and folds the REAL
      CLAIMS.md, asserting zero violations and the no-loss invariant. SKIPPED (a
      printed notice, counted as pass) where the dos kernel is not importable, so the
      file passes in hermetic CI and proves the real wiring where dos is present.

Run: `python tools/claims_salience_register_test.py`  (exit 0 = all pass),
or `python -m pytest tools/claims_salience_register_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims_salience_register as m  # noqa: E402

# Is the real dos kernel importable here? (Gates the tier-2 live smoke.)
_SAL, _ = m.load_salience()


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #

SAMPLE = """# CLAIMS.md — the fak honesty ledger

Every capability claim carries **exactly one** tag:

- `[SHIPPED]` — real code on the critical path.
- `[SIMULATED]` — modeled with labeled stand-in data.
- `[STUB]` — plumbing present, behavior deferred.

## The product

- [SHIPPED] One statically-linked Go binary runs the loop. Witness: go build.
- [SHIPPED] Process-level fusion collapses the decide path. Witness: TestNoOsExec.

## Tool vDSO

- [SIMULATED] Real-world vDSO hit-rate is ~0.7%, far below a useful threshold.
- [STUB] Decode-time logit-mask requires owning the decode loop; not in v0.1.

Some prose mentioning a [STUB] inline that must NOT be parsed as a claim line.
"""


def _claims():
    return m.parse_claims(SAMPLE)


class _Part:
    """Minimal stand-in for dos's SaliencePartition (only `.total` is read)."""

    def __init__(self, total: int):
        self.total = total


def _result(claims, *, total=None, by_state=None, mutate_rows=None):
    """A verdict-result dict consistent with `claims` (the shape classify_corpus
    returns), so verify_invariants can be exercised with NO dos import. Perturb via
    total / by_state / mutate_rows to trigger each violation path."""
    sh = sum(1 for c in claims if c["tag"] == "SHIPPED")
    sim = sum(1 for c in claims if c["tag"] == "SIMULATED")
    st = sum(1 for c in claims if c["tag"] == "STUB")
    if by_state is None:
        by_state = {"LIVE": sh, "PARKED": sim + st, "INDETERMINATE": 0}
    if total is None:
        total = len(claims)
    rows = []
    for c in claims:
        if c["tag"] == "SHIPPED":
            rows.append({"label": c["label"], "tag": "SHIPPED", "section": c["section"],
                         "text": c["text"], "state": "LIVE", "reason_class": "",
                         "reactivation": "", "fak_reactivation": "", "reason": "ok"})
        else:
            rows.append({"label": c["label"], "tag": c["tag"], "section": c["section"],
                         "text": c["text"], "state": "PARKED", "reason_class": c["tag"],
                         "reactivation": "kernel recovery line",
                         "fak_reactivation": m.FAK_REACTIVATION[c["tag"]], "reason": "parked"})
    if mutate_rows:
        mutate_rows(rows)
    return {"partition": _Part(total), "by_state": by_state, "rows": rows}


# --------------------------------------------------------------------------- #
# (1) HERMETIC — parsing + counts + slug.
# --------------------------------------------------------------------------- #


def test_parse_excludes_legend_and_inline_mentions():
    claims = _claims()
    # 2 SHIPPED + 1 SIMULATED + 1 STUB = 4 real claim lines; the three legend lines
    # (`- `[TAG]``, backtick) and the inline `[STUB]` prose line are NOT claims.
    assert len(claims) == 4, [c["tag"] for c in claims]
    assert [c["tag"] for c in claims] == ["SHIPPED", "SHIPPED", "SIMULATED", "STUB"]
    # The inline-mention prose line did not sneak in.
    assert all("Some prose mentioning" not in c["text"] for c in claims)


def test_parse_tracks_sections():
    claims = _claims()
    assert claims[0]["section"] == "The product"
    assert claims[2]["section"] == "Tool vDSO"
    assert claims[3]["section"] == "Tool vDSO"


def test_parse_labels_unique_and_deterministic():
    # Two claims with identical lead text must get distinct, stable labels.
    text = ("## S\n- [STUB] same lead text here for both.\n"
            "- [STUB] same lead text here for both.\n")
    a = m.parse_claims(text)
    b = m.parse_claims(text)
    labels = [c["label"] for c in a]
    assert len(set(labels)) == 2, labels          # unique
    assert labels[1].endswith("-2"), labels        # deterministic collision suffix
    assert [c["label"] for c in b] == labels       # stable across runs


def test_tag_counts():
    counts = m.tag_counts(_claims())
    assert counts == {"SHIPPED": 2, "SIMULATED": 1, "STUB": 1}


def test_slug_is_kebab_strips_markdown_and_stable():
    s = m._slug("`Decode-time` logit-mask (grammar) requires the loop!")
    assert s == m._slug("`Decode-time` logit-mask (grammar) requires the loop!")
    assert "`" not in s and "(" not in s and " " not in s
    assert s.startswith("decode-time-logit-mask")


# --------------------------------------------------------------------------- #
# (1) HERMETIC — the invariant verifier (the part that makes "never lose" a gate).
# --------------------------------------------------------------------------- #


def test_verify_invariants_clean():
    claims = _claims()
    assert m.verify_invariants(claims, _result(claims)) == []


def test_verify_detects_silent_drop():
    claims = _claims()
    # partition.total < #claims ⇒ something was dropped ledger→fold: the load-bearing catch.
    vios = m.verify_invariants(claims, _result(claims, total=len(claims) - 1))
    assert any("no-loss" in v.lower() for v in vios), vios


def test_verify_detects_count_drift_vs_ledger():
    claims = _claims()
    bad = {"LIVE": 1, "PARKED": 2, "INDETERMINATE": 0}   # live should be 2 (#SHIPPED)
    vios = m.verify_invariants(claims, _result(claims, by_state=bad))
    assert any("live=" in v for v in vios), vios


def test_verify_detects_indeterminate():
    claims = _claims()
    bad = {"LIVE": 2, "PARKED": 1, "INDETERMINATE": 1}   # a tagged claim never abstains
    vios = m.verify_invariants(claims, _result(claims, by_state=bad))
    assert any("indeterminate=" in v for v in vios), vios


def test_verify_detects_missing_reactivation():
    claims = _claims()

    def strip(rows):
        for r in rows:
            if r["state"] == "PARKED":
                r["reactivation"] = ""   # a parked thing with no path back == a slow drop
                break

    vios = m.verify_invariants(claims, _result(claims, mutate_rows=strip))
    assert any("recoverable" in v.lower() for v in vios), vios


def test_verify_detects_reason_class_drift():
    claims = _claims()

    def flip(rows):
        for r in rows:
            if r["reason_class"] == "STUB":
                r["reason_class"] = "SIMULATED"   # now SIMULATED count != #[SIMULATED]
                break

    vios = m.verify_invariants(claims, _result(claims, mutate_rows=flip))
    assert any("reason_class" in v for v in vios), vios


# --------------------------------------------------------------------------- #
# (1) HERMETIC — payload + renderers don't crash and carry the right shape.
# --------------------------------------------------------------------------- #


def test_build_payload_skipped_shape():
    claims = _claims()
    p = m.build_payload("/ws", "", claims, None, [], skipped=True)
    assert p["skipped"] is True and p["dos_available"] is False
    assert p["claims_total"] == 4 and p["tag_counts"]["SHIPPED"] == 2
    assert "reason" in p


def test_renderers_do_not_crash():
    claims = _claims()
    res = _result(claims)
    payload = m.build_payload("/ws", "x/salience.py", claims, res, [], skipped=False)
    out = m.render(payload, res)
    assert "PARKED register" in out and "no-loss" in out
    md = m.render_markdown(payload, res)
    assert "PARKED ≠ dropped" in md and "## PARKED · `SIMULATED`" in md
    skipped = m.build_payload("/ws", "", claims, None, [], skipped=True)
    assert "SKIPPED" in m.render(skipped, None)


# --------------------------------------------------------------------------- #
# (2) DOS-GATED — the live smoke over the REAL CLAIMS.md (skips without dos).
# --------------------------------------------------------------------------- #


def test_live_smoke_real_ledger_zero_violations():
    if _SAL is None:
        print("  [skip] dos kernel not importable — live salience smoke skipped (advisory)")
        return
    root = Path(__file__).resolve().parent.parent
    payload, result, claims = m.collect(root)
    assert not payload.get("skipped"), "dos present but collect reported skipped"
    # The real ledger folds with the no-loss invariant intact and no honesty drift.
    assert payload["no_loss_ok"] is True
    assert payload["by_state"]["INDETERMINATE"] == 0
    by = payload["by_state"]
    assert by["LIVE"] + by["PARKED"] == payload["claims_total"]
    assert payload["violations"] == [], payload["violations"]
    assert payload["ok"] is True
    # Every parked claim is recoverable: a non-empty reason class + reactivation line.
    for r in result["rows"]:
        if r["state"] == "PARKED":
            assert r["reason_class"] in m.PARK_TAGS
            assert r["reactivation"] and r["fak_reactivation"]


def test_live_smoke_uses_the_real_kernel_partition():
    if _SAL is None:
        print("  [skip] dos kernel not importable — partition-identity smoke skipped (advisory)")
        return
    # The partition's no-loss fold is what we route on: total == input, nothing dropped.
    claims = _claims()
    res = m.classify_corpus(claims, _SAL)
    assert res["partition"].total == len(claims)
    assert res["by_state"]["LIVE"] == 2
    assert res["by_state"]["PARKED"] == 2
    assert m.verify_invariants(claims, res) == []


# --------------------------------------------------------------------------- #
# Runner (mirrors the sibling tool tests: collect test_*, report, exit code).
# --------------------------------------------------------------------------- #


def main() -> int:
    failures: list[str] = []

    def check(name, fn):
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    gated = "" if _SAL is not None else " (dos-gated live smoke skipped: kernel absent)"
    print(f"ok ({len(tests)} tests){gated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
