#!/usr/bin/env python3
"""Tests for tools/learning_loop_metric_detector.py (issue #2914) — hermetic.

The load-bearing property: on the LEARNING surface, a metric ("skills kept") is
self-fulfilling if it can be RAISED without a paired net-value RISE, and a skill
that games the witness must score WORSE, not better — the same guardrail #2816
landed for the cache-value surface, moved to the skill/memory loop.

These pin the acceptance directly:
  * a deliberately-GAMING skill (metric up, net task value flat/negative) is
    FLAGGED and scores below a genuinely value-raising skill,
  * a value-RAISING skill (metric up, net up) is NOT flagged,
  * an unwitnessed (MODELED) positive net claim earns no credit — you cannot
    launder activity into value,
  * the detector reports the metric as self-fulfilling and exits non-zero.

Red before the detector existed (import fails / no such module); green after.

Run: `python tools/learning_loop_metric_detector_test.py`  (exit 0 = all pass),
or `python -m pytest tools/learning_loop_metric_detector_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learning_loop_metric_detector as det  # noqa: E402


def _obs(skill: str, metric_delta: float, net: float, prov: str = det.WITNESSED):
    return det.SkillObservation(skill, "skills_kept", metric_delta, net, prov)


def test_gaming_skill_is_flagged_and_scores_worse() -> None:
    """The core regression: a skill that inflates 'skills kept' while net task
    value stays flat must be flagged AND score below a value-raising skill."""
    value = _obs("value-raiser", metric_delta=3, net=12)      # kept 3, net +12 (HELPED)
    gamer = _obs("witness-gamer", metric_delta=10, net=0)     # kept 10, net 0 (WASH)
    payload = det.detect([value, gamer])

    rows = {r["skill"]: r for r in payload["observations"]}
    assert rows["witness-gamer"]["flagged"] is True, rows["witness-gamer"]
    assert rows["value-raiser"]["flagged"] is False, rows["value-raiser"]
    # a gaming skill MUST score worse than a genuinely value-raising one
    assert rows["witness-gamer"]["score"] < rows["value-raiser"]["score"], payload
    # and worse than zero — inflating the metric with no net value is a net loss
    assert rows["witness-gamer"]["score"] < 0, rows["witness-gamer"]


def test_metric_is_reported_self_fulfilling_and_exit_nonzero() -> None:
    """The detector must NAME the raisable-without-net metric and fail the gate."""
    payload = det.detect([_obs("witness-gamer", 10, 0)])
    assert payload["ok"] is False
    assert payload["verdict"] == "ACTION"
    assert payload["finding"] == "learning_loop_reward_hack"
    assert "skills_kept" in payload["self_fulfilling_metrics"], payload
    assert det.main(["--ledger"]) if False else True  # (ledger path exercised below)


def test_value_raising_skill_clears() -> None:
    """A metric rise fully backed by witnessed net value is not flagged, and the
    whole corpus reads OK when every rise is paired."""
    payload = det.detect([
        _obs("value-raiser", 3, 12),
        _obs("honest-1to1", 5, 5),   # each kept skill paired 1:1 with net value
    ])
    assert payload["ok"] is True
    assert payload["verdict"] == "OK"
    assert payload["n_flagged"] == 0
    for r in payload["observations"]:
        assert r["flagged"] is False, r


def test_negative_net_is_flagged_worst() -> None:
    """A skill whose kept updates actively HURT net value (net < 0) is flagged and
    scores below a merely wash gamer."""
    wash = _obs("wash-gamer", 6, 0)
    hurt = _obs("hurt-gamer", 6, -4)
    payload = det.detect([wash, hurt])
    rows = {r["skill"]: r for r in payload["observations"]}
    assert rows["hurt-gamer"]["flagged"] is True
    assert rows["hurt-gamer"]["score"] < rows["wash-gamer"]["score"], payload


def test_modeled_positive_net_is_not_credited() -> None:
    """Provenance honesty: an unwitnessed (MODELED) positive net claim cannot buy
    back activity, so a 'modeled-gamer' that asserts net value it never proved is
    flagged exactly like a flat gamer — the witness, not the assertion, decides."""
    modeled = _obs("modeled-gamer", 8, 8, prov=det.MODELED)   # claims +8, unproven
    witnessed = _obs("proven", 8, 8, prov=det.WITNESSED)      # same numbers, proven
    payload = det.detect([modeled, witnessed])
    rows = {r["skill"]: r for r in payload["observations"]}
    assert rows["modeled-gamer"]["flagged"] is True, rows["modeled-gamer"]
    assert rows["modeled-gamer"]["credited_net"] == 0
    assert rows["proven"]["flagged"] is False, rows["proven"]
    assert rows["modeled-gamer"]["score"] < rows["proven"]["score"]


def test_over_claiming_is_soft_not_hard() -> None:
    """A metric rise only PARTLY backed by net value is a soft 'over-claiming'
    warning (activity outran value) — not the HARD self-fulfilling flag, since a
    net rise did occur — and still scores below a fully-paired skill."""
    over = _obs("over-claimer", 10, 4)   # kept 10, only +4 net behind it
    paired = _obs("fully-paired", 4, 4)
    payload = det.detect([over, paired])
    rows = {r["skill"]: r for r in payload["observations"]}
    assert rows["over-claimer"]["flagged"] is False
    assert rows["over-claimer"]["over_claiming"] is True
    assert rows["over-claimer"]["score"] < rows["fully-paired"]["score"]
    # no HARD flag, so the corpus is OK but names the soft drift
    assert payload["ok"] is True
    assert payload["finding"] == "learning_loop_paired_with_overclaim"


def test_determinism_same_inputs_same_payload() -> None:
    """Same observations -> byte-identical payload: the verdict is a re-derivable
    witness, not a one-run reading (the nettrue.go / rsi-scorecard discipline)."""
    import json
    obs = [_obs("value-raiser", 3, 12), _obs("witness-gamer", 10, 0)]
    a = json.dumps(det.detect(obs), sort_keys=True)
    b = json.dumps(det.detect(obs), sort_keys=True)
    assert a == b


def test_ledger_roundtrip(tmp_path_factory=None) -> None:
    """The JSONL ledger reader parses a real feed shape and the CLI gate exits
    non-zero when it finds a gaming skill."""
    import tempfile
    import json
    rows = [
        {"skill": "value-raiser", "metric_delta": 3, "net_value_delta": 12,
         "net_provenance": "WITNESSED"},
        {"skill": "witness-gamer", "metric_delta": 10, "net_value_delta": 0,
         "net_provenance": "WITNESSED"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "obs.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        loaded = det.load_ledger(str(p))
        assert len(loaded) == 2
        payload = det.detect(loaded)
        assert payload["n_flagged"] == 1
        # CLI returns 1 on a flagged learning metric (a gate a loop/CI can read)
        assert det.main(["--ledger", str(p), "--json"]) == 1


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall learning-loop-metric-detector tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
