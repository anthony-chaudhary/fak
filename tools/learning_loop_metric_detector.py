#!/usr/bin/env python3
"""Self-fulfilling-metric detector for the skill/memory LEARNING loop (#2914).

The self-fulfilling / vanity-metric *concept* (#2818,
``docs/notes/CACHE-HIT-VANITY-METRIC-SELF-FULFILLING-2026-07-01.md``) names the
trap: a number that rises for the *same reason* the pathology it appears to
reward grows. Optimize it and you optimize the pathology. The anti-reward-hack
guardrail already landed for the CACHE-VALUE surface (#2816): a loop optimizing
``fak_share_gross`` MUST optimize ``fak_share_net`` — gross paired with net, or a
burst that games the gross scores worse.

This is the SAME guardrail moved to the LEARNING surface. Hermes' learning loop
rewards *activity* ("most sessions produce at least one skill update") with no
counter-metric — the exact reward-hack shape. If fak builds a witnessed learning
loop, its own activity metric ("skills kept", skill-updates-per-session)
immediately becomes a target an RSI process can game: keep skills that inflate
the count without raising real task value. This detector keeps that loop honest
by construction:

  A learning-loop metric ("skills kept") MUST be paired with an INDEPENDENT
  net-value witness. Any learning-loop metric that can be RAISED without a
  paired net-value RISE is self-fulfilling — flagged. A skill that games the
  witness (inflates the metric while net task value stays flat/negative) scores
  WORSE, not better.

"Net task value" is the independent OUTCOME witness — the per-session net-true
verdict (``internal/sessionobs/nettrue.go``: HELPED / WASH / HURT, a signed
``net_tokens`` judged against the cost the mediation added) — NOT the learning
loop's own activity count. Provenance is kept honest exactly as net-true keeps
it (``WITNESSED`` / ``OBSERVED`` / ``MODELED``): an unwitnessed (MODELED)
positive net claim earns NO credit, so a skill cannot launder activity into
value by asserting a net gain it never proved.

The detector is a pure, deterministic function of the observations it is fed —
stdlib-only, no clock, no RNG — so a verdict is a witness a third party can
re-derive, the same discipline nettrue.go and the rsi scorecards obey. Feed it
one ``SkillObservation`` per skill/learning-loop candidate; it returns per-skill
rows (flagged?, score), a per-metric rollup (is this metric raisable without a
paired net rise?), and a corpus verdict.

    python tools/learning_loop_metric_detector.py            # human report (self-check demo)
    python tools/learning_loop_metric_detector.py --json     # machine payload
    python tools/learning_loop_metric_detector.py --ledger obs.jsonl   # score a real feed

Exit code: 0 = no self-fulfilling learning metric flagged · 1 = at least one
learning-loop metric is raisable without a paired net-value rise (a reward-hack
surface to close). The companion regression test
(``tools/learning_loop_metric_detector_test.py``) pins a deliberately-gaming
skill to a WORSE score than a genuinely value-raising one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-learning-loop-metric-detector/1"

# Provenance of the NET-VALUE witness — mirrors internal/sessionobs.Provenance so
# the learning-surface detector and the net-true ledger keep provenance the same
# way. Only WITNESSED / OBSERVED net value is credited; a MODELED (unproven) net
# claim cannot buy back activity.
WITNESSED = "WITNESSED"
OBSERVED = "OBSERVED"
MODELED = "MODELED"
_CREDITED_PROVENANCE = frozenset({WITNESSED, OBSERVED})

# Each unit of learning-metric activity that outran the witnessed net value is
# debited one net-value unit. A naive activity-rewarding loop (Hermes) credits
# +1 per "skill kept"; this guardrail credits ONLY witnessed net, so unpaired
# activity flips from a +1 reward to a -1 penalty. Symmetric and defensible: the
# gain a gamer expected becomes an equal loss, guaranteeing a gaming skill scores
# below an honest one.
GAMING_PENALTY = 1.0


class SkillObservation:
    """One skill's (or learning-loop candidate's) paired reading: the cheap-to-
    raise learning ACTIVITY metric vs the independent NET-VALUE witness.

    metric_delta      how much the learning-loop metric ("skills_kept",
                      "skill_updates") rose because this skill was kept. The
                      thing an RSI process can inflate. Clamped at >= 0.
    net_value_delta   the independent net-value witness delta attributable to the
                      skill — SIGNED net-true value (>0 HELPED, 0 WASH, <0 HURT).
                      NOT the activity count.
    net_provenance    WITNESSED / OBSERVED / MODELED for net_value_delta. A
                      MODELED positive claim is not credited (see credited_net).
    """

    __slots__ = ("skill", "metric", "metric_delta", "net_value_delta", "net_provenance")

    def __init__(self, skill: str, metric: str, metric_delta: float,
                 net_value_delta: float, net_provenance: str = MODELED) -> None:
        self.skill = str(skill)
        self.metric = str(metric or "skills_kept")
        self.metric_delta = max(0.0, float(metric_delta))
        self.net_value_delta = float(net_value_delta)
        self.net_provenance = (net_provenance or MODELED).upper()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SkillObservation":
        return cls(
            skill=row.get("skill", row.get("name", "")),
            metric=row.get("metric", "skills_kept"),
            metric_delta=row.get("metric_delta", row.get("skills_kept_delta", 0)),
            net_value_delta=row.get("net_value_delta", row.get("net_tokens", 0)),
            net_provenance=row.get("net_provenance", row.get("provenance", MODELED)),
        )


# ---------------------------------------------------------------------------
# Pure detector core — same inputs -> identical verdict, always.
# ---------------------------------------------------------------------------

def credited_net(net_value_delta: float, net_provenance: str) -> float:
    """The net value the detector is willing to CREDIT, after provenance.

    WITNESSED / OBSERVED net value counts as-is. An unproven (MODELED / unknown)
    row cannot claim a positive net gain — it is floored to 0 — but a MODELED
    HURT is kept, so a gamer cannot hide harm behind "modeled" either. This is
    the net-true provenance law: the witness, not the assertion, decides.
    """
    prov = (net_provenance or MODELED).upper()
    if prov in _CREDITED_PROVENANCE:
        return float(net_value_delta)
    return min(0.0, float(net_value_delta))


def is_self_fulfilling(metric_delta: float, credited: float) -> bool:
    """The HARD flag: the learning metric ROSE while credited net value did NOT.

    metric_delta > 0 and credited <= 0 means this metric was raised without a
    paired net-value rise — the self-fulfilling shape. Its existence proves the
    metric is *raisable without a paired net rise*.
    """
    return metric_delta > 0 and credited <= 0


def unpaired_activity(metric_delta: float, credited: float) -> float:
    """Learning-metric activity that outran the witnessed net value (metric units).

    Each unit of credited positive net "pays for" one unit of activity (the 1:1
    assumption a naive activity loop makes); activity beyond that is unpaired.
    """
    paid = max(0.0, credited)
    return max(0.0, metric_delta - paid)


def skill_score(metric_delta: float, credited: float) -> float:
    """Value-adjusted learning score, in NET-VALUE units.

    Score the WITNESS, never the activity count: the credited net value minus a
    gaming penalty for activity that was not backed by net value. A genuinely
    value-raising skill (metric backed 1:1 or better by net value) keeps its full
    net score; a gaming skill (metric with no net behind it) is driven negative.
    """
    return credited - GAMING_PENALTY * unpaired_activity(metric_delta, credited)


def score_observation(obs: SkillObservation) -> dict[str, Any]:
    credited = credited_net(obs.net_value_delta, obs.net_provenance)
    flagged = is_self_fulfilling(obs.metric_delta, credited)
    unpaired = unpaired_activity(obs.metric_delta, credited)
    if flagged:
        reason = (f"self-fulfilling: {obs.metric} rose {obs.metric_delta:g} with "
                  f"no paired net-value rise (credited net {credited:g})")
    elif unpaired > 0:
        reason = (f"over-claiming: {obs.metric} rose {obs.metric_delta:g} but only "
                  f"{max(0.0, credited):g} net-value units back it ({unpaired:g} unpaired)")
    else:
        reason = (f"paired: {obs.metric} rise {obs.metric_delta:g} is backed by "
                  f"net-value {credited:g}")
    return {
        "skill": obs.skill,
        "metric": obs.metric,
        "metric_delta": obs.metric_delta,
        "net_value_delta": obs.net_value_delta,
        "net_provenance": obs.net_provenance,
        "credited_net": credited,
        "unpaired_activity": unpaired,
        "flagged": flagged,
        "over_claiming": (not flagged) and unpaired > 0,
        "score": round(skill_score(obs.metric_delta, credited), 4),
        "reason": reason,
    }


def detect(observations: list[SkillObservation]) -> dict[str, Any]:
    """Score every observation, roll up per learning-loop metric, and judge the
    corpus. A metric is self-fulfilling iff SOME observation raised it without a
    paired net-value rise — that witnesses the metric is raisable without net."""
    rows = [score_observation(o) for o in observations]

    metrics: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = metrics.setdefault(r["metric"], {
            "metric": r["metric"], "n": 0, "flagged_skills": [],
            "self_fulfilling": False, "min_score": None, "max_score": None,
        })
        m["n"] += 1
        if r["flagged"]:
            m["self_fulfilling"] = True
            m["flagged_skills"].append(r["skill"])
        m["min_score"] = r["score"] if m["min_score"] is None else min(m["min_score"], r["score"])
        m["max_score"] = r["score"] if m["max_score"] is None else max(m["max_score"], r["score"])
    metric_rollup = [metrics[k] for k in sorted(metrics)]

    flagged = [r for r in rows if r["flagged"]]
    over = [r for r in rows if r["over_claiming"]]
    self_fulfilling_metrics = sorted(m["metric"] for m in metric_rollup if m["self_fulfilling"])
    ok = not flagged

    if ok and not over:
        verdict, finding = "OK", "learning_loop_paired"
        reason = (f"{len(rows)} learning-loop observation(s): every metric rise is "
                  f"paired with a witnessed net-value rise; no self-fulfilling metric")
        next_action = "hold the line; keep scoring the net witness, never the activity count"
    elif ok:
        verdict, finding = "OK", "learning_loop_paired_with_overclaim"
        reason = (f"no self-fulfilling metric, but {len(over)} skill(s) over-claim "
                  f"(activity outran witnessed net value) — a soft drift to watch")
        next_action = ("no metric is raisable without a net rise; review the over-claiming "
                       "skills so activity stays backed by net value")
    else:
        verdict, finding = "ACTION", "learning_loop_reward_hack"
        reason = (f"{len(flagged)} skill(s) raised a learning metric with NO paired "
                  f"net-value rise; self-fulfilling metric(s): "
                  f"{', '.join(self_fulfilling_metrics)}")
        worst = min(rows, key=lambda r: r["score"])
        next_action = (f"a skill can game {', '.join(self_fulfilling_metrics)} without raising "
                       f"net task value (worst: {worst['skill']} @ score {worst['score']}); "
                       f"pair the metric to the net witness or drop it")

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "n_observations": len(rows),
        "n_flagged": len(flagged),
        "n_over_claiming": len(over),
        "self_fulfilling_metrics": self_fulfilling_metrics,
        "metrics": metric_rollup,
        "observations": rows,
    }


# ---------------------------------------------------------------------------
# I/O boundary — ledger reader + renderers. The logic above needs no disk.
# ---------------------------------------------------------------------------

def load_ledger(path: str) -> list[SkillObservation]:
    """Read a JSONL feed of observation rows (one JSON object per line). Blank
    lines and ``#`` comment lines are skipped so a hand-authored fixture reads."""
    out: list[SkillObservation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(SkillObservation.from_row(json.loads(line)))
    return out


def demo_observations() -> list[SkillObservation]:
    """The self-check corpus: a gaming skill (metric up, net flat) that MUST be
    flagged and score worse, and a value-raising skill (metric up, net up) that
    must not be flagged. The human/JSON default run shows the detector firing."""
    return [
        SkillObservation("value-raiser", "skills_kept", metric_delta=3,
                         net_value_delta=12, net_provenance=WITNESSED),
        SkillObservation("witness-gamer", "skills_kept", metric_delta=10,
                         net_value_delta=0, net_provenance=WITNESSED),
        SkillObservation("modeled-gamer", "skills_kept", metric_delta=8,
                         net_value_delta=8, net_provenance=MODELED),
    ]


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"learning-loop metric detector — {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  observations: {payload['n_observations']}   flagged: {payload['n_flagged']}"
        f"   over-claiming: {payload['n_over_claiming']}",
        "",
    ]
    for r in payload["observations"]:
        mark = "FLAG" if r["flagged"] else ("warn" if r["over_claiming"] else "  ok")
        lines.append(f"  [{mark}] {r['skill']:<16} {r['metric']} +{r['metric_delta']:g}"
                     f"  net {r['net_value_delta']:+g} [{r['net_provenance']}]"
                     f"  score {r['score']:+g}")
        lines.append(f"         {r['reason']}")
    lines.extend(["", f"  -> {payload['next_action']}"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Self-fulfilling-metric detector for the skill/memory learning "
                    "loop (#2914): flag any learning metric raisable without a paired "
                    "net-value rise; score a gaming skill worse.")
    ap.add_argument("--ledger", help="JSONL feed of observation rows "
                    "(skill, metric_delta, net_value_delta, net_provenance). "
                    "Default: a built-in self-check corpus.")
    ap.add_argument("--json", action="store_true", help="emit the machine payload")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    try:
        observations = load_ledger(args.ledger) if args.ledger else demo_observations()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ledger error: {exc}", file=sys.stderr)
        return 2

    payload = detect(observations)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
