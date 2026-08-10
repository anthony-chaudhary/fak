#!/usr/bin/env python3
"""Lightgap scorecard — how usable is fak, on an UNBOUNDED scale, per use case?

The sibling scorecards all answer bounded questions. ``industry_scorecard`` asks
whether the competitive map is *complete and honest* (coverage + parity-debt).
``product_scorecard`` asks whether each concept is *real and useful today*.
``persona_*`` asks whether each persona is *served*. Every one of them tops out:
you can score 100 on all of them and still not know whether anyone should switch.

This one answers the question a buyer actually holds:

  **Relative to the next best thing I could do instead, and relative to the best
  that is physically possible, how far does fak actually get me — and is that
  worth what it costs me to adopt it?**

That question has no ceiling of 100, so this scorecard has no ceiling either.

===========================================================================
THE MODEL: two anchors, one unbounded coordinate
===========================================================================

Every cell is pinned between two anchors, never graded in the abstract:

  N  the NEXT-BEST OPTION — what a competent team does today WITHOUT fak.
     This is the origin. You are already here and it costs you nothing.
  c  the CEILING — "the speed of light in a vacuum": the best that is possible
     for anyone, ever, on this axis. Derived, not guessed, and labeled with the
     kind of derivation (physical / definitional / lower-bound).
  F  where fak actually measures, with committed provenance.

From those, the closure fraction:

     beta = (F - N) / (c - N)

  beta = 0    fak is exactly the next-best option. No reason to move.
  beta = 1    fak is AT the ceiling. Nothing better can exist.
  beta < 0    fak is WORSE than the next-best option. This happens, a lot.

beta is direction-agnostic by construction: for a lower-is-better metric (ASR,
latency, hours) the ceiling sits below the alternative, both differences flip
sign, and the ratio comes out right with no special case.

beta is bounded above by 1, which is exactly the property we do not want, so the
score is its RAPIDITY:

     w = artanh(beta)          the LIGHTGAP SCORE, in nats of approach

Rapidity is not a decorative log. It is the correct object here for four reasons:

  1. UNBOUNDED AND SIGNED. w runs (-inf, +inf). Being worse than the alternative
     is a negative score, not a low one.
  2. ADDITIVE. Relativistic velocities compose by a messy formula; rapidities
     just add. So layered gains add — and, the point, the ADOPTION COST
     SUBTRACTS on the same scale.
  3. IT DIVERGES AT THE CEILING. Closing the last 1% of the gap to physics is
     unboundedly harder than the first 50%, and w says so. A system that is the
     only thing in the world that does X scores arbitrarily high. That is what
     a moat IS.
  4. ZERO AT THE ALTERNATIVE. "No better than what you already have" is 0, not
     50. A positive number has to be EARNED.

===========================================================================
THE TAX: mental effort, on the same scale, and DIFFERENTIAL
===========================================================================

A gain you cannot afford to collect is not a gain. Each segment declares a
TOLERANCE: the engineer-hours of unfamiliarity it absorbs before walking away.
Each cell declares what it costs to get that facet working — AND what the
alternative costs to adopt, because the alternative is not always free either
(a formal-isolation defense makes you restructure your agent; fak is a wrapper).

     load = (cost_hours - alt_cost_hours) / tolerance_hours
     tau  = artanh(load)                    the tax, on the same scale
     w_net = w - tau                        the verdict

load is SIGNED. When fak is cheaper to adopt than the alternative, tau is
negative and *adds* to the score — which is the honest way to price "same
result, far less restructuring". Because artanh is monotone, w_net > 0 exactly
when beta > load. That is the entire system in one line:

     ADOPT IFF YOU CLOSE MORE OF THE GAP TO PHYSICS
     THAN YOU CONSUME OF THE ADOPTER'S PATIENCE.

Note what this makes true, correctly:
  - A huge win nobody can afford to install scores NEGATIVE.
  - A tiny win that costs nothing scores POSITIVE.
  - The SAME feature scores differently per segment, because tolerance differs.
    A regulated buyer absorbs 240h without blinking; a solo dev on a flat-rate
    subscription absorbs 2h. That is not a bug in the metric, that is the finding.
  - An alternative that costs MORE than the segment's whole tolerance is not
    actually available to that segment, so it cannot be the next-best option.
    The engine refuses it (ALT_UNAFFORDABLE) rather than let fak win by
    comparing against something the buyer would never have deployed.

===========================================================================
THREE ANCHOR SHAPES THE NAIVE FORM GETS WRONG
===========================================================================

  pure_tax          The alternative IS the ceiling — you already run the best
                    thing and fak can only take a cut (mediation overhead, a
                    narrower model matrix). (F-N)/(c-N) is 0/0 there, so beta is
                    the fractional shortfall (F-N)/N instead. Such a cell can
                    never score positive on the axis, which is the honest shape
                    of a tax.
  parity_at_ceiling Both fak AND the next-best already sit at the limit (zero
                    attack success). beta = 0: no advantage on the axis. Any
                    reason to switch has to come from the differential adoption
                    cost, not from the number.
  ceiling override  A facet's ceiling is per-hardware (a bandwidth roofline).
                    A cell may state its own c with its own derivation.

===========================================================================
CLAIM CAPS: you cannot outrun your own evidence
===========================================================================

Two caps limit how large a verdict the evidence can support, independent of the
arithmetic. The raw w_net is always reported; w_eff is what decisions use.

  provenance   MODELED / PROJECTED  -> at most CRUISE
               OBSERVED             -> at most RELATIVISTIC
               MEASURED / WITNESSED -> uncapped
  ceiling kind lower-bound          -> at most RELATIVISTIC

You cannot claim a category-defining lead on a simulated corpus, and you cannot
claim to be near light-speed when all you know is where the current record sits.

===========================================================================
THE SPHERE: why one number is a lie
===========================================================================

Collapsing this to a scalar destroys the only useful information in it. So the
output is a SPHERE, not a score:

     azimuth   = facet   (what kind of value: 8 facets in 4 bands)
     elevation = segment (who is buying: the use cases)
     radius    = w_net   (how far out fak sits — NEGATIVE radii dent inward)

fak bulges hugely on some (segment, facet) cells and dents INWARD on others.
Reporting the mean would hide both. So there is no mean. Per use case we report
the PULL (best cell — the actual reason to adopt) and the DRAG (worst cell —
what you eat to get it). A material axis that is REGRESSIVE BLOCKS adoption no
matter how high the pull. And a use case whose deciding comparison has never
been run comes back UNDECIDABLE rather than being quietly scored on the cells
that happen to exist.

  python tools/lightgap_scorecard.py                 # the sphere + per-use-case verdicts
  python tools/lightgap_scorecard.py --json          # machine payload
  python tools/lightgap_scorecard.py --segment ID    # one use case, in full
  python tools/lightgap_scorecard.py --facet ID      # one facet across all use cases
  python tools/lightgap_scorecard.py --dents         # every cell where fak LOSES, worst first
  python tools/lightgap_scorecard.py --unrun         # the comparisons that would decide it
  python tools/lightgap_scorecard.py --ceilings      # the c anchors and their derivations
  python tools/lightgap_scorecard.py --check         # honesty gate (exit 1 on lightgap_debt)
  python tools/lightgap_scorecard.py --markdown-dir docs/lightgap-scorecard

Pure-stdlib, repo root, deterministic. Data lives in ``tools/lightgap_scorecard.data/``;
the doc folder is GENERATED — never hand-edit it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "lightgap_scorecard.data")
REPO_ROOT = os.path.dirname(HERE)

# The display horizon. beta and load are clamped here so the arithmetic never
# produces an infinity (JSON cannot hold one) while staying effectively
# unbounded for reporting. artanh(0.999) = 3.8002.
HORIZON_BETA = 0.999
HORIZON = math.atanh(HORIZON_BETA)
EPS = 0.001

# Verdict ladder on w_eff. Ordered high -> low; first match wins.
LADDER = [
    ("AT-CEILING", 3.80, "at the limit — nothing better can exist on this axis"),
    ("NEAR-C", 2.00, "category-defining; the alternative is not in the running"),
    ("RELATIVISTIC", 1.00, "a real, large, net win worth restructuring for"),
    ("CRUISE", 0.50, "a solid net win; adopt if this facet is what you came for"),
    ("DRIFT", 0.10, "marginally better once effort is counted; easy to regret"),
    ("REST", -0.10, "indistinguishable from doing nothing"),
    ("DRAG", -1.00, "the alternative wins once you count the effort"),
    ("REGRESSIVE", float("-inf"), "actively worse than what you already have"),
]
RUNG = {name: floor for name, floor, _ in LADDER}
ORDER = [name for name, _, _ in LADDER]

PROVENANCE_CAP = {"MODELED": "CRUISE", "PROJECTED": "CRUISE", "OBSERVED": "RELATIVISTIC"}
CEILING_KIND_CAP = {"lower-bound": "RELATIVISTIC"}

PROVENANCE = {"MEASURED", "WITNESSED", "OBSERVED", "MODELED", "PROJECTED", "NONE"}
CEILING_KINDS = {"physical", "definitional", "lower-bound"}


def _fail(msg: str) -> None:
    print(f"lightgap_scorecard: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name: str) -> Any:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        _fail(f"missing data file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            _fail(f"{name}: {exc}")


def _clamp(x: float) -> float:
    return max(-HORIZON_BETA, min(HORIZON_BETA, x))


def rapidity(beta: float) -> float:
    return math.atanh(_clamp(beta))


def cap_value(cap: str | None) -> float:
    """Highest w_eff a capped cell may report: just under the next rung up."""
    if cap is None:
        return float("inf")
    idx = ORDER.index(cap)
    return float("inf") if idx == 0 else RUNG[ORDER[idx - 1]] - EPS


def verdict_for(w_eff: float) -> tuple[str, str]:
    for label, floor, blurb in LADDER:
        if w_eff >= floor:
            return label, blurb
    return LADDER[-1][0], LADDER[-1][2]


@dataclass
class Defect:
    code: str
    where: str
    detail: str
    next_action: str = ""

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "where": self.where,
            "detail": self.detail,
            "next_action": self.next_action,
        }


@dataclass
class Cell:
    segment: str
    facet: str
    weight: float
    shape: str
    mode: str
    beta: float
    w: float
    load: float
    tau: float
    w_net: float
    w_eff: float
    cap: str
    cap_reason: str
    verdict: str
    why: str
    fak_value: float
    fak_source: str
    provenance: str
    alt_name: str
    # The alternatives-file key, carried so a reader (and the tests) can join a
    # cell back to the alternative's declared `class` -- which is what says
    # whether the comparison is against a real option or a strawman.
    alt_id: str
    alt_value: float
    alt_source: str
    alt_cost_hours: float
    ceiling: float
    ceiling_kind: str
    cost_hours: float
    cost_basis: list
    note: str = ""
    fence: str = ""
    defects: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["defects"] = [x.as_dict() for x in self.defects]
        for k in ("beta", "w", "load", "tau", "w_net", "w_eff"):
            d[k] = round(d[k], 4)
        return d


class Scorecard:
    def __init__(self) -> None:
        self.meta = _load("_meta.json")
        self.ceilings = {c["id"]: c for c in _load("_ceilings.json")["ceilings"]}
        self.alternatives = {a["id"]: a for a in _load("_alternatives.json")["alternatives"]}
        self.facets = {f["id"]: f for f in self.meta["facets"]}
        self.segments = {s["id"]: s for s in self.meta["segments"]}
        self.bands = {b["id"]: b for b in self.meta["bands"]}
        self.cells: list[Cell] = []
        self.defects: list[Defect] = []
        self.unrun: list[dict] = []
        self._build()

    # -- construction -----------------------------------------------------
    def _build(self) -> None:
        raw: dict[tuple[str, str], dict] = {}
        for fname in sorted(
            f for f in os.listdir(DATA_DIR) if f.startswith("cells-") and f.endswith(".json")
        ):
            for row in _load(fname).get("cells", []):
                seg, fac = row.get("segment"), row.get("facet")
                if seg not in self.segments:
                    self.defects.append(Defect("UNKNOWN_SEGMENT", fname, f"segment {seg!r}"))
                    continue
                if fac not in self.facets:
                    self.defects.append(Defect("UNKNOWN_FACET", fname, f"facet {fac!r}"))
                    continue
                if (seg, fac) in raw:
                    self.defects.append(Defect("DUPLICATE_CELL", f"{seg}/{fac}", "scored twice"))
                    continue
                raw[(seg, fac)] = row

        for cid, c in self.ceilings.items():
            if not c.get("derivation"):
                self.defects.append(Defect("NO_DERIVATION", f"ceiling/{cid}", "no derivation"))
            if c.get("kind") not in CEILING_KINDS:
                self.defects.append(
                    Defect("BAD_CEILING_KIND", f"ceiling/{cid}", f"{c.get('kind')!r}")
                )

        for sid, seg in self.segments.items():
            for fid, weight in seg.get("weights", {}).items():
                if fid not in self.facets:
                    self.defects.append(Defect("UNKNOWN_FACET", sid, f"weight on {fid!r}"))
                    continue
                if weight <= 0:
                    continue
                row = raw.pop((sid, fid), None)
                if row is None:
                    gap = seg.get("unrun", {}).get(fid, {})
                    self.defects.append(
                        Defect(
                            "UNCOVERED",
                            f"{sid}/{fid}",
                            gap.get("why", f"material facet (weight {weight:.2f}) has no cell"),
                            gap.get("next_action", "run the head-to-head, then add the cell"),
                        )
                    )
                    self.unrun.append(
                        {
                            "segment": sid,
                            "facet": fid,
                            "weight": weight,
                            "why": gap.get("why", "no scored cell"),
                            "next_action": gap.get("next_action", ""),
                        }
                    )
                    continue
                self.cells.append(self._score(sid, fid, weight, row))

        for (sid, fid) in sorted(raw):
            self.defects.append(
                Defect("IMMATERIAL_CELL", f"{sid}/{fid}", "scored, but the segment weights it 0")
            )

    def _score(self, sid: str, fid: str, weight: float, row: dict) -> Cell:
        defects: list[Defect] = []
        where = f"{sid}/{fid}"
        facet_ceiling = self.ceilings.get(fid)
        if facet_ceiling is None:
            _fail(f"{where}: facet has no ceiling entry")

        override = row.get("ceiling") or {}
        c = float(override.get("c", facet_ceiling["c"]))
        kind = override.get("kind", facet_ceiling.get("kind", "definitional"))
        if override and not override.get("derivation"):
            defects.append(Defect("NO_DERIVATION", where, "cell overrides c with no derivation"))

        fak, alt_ref, cost = row.get("fak", {}), row.get("alt", {}), row.get("cost", {})
        mode = "standard"

        F = float(fak.get("value", 0.0))
        prov = fak.get("provenance", "NONE")
        if prov not in PROVENANCE:
            defects.append(Defect("BAD_PROVENANCE", where, f"{prov!r}"))
        if prov != "NONE" and not fak.get("source"):
            defects.append(Defect("NO_FAK_SOURCE", where, "a fak value with no committed source"))

        alt_id = alt_ref.get("id")
        alt = self.alternatives.get(alt_id)
        if alt is None:
            defects.append(Defect("NO_ALT_NAMED", where, f"unknown alternative {alt_id!r}"))
            alt = {"name": alt_id or "(unnamed)", "source": "", "class": "unknown"}
        if alt.get("class") == "naive":
            defects.append(
                Defect(
                    "NAIVE_BASELINE",
                    where,
                    f"{alt.get('name')} is a strawman; state the gain vs the tuned/SOTA option",
                )
            )
        N = float(alt_ref.get("value", 0.0))
        alt_source = alt_ref.get("source") or alt.get("source", "")
        if not alt_source:
            defects.append(Defect("NO_ALT_SOURCE", where, "next-best value with no source"))

        # -- beta, in one of three anchor shapes
        if row.get("parity_at_ceiling"):
            mode = "parity_at_ceiling"
            beta = 0.0
        elif row.get("pure_tax"):
            mode = "pure_tax"
            if N == 0:
                defects.append(Defect("DEGENERATE_CEILING", where, "pure_tax with N = 0"))
                beta = 0.0
            else:
                beta = (F - N) / abs(N)
            if beta > 0:
                defects.append(
                    Defect("CEILING_BREACHED", where, "pure_tax cell scores above its ceiling")
                )
        elif math.isclose(c, N):
            defects.append(
                Defect(
                    "DEGENERATE_CEILING",
                    where,
                    "ceiling equals the alternative — mark pure_tax or fix the anchor",
                )
            )
            beta = 0.0
        else:
            beta = (F - N) / (c - N)
            if beta > 1.0 + 1e-9:
                defects.append(
                    Defect("CEILING_BREACHED", where, f"beta={beta:.3f} — the ceiling is wrong")
                )

        # -- the differential tax
        hours = float(cost.get("hours", 0.0))
        alt_hours = float(cost.get("alt_hours", 0.0))
        basis = cost.get("basis", [])
        if not basis:
            defects.append(Defect("NO_COST_BASIS", where, "adoption cost with no measured basis"))
        tol = float(self.segments[sid]["tolerance_hours"])
        if alt_hours > tol:
            defects.append(
                Defect(
                    "ALT_UNAFFORDABLE",
                    where,
                    f"{alt['name']} costs {alt_hours:g}h against a {tol:g}h tolerance -- this "
                    "buyer would never deploy it, so it is not their next-best option",
                )
            )
        load = (hours - alt_hours) / tol if tol > 0 else 1.0

        w = rapidity(beta)
        tau = rapidity(load)
        w_net = w - tau

        caps = [c_ for c_ in (PROVENANCE_CAP.get(prov), CEILING_KIND_CAP.get(kind)) if c_]
        cap = min(caps, key=lambda x: ORDER.index(x)) if caps else ""
        cap_reason = ""
        if cap:
            reasons = []
            if PROVENANCE_CAP.get(prov) == cap:
                reasons.append(f"{prov} evidence")
            if CEILING_KIND_CAP.get(kind) == cap:
                reasons.append(f"{kind} ceiling")
            cap_reason = " + ".join(reasons)
        w_eff = min(w_net, cap_value(cap or None))
        verdict, why = verdict_for(w_eff)

        if w_eff >= RUNG["RELATIVISTIC"] and prov in ("MODELED", "PROJECTED") and not row.get("fence"):
            defects.append(
                Defect("UNFENCED_MODELED_LEAD", where, f"{verdict} on {prov} evidence, no fence")
            )

        return Cell(
            segment=sid,
            facet=fid,
            weight=weight,
            shape=row.get("shape", "wrapper"),
            mode=mode,
            beta=beta,
            w=w,
            load=load,
            tau=tau,
            w_net=w_net,
            w_eff=w_eff,
            cap=cap,
            cap_reason=cap_reason,
            verdict=verdict,
            why=why,
            fak_value=F,
            fak_source=fak.get("source", ""),
            provenance=prov,
            alt_name=alt.get("name", alt_id or ""),
            alt_id=alt_id or "",
            alt_value=N,
            alt_source=alt_source,
            alt_cost_hours=alt_hours,
            ceiling=c,
            ceiling_kind=kind,
            cost_hours=hours,
            cost_basis=basis,
            note=row.get("note", ""),
            fence=row.get("fence", ""),
            defects=defects,
        )

    # -- aggregation ------------------------------------------------------
    @property
    def debt(self) -> int:
        return len(self.defects) + sum(len(c.defects) for c in self.cells)

    def by_segment(self, sid: str) -> list[Cell]:
        return sorted([c for c in self.cells if c.segment == sid], key=lambda c: -c.w_eff)

    def by_facet(self, fid: str) -> list[Cell]:
        return sorted([c for c in self.cells if c.facet == fid], key=lambda c: -c.w_eff)

    def segment_profile(self, sid: str) -> dict:
        seg = self.segments[sid]
        cells = self.by_segment(sid)
        unrun = [u for u in self.unrun if u["segment"] == sid]
        unrun_weight = sum(u["weight"] for u in unrun)
        base = {
            "segment": sid,
            "name": seg["name"],
            "abbr": seg["abbr"],
            "tolerance_hours": seg["tolerance_hours"],
            "switch_bar": float(seg.get("switch_bar", 0.5)),
            "unrun_weight": round(unrun_weight, 3),
            "unrun": [u["facet"] for u in unrun],
            "cells": len(cells),
        }
        if not cells:
            return {**base, "verdict": "UNSCORED", "because": "no scored cells", "pull": None,
                    "drag": None}

        pull, drag = cells[0], cells[-1]
        bar = base["switch_bar"]
        blockers = [
            c
            for c in cells
            if c.w_eff <= RUNG["DRAG"] and c.weight >= float(seg.get("block_weight", 0.20))
        ]
        undecidable_at = float(seg.get("undecidable_weight", 0.25))

        if blockers:
            verdict = "BLOCKED"
            because = (
                f"{blockers[0].facet} is REGRESSIVE at weight {blockers[0].weight:.2f} -- a "
                "material axis goes backwards, and no pull elsewhere buys it back"
            )
        elif unrun_weight >= undecidable_at:
            verdict = "UNDECIDABLE"
            because = (
                f"{unrun_weight:.0%} of what this buyer weights has never been measured against "
                f"their actual alternative ({', '.join(u['facet'] for u in unrun)})"
            )
        elif pull.w_eff >= bar and drag.w_eff <= RUNG["DRAG"]:
            verdict = "ADOPT-WITH-SCARS"
            because = f"{pull.facet} carries it; you eat {drag.facet} to get it"
        elif pull.w_eff >= bar:
            verdict = "ADOPT"
            because = f"{pull.facet} clears the switch bar ({bar:+.2f})"
        elif pull.w_eff >= RUNG["DRIFT"]:
            verdict = "PILOT-ONLY"
            because = f"best axis {pull.facet} is {pull.w_eff:+.2f}, under the bar {bar:+.2f}"
        else:
            verdict = "HOLD"
            because = f"nothing clears rest; best axis {pull.facet} at {pull.w_eff:+.2f}"

        return {
            **base,
            "verdict": verdict,
            "because": because,
            "pull": {"facet": pull.facet, "w_eff": round(pull.w_eff, 3), "verdict": pull.verdict},
            "drag": {"facet": drag.facet, "w_eff": round(drag.w_eff, 3), "verdict": drag.verdict},
            "positive": sum(1 for c in cells if c.w_eff > RUNG["DRIFT"]),
            "negative": sum(1 for c in cells if c.w_eff < RUNG["REST"]),
        }

    def facet_profile(self, fid: str) -> dict:
        cells = self.by_facet(fid)
        f = self.facets[fid]
        base = {
            "facet": fid,
            "name": f["name"],
            "band": f["band"],
            "ceiling_kind": self.ceilings[fid]["kind"],
            "cells": len(cells),
        }
        if not cells:
            return base
        return {
            **base,
            "best": {"segment": cells[0].segment, "w_eff": round(cells[0].w_eff, 3)},
            "worst": {"segment": cells[-1].segment, "w_eff": round(cells[-1].w_eff, 3)},
            "spread": round(cells[0].w_eff - cells[-1].w_eff, 3),
        }

    def hull(self) -> dict:
        if not self.cells:
            return {}
        best = max(self.cells, key=lambda c: c.w_eff)
        worst = min(self.cells, key=lambda c: c.w_eff)
        return {
            "peak": {
                "segment": best.segment,
                "facet": best.facet,
                "w_eff": round(best.w_eff, 3),
                "verdict": best.verdict,
            },
            "dent": {
                "segment": worst.segment,
                "facet": worst.facet,
                "w_eff": round(worst.w_eff, 3),
                "verdict": worst.verdict,
            },
            "eccentricity": round(best.w_eff - worst.w_eff, 3),
            "negative_cells": sum(1 for c in self.cells if c.w_eff < 0),
            "scored_cells": len(self.cells),
        }

    def payload(self) -> dict:
        return {
            "schema": "fak-lightgap-scorecard/1",
            "meta": self.meta["meta"],
            "horizon": round(HORIZON, 4),
            "lightgap_debt": self.debt,
            "hull": self.hull(),
            "segments": [self.segment_profile(s) for s in self.segments],
            "facets": [self.facet_profile(f) for f in self.facets],
            "cells": [c.as_dict() for c in self.cells],
            "unrun": self.unrun,
            "defects": [d.as_dict() for d in self.defects],
        }


# -- rendering ------------------------------------------------------------

GLYPH = {
    "AT-CEILING": "***",
    "NEAR-C": "**",
    "RELATIVISTIC": "*",
    "CRUISE": "+",
    "DRIFT": ".",
    "REST": "o",
    "DRAG": "-",
    "REGRESSIVE": "--",
}


def _bar(w: float, width: int = 24) -> str:
    half = width // 2
    n = int(round(min(abs(w), 4.0) / 4.0 * half))
    if w >= 0:
        return " " * half + "|" + "#" * n
    return " " * (half - n) + "#" * n + "|"


def render(sc: Scorecard) -> str:
    m = sc.meta["meta"]
    out = [
        "",
        f"  LIGHTGAP SCORECARD -- {m['title']}",
        f"  as of {m['as_of']} | fak {m['fak_version']} | horizon +-{HORIZON:.2f} nats",
        "",
        "  w_net = artanh(beta) - artanh(load)",
        "    beta = (fak - next_best) / (ceiling - next_best)",
        "           how much of the gap between the alternative and physics fak closes",
        "    load = (fak_hours - alt_hours) / segment_tolerance_hours",
        "           how much of the adopter's patience it consumes (signed)",
        "  Positive iff beta > load. Unbounded both ways. Zero = the alternative.",
        "",
    ]

    fids = list(sc.facets)
    out.append("  " + " " * 26 + "".join(f"{sc.facets[f]['abbr']:>7}" for f in fids))
    grid = {(c.segment, c.facet): c for c in sc.cells}
    unrun = {(u["segment"], u["facet"]) for u in sc.unrun}
    for sid, seg in sc.segments.items():
        row = f"  {seg['abbr']:<26}"
        for fid in fids:
            cell = grid.get((sid, fid))
            if cell:
                row += f"{GLYPH.get(cell.verdict, '?'):>7}"
            elif (sid, fid) in unrun:
                row += f"{'?':>7}"
            else:
                row += f"{'':>7}"
        out.append(row)
    out += [
        "",
        "  *** at-ceiling  ** near-c  * relativistic  + cruise  . drift",
        "  o rest  - drag  -- regressive  ? never measured  (blank: immaterial)",
        "",
        "  PER USE CASE -- the pull, the drag, and the call",
        "  " + "-" * 72,
    ]
    for sid in sc.segments:
        p = sc.segment_profile(sid)
        out.append(f"  {p['name']}  ->  {p['verdict']}")
        if p.get("pull"):
            out.append(
                f"      pull {p['pull']['w_eff']:+.2f} {p['pull']['facet']}"
                f"   drag {p['drag']['w_eff']:+.2f} {p['drag']['facet']}"
            )
        out.append(f"      {p['because']}")
        out.append("")

    hull = sc.hull()
    if hull:
        out += [
            "  THE SHAPE",
            "  " + "-" * 72,
            f"  peak  {hull['peak']['w_eff']:+.2f}  {hull['peak']['segment']}"
            f" x {hull['peak']['facet']}  ({hull['peak']['verdict']})",
            f"  dent  {hull['dent']['w_eff']:+.2f}  {hull['dent']['segment']}"
            f" x {hull['dent']['facet']}  ({hull['dent']['verdict']})",
            f"  eccentricity {hull['eccentricity']:.2f} nats | "
            f"{hull['negative_cells']}/{hull['scored_cells']} scored cells are net-negative",
            "  fak is not a sphere. It is a spike on a few axes attached to a body",
            "  that dents inward on others.",
            "",
        ]
    out.append(f"  lightgap_debt = {sc.debt}   (--check for the list, --unrun for the work)")
    out.append("")
    return "\n".join(out)


def render_segment(sc: Scorecard, sid: str) -> str:
    if sid not in sc.segments:
        _fail(f"unknown segment {sid!r}; have: {', '.join(sc.segments)}")
    seg, p = sc.segments[sid], sc.segment_profile(sid)
    out = [
        "",
        f"  {seg['name']} -- {p['verdict']}",
        f"  {seg['description']}",
        f"  next best overall: {seg['next_best_summary']}",
        f"  tolerance {seg['tolerance_hours']:g} engineer-hours | switch bar"
        f" {p['switch_bar']:+.2f}",
        f"  {p['because']}",
        "",
    ]
    for c in sc.by_segment(sid):
        f = sc.facets[c.facet]
        out.append(f"  {f['name']}   [{c.verdict}]  w_eff {c.w_eff:+.3f}  (weight {c.weight:.2f})")
        out.append(f"  {_bar(c.w_eff)}")
        out.append(
            f"    beta {c.beta:+.3f}   fak {c.fak_value:g} | next-best {c.alt_value:g}"
            f" | ceiling {c.ceiling:g} ({c.ceiling_kind})"
        )
        out.append(
            f"    load {c.load:+.3f}   {c.cost_hours:g}h fak vs {c.alt_cost_hours:g}h alt,"
            f" of {seg['tolerance_hours']:g}h   tau {c.tau:+.3f}"
        )
        out.append(f"    w {c.w:+.3f} - tau = w_net {c.w_net:+.3f}")
        if c.cap:
            out.append(f"    CAPPED at {c.cap} ({c.cap_reason}) -> w_eff {c.w_eff:+.3f}")
        out.append(f"    vs {c.alt_name}")
        out.append(f"    {c.provenance}" + (f" | {c.fak_source}" if c.fak_source else ""))
        if c.mode != "standard":
            out.append(f"    mode {c.mode}")
        if c.note:
            out.append(f"    note  {c.note}")
        if c.fence:
            out.append(f"    FENCE {c.fence}")
        out.append("")
    for u in [x for x in sc.unrun if x["segment"] == sid]:
        out.append(f"  {sc.facets[u['facet']]['name']}   [NEVER MEASURED]  (weight {u['weight']:.2f})")
        out.append(f"    {u['why']}")
        out.append(f"    next: {u['next_action']}")
        out.append("")
    return "\n".join(out)


def render_facet(sc: Scorecard, fid: str) -> str:
    if fid not in sc.facets:
        _fail(f"unknown facet {fid!r}; have: {', '.join(sc.facets)}")
    f, ceil = sc.facets[fid], sc.ceilings[fid]
    out = [
        "",
        f"  {f['name']}  ({f['band']} band)",
        f"  {f['question']}",
        "",
        f"  metric   {ceil['metric']} [{ceil['unit']}, {ceil['direction']}-is-better]",
        f"  ceiling  c = {ceil['c']:g}  ({ceil['kind']})",
        f"  why      {ceil['derivation']}",
        "",
    ]
    for c in sc.by_facet(fid):
        out.append(
            f"  {sc.segments[c.segment]['abbr']:<22} {c.verdict:<13} w_eff {c.w_eff:+.3f}"
            f"   beta {c.beta:+.3f}  load {c.load:+.2f}"
        )
        out.append(f"      vs {c.alt_name}")
    for u in [x for x in sc.unrun if x["facet"] == fid]:
        out.append(f"  {sc.segments[u['segment']]['abbr']:<22} NEVER MEASURED")
    out.append("")
    return "\n".join(out)


def render_dents(sc: Scorecard) -> str:
    bad = sorted([c for c in sc.cells if c.w_eff < 0], key=lambda c: c.w_eff)
    out = ["", "  WHERE fak LOSES -- every cell below the next-best option, worst first", ""]
    if not bad:
        return "\n".join(out + ["  (none -- which would itself be suspicious)", ""])
    for c in bad:
        cause = "WORSE ON THE AXIS" if c.beta < 0 else "adoption cost eats the gain"
        out.append(f"  {c.w_eff:+.3f}  {c.segment} x {c.facet:<22} {c.verdict:<12} {cause}")
        out.append(
            f"          beta {c.beta:+.3f} vs {c.alt_name} | load {c.load:+.2f}"
            f" ({c.cost_hours:g}h vs {c.alt_cost_hours:g}h)"
        )
        if c.note:
            out.append(f"          {c.note}")
    out += [
        "",
        f"  {len(bad)} of {len(sc.cells)} scored cells are net-negative.",
        "  beta < 0 means fak is worse ON THE AXIS -- no amount of easier onboarding fixes it.",
        "  beta > 0 with w_net < 0 means the gain is real but the surface eats it -- that one",
        "  is fixed by cutting the surface, not by shipping more features.",
        "",
    ]
    return "\n".join(out)


def render_unrun(sc: Scorecard) -> str:
    out = [
        "",
        "  NEVER MEASURED -- the comparisons that would actually decide it",
        "  Ranked by the weight the buyer puts on them.",
        "",
    ]
    if not sc.unrun:
        return "\n".join(out + ["  (none)", ""])
    for u in sorted(sc.unrun, key=lambda u: -u["weight"]):
        out.append(f"  weight {u['weight']:.2f}  {u['segment']} x {u['facet']}")
        out.append(f"            {u['why']}")
        out.append(f"    next -> {u['next_action']}")
        out.append("")
    return "\n".join(out)


def render_ceilings(sc: Scorecard) -> str:
    out = ["", "  THE CEILINGS -- 'the speed of light in a vacuum', per facet", ""]
    for fid, f in sc.facets.items():
        c = sc.ceilings.get(fid)
        if not c:
            continue
        out += [
            f"  {f['name']}  [{c['kind']}]",
            f"    metric      {c['metric']} ({c['unit']}, {c['direction']}-is-better)",
            f"    c           {c['c']:g}",
            f"    derivation  {c['derivation']}",
        ]
        if c.get("caveat"):
            out.append(f"    caveat      {c['caveat']}")
        out.append("")
    out += [
        "  physical     a derived limit (bandwidth roofline, information content).",
        "  definitional the metric's own bound (zero attack success, zero lost work).",
        "  lower-bound  the best system KNOWN; the true c is at least this and probably",
        "               higher. Cells against one are capped at RELATIVISTIC -- you cannot",
        "               claim to be near light-speed knowing only where the record sits.",
        "",
    ]
    return "\n".join(out)


def render_check(sc: Scorecard) -> str:
    out = ["", "  LIGHTGAP HONESTY GATE", ""]
    all_d = list(sc.defects) + [d for c in sc.cells for d in c.defects]
    if not all_d:
        return "\n".join(out + ["  OK -- lightgap_debt = 0", ""])
    for d in all_d:
        out.append(f"  {d.code:<22} {d.where:<26} {d.detail}")
        if d.next_action:
            out.append(f"  {'':<22} {'':<26} next -> {d.next_action}")
    out += ["", f"  lightgap_debt = {len(all_d)}", ""]
    return "\n".join(out)


# -- markdown -------------------------------------------------------------


def _md_index(sc: Scorecard) -> str:
    m, hull = sc.meta["meta"], sc.hull()
    lines = [
        "# Lightgap scorecard — unbounded usability, per use case",
        "",
        "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
        "",
        f"**As of {m['as_of']} · fak {m['fak_version']} · `lightgap_debt` = {sc.debt}**",
        "",
        m["summary"],
        "",
        "## The one line",
        "",
        "> **Adopt iff you close more of the gap to physics than you consume of the"
        " adopter's patience.**",
        "",
        "`w_net = artanh(β) − artanh(load)`, where `β = (fak − next_best) / (ceiling −"
        " next_best)` is the fraction of the gap between *the thing you would otherwise do*"
        " and *the best that is physically possible* that fak closes, and `load = (fak_hours"
        " − alt_hours) / tolerance_hours` is the share of the buyer's patience it costs."
        " Because `artanh` is monotone, `w_net > 0` exactly when `β > load`.",
        "",
        "The score is unbounded in both directions: being worse than the alternative is a"
        " *negative* score, not a low one, and approaching the ceiling diverges — which is"
        " the honest description of a moat. [Full model](model.md) ·"
        " [ceilings and their derivations](ceilings.md) ·"
        " [every cell where fak loses](dents.md) ·"
        " [the comparisons nobody has run](unrun.md)",
        "",
        "## The shape",
        "",
    ]
    if hull:
        lines += [
            f"- **Peak** `{hull['peak']['w_eff']:+.2f}` — {hull['peak']['segment']} ×"
            f" {hull['peak']['facet']} ({hull['peak']['verdict']})",
            f"- **Dent** `{hull['dent']['w_eff']:+.2f}` — {hull['dent']['segment']} ×"
            f" {hull['dent']['facet']} ({hull['dent']['verdict']})",
            f"- **Eccentricity** {hull['eccentricity']:.2f} nats;"
            f" {hull['negative_cells']} of {hull['scored_cells']} scored cells are"
            " net-negative.",
            "",
            "fak is not a sphere. It is a spike on a few axes attached to a body that dents"
            " inward on others — and the dents are not where the marketing is.",
            "",
        ]
    lines += [
        "## The sphere",
        "",
        "Rows are use cases, columns are facets, each entry is `w_eff` (the claim-capped"
        " score decisions use).",
        "",
        "| use case | " + " | ".join(sc.facets[f]["abbr"] for f in sc.facets) + " |",
        "|---|" + "---|" * len(sc.facets),
    ]
    grid = {(c.segment, c.facet): c for c in sc.cells}
    unrun = {(u["segment"], u["facet"]) for u in sc.unrun}
    for sid, seg in sc.segments.items():
        row = f"| **{seg['abbr']}** |"
        for fid in sc.facets:
            cell = grid.get((sid, fid))
            if cell:
                row += f" `{cell.w_eff:+.2f}` |"
            elif (sid, fid) in unrun:
                row += " ? |"
            else:
                row += " – |"
        lines.append(row)
    lines += [
        "",
        "`?` = the buyer weights it and nobody has measured it. `–` = immaterial to that"
        " buyer (weight 0), not unscored.",
        "",
        "## Per use case",
        "",
        "| use case | verdict | pull (the reason) | drag (the price) | tolerance |",
        "|---|---|---|---|---|",
    ]
    for sid in sc.segments:
        p = sc.segment_profile(sid)
        pull = f"`{p['pull']['w_eff']:+.2f}` {p['pull']['facet']}" if p.get("pull") else "—"
        drag = f"`{p['drag']['w_eff']:+.2f}` {p['drag']['facet']}" if p.get("drag") else "—"
        lines.append(
            f"| [{p['name']}](segment-{sid}.md) | **{p['verdict']}** | {pull} | {drag} |"
            f" {p['tolerance_hours']:g} h |"
        )
    lines += ["", "### What each verdict means", "",
              "| verdict | means |", "|---|---|",
              "| **ADOPT** | one axis clears the switch bar and nothing material goes"
              " backwards |",
              "| **ADOPT-WITH-SCARS** | the pull is real, but a material axis is `DRAG` or"
              " worse — go in knowing what you are eating |",
              "| **PILOT-ONLY** | the best axis is positive but under the bar; worth a trial,"
              " not a migration |",
              "| **BLOCKED** | a material axis is `REGRESSIVE` — no pull elsewhere buys it"
              " back |",
              "| **UNDECIDABLE** | too much of what this buyer weights has never been measured"
              " against their actual alternative |",
              "| **HOLD** | nothing clears rest |",
              "",
              "## Verdict ladder", "", "| w_eff | verdict | means |", "|---|---|---|"]
    for label, floor, blurb in LADDER:
        bound = "−∞" if floor == float("-inf") else f"{floor:+.2f}"
        lines.append(f"| ≥ {bound} | **{label}** | {blurb} |")
    lines += [
        "",
        "## Regenerate",
        "",
        "```bash",
        "python tools/lightgap_scorecard.py                      # the sphere",
        "python tools/lightgap_scorecard.py --dents --unrun      # the honest parts",
        "python tools/lightgap_scorecard.py --markdown-dir docs/lightgap-scorecard",
        "```",
        "",
    ]
    return "\n".join(lines)


def _md_model(sc: Scorecard) -> str:
    doc = (__doc__ or "").split(
        "==========================================================================="
    )
    return "\n".join(
        [
            "# The lightgap model",
            "",
            "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
            "",
            doc[0].strip(),
            "",
            "## Two anchors",
            "",
            "| symbol | meaning |",
            "|---|---|",
            "| `N` | the **next-best option** — what a competent team does today without fak."
            " This is the origin; you are already here and it costs nothing. |",
            "| `c` | the **ceiling** — the best possible for anyone, ever, on this axis."
            " Derived, and labeled `physical` / `definitional` / `lower-bound`. |",
            "| `F` | where fak actually measures, with committed provenance. |",
            "",
            "```",
            "beta  = (F - N) / (c - N)                    0 = the alternative, 1 = physics",
            "w     = artanh(beta)                         the lightgap score, in nats",
            "load  = (fak_hours - alt_hours) / tolerance  signed share of patience spent",
            "tau   = artanh(load)                         the tax, on the same scale",
            "w_net = w - tau                              the verdict",
            "```",
            "",
            "`beta` needs no direction flag: for a lower-is-better metric the ceiling sits"
            " *below* the alternative, both differences flip sign, and the ratio comes out"
            " right with no special case.",
            "",
            "## Why rapidity and not a percentage",
            "",
            "1. **Unbounded and signed.** `w ∈ (−∞, +∞)`. Being worse than the alternative is"
            " a negative score, not a low one.",
            "2. **Additive.** Relativistic velocities compose by a messy formula; rapidities"
            " just add. So layered gains add — and the adoption cost subtracts — on one scale.",
            "3. **It diverges at the ceiling.** Closing the last 1% of the gap to physics is"
            " unboundedly harder than the first 50%. A system that is the only thing that does"
            " X scores arbitrarily high. That is what a moat *is*.",
            "4. **Zero at the alternative.** \"No better than what you already have\" is 0,"
            " not 50.",
            "",
            "## The tax is differential",
            "",
            "`load` subtracts the **alternative's** adoption cost, because the alternative is"
            " not always free either. Reaching zero attack-success with a formal-isolation"
            " defense means restructuring your agent; reaching it with fak means a wrapper."
            " Same number on the axis, very different `load` — and that difference, not the"
            " ASR, is the actual reason to switch. Conversely, an alternative that costs more"
            " than the segment's entire tolerance is not *available* to that buyer, so it"
            " cannot be their next-best option: the engine refuses it (`ALT_UNAFFORDABLE`)"
            " rather than let fak win against something nobody would deploy.",
            "",
            "## Three anchor shapes the naive form gets wrong",
            "",
            "| mode | when | what changes |",
            "|---|---|---|",
            "| `pure_tax` | the alternative **is** the ceiling — you already run the best"
            " thing and fak can only take a cut | `β = (F−N)/│N│`, so the cell can never score"
            " positive on the axis. That is the honest shape of a tax. |",
            "| `parity_at_ceiling` | fak **and** the next-best both sit at the limit | `β = 0`."
            " No advantage on the axis; any reason to switch must come from the differential"
            " adoption cost. |",
            "| ceiling override | the ceiling is per-hardware (a bandwidth roofline) | the cell"
            " states its own `c` and derivation. |",
            "",
            "## Claim caps — you cannot outrun your own evidence",
            "",
            "| condition | cap |",
            "|---|---|",
            "| `MODELED` / `PROJECTED` provenance | at most `CRUISE` |",
            "| `OBSERVED` provenance | at most `RELATIVISTIC` |",
            "| `lower-bound` ceiling | at most `RELATIVISTIC` |",
            "",
            "`w_net` (raw) is always reported; `w_eff` is the capped value decisions use. You"
            " cannot claim a category-defining lead on a simulated corpus, and you cannot"
            " claim to be near light-speed when all you know is where the current record sits.",
            "",
            "## Why there is no overall score",
            "",
            "A mean over the sphere is precisely the lie this scorecard exists to avoid: it"
            " lets a spike on one axis pay for a dent on another that a given buyer actually"
            " cares about. So each use case reports its **pull** (best cell — the real reason"
            " to adopt) and its **drag** (worst cell — what you eat to get it). A material"
            " axis that is `REGRESSIVE` **blocks** adoption regardless of pull. And a use case"
            " whose deciding comparison has never been run comes back **UNDECIDABLE** instead"
            " of being quietly scored on whichever cells happen to exist.",
            "",
            "## Honesty gates (`lightgap_debt`)",
            "",
            "| code | refuses |",
            "|---|---|",
            "| `NO_FAK_SOURCE` | a fak value with no committed artifact |",
            "| `NO_ALT_SOURCE` / `NO_ALT_NAMED` | a next-best value that is a guess |",
            "| `NAIVE_BASELINE` | a win measured against a strawman |",
            "| `NO_COST_BASIS` | an adoption cost with no measured basis |",
            "| `CEILING_BREACHED` | `β > 1` — the ceiling is wrong, not the result |",
            "| `DEGENERATE_CEILING` | the ceiling equals the alternative and the cell did not"
            " declare `pure_tax` |",
            "| `ALT_UNAFFORDABLE` | an alternative this buyer could never deploy |",
            "| `NO_DERIVATION` | a ceiling asserted without a derivation |",
            "| `UNFENCED_MODELED_LEAD` | a large win on `MODELED`/`PROJECTED` evidence, no"
            " fence |",
            "| `UNCOVERED` | a facet a buyer materially weights, with no cell — the work-list |",
            "| `IMMATERIAL_CELL` | a cell scored for a facet the segment weights 0 |",
            "",
        ]
    )


def _md_ceilings(sc: Scorecard) -> str:
    lines = [
        "# Ceilings — 'the speed of light in a vacuum', per facet",
        "",
        "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
        "",
        "Each facet's `c` is the best that is possible for **anyone**, not the best fak or any"
        " competitor has achieved. Three kinds:",
        "",
        "- **physical** — a derived limit (bandwidth roofline, information content).",
        "- **definitional** — the metric's own bound (zero attack success, zero lost work).",
        "- **lower-bound** — the best system known; the true ceiling is at least this and"
        " probably higher. Cells against one are capped at `RELATIVISTIC`.",
        "",
    ]
    for fid, f in sc.facets.items():
        c = sc.ceilings.get(fid)
        if not c:
            continue
        lines += [
            f"## {f['name']}",
            "",
            f"*{f['question']}*",
            "",
            f"- **metric** — {c['metric']} ({c['unit']}, {c['direction']}-is-better)",
            f"- **c** — `{c['c']:g}` ({c['kind']})",
            f"- **derivation** — {c['derivation']}",
        ]
        if c.get("caveat"):
            lines.append(f"- **caveat** — {c['caveat']}")
        lines.append("")
    return "\n".join(lines)


def _dent_split(bad: list[Cell]) -> str:
    """Which of the two failure modes the dents actually are.

    Worth stating outright rather than leaving the reader to tally the table,
    because the two modes imply completely different work: an axis loss is an
    engineering problem, an adoption-cost loss is a surface-area problem. When
    every dent falls on one side, that IS the finding.
    """
    axis = [c for c in bad if c.beta < 0]
    tax = [c for c in bad if c.beta >= 0]
    if not bad:
        return "No net-negative cells on the current board."
    if not tax:
        return (
            f"**On this board all {len(axis)} dents are the first kind.** Not one negative cell"
            " is a case of a real gain being eaten by the adoption surface — every one of them"
            " is fak being genuinely behind on the axis. That is the more expensive kind to"
            " fix: trimming verbs or shortening docs would not move a single one of them."
        )
    if not axis:
        return (
            f"**On this board all {len(tax)} dents are the second kind.** Every negative cell is"
            " a real gain the adoption surface ate. None of them is a case of fak being behind"
            " on the axis, so all of them are reachable by cutting surface area."
        )
    return (
        f"**{len(axis)} of {len(bad)} dents are axis losses; {len(tax)} are the adoption surface"
        " eating a real gain.** The first group needs engineering, the second needs less of it."
    )


def _md_dents(sc: Scorecard) -> str:
    bad = sorted([c for c in sc.cells if c.w_eff < 0], key=lambda c: c.w_eff)
    lines = [
        "# Dents — every cell where fak loses",
        "",
        "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
        "",
        f"**{len(bad)} of {len(sc.cells)} scored cells are net-negative.** A scorecard with no"
        " dents would not be measuring anything.",
        "",
        "Two distinct failure modes, and the distinction is the whole point: `β < 0` means fak"
        " is worse **on the axis itself** — no amount of easier onboarding fixes that. `β > 0`"
        " with `w_net < 0` means the gain is real but the adoption surface eats it — that one"
        " is fixed by cutting the surface, not by shipping more features.",
        "",
        _dent_split(bad),
        "",
        "| w_eff | use case × facet | verdict | cause | vs |",
        "|---|---|---|---|---|",
    ]
    for c in bad:
        cause = "**worse on the axis**" if c.beta < 0 else "adoption cost eats the gain"
        lines.append(
            f"| `{c.w_eff:+.2f}` | {c.segment} × {c.facet} | {c.verdict} | {cause} |"
            f" {c.alt_name} |"
        )
    lines.append("")
    for c in bad:
        lines += [
            f"### {c.segment} × {c.facet} (`{c.w_eff:+.2f}`)",
            "",
            f"- **β** `{c.beta:+.3f}` — fak `{c.fak_value:g}` vs next-best `{c.alt_value:g}`"
            f" ({c.alt_name}), ceiling `{c.ceiling:g}`",
            f"- **load** `{c.load:+.2f}` — {c.cost_hours:g} h fak vs {c.alt_cost_hours:g} h"
            f" alternative, against a {sc.segments[c.segment]['tolerance_hours']:g} h tolerance",
            f"- **provenance** {c.provenance}" + (f" — {c.fak_source}" if c.fak_source else ""),
        ]
        if c.note:
            lines.append(f"- {c.note}")
        if c.fence:
            lines.append(f"- **Fence:** {c.fence}")
        lines.append("")
    return "\n".join(lines)


def _md_unrun(sc: Scorecard) -> str:
    lines = [
        "# Never measured — the comparisons that would actually decide it",
        "",
        "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
        "",
        "Each row is a facet a buyer materially weights where fak has **no head-to-head**"
        " against the thing that buyer would otherwise use. These are not gaps in fak; they"
        " are gaps in the evidence, and they are the highest-value benchmark work in the repo"
        " — every one of them is currently deciding a purchase by default.",
        "",
        "| weight | use case × facet | why it is missing | next |",
        "|---|---|---|---|",
    ]
    for u in sorted(sc.unrun, key=lambda u: -u["weight"]):
        lines.append(
            f"| {u['weight']:.2f} | {u['segment']} × {u['facet']} | {u['why']} |"
            f" {u['next_action']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_segment(sc: Scorecard, sid: str) -> str:
    seg, p = sc.segments[sid], sc.segment_profile(sid)
    lines = [
        f"# {seg['name']} — {p['verdict']}",
        "",
        "<!-- GENERATED by tools/lightgap_scorecard.py — do not hand-edit. -->",
        "",
        seg["description"],
        "",
        f"- **Next best option overall** — {seg['next_best_summary']}",
        f"- **Tolerance** — {seg['tolerance_hours']:g} engineer-hours of unfamiliarity before"
        " this buyer walks away. Every cell's `load` is measured against it.",
        f"- **Switch bar** — `{p['switch_bar']:+.2f}`: the `w_eff` one axis must clear before"
        " switching is rational.",
        f"- **Verdict** — **{p['verdict']}**. {p['because']}",
        "",
        "| facet | weight | w_eff | verdict | β | load | vs |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sc.by_segment(sid):
        lines.append(
            f"| {sc.facets[c.facet]['name']} | {c.weight:.2f} | `{c.w_eff:+.2f}` |"
            f" **{c.verdict}** | `{c.beta:+.2f}` | `{c.load:+.2f}` | {c.alt_name} |"
        )
    for u in [x for x in sc.unrun if x["segment"] == sid]:
        lines.append(
            f"| {sc.facets[u['facet']]['name']} | {u['weight']:.2f} | ? | **NEVER MEASURED** |"
            " – | – | – |"
        )
    lines.append("")
    for c in sc.by_segment(sid):
        f = sc.facets[c.facet]
        lines += [
            f"## {f['name']} — `{c.w_eff:+.2f}` ({c.verdict})",
            "",
            f"*{f['question']}*",
            "",
            f"- **fak** `{c.fak_value:g}` — {c.provenance}"
            + (f", {c.fak_source}" if c.fak_source else ""),
            f"- **next best** `{c.alt_value:g}` — {c.alt_name}"
            + (f", {c.alt_source}" if c.alt_source else ""),
            f"- **ceiling** `{c.ceiling:g}` ({c.ceiling_kind})",
            f"- **β** `{c.beta:+.3f}` → `w` `{c.w:+.3f}`"
            + (f"  ·  mode `{c.mode}`" if c.mode != "standard" else ""),
            f"- **adoption** {c.cost_hours:g} h fak vs {c.alt_cost_hours:g} h alternative, of"
            f" {seg['tolerance_hours']:g} h → load `{c.load:+.2f}` → `τ` `{c.tau:+.3f}`",
            f"- **w_net** `{c.w_net:+.3f}`"
            + (f" → **capped at {c.cap}** ({c.cap_reason}) → `w_eff` `{c.w_eff:+.3f}`"
               if c.cap else ""),
            f"- **cost basis** {'; '.join(c.cost_basis)}",
        ]
        if c.note:
            lines.append(f"- **Read** {c.note}")
        if c.fence:
            lines.append(f"- **Fence** {c.fence}")
        lines.append("")
    for u in [x for x in sc.unrun if x["segment"] == sid]:
        lines += [
            f"## {sc.facets[u['facet']]['name']} — never measured (weight {u['weight']:.2f})",
            "",
            u["why"],
            "",
            f"**Next:** {u['next_action']}",
            "",
        ]
    return "\n".join(lines)


def write_markdown(sc: Scorecard, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    files = {
        "README.md": _md_index(sc),
        "model.md": _md_model(sc),
        "ceilings.md": _md_ceilings(sc),
        "dents.md": _md_dents(sc),
        "unrun.md": _md_unrun(sc),
    }
    for sid in sc.segments:
        if sc.by_segment(sid) or any(u["segment"] == sid for u in sc.unrun):
            files[f"segment-{sid}.md"] = _md_segment(sc, sid)
    written = []
    for name, body in files.items():
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body.rstrip() + "\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lightgap scorecard — unbounded usability per use case")
    ap.add_argument("--json", action="store_true", help="machine payload")
    ap.add_argument("--segment", help="one use case, in full")
    ap.add_argument("--facet", help="one facet across all use cases")
    ap.add_argument("--dents", action="store_true", help="every cell where fak loses")
    ap.add_argument("--unrun", action="store_true", help="comparisons that would decide it")
    ap.add_argument("--ceilings", action="store_true", help="the c anchors and derivations")
    ap.add_argument("--check", action="store_true", help="honesty gate; exit 1 on debt")
    ap.add_argument("--markdown-dir", help="regenerate the doc folder")
    args = ap.parse_args(argv)

    sc = Scorecard()

    if args.json:
        print(json.dumps(sc.payload(), indent=2))
        return 0
    if args.markdown_dir:
        for p in write_markdown(sc, args.markdown_dir):
            print(f"wrote {os.path.relpath(p, REPO_ROOT)}")
        return 0
    if args.check:
        print(render_check(sc))
        return 1 if sc.debt else 0

    shown = False
    for flag, fn in (
        (args.ceilings, render_ceilings),
        (args.dents, render_dents),
        (args.unrun, render_unrun),
    ):
        if flag:
            print(fn(sc))
            shown = True
    if args.segment:
        print(render_segment(sc, args.segment))
        shown = True
    if args.facet:
        print(render_facet(sc, args.facet))
        shown = True
    if not shown:
        print(render(sc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
