#!/usr/bin/env python3
"""conflation_scorecard  -  the anti-conflation / provenance-honesty stick.

The lesson this generalizes (from the compaction-metrics fix): every number or status a
surface reports should declare its PROVENANCE  -  is it a fact fak AUTHORED/WITNESSED, or a
value fak OBSERVED from an external party and merely RELAYS?  -  and a bad OBSERVED value must
never be attributed to a fak ACTION unless a WITNESSED signal proves the fault is fak's.

Conflating the two is a truth-maintenance bug: it makes a provider-side miss (a cache TTL
expiry, an eviction, a client moving its own breakpoint) read as "fak broke the cache", which
erodes trust in the one number fak can actually stand behind. This scorecard reads the
fact-REPORTING surfaces (Prometheus metric help strings, the `fak guard` exit summary, other
operator-facing summaries) and scores whether they keep that line.

It is a TREE-READING scorecard (no data dir): the rendered fact-strings ARE the data, read
from the Go source, so it cannot be gamed by editing a JSON file  -  only by fixing the prose.

KPIs (pure functions over the extracted fact-strings):
  - provenance_labeled (HARD): a help/summary string that reports an EXTERNAL/relayed value
    (a provider counter, upstream-reported usage) must carry an OBSERVED/provider qualifier;
    a self-authored counter should carry a WITNESSED/authored qualifier. Each unlabeled
    external value is one debt.
  - no_false_attribution (HARD): no help/summary prose may attribute a bad OBSERVED value to
    a fak ACTION ("X broke", "the splice is producing", "we re-bill") UNLESS a co-located
    witnessed qualifier disambiguates it. Each violation is one debt.
  - fault_signal_isolated (SOFT): a metric family that mixes witnessed + observed should name
    exactly one fault signal the help points at (the "only X>0 is our bug" pattern).

Usage:
    python tools/conflation_scorecard.py                 # terminal work-list
    python tools/conflation_scorecard.py --json          # control-pane payload
    python tools/conflation_scorecard.py --compare b.json # prove the debt moved
    python tools/conflation_scorecard.py --markdown      # committed snapshot body
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# The fact-reporting surfaces this scorecard gardens. Each is a Go file that renders
# operator-facing numbers/statuses (metric help, exit summaries). Add a surface here when a
# new reporting file lands; the scorecard then holds it to the same provenance discipline.
REPORTING_SURFACES = [
    "internal/gateway/metrics.go",
    "cmd/fak/guard.go",
]

# Tokens that mark a reported value as coming from an EXTERNAL party fak relays (provider,
# upstream, remote). If a help/summary string mentions one of these, it is reporting an
# OBSERVED value and must label it as such.
EXTERNAL_VALUE_TOKENS = [
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "provider cache",
    "upstream",
    "remote prompt cache",
    "provider-reported",
    "provider billed",
    "what the provider billed",
]

# Provenance qualifiers. A string reporting an external value is honestly labeled if it
# carries one of these (the OBSERVED side); a self-authored counter the WITNESSED side.
OBSERVED_QUALIFIERS = ["OBSERVED", "provider-reported", "relayed", "relays", "not a fak claim",
                       "not a claim about what the provider", "attribute nothing to itself",
                       "attributes nothing to itself", "the provider's call", "the provider's number",
                       # honest phrasings already in the tree that genuinely scope a value as
                       # external/provider-side without the literal OBSERVED token  -  recognizing
                       # these is not weakening the detector, it is seeing real honesty:
                       "provider-side", "performance evidence", "distinct from the local",
                       "distinct from local"]
WITNESSED_QUALIFIERS = ["WITNESSED", "fak authored", "fak SENT", "what fak SENT", "byte-identical"]

# Attribution phrases that BLAME a fak action for an outcome. Applied to an external value
# WITHOUT a co-located disambiguating qualifier, each is a false-attribution defect. The
# phrases are deliberately the ones the original conflation used.
ATTRIBUTION_PHRASES = [
    r"the cache broke",
    r"broke the cache",
    r"the splice is producing a body the provider re-bills",
    r"the splice broke",
    r"every fire is re-billing",
    r"compaction (?:is )?costing money",
    r"means the cache broke",
]

# A disambiguating qualifier near an attribution phrase makes it honest (it is describing,
# not asserting, the failure  -  e.g. "Reading the crater as 'the cache broke' is the conflation
# this prevents"). These markers neutralize an attribution phrase on the same string.
DISAMBIGUATION_MARKERS = [
    "conflation", "is the conflation", "not something fak", "does NOT control",
    "does not control", "is NOT that bug", "is not that bug", "reading the crater",
    "reading a low", "unless", "provider-side", "only bail_reason", "the ONLY fak-fault",
]

CLEAN_FLOOR = 0  # the disciplined tree emits zero conflation debt; the smoke test pins this.


# ---- fact extraction (the impure shell, kept thin) --------------------------------------

def _read(path: str) -> str:
    p = REPO / path
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def extract_help_strings(text: str) -> list[str]:
    """Pull the operator-facing fact-strings out of a reporting source file: the help
    arguments of writeHelpType/writeCounter and the format strings of the exit-summary
    Fprintf calls. These are the strings a human reads, so they are what we grade."""
    strings: list[str] = []
    # writeHelpType(b, "name", "HELP...", "type")  and  writeCounter(b, "name", "HELP...", n)
    for m in re.finditer(r'write(?:HelpType|Counter)\(\s*[^,]+,\s*"[^"]*",\s*"((?:[^"\\]|\\.)*)"',
                         text):
        strings.append(_unescape(m.group(1)))
    # fmt.Fprintf(&b, "fak guard: ... %d ...", ...)   -  operator exit-summary lines.
    for m in re.finditer(r'Fprintf\(\s*&?\w+\s*,\s*"((?:[^"\\]|\\.)*)"', text):
        s = _unescape(m.group(1))
        if "fak guard:" in s or "compaction" in s.lower() or "cache" in s.lower():
            strings.append(s)
    return strings


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")


# ---- KPIs (pure) -------------------------------------------------------------------------

def _has_any(s: str, needles: list[str]) -> bool:
    low = s.lower()
    return any(n.lower() in low for n in needles)


def kpi_provenance_labeled(surfaces: dict[str, list[str]]) -> dict[str, Any]:
    """Every help/summary string that reports an EXTERNAL value must carry an OBSERVED-side
    qualifier so a reader knows fak is relaying, not asserting, that number."""
    defects: list[str] = []
    total_external = 0
    for path, strings in surfaces.items():
        sid = path.split("/")[-1]
        for s in strings:
            if _has_any(s, EXTERNAL_VALUE_TOKENS):
                total_external += 1
                if not _has_any(s, OBSERVED_QUALIFIERS):
                    defects.append(f"{sid}: external value reported without an OBSERVED/"
                                   f"provider-relayed qualifier: \"{_clip(s)}\"")
    score = 100.0 if not defects else max(0.0, 100.0 * (1 - len(defects) / max(total_external, 1)))
    return {"kpi": "provenance_labeled", "group": "honesty", "score": score,
            "detail": f"{total_external - len(defects)}/{total_external} external-value strings "
                      f"carry a provenance label",
            "defects": defects, "soft": []}


def kpi_no_false_attribution(surfaces: dict[str, list[str]]) -> dict[str, Any]:
    """No help/summary prose may blame a fak ACTION for a bad OBSERVED value unless a
    co-located qualifier disambiguates it (describing the failure, not asserting it)."""
    defects: list[str] = []
    for path, strings in surfaces.items():
        sid = path.split("/")[-1]
        for s in strings:
            for pat in ATTRIBUTION_PHRASES:
                if re.search(pat, s, re.IGNORECASE):
                    if not _has_any(s, DISAMBIGUATION_MARKERS):
                        defects.append(f"{sid}: attributes an observed miss to a fak action "
                                       f"with no disambiguation: \"{_clip(s)}\"")
                    break
    return {"kpi": "no_false_attribution", "group": "honesty",
            "score": 100.0 if not defects else 0.0,
            "detail": "no observed-miss-blamed-on-fak prose" if not defects
                      else f"{len(defects)} false-attribution string(s)",
            "defects": defects, "soft": []}


def kpi_fault_signal_isolated(surfaces: dict[str, list[str]], sources: dict[str, str]) -> dict[str, Any]:
    """SOFT: a metric family that mixes witnessed + observed values should name exactly one
    fault signal the help points at (the 'only X>0 is our bug' pattern), so a reader knows
    which single signal means the fault is genuinely fak's."""
    soft: list[str] = []
    # Heuristic: if a surface renders BOTH an OBSERVED external value AND a WITNESSED counter,
    # its prose should name a single fak-fault signal (e.g. "only ... prefix_mismatch ... is").
    for path, strings in surfaces.items():
        sid = path.split("/")[-1]
        blob = " ".join(strings)
        has_observed = _has_any(blob, EXTERNAL_VALUE_TOKENS)
        has_witnessed = _has_any(blob, WITNESSED_QUALIFIERS)
        if has_observed and has_witnessed:
            names_fault = bool(re.search(r"only\b.{0,40}\bis\s+(?:fak's|the\s+\w+\s+)?bug", blob, re.I)) \
                or "fak-fault" in blob.lower() or "prefix_mismatch" in blob.lower()
            if not names_fault:
                soft.append(f"{sid}: mixes WITNESSED + OBSERVED values but names no single "
                            f"fak-fault signal  -  a reader can't tell which signal means our bug")
    return {"kpi": "fault_signal_isolated", "group": "honesty",
            "score": 100.0 if not soft else 70.0,
            "detail": "fault signal isolated where families mix provenance" if not soft
                      else f"{len(soft)} mixed family without an isolated fault signal",
            "defects": [], "soft": soft}


def _clip(s: str, n: int = 90) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "..."


# ---- fold + render -----------------------------------------------------------------------

GROUP_WEIGHTS = {"honesty": 1.0}


def grade_letter(score: float) -> str:
    if score >= 95: return "A"
    if score >= 85: return "B"
    if score >= 75: return "C"
    if score >= 60: return "D"
    return "F"


def run() -> dict[str, Any]:
    sources = {p: _read(p) for p in REPORTING_SURFACES}
    surfaces = {p: extract_help_strings(t) for p, t in sources.items()}
    kpis = [
        kpi_provenance_labeled(surfaces),
        kpi_no_false_attribution(surfaces),
        kpi_fault_signal_isolated(surfaces, sources),
    ]
    composite = sum(k["score"] for k in kpis) / len(kpis)
    debt = sum(len(k["defects"]) for k in kpis)
    grade = grade_letter(composite)
    verdict = "OK" if debt == 0 else "ACTION"
    finding = ("every reported fact labels its provenance and blames no provider-side miss on fak"
               if debt == 0 else
               f"{debt} conflation defect(s): a reported value is unlabeled or a provider miss is "
               f"attributed to a fak action")
    worst = max(kpis, key=lambda k: len(k["defects"]))
    next_action = ("hold  -  re-run after a new reporting surface lands" if debt == 0 else
                   f"fix {worst['kpi']}: label external values OBSERVED / correct the attribution")
    return {
        "schema": "fak-conflation-scorecard/1",
        "ok": debt == 0,
        "verdict": verdict,
        "finding": finding,
        "reason": "; ".join(d for k in kpis for d in k["defects"]) or "clean",
        "next_action": next_action,
        "corpus": {"score": round(composite, 1), "grade": grade,
                   "conflation_debt": debt,
                   "surfaces": len(REPORTING_SURFACES),
                   "external_values_seen": sum(1 for p in surfaces for s in surfaces[p]
                                               if _has_any(s, EXTERNAL_VALUE_TOKENS))},
        "kpis": kpis,
    }


def render(payload: dict[str, Any]) -> str:
    c = payload["corpus"]
    out = [f"conflation scorecard: {c['grade']} (score {c['score']}, conflation_debt {c['conflation_debt']})",
           f"  {payload['finding']}", ""]
    for k in payload["kpis"]:
        mark = "ok " if not k["defects"] and not k["soft"] else "!! " if k["defects"] else "~  "
        out.append(f"  {mark}{k['kpi']:<22} {k['score']:5.1f}  {k['detail']}")
        for d in k["defects"]:
            out.append(f"       HARD  {d}")
        for s in k["soft"]:
            out.append(f"       soft  {s}")
    out.append("")
    out.append(f"  next: {payload['next_action']}")
    return "\n".join(out)


def markdown(payload: dict[str, Any]) -> str:
    c = payload["corpus"]
    lines = [
        "# Conflation / provenance-honesty scorecard",
        "",
        "_Auto-generated by `tools/conflation_scorecard.py`. Do not hand-edit; re-run the tool._",
        "",
        f"**Grade {c['grade']}** - score {c['score']} - conflation_debt **{c['conflation_debt']}** - "
        f"{c['surfaces']} reporting surface(s) - {c['external_values_seen']} external-value string(s)",
        "",
        "The law: every reported number/status labels its provenance  -  WITNESSED (a fact fak "
        "authored) vs OBSERVED (a value fak relays from an external party)  -  and a bad OBSERVED "
        "value is never blamed on a fak action unless a witnessed signal proves the fault is fak's.",
        "",
        "| KPI | score | detail |",
        "|---|---|---|",
    ]
    for k in payload["kpis"]:
        lines.append(f"| {k['kpi']} | {k['score']:.0f} | {k['detail']} |")
    work = [d for k in payload["kpis"] for d in k["defects"]]
    if work:
        lines += ["", "## Work list (HARD)", ""]
        lines += [f"- {d}" for d in work]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="anti-conflation / provenance-honesty scorecard")
    ap.add_argument("--json", action="store_true", help="emit the control-pane payload")
    ap.add_argument("--compare", metavar="BASELINE", help="print the debt delta vs a prior --json")
    ap.add_argument("--markdown", action="store_true", help="emit the committed snapshot body")
    args = ap.parse_args(argv)

    payload = run()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    if args.markdown:
        print(markdown(payload))
        return 0
    if args.compare:
        try:
            prior = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            pd = prior.get("corpus", {}).get("conflation_debt")
        except (OSError, json.JSONDecodeError):
            pd = None
        cur = payload["corpus"]["conflation_debt"]
        print(render(payload))
        if pd is not None:
            delta = cur - pd
            verdict = "improved" if delta < 0 else "regressed" if delta > 0 else "flat"
            print(f"\n  compare: conflation_debt {pd} -> {cur} ({verdict} by {abs(delta)})")
        return 0 if payload["ok"] else 1
    print(render(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
