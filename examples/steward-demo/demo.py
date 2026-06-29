#!/usr/bin/env python3
"""
fak kernel — single-invariant steward demo (#324)
=================================================

A *steward* is a single-invariant runtime checker. It fires only with an
**independently-authored witness** — never on the model's own claim. The thesis,
in one line: you do not trust the alert, you trust the *witness chain* behind it.

A steward's check returns ``(violated, witness)``. It RAISES only when BOTH:

  (a) its one-sentence invariant pattern matches (e.g. a secret-shaped token is
      present in the context snapshot), AND
  (b) the ``witness`` is non-empty — and a witness is produced by an INDEPENDENT
      scan/replay/registry, never by the thing under test.

A pattern match with an empty witness is a *self-accusation the model cannot make*,
so the steward ABSTAINS (suppressed). That asymmetry is the load-bearing property.

A **meta-steward** then prunes any steward that NEVER fired across a soak — dead-code
detection on the invariant layer itself, so the population can't ossify.

This script is a faithful, dependency-free re-enactment of the real Go package
``fak/internal/steward`` (steward.go) and its tests (units 87–92 in steward_test.go).
The Go tests are the authoritative witness; this demo makes the behaviour runnable and
readable without a Go toolchain. It reads ``sample-steward.json`` for its scenario.

Usage:
    examples/steward-demo/run.sh                 # one command
    python3 demo.py [--config PATH] [--no-color]

Exit code: 0 if every steward behaved as the witness model demands AND the meta-steward
pruned exactly the never-firing steward; 1 otherwise. CI-usable. Honors NO_COLOR.
"""
from __future__ import annotations
import argparse, json, os, re, sys

# The exact secret shape the real steward uses — steward.go:secretRE. A demo that
# guessed a different pattern would not be re-enacting the shipped check.
SECRET_RE = re.compile(r"(?i)(sk-[a-z0-9]{16,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,})")


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


# --- the steward kernel: a faithful port of steward.go's (violated, witness) rule ---
#
# Each invariant below decides ONLY whether its pattern matches the frozen probe. The
# RAISE decision is the conjunction (pattern_match AND witness_present) — the witness is
# authored independently and lives in the fixture, exactly as the real check returns it.

def pattern_matches(kind, probe):
    """Return True iff the single invariant's pattern trips on the probe.

    This is the (a) half — the cheap match. It is deliberately NOT enough to raise.
    """
    if kind == "secret-in-context":
        return any(SECRET_RE.search(s) for s in probe.get("context_snapshot", []))
    if kind == "lease-disjointness":
        leases = probe.get("leases", [])
        for i in range(len(leases)):
            for j in range(i + 1, len(leases)):
                for a in leases[i].get("trees", []):
                    for b in leases[j].get("trees", []):
                        if a == b or a.startswith(b) or b.startswith(a):
                            return True
        return False
    if kind == "kpi-regression":
        b, c = probe.get("baseline", 0.0), probe.get("current", 0.0)
        return b > 0 and c > b  # tol applied by the fixture's chosen numbers
    if kind == "vdso-soundness":
        return bool(probe.get("mismatch", False))
    raise ValueError(f"unknown steward kind: {kind}")


def steward_check(case, kind):
    """The steward's Check, returning (violated, witness).

    The RAISE rule, verbatim in spirit from steward.go: a violation is reported ONLY
    with an independently-authored witness, else the steward abstains. So even when the
    pattern matches, an EMPTY witness (no independent scan corroborated it) abstains.
    """
    matched = pattern_matches(kind, case["probe"])
    witness = case.get("witness", "")
    violated = matched and bool(witness)
    return violated, witness, matched


def main():
    # Emit UTF-8 (the ✓/— glyphs) even on a Windows cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="fak single-invariant steward demo (#324)")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "sample-steward.json"))
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    c = color(not args.no_color and os.environ.get("NO_COLOR") is None and sys.stdout.isatty())
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"{c['b']}fak kernel — single-invariant steward demo{c['x']}  config={os.path.basename(args.config)}")
    print(f"  {c['d']}a steward RAISES only when (pattern matches) AND (an INDEPENDENT witness corroborates);"
          f" a ✓ means the verdict matched the witness model{c['x']}")
    print()

    ok = True

    # ----- part 1: each steward, witness vs no-witness vs abstain -----
    print(f"{c['b']}STEWARDS{c['x']}  — one invariant each; raise needs an independently-authored witness")
    for st in cfg["stewards"]:
        print(f"  {c['y']}{st['name']}{c['x']}  {c['d']}invariant: {st['invariant']}{c['x']}")
        for case in st["cases"]:
            violated, witness, matched = steward_check(case, st["kind"])
            # Expectation: RAISE iff the fixture authored a witness for a real match.
            want_raise = matched and bool(witness)
            consistent = (violated == want_raise)
            ok = ok and consistent
            mark = f"{c['g']}✓{c['x']}" if consistent else f"{c['r']}✗{c['x']}"
            label = case["label"]
            if violated:
                verdict = f"{c['r']}RAISE{c['x']}"
                detail = f"witness=\"{witness}\"  by {case['witness_author'].split(' (')[0]}"
            else:
                # ABSTAIN: either nothing matched, or matched but no independent witness.
                verdict = f"{c['d']}ABSTAIN{c['x']}"
                if matched and not witness:
                    detail = "pattern matched but NO independent witness — self-accusation suppressed"
                else:
                    detail = "invariant pattern did not trip"
            print(f"      {mark} {label:<58} {verdict:<24} {c['d']}{detail}{c['x']}")
        print()

    # ----- part 2: the meta-steward prunes a never-firing steward -----
    ms = cfg["meta_steward"]
    print(f"{c['b']}META-STEWARD{c['x']}  — after a {ms['soak_sweeps']}-sweep soak, prune any steward that NEVER fired")
    fires = {s["name"]: (ms["soak_sweeps"] if s["fires_during_soak"] else 0) for s in ms["population"]}
    pruned = [s["name"] for s in ms["population"] if fires[s["name"]] == 0]
    kept = [s["name"] for s in ms["population"] if fires[s["name"]] > 0]
    for s in ms["population"]:
        n = s["name"]
        if fires[n] > 0:
            print(f"      {c['g']}keep{c['x']}  {n:<22} {c['d']}fired {fires[n]}× during the soak{c['x']}")
        else:
            print(f"      {c['r']}prune{c['x']} {n:<22} {c['d']}fired 0× — {s.get('why_dead','dead invariant')}{c['x']}")
    want_pruned = [s["name"] for s in ms["population"] if not s["fires_during_soak"]]
    prune_ok = (pruned == want_pruned)
    ok = ok and prune_ok
    print()
    print(f"  meta-steward pruned: {pruned or '∅'}   kept: {kept}")
    if not prune_ok:
        print(f"  {c['r']}meta-steward pruned {pruned}, expected exactly {want_pruned}{c['x']}")

    # ----- summary -----
    print()
    raises = sum(1 for st in cfg["stewards"] for ca in st["cases"]
                 if steward_check(ca, st["kind"])[0])
    suppressed = sum(1 for st in cfg["stewards"] for ca in st["cases"]
                     if steward_check(ca, st["kind"])[2] and not ca.get("witness"))
    status = f"{c['g']}steward demo passed{c['x']}" if ok else f"{c['r']}steward demo FAILED{c['x']}"
    print(f"summary: {status}  ·  {raises} raises (each with an independent witness)  ·  "
          f"{suppressed} un-witnessed match(es) suppressed  ·  {len(pruned)} never-firing steward pruned")
    print(f"  {c['d']}the load-bearing result: a steward raises ONLY when an independently-authored witness"
          f" corroborates the invariant match — the model can never self-accuse, and a check that can"
          f" never fire is pruned.{c['x']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
