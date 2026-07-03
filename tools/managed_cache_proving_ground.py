#!/usr/bin/env python3
"""The managed-cache proving ground — real guarded sessions as the regression floor (epic #1844).

fak's durable ledgers already record REAL traffic: every `fak guard -- claude` exit
appends an OBSERVED-$ row to ``docs/nightrun/cache-savings.jsonl``, every MCP serve
exit appends a counter snapshot to ``docs/nightrun/gateway-usage.jsonl``, and every
`fak run` kernel session with KV reuse appends a WITNESSED row to
``docs/nightrun/cache-value.jsonl``. Those ledgers are the one population where a
managed-cache lever's effect is *provable* rather than asserted — the same sessions
the FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01 note folded into the "fak share = 0.0000%
on guard" baseline.

This tool is the durable spine that turns those ledgers into a reusable proving
ground for the managed-cache concepts. It is read-only, stdlib-only, deterministic
(the report is a pure function of the ledger bytes; ``as_of`` is the max row
timestamp, never the wall clock), and it does three things:

  1. VALIDATE — every ledger row is checked against its schema contract: exact
     schema string, closed mechanism/session_type vocabulary, non-negative counters,
     and the Track-2 savings identity ``saved = 0.9*cache_read - 0.25*cache_creation``
     (internal/cachevaluereport/track2.go providerSavedTokenEquiv). A row that
     drifts is a named violation, not a silent skip.

  2. FOLD — each managed-cache concept (the #1844 lever family) is resolved to a
     rung on a closed evidence ladder, from what the real rows can prove:

         0 UNIMPLEMENTED   the lever does not exist in the tree
         1 UNWIRED         the lever exists but has NO durable witness channel
                           (an in-process /metrics counter or a test-only witness
                           dies with the session — the proving ground cannot see it)
         2 CHANNEL_READY   a durable channel exists but no real-session row has
                           reached it yet
         3 SILENT_ZERO     the writer is PROVEN live on real sessions and the
                           value is zero — a witnessed zero, not a recording gap
         4 EVIDENCED       nonzero real-session evidence rows exist

     Rung upgrades are auto-detected where possible: e.g. the moment a
     ``ttl_upgrade``-bearing counter key appears in a gateway-usage row, the C6
     concept climbs from UNWIRED without a code change here.

  3. RATCHET — ``--check`` compares the fold against the checked-in baseline
     (``tools/managed_cache_proving_ground.data/baseline.json``) and fails on any
     of a closed set of regressions:

         SCHEMA_DRIFT            a ledger's schema string changed
         REGRESSION_ROW_COUNT    an append-only ledger shrank (rows were rewritten)
         REGRESSION_RUNG         a concept fell down the evidence ladder
         REGRESSION_VIOLATIONS   row-contract violations increased

     Counts may only grow, rungs may only climb, violations may only shrink — so
     concurrent sessions appending rows keep the gate green, while a lever that
     silently stops witnessing, a ledger rewrite, or a schema drift turns it red.
     ``--write-baseline`` re-snapshots after an intentional rung climb.

Usage:
    python3 tools/managed_cache_proving_ground.py                  # human card
    python3 tools/managed_cache_proving_ground.py --json           # full report
    python3 tools/managed_cache_proving_ground.py --check          # ratchet gate
    python3 tools/managed_cache_proving_ground.py --write-baseline # re-snapshot

Exit 0 = OK. Exit 1 = ratchet regression (each named). Exit 2 = harness error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPORT_SCHEMA = "fak-managed-cache-proving-ground/1"

SAVINGS_LEDGER_REL = "docs/nightrun/cache-savings.jsonl"
USAGE_LEDGER_REL = "docs/nightrun/gateway-usage.jsonl"
VALUE_LEDGER_REL = "docs/nightrun/cache-value.jsonl"

SAVINGS_SCHEMA = "fak-cache-savings-ledger/1"
USAGE_SCHEMA = "fak-gateway-usage-ledger/1"
VALUE_SCHEMA = "fak-cache-value-ledger/1"

# The closed Track-2 mechanism vocabulary (internal/cachevaluereport/track2.go
# NewSavingsRows). A new mechanism landing there must be added here — that is the
# point: an unregistered mechanism on real traffic is a contract drift to notice.
SAVINGS_MECHANISMS = {"provider_prompt_cache", "compaction_shed"}
SESSION_TYPES = {"guard", "serve", "run"}
USAGE_KINDS = {"exit", "periodic"}

# providerSavedTokenEquiv: read*(1-0.1) + creation*(1-1.25) — the 5m-tier cache
# economics (0.1x reads, 1.25x writes) the guard-exit writer bakes into every row.
PROVIDER_READ_MULT = 0.1
PROVIDER_WRITE_MULT_5M = 1.25
FORMULA_TOLERANCE = 0.51  # one rounding ulp across the float sum

# The evidence ladder (closed; ordered).
RUNGS = ["UNIMPLEMENTED", "UNWIRED", "CHANNEL_READY", "SILENT_ZERO", "EVIDENCED"]


def rung_rank(name: str) -> int:
    return RUNGS.index(name)


# ---------------------------------------------------------------------------
# Ledger loading + row contracts
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    violations: list[str] = []
    if not path.exists():
        return rows, violations
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            violations.append(f"{path.name}:{lineno}: UNPARSEABLE_ROW")
            continue
        if not isinstance(row, dict):
            violations.append(f"{path.name}:{lineno}: NON_OBJECT_ROW")
            continue
        rows.append(row)
    return rows, violations


def _nonneg(row: dict, key: str) -> bool:
    v = row.get(key, 0)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def validate_savings_row(row: dict, where: str) -> list[str]:
    v: list[str] = []
    if row.get("schema") != SAVINGS_SCHEMA:
        v.append(f"{where}: SCHEMA_MISMATCH got {row.get('schema')!r}")
        return v
    mech = row.get("mechanism")
    if mech not in SAVINGS_MECHANISMS:
        v.append(f"{where}: UNKNOWN_MECHANISM {mech!r}")
    if row.get("session_type") not in SESSION_TYPES:
        v.append(f"{where}: UNKNOWN_SESSION_TYPE {row.get('session_type')!r}")
    for key in ("input_tokens", "cache_read_tokens", "cache_creation_tokens",
                "output_tokens", "compaction_shed_tokens"):
        if key in row and not _nonneg(row, key):
            v.append(f"{where}: NEGATIVE_COUNTER {key}")
    if mech == "provider_prompt_cache":
        read = row.get("cache_read_tokens", 0) or 0
        creation = row.get("cache_creation_tokens", 0) or 0
        want = read * (1 - PROVIDER_READ_MULT) + creation * (1 - PROVIDER_WRITE_MULT_5M)
        got = row.get("saved_token_equiv")
        if not isinstance(got, (int, float)) or abs(want - got) > FORMULA_TOLERANCE:
            v.append(f"{where}: SAVED_FORMULA_MISMATCH want {want} got {got}")
    return v


def validate_usage_row(row: dict, where: str) -> list[str]:
    v: list[str] = []
    if row.get("schema") != USAGE_SCHEMA:
        v.append(f"{where}: SCHEMA_MISMATCH got {row.get('schema')!r}")
        return v
    if row.get("kind") not in USAGE_KINDS:
        v.append(f"{where}: UNKNOWN_KIND {row.get('kind')!r}")
    if row.get("session_type") not in SESSION_TYPES:
        v.append(f"{where}: UNKNOWN_SESSION_TYPE {row.get('session_type')!r}")
    counters = row.get("counters")
    if not isinstance(counters, dict):
        v.append(f"{where}: MISSING_COUNTERS")
        return v
    for key, val in counters.items():
        if key == "by_reason":
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            v.append(f"{where}: BAD_COUNTER {key}={val!r}")
    return v


def validate_value_row(row: dict, where: str) -> list[str]:
    v: list[str] = []
    if row.get("schema") != VALUE_SCHEMA:
        v.append(f"{where}: SCHEMA_MISMATCH got {row.get('schema')!r}")
        return v
    if row.get("session_type") not in SESSION_TYPES:
        v.append(f"{where}: UNKNOWN_SESSION_TYPE {row.get('session_type')!r}")
    for key in ("turns", "prompt_tokens", "reused_tokens"):
        if not _nonneg(row, key):
            v.append(f"{where}: NEGATIVE_COUNTER {key}")
    prompt = row.get("prompt_tokens", 0) or 0
    reused = row.get("reused_tokens", 0) or 0
    if isinstance(prompt, (int, float)) and isinstance(reused, (int, float)) and reused > prompt:
        v.append(f"{where}: REUSE_EXCEEDS_PROMPT {reused} > {prompt}")
    return v


# ---------------------------------------------------------------------------
# Fold: ledger facts
# ---------------------------------------------------------------------------

def load_facts(root: Path) -> dict:
    """Read the three ledgers into the fact fold the concept rungs are decided from."""
    facts: dict = {"ledgers": {}, "violations": {}}
    as_of = ""

    path = root / SAVINGS_LEDGER_REL
    rows, viol = _load_jsonl(path)
    by_mech: dict[str, dict] = {}
    by_session: dict[str, int] = {}
    for i, row in enumerate(rows, 1):
        viol.extend(validate_savings_row(row, f"{path.name}:{i}"))
        mech = str(row.get("mechanism"))
        b = by_mech.setdefault(mech, {"rows": 0, "saved_token_equiv": 0.0, "net_saved_token_equiv": 0.0})
        b["rows"] += 1
        for key in ("saved_token_equiv", "net_saved_token_equiv"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                b[key] += val
        st = str(row.get("session_type"))
        by_session[st] = by_session.get(st, 0) + 1
        as_of = max(as_of, str(row.get("generated_at") or ""))
    facts["ledgers"]["cache_savings"] = {
        "path": SAVINGS_LEDGER_REL,
        "schema": SAVINGS_SCHEMA,
        "rows": len(rows),
        "by_mechanism": by_mech,
        "by_session_type": by_session,
    }
    facts["violations"][SAVINGS_LEDGER_REL] = viol

    path = root / USAGE_LEDGER_REL
    rows, viol = _load_jsonl(path)
    by_session = {}
    nonzero_rows = 0
    guard_rows = 0
    ttl_keys_present = False
    compaction_fired = 0
    kv_reused = 0
    for i, row in enumerate(rows, 1):
        viol.extend(validate_usage_row(row, f"{path.name}:{i}"))
        st = str(row.get("session_type"))
        by_session[st] = by_session.get(st, 0) + 1
        if st == "guard":
            guard_rows += 1
        counters = row.get("counters") if isinstance(row.get("counters"), dict) else {}
        numeric = {k: v for k, v in counters.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if any(numeric.values()):
            nonzero_rows += 1
        # Auto-detect the C6 durable channel: the moment a ttl_upgrade counter key
        # ships in the usage-ledger row shape, the concept climbs past UNWIRED here
        # without editing this tool.
        if any("ttl" in k for k in counters):
            ttl_keys_present = True
        compaction_fired += int(numeric.get("compaction_fired", 0))
        kv_reused += int(numeric.get("kv_prefix_reused_tokens", 0))
        as_of = max(as_of, str(row.get("generated_at") or ""))
    facts["ledgers"]["gateway_usage"] = {
        "path": USAGE_LEDGER_REL,
        "schema": USAGE_SCHEMA,
        "rows": len(rows),
        "by_session_type": by_session,
        "guard_rows": guard_rows,
        "nonzero_counter_rows": nonzero_rows,
        "ttl_upgrade_keys_present": ttl_keys_present,
        "compaction_fired_total": compaction_fired,
        "kv_prefix_reused_tokens_total": kv_reused,
    }
    facts["violations"][USAGE_LEDGER_REL] = viol

    path = root / VALUE_LEDGER_REL
    rows, viol = _load_jsonl(path)
    by_session = {}
    reused_total = 0
    for i, row in enumerate(rows, 1):
        viol.extend(validate_value_row(row, f"{path.name}:{i}"))
        st = str(row.get("session_type"))
        by_session[st] = by_session.get(st, 0) + 1
        val = row.get("reused_tokens")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            reused_total += int(val)
        as_of = max(as_of, str(row.get("generated_at") or ""))
    facts["ledgers"]["cache_value"] = {
        "path": VALUE_LEDGER_REL,
        "schema": VALUE_SCHEMA,
        "rows": len(rows),
        "by_session_type": by_session,
        "reused_tokens_total": reused_total,
    }
    facts["violations"][VALUE_LEDGER_REL] = viol

    facts["as_of"] = as_of
    return facts


# ---------------------------------------------------------------------------
# Concepts: the #1844 lever family, resolved to rungs
# ---------------------------------------------------------------------------

def _savings_bucket(facts: dict, mechanism: str) -> dict:
    return facts["ledgers"]["cache_savings"]["by_mechanism"].get(
        mechanism, {"rows": 0, "saved_token_equiv": 0.0, "net_saved_token_equiv": 0.0})


def resolve_concepts(facts: dict) -> list[dict]:
    """Every managed-cache concept, its rung, and the evidence the rung rests on.

    Rung logic reads ONLY the fact fold, so the same fixture rows that drive the
    tests drive real reports — and a durable channel landing in a ledger auto-
    upgrades its concept here.
    """
    savings = facts["ledgers"]["cache_savings"]
    usage = facts["ledgers"]["gateway_usage"]
    value = facts["ledgers"]["cache_value"]
    concepts: list[dict] = []

    # 1. Provider prompt-cache passthrough — the baseline every lever is measured
    # against: the client's own cache_control forwarded untouched (OBSERVED).
    provider = _savings_bucket(facts, "provider_prompt_cache")
    concepts.append({
        "id": "provider_prompt_cache_passthrough",
        "title": "provider prompt-cache passthrough (baseline)",
        "provenance": "observed",
        "refs": ["#1844", "docs/notes/FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01.md"],
        "rung": "EVIDENCED" if provider["rows"] > 0 else "CHANNEL_READY",
        "evidence": {
            "savings_rows": provider["rows"],
            "net_saved_token_equiv": round(provider["net_saved_token_equiv"], 1),
            "guard_sessions": savings["by_session_type"].get("guard", 0),
        },
        "next_step": "grow the guard-session population; every exit appends a row",
    })

    # 2. Compaction shed (#1407) — fak-authored. The provider rows prove the same
    # writer ran at every guard exit, so zero shed rows is a WITNESSED zero
    # (anchor-starvation), not a recording gap.
    shed = _savings_bucket(facts, "compaction_shed")
    if shed["rows"] > 0:
        shed_rung = "EVIDENCED"
    elif provider["rows"] > 0:
        shed_rung = "SILENT_ZERO"
    else:
        shed_rung = "CHANNEL_READY"
    concepts.append({
        "id": "compaction_shed",
        "title": "compaction shed (fak-authored context shrink)",
        "provenance": "witnessed",
        "refs": ["#1407", "#1844"],
        "rung": shed_rung,
        "evidence": {
            "savings_rows": shed["rows"],
            "net_saved_token_equiv": round(shed["net_saved_token_equiv"], 1),
            "writer_proven_by_provider_rows": provider["rows"],
            "usage_compaction_fired_total": usage["compaction_fired_total"],
        },
        "next_step": "de-starve anchors (#1407) so real guard sessions shed; first shed row climbs this to EVIDENCED",
    })

    # 3. KV-prefix reuse (Track 1) — fak-authored, kernel (`fak run`) sessions only;
    # structurally 0 on the guard proxy path (Decide increments no KV counters).
    kv_total = value["reused_tokens_total"] + usage["kv_prefix_reused_tokens_total"]
    if kv_total > 0:
        kv_rung = "EVIDENCED"
    elif value["rows"] > 0:
        kv_rung = "SILENT_ZERO"
    else:
        kv_rung = "CHANNEL_READY"
    concepts.append({
        "id": "kv_prefix_reuse",
        "title": "in-kernel KV-prefix reuse (Track 1)",
        "provenance": "witnessed",
        "refs": ["#1844", "docs/nightrun/cache-value.jsonl"],
        "rung": kv_rung,
        "evidence": {
            "value_rows": value["rows"],
            "reused_tokens_total": kv_total,
            "kernel_sessions": value["by_session_type"].get("run", 0),
        },
        "next_step": "kernel-session population only; a guard-path KV witness would need the proxy to adjudicate through the kernel counters",
    })

    # 4. C6 — managed 1h cache_control TTL upgrade (cmd/fak/guard_managed_cache.go,
    # #1614). The lever SHIPPED, but its only witness is the in-process /metrics
    # counter fak_gateway_cache_ttl_upgrade_total — no durable ledger field exists,
    # so the proving ground cannot see it fire. The probe auto-upgrades this rung
    # the moment a ttl-bearing counter key lands in a usage-ledger row.
    concepts.append({
        "id": "ttl_upgrade_1h",
        "title": "managed 1h TTL upgrade on the stable prefix (C6)",
        "provenance": "witnessed",
        "refs": ["#1844(C6)", "#1614", "cmd/fak/guard_managed_cache.go"],
        "rung": "CHANNEL_READY" if usage["ttl_upgrade_keys_present"] else "UNWIRED",
        "evidence": {
            "lever_shipped": True,
            "durable_ttl_counter_keys": usage["ttl_upgrade_keys_present"],
            "metrics_only_witness": "fak_gateway_cache_ttl_upgrade_total",
        },
        "next_step": "add ttl_upgrade outcome counters to gatewayusageledger.Counters so real API-key-billed sessions leave durable evidence",
    })

    # 5. Breakpoint placement for no-cache_control callers (#1603/#806). Witnessed
    # in test only; on real rows a fak-placed breakpoint's saving is
    # indistinguishable from client passthrough — no attribution channel.
    concepts.append({
        "id": "breakpoint_placement",
        "title": "fak-placed cache_control breakpoints (no-cache_control callers)",
        "provenance": "witnessed",
        "refs": ["#1603", "#806", "internal/gateway/provider_cache_fak_placement_savings_test.go"],
        "rung": "UNWIRED",
        "evidence": {
            "lever_shipped": True,
            "population_note": "Claude Code marks its own head — placement is identity on the recorded guard population",
        },
        "next_step": "stamp placement attribution (fak-placed vs already_set) onto the savings row so a no-breakpoint caller population becomes provable",
    })

    # 6. C7 — uncached-remainder shrink. Not implemented anywhere in the tree.
    concepts.append({
        "id": "uncached_remainder_shrink",
        "title": "uncached input-remainder shrink post-breakpoint (C7)",
        "provenance": "witnessed",
        "refs": ["#1844(C7)"],
        "rung": "UNIMPLEMENTED",
        "evidence": {"lever_shipped": False},
        "next_step": "implement the lever; register it as an ablation feature and a savings-row mechanism in the same change",
    })

    # 7. Guard usage-plane rows (#1601) — infrastructure every guard-path lever
    # depends on: the counter family exists and serve exits write it, but guard
    # exits do not, so guard sessions have no durable counter trail here.
    if usage["guard_rows"] > 0:
        guard_rung = "EVIDENCED" if usage["nonzero_counter_rows"] > 0 else "SILENT_ZERO"
    else:
        guard_rung = "CHANNEL_READY"
    concepts.append({
        "id": "guard_usage_plane",
        "title": "guard sessions in the gateway-usage ledger (witness infrastructure)",
        "provenance": "observed",
        "refs": ["#1601"],
        "rung": guard_rung,
        "evidence": {
            "usage_rows": usage["rows"],
            "guard_rows": usage["guard_rows"],
            "serve_rows": usage["by_session_type"].get("serve", 0),
            "nonzero_counter_rows": usage["nonzero_counter_rows"],
        },
        "next_step": "append a gateway-usage exit row from guard teardown, same writer as serve",
    })

    return concepts


# ---------------------------------------------------------------------------
# Report + baseline ratchet
# ---------------------------------------------------------------------------

def build_report(root: Path) -> dict:
    facts = load_facts(root)
    concepts = resolve_concepts(facts)
    return {
        "schema": REPORT_SCHEMA,
        "as_of": facts["as_of"],
        "ledgers": facts["ledgers"],
        "violations": facts["violations"],
        "violation_count": sum(len(v) for v in facts["violations"].values()),
        "concepts": concepts,
    }


def baseline_snapshot(report: dict) -> dict:
    """The subset of the report the ratchet compares: counts, rungs, violations."""
    return {
        "schema": REPORT_SCHEMA,
        "as_of": report["as_of"],
        "ledger_rows": {name: led["rows"] for name, led in report["ledgers"].items()},
        "ledger_schemas": {name: led["schema"] for name, led in report["ledgers"].items()},
        "savings_mechanism_rows": {
            mech: bucket["rows"]
            for mech, bucket in report["ledgers"]["cache_savings"]["by_mechanism"].items()
        },
        "violation_count": report["violation_count"],
        "concept_rungs": {c["id"]: c["rung"] for c in report["concepts"]},
    }


def check_against_baseline(report: dict, baseline: dict) -> list[str]:
    """The closed regression vocabulary. Empty list = the ratchet holds."""
    problems: list[str] = []
    snap = baseline_snapshot(report)

    for name, schema in baseline.get("ledger_schemas", {}).items():
        got = snap["ledger_schemas"].get(name)
        if got != schema:
            problems.append(f"SCHEMA_DRIFT {name}: baseline {schema!r} now {got!r}")

    for name, rows in baseline.get("ledger_rows", {}).items():
        got = snap["ledger_rows"].get(name, 0)
        if got < rows:
            problems.append(
                f"REGRESSION_ROW_COUNT {name}: append-only ledger shrank {rows} -> {got}")

    for mech, rows in baseline.get("savings_mechanism_rows", {}).items():
        got = snap["savings_mechanism_rows"].get(mech, 0)
        if got < rows:
            problems.append(
                f"REGRESSION_ROW_COUNT cache_savings[{mech}]: {rows} -> {got}")

    base_viol = baseline.get("violation_count", 0)
    if report["violation_count"] > base_viol:
        problems.append(
            f"REGRESSION_VIOLATIONS: {base_viol} -> {report['violation_count']} "
            f"(first: {_first_violation(report)})")

    for cid, rung in baseline.get("concept_rungs", {}).items():
        got = snap["concept_rungs"].get(cid)
        if got is None:
            problems.append(f"REGRESSION_RUNG {cid}: concept vanished from the registry")
        elif rung_rank(got) < rung_rank(rung):
            problems.append(f"REGRESSION_RUNG {cid}: fell {rung} -> {got}")

    return problems


def _first_violation(report: dict) -> str:
    for viols in report["violations"].values():
        if viols:
            return viols[0]
    return "-"


def render_card(report: dict) -> str:
    lines = [f"managed-cache proving ground — real-session evidence as of {report['as_of'] or '-'}"]
    led = report["ledgers"]
    lines.append(
        f"  ledgers: cache-savings {led['cache_savings']['rows']} rows | "
        f"gateway-usage {led['gateway_usage']['rows']} rows "
        f"(guard {led['gateway_usage']['guard_rows']}) | "
        f"cache-value {led['cache_value']['rows']} rows")
    lines.append(f"  row-contract violations: {report['violation_count']}")
    lines.append("  concept rungs (0 UNIMPLEMENTED .. 4 EVIDENCED):")
    for c in report["concepts"]:
        marker = {"EVIDENCED": "##", "SILENT_ZERO": "==", "CHANNEL_READY": "--",
                  "UNWIRED": "!!", "UNIMPLEMENTED": "  "}[c["rung"]]
        lines.append(f"    [{rung_rank(c['rung'])}] {marker} {c['id']:34s} {c['rung']:13s} {c['title']}")
        lines.append(f"           next: {c['next_step']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT), help="repo root holding docs/nightrun/")
    parser.add_argument("--baseline", default=None,
                        help="baseline path (default tools/managed_cache_proving_ground.data/baseline.json under --root)")
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    parser.add_argument("--check", action="store_true", help="ratchet against the baseline; exit 1 on regression")
    parser.add_argument("--write-baseline", action="store_true", help="snapshot the current fold as the baseline")
    args = parser.parse_args(argv)

    root = Path(args.root)
    baseline_path = Path(args.baseline) if args.baseline else (
        root / "tools" / "managed_cache_proving_ground.data" / "baseline.json")

    try:
        report = build_report(root)
    except OSError as err:
        print(f"managed-cache proving ground: HARNESS ERROR reading ledgers: {err}", file=sys.stderr)
        return 2

    if args.write_baseline:
        snap = baseline_snapshot(report)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {baseline_path}")
        return 0

    if args.check:
        if not baseline_path.exists():
            print(f"managed-cache proving ground: HARNESS ERROR: no baseline at {baseline_path} "
                  f"(run --write-baseline once)", file=sys.stderr)
            return 2
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            print(f"managed-cache proving ground: HARNESS ERROR reading baseline: {err}", file=sys.stderr)
            return 2
        problems = check_against_baseline(report, baseline)
        if problems:
            print("managed-cache proving ground: RATCHET REGRESSION")
            for p in problems:
                print(f"  {p}")
            return 1
        print(f"managed-cache proving ground OK — {report['ledgers']['cache_savings']['rows']} savings rows, "
              f"{report['violation_count']} violations, rungs hold vs baseline")
        return 0

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_card(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
