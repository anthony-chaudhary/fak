#!/usr/bin/env python3
"""guard_hop_bench.py — the kernel-in-the-loop guard-hop overhead + prompt-cache
preservation harness (issue #734).

When a dispatch/interactive worker is fronted with `fak guard` (the dogfood default),
every tool call the agent proposes crosses the kernel before it reaches the provider.
That hop buys adjudication (deny / repair / quarantine) and a decision journal — but it
is not free, and a serving stack's whole value proposition is that the safety hop does
NOT (a) add meaningful latency or (b) break the provider prompt-cache. This harness
measures both, as a BENCHMARK-AUTHORITY-shaped row.

HONESTY (BENCHMARK-AUTHORITY.md rules). A number is only ever emitted with an explicit
`status`:

  * MEASURED  — a real wall-clock measured on THIS box against a live gateway
                (`measure --gateway-url … --direct-url …`). Single-box, disclosed.
  * PROJECTED — a closed-form estimate from a COMMITTED pure-kernel latency artifact
                (the 362 ns Decide / ~2.4 µs in-process adjudication rows), NOT a
                wall-clock and NEVER labeled "measured".
  * PENDING   — no number: the live-hardware run has not happened. The field carries
                no value, only the reproduce command.

A bare number with no `status` (or a PENDING field that smuggles a number) FAILS the
`--check` gate. That gate is the structural guarantee this harness can't fabricate a
"measured" guard-hop number it didn't measure.

Usage:
  python tools/guard_hop_bench.py describe            # PROJECTED + PENDING row (no hardware)
  python tools/guard_hop_bench.py describe --json
  python tools/guard_hop_bench.py measure \
      --gateway-url http://127.0.0.1:8080 --direct-url http://127.0.0.1:8099  # MEASURED
  python tools/guard_hop_bench.py --check row.json    # honesty gate over an emitted row
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "guard-hop-bench/1"

# Committed pure-kernel latency anchors (BENCHMARK-AUTHORITY.md, commit bcad56e,
# experiments/mac-m3pro-kernel-20260620/). These are the basis for the PROJECTED
# overhead — a closed-form estimate, NOT a wall-clock of the full serving loop.
DECIDE_NS = 362          # p50 ns/op, canonical ALLOW (Decide)
INPROC_ADJUDICATE_NS = 2427  # in-process syscall p50 (vs 6.913 ms spawned `fak hook`)
ANCHOR_COMMIT = "bcad56e"
ANCHOR_ARTIFACT = "experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json"


def project_overhead(calls_per_turn: int, turns: int,
                     decide_ns: int = DECIDE_NS,
                     inproc_ns: int = INPROC_ADJUDICATE_NS) -> dict[str, Any]:
    """Closed-form PROJECTED guard-hop overhead per turn and per session.

    The guard process is long-lived (one `fak guard` fronts the whole worker session),
    so the marginal cost of the hop is the IN-PROCESS adjudication per tool call — NOT
    the ~6.9 ms spawned-`fak hook` boundary (which the in-process path replaces). Two
    bounds are reported so the estimate is honest about its own spread:

      * floor : decide-only (362 ns/call) — the cheapest ALLOW path.
      * ceil  : full in-process adjudication (2.427 µs/call) — the syscall p50.
    """
    if calls_per_turn < 0 or turns < 0:
        raise ValueError("calls_per_turn and turns must be non-negative")
    floor_us_turn = decide_ns * calls_per_turn / 1000.0
    ceil_us_turn = inproc_ns * calls_per_turn / 1000.0
    return {
        "status": "PROJECTED",
        "calls_per_turn": calls_per_turn,
        "turns": turns,
        "per_turn_overhead_us": {"floor": round(floor_us_turn, 3),
                                 "ceil": round(ceil_us_turn, 3)},
        "per_session_overhead_ms": {"floor": round(floor_us_turn * turns / 1000.0, 4),
                                    "ceil": round(ceil_us_turn * turns / 1000.0, 4)},
        "basis": {"decide_ns": decide_ns, "inproc_adjudicate_ns": inproc_ns,
                  "commit": ANCHOR_COMMIT, "artifact": ANCHOR_ARTIFACT},
        "note": "closed-form from the committed pure-kernel decide-latency rows; "
                "NOT a wall-clock of the full serving loop (that is the MEASURED path).",
    }


def _post_messages(base_url: str, timeout: float) -> tuple[float, dict[str, Any]]:
    """POST a trivial /v1/messages and return (elapsed_s, parsed_json). Used by the
    MEASURED path against a live gateway or a direct upstream that speaks the wire."""
    body = json.dumps({
        "model": "guard-hop-bench",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/messages",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (loopback bench)
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return elapsed, parsed


def measure_hop(gateway_url: str, direct_url: str, reps: int = 20,
                timeout: float = 30.0) -> dict[str, Any]:
    """MEASURED guard-hop overhead: median roundtrip THROUGH the kernel gateway minus
    the median roundtrip DIRECT to the same mock upstream, on this box. Single-box and
    disclosed as such. Requires both endpoints reachable (e.g. `fak serve` as the
    gateway and the in-tree shim as the direct upstream)."""
    def median_ms(url: str) -> float:
        samples = sorted(_post_messages(url, timeout)[0] for _ in range(reps))
        mid = samples[len(samples) // 2]
        return round(mid * 1000.0, 3)

    direct_ms = median_ms(direct_url)
    gateway_ms = median_ms(gateway_url)
    return {
        "status": "MEASURED",
        "reps": reps,
        "direct_p50_ms": direct_ms,
        "gateway_p50_ms": gateway_ms,
        "guard_hop_overhead_ms": round(gateway_ms - direct_ms, 3),
        "disclosure": "single-box wall-clock on the measuring host; not a fleet SLA.",
    }


def cache_preservation_pending() -> dict[str, Any]:
    """The prompt-cache-preservation arm. The kernel forwards inbound request bytes
    verbatim, so `cache_control` markers should survive the guard hop byte-for-byte and
    the provider's prompt-cache hit-rate should be UNCHANGED vs a direct connection.
    Confirming that needs a live provider that reports cache tokens — hardware/credential
    gated — so the live number is PENDING. The structural forwarding is exercised by the
    gateway's own tests (`internal/gateway`); this row is the end-to-end cache-rate
    witness that is still owed."""
    return {
        "status": "PENDING",
        "claim": "provider prompt-cache hit-rate is preserved across the fak guard hop "
                 "(cache_control forwarded byte-for-byte; cache_read tokens unchanged).",
        "reproduce": "run a guarded vs direct A/B on a cache-reporting provider and "
                     "compare provider_cache_read tokens per turn (see "
                     "tools/cross_agent_ablate.py for the token-decomposition harness).",
        "blocked_on": "a live provider that reports cache tokens (hardware/credential gated).",
    }


def build_row(calls_per_turn: int = 8, turns: int = 50,
              measured: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the BENCHMARK-AUTHORITY-shaped guard-hop row. `measured` (when present)
    supplies the MEASURED overhead arm; otherwise the overhead arm is PROJECTED."""
    overhead = measured if measured is not None else project_overhead(calls_per_turn, turns)
    return {
        "schema": SCHEMA,
        "claim": "kernel-in-the-loop guard-hop overhead + prompt-cache preservation",
        "guard_hop_overhead": overhead,
        "prompt_cache_preservation": cache_preservation_pending(),
        "reproduce": "python tools/guard_hop_bench.py measure "
                     "--gateway-url http://127.0.0.1:8080 --direct-url http://127.0.0.1:8099",
    }


def check_row(row: dict[str, Any]) -> list[str]:
    """The honesty gate: return a list of violations (empty == honest).

    Rules (BENCHMARK-AUTHORITY): every metric arm carries a `status` in
    {MEASURED, PROJECTED, PENDING}; a PENDING arm carries NO number; a MEASURED arm
    carries a number AND a single-box disclosure; a PROJECTED arm names its committed
    basis (commit + artifact). This is what stops a fabricated "measured" number."""
    violations: list[str] = []
    valid = {"MEASURED", "PROJECTED", "PENDING"}
    number_keys = ("guard_hop_overhead_ms", "per_turn_overhead_us",
                   "per_session_overhead_ms", "direct_p50_ms", "gateway_p50_ms")

    if row.get("schema") != SCHEMA:
        violations.append(f"schema must be {SCHEMA!r}, got {row.get('schema')!r}")

    for arm_name in ("guard_hop_overhead", "prompt_cache_preservation"):
        arm = row.get(arm_name)
        if not isinstance(arm, dict):
            violations.append(f"{arm_name}: missing or not an object")
            continue
        status = arm.get("status")
        if status not in valid:
            violations.append(f"{arm_name}.status must be one of {sorted(valid)}, got {status!r}")
            continue
        has_number = any(k in arm for k in number_keys)
        if status == "PENDING" and has_number:
            violations.append(f"{arm_name}: PENDING arm must carry NO number (found one)")
        if status == "MEASURED":
            if "guard_hop_overhead_ms" not in arm:
                violations.append(f"{arm_name}: MEASURED arm must report guard_hop_overhead_ms")
            if not arm.get("disclosure"):
                violations.append(f"{arm_name}: MEASURED arm must carry a single-box disclosure")
        if status == "PROJECTED":
            basis = arm.get("basis") or {}
            if not (basis.get("commit") and basis.get("artifact")):
                violations.append(f"{arm_name}: PROJECTED arm must name its committed basis "
                                  "(commit + artifact)")
    return violations


def render(row: dict[str, Any]) -> str:
    oh = row["guard_hop_overhead"]
    cache = row["prompt_cache_preservation"]
    lines = [f"guard-hop-bench: {row['claim']}"]
    if oh["status"] == "PROJECTED":
        pt = oh["per_turn_overhead_us"]
        ps = oh["per_session_overhead_ms"]
        lines.append(f"  overhead [PROJECTED]  per-turn {pt['floor']}–{pt['ceil']} µs "
                     f"({oh['calls_per_turn']} calls/turn) · "
                     f"per-session {ps['floor']}–{ps['ceil']} ms ({oh['turns']} turns)")
        lines.append(f"    basis: decide {oh['basis']['decide_ns']} ns / "
                     f"in-proc {oh['basis']['inproc_adjudicate_ns']} ns "
                     f"({oh['basis']['commit']})")
    elif oh["status"] == "MEASURED":
        lines.append(f"  overhead [MEASURED]   guard-hop {oh['guard_hop_overhead_ms']} ms "
                     f"(gateway {oh['gateway_p50_ms']} − direct {oh['direct_p50_ms']}, "
                     f"n={oh['reps']})")
    lines.append(f"  prompt-cache [{cache['status']}]  {cache['claim']}")
    lines.append(f"  reproduce: {row['reproduce']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Guard-hop overhead + prompt-cache preservation harness.")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("describe", help="emit the PROJECTED + PENDING row (no hardware)")
    d.add_argument("--calls-per-turn", type=int, default=8)
    d.add_argument("--turns", type=int, default=50)
    d.add_argument("--json", action="store_true")
    d.add_argument("--out", default="")

    m = sub.add_parser("measure", help="MEASURED guard-hop overhead against a live gateway")
    m.add_argument("--gateway-url", required=True, help="fak serve base URL (the kernel hop)")
    m.add_argument("--direct-url", required=True, help="direct upstream base URL (same mock)")
    m.add_argument("--reps", type=int, default=20)
    m.add_argument("--json", action="store_true")
    m.add_argument("--out", default="")

    ap.add_argument("--check", metavar="ROW.json", default="",
                    help="honesty-gate an emitted row (exit 1 on any violation)")
    args = ap.parse_args(argv)

    if args.check:
        row = json.loads(Path(args.check).read_text(encoding="utf-8"))
        violations = check_row(row)
        if violations:
            print("guard-hop-bench --check: FAIL")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("guard-hop-bench --check: OK (row is honest)")
        return 0

    if args.cmd == "measure":
        measured = measure_hop(args.gateway_url, args.direct_url, reps=args.reps)
        row = build_row(measured=measured)
    else:  # describe (default)
        cpt = getattr(args, "calls_per_turn", 8)
        turns = getattr(args, "turns", 50)
        row = build_row(calls_per_turn=cpt, turns=turns)

    out = json.dumps(row, indent=2)
    if getattr(args, "out", ""):
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    if getattr(args, "json", False):
        print(out)
    else:
        print(render(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
