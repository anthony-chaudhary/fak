# Captured run

A real run of [`run.sh`](run.sh), color stripped:

```console
$ examples/trace-reset/run.sh
[reset] building fak -> /tmp/tmp.aGuKI6tDsy/fak
[reset] starting gateway: fak serve http://127.0.0.1:8088
[reset] gateway healthy (PID 120767): {"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}

[reset] 1) baseline: a fresh trace reads the clean default
  ✓ GET /v1/fak/trace/sess-boundary-A (before any admit) -> trusted
[reset] 2) admit an UNTRUSTED result onto trace A (an injection-shaped tool result)
[reset] 3) the trace's IFC high-water mark rose above Trusted:
  ✓ GET /v1/fak/trace/sess-boundary-A -> quarantined
[reset] 4) admit an untrusted result onto a DIFFERENT trace B
[reset] 5) trace B's mark is up too (the neighbour we will prove is left untouched):
  ✓ GET /v1/fak/trace/sess-boundary-B -> quarantined
[reset] 6) global forensic quarantine counter (no-rollback baseline): quarantines=2

[reset] 7) operator-approved session boundary: reset ONLY trace A
  ✓ POST /v1/fak/trace/reset {"trace_id":"sess-boundary-A"} -> reset:true
[reset] 8) trace A's high-water mark is back to baseline:
  ✓ GET /v1/fak/trace/sess-boundary-A after reset -> trusted
[reset] 9) per-trace scope: the reset did NOT touch trace B:
  ✓ GET /v1/fak/trace/sess-boundary-B still tainted after A's reset -> quarantined
[reset] 10) the reset cleared a per-trace mark, NOT the global forensic tally:
  ✓ quarantines counter unchanged (2 == 2) — reset is per-trace, not a counter rollback

[reset] 11) fail-loud: a reset with a blank trace_id is refused
  ✓ reset with a blank trace_id -> 400 (refused: trace_id is required)

[reset] all witnesses passed — trace A's IFC mark rose, was reset to baseline, and the reset touched neither trace B nor the global forensic counter.
```

## What the capture proves

- **The mark rose, then reset to baseline.** Trace A read `trusted` before any
  admit (step 1), `quarantined` after an untrusted result was admitted (step 3),
  and `trusted` again after the reset (step 8) — the only thing between steps 3
  and 8 was the `POST /v1/fak/trace/reset` in step 7.
- **The reset is per-trace.** Trace B was `quarantined` in step 5 and **still**
  `quarantined` in step 9, after A was reset. Clearing A's mark did not leak
  across to B.
- **The reset is not a forensics rollback.** The global `kernel.quarantines`
  counter on `/debug/vars` was `2` before the reset (step 6) and `2` after it
  (step 10). The append-only record that two results were quarantined survives;
  the operator ended a session, they did not erase the audit trail.
- **Fail-loud held.** A reset with a blank `trace_id` was refused with `400`
  (`trace_id is required`) — the route does not silently no-op on a missing id.

The raw bodies behind the witnesses (for reference):

```jsonc
// step 1 — GET /v1/fak/trace/sess-boundary-A (fresh trace, clean default)
{"trace_id":"sess-boundary-A","taint":"trusted","dangerous":false}

// step 2 — POST /v1/fak/admit (an injection-shaped untrusted result)
// The result-side stack quarantines the bytes AND raises the trace's IFC mark.
{"verdict":{"kind":"QUARANTINE","reason":"TRUST_VIOLATION","by":"normgate"},
 "result":{"status":"OK","meta":{"admit":"quarantined","ifc_taint":"quarantined",
   "normgate":"quarantined","quarantine_id":"ng-q1"}},
 "trace_id":"sess-boundary-A"}

// step 3 — GET /v1/fak/trace/sess-boundary-A (mark rose above Trusted)
{"trace_id":"sess-boundary-A","taint":"quarantined","dangerous":true}

// step 7 — POST /v1/fak/trace/reset {"trace_id":"sess-boundary-A"}
{"reset":true,"trace_id":"sess-boundary-A"}

// step 8 — GET /v1/fak/trace/sess-boundary-A (back to the clean default)
{"trace_id":"sess-boundary-A","taint":"trusted","dangerous":false}

// step 11 — POST /v1/fak/trace/reset {"trace_id":"  "}
// HTTP 400
{"error":{"message":"trace_id is required","type":"invalid_request_error"}}
```

> The `mock` engine/model in the health line is the in-kernel default `fak serve`
> boots with no `--base-url` — this example needs no upstream, so the admit,
> observe, and reset routes answer deterministically without one.
