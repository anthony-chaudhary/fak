---
title: "Debug a fak verdict: why was my call denied, transformed, or not cached?"
description: "An end-to-end runbook for diagnosing a fak adjudication — reproduce the verdict offline with the per-rung decision trace, correlate it across the live gateway by trace_id, read who refused and why from /metrics and the audit journal, and replay a finished session. No request bodies or tool arguments are ever logged."
---

# Debugging guide: why did fak do that to my call?

A tool call came back denied, repaired, or quarantined and you want to know **why** —
which of the eight folded rungs decided it, what reason it cited, and what would make it
pass. fak answers this from evidence at every layer, so you never have to read kernel
source or guess. This runbook is the loop, from the fastest offline check to a full
post-hoc session replay.

> The companion [observability guide](../fak/observability.md) is the *reference* for the
> metrics, logs, and `trace_id` surfaces. This page is the *task*: diagnosing one
> surprising verdict, end to end. Every surface below is designed to never log a request
> body, a tool argument, or result content — only digests and bounded disclosures.

---

## 1. Reproduce it offline, deterministically — `fak preflight --explain`

The capability floor is the same code whether a model is in the loop or not, so you can
reproduce any verdict with no key, no model, and no gateway. The default one-liner tells
you *what*; `--explain` tells you **why** — every rung that folded, what each returned,
and which one won:

```bash
fak preflight --tool write_file --args '{"path":"internal/abi/x.go","content":"…"}' \
  --policy your-policy.json --explain
```

```
tool: write_file   args: 54 bytes (sha 8b3b16b5d7c8)
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: internal/abi/
explanation: write_file denied by monitor: SELF_MODIFY (ESCALATE). offending set: internal/abi/.

decision chain (8 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
   [2] preflight.Ladder           DEFER     by=preflight
   [3] engine.residencyGate       DEFER     by=engine-residency
   [4] plancfi.Adjudicator        DEFER     by=plancfi
   [5] ifc.SinkGate               DEFER     by=ifc-sink
   [6] shipgate.ShipAdjudicator   DEFER     by=shipgate
=> [7] adjudicator.Adjudicator    DENY      SELF_MODIFY by=monitor   {internal/abi/}   <- winner (rank 100)
```

This is the single most useful command in the box. It tells you:

- **which rung decided** (the `=>` winner, with its lattice `rank`) — so when two rungs
  both refuse, you can see the earlier one shadowed the later one (a `preflight`
  `MALFORMED` deny will win the tie over a later `monitor` `DEFAULT_DENY`);
- **the bounded-disclosure witness** — the exact offending fragment (`internal/abi/`
  here), so you know whether your *new* narrow glob fired or an old catch-all did;
- **the disposition** — `RETRYABLE` / `WAIT` / `ESCALATE` / `TERMINAL`, i.e. whether a
  retry could ever succeed.

For tooling (or a debugging skill), `--json` emits the same `Decision` structured and
**safe to log** — it carries an args *digest*, never the raw args:

```bash
fak preflight --tool refund_payment --args '{}' --policy your-policy.json --json
```

A `TRANSFORM` verdict additionally lists the `redacted` arg keys it rewrote; an
admit-and-log posture surfaces `posture` and `would_deny`.

---

## 2. Find the call in a live `fak serve`

Over the wire, every adjudication rides back **inline** on the response — `fak` on
`/v1/chat/completions`, `_fak` on `/v1/messages` — with the verdict, the deciding `by`,
the reason, and the `trace_id`:

```json
"_fak": { "version": "fak/v1", "admissions": [
  { "tool": "Bash", "verdict": "DENY", "reason": "POLICY_BLOCK", "by": "monitor", "trace_id": "…" }
]}
```

That same `trace_id` is returned in the **`X-Trace-Id`** response header and stamped on
the structured access log. To follow one request across every surface, filter the access
log by it — the per-operation line now names **who** decided (`by`):

```bash
# the structured op log is always on (not gated behind the audit journal)
jq 'select(.event=="gateway_operation" and .trace_id=="<id>")' fak-serve.log
# -> {"operation":"adjudicate","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","disposition":"TERMINAL", …}
```

---

## 3. See the pattern across the fleet — `/metrics`

When the question is "why is my deny rate / miss rate high across the workload", the
Prometheus surface answers in aggregate. Two families added for exactly this:

```promql
# WHO is refusing (which rung), not just that something was refused:
sum by (by, reason) (fak_gateway_operations_total{verdict="DENY"})

# WHY the vDSO fast path isn't hitting — each miss attributed to a cause instead of a
# bare ok=false (DESTRUCTIVE / MISSING_HINTS / RESOURCE_MISNAMED / WITNESS_REVOKED / NOT_CACHED):
sum by (reason) (fak_vdso_misses_total)
```

A high `fak_vdso_misses_total{reason="MISSING_HINTS"}` means your client is dropping the
`readOnlyHint`/`idempotentHint` metadata; a high `DESTRUCTIVE` means the tools simply
aren't fast-path eligible. The full metric catalog is in the
[observability guide](../fak/observability.md).

---

## 4. Pull the durable record — `GET /v1/fak/events`

With the audit journal enabled (`FAK_AUDIT_JOURNAL=/path/to/journal.jsonl`), every
decision is a hash-chained, append-only row you can drain after the fact:

```bash
curl -s 'http://127.0.0.1:8080/v1/fak/events?since=0' | jq '.'
```

Each row carries the verdict, reason, `by`, and content digests, plus two fields for
correlation and forensics:

- **`call_seq`** — the kernel's per-call id. A call's `DECIDE` row and its later
  `QUARANTINE` row share it, so you can pull one call's whole timeline instead of
  guessing from `tool` + `args_digest` when calls interleave.
- **`witness`** — the bounded-disclosure claim (the offending self-modify glob, the
  `tool.arg` bound that broke), persisted so the durable record names *which* thing
  tripped the deny, not just that one did.

---

## 5. Replay a finished session — `fak debug`

To diagnose a *whole* Claude Code session after the fact, attach to its transcript as a
core image. You don't need to remember where transcripts live — discover them:

```bash
fak debug --list
# found 42 Claude Code transcript(s); most recent first:
#   [ 1] 2026-06-22 01:56     1.1M  7ef0a89e-….jsonl
#        fak debug --session "…/projects/…/7ef0a89e-….jsonl"
```

Then attach the one you want and ask a question — fak demand-pages only the working set
the question touches, and drives every tool result back through the shipped gate (so a
poisoned/secret result *seals* on ingest, exactly as it would live):

```bash
fak debug --session "<path>.jsonl" --cmd bt          # backtrace of pages, sealed ones marked
fak debug --session "<path>.jsonl" --cmd ws --query "what did the kernel quarantine?"
```

(With no `--session`, `fak debug` runs a hermetic demo over a committed fixture and says
so — run `--list` to point it at your real data.)

---

## 6. Debug an agent A/B run — `agent-report.json`

`fak agent` writes a per-call decision trace into its report artifact, so a run that went
wrong is debuggable from the JSON alone (no separate `--log` file needed). Each row in
`calls[]` carries the arm, tool, verdict, the closed `reason` name and `disposition` on a
deny, the deciding `by`, and a bounded args preview:

```bash
fak agent --offline --out agent-report.json
jq '.calls[] | select(.arm=="fak")' agent-report.json
# {"arm":"fak","turn":2,"tool":"fetch_policy","verdict":"DENY","reason":"TRUST_VIOLATION","by":"ifc-sink", …}
```

---

## Decision vocabulary (quick reference)

| Verdict | Meaning | Where it shows |
|---|---|---|
| `ALLOW` | admitted (or vDSO-served — `by=vdso`) | inline `fak`/`_fak`, metrics, journal |
| `DENY` | refused by structure (carries a `reason`) | everywhere |
| `TRANSFORM` | args rewritten before dispatch (e.g. secret redaction) | inline, `--explain` `redacted` |
| `QUARANTINE` | result paged out of the model's context | journal, `agent` quarantines |

| Disposition | What a loop should do |
|---|---|
| `RETRYABLE` | the model can fix it (e.g. `MALFORMED`) and retry |
| `WAIT` | transient (rate-limited / lease held) — back off |
| `ESCALATE` | a trust/self-modify event — needs a human, not a retry |
| `TERMINAL` | structurally refused — a retry will not help |

---

## Cross-references

- [Observability guide](../fak/observability.md) — the metrics / logs / `trace_id` reference.
- [Integration index](README.md) — put fak in front of your agent.
- [Policy / permissions](../../POLICY.md) — author and review the capability floor.
- [MCP tools](../../examples/mcp/README.md) — the `fak_*` tools an MCP client calls directly.
