---
title: "Cold-tool deferral (epic #3229) — validation status + the #3536 dogfood runbook"
description: "The three synthetic QA gates for the DeferColdTools default-on flip are green; the live fable dogfood (#3536) is the remaining gate. This note records the deterministic evidence and the exact operator runbook to capture the live numbers and clear #3537."
---

# Cold-tool deferral (epic #3229) — validation status

**Date:** 2026-07-09. **Scope:** the `DeferColdTools` lever (#3232) and the four QA gates that
block flipping it default-on for the flagship `fak manage -- claude` path (#3537).

`DeferColdTools` marks every allowed-but-cold custom tool `defer_loading:true` on the outbound
Anthropic body and injects one `tool_search_tool`, so the provider loads only the hot core into
context and faults a cold schema in on demand. It ships **default-off** because it is the epic's
highest-risk lever; #3537 flips it on **only when all four QA gates are green**.

## Gate status

| Gate | Ticket | Status | Witness (deterministic, in-repo) |
|------|--------|--------|----------------------------------|
| Token-delta A/B | #3532 | **GREEN** | `fak footprint --ab` — resident tool-slice ARMED vs ABLATED; delta sign + cache-prefix byte-parity pinned. |
| Held-accuracy (fault-in recall) | #3533 | **GREEN** | `fak footprint --held-accuracy` — armed **3/3** vs ablated 3/3, gate HOLDS; every representative cold tool (fak_*, dos_*, cold built-in) stays faultable-in (present + searchable + schema-intact). |
| Poison / quarantine (no bypass) | #3534 | **GREEN** | `internal/gateway/tooldefer_no_bypass_test.go` — a deferred floor-denied tool is still DENY'd; a deferred tool's poisoned result is still QUARANTINE'd; the transform adds no trust/allowlist key. |
| **Live dogfood** | **#3536** | **NOT YET** | needs a real `claude-fable-5` worker's traffic through the guard — see the runbook below. |

Three of four gates are green and deterministic. Their honesty labels matter: #3532's delta is
**ESTIMATED** (house tokenizer on the resident slice — `defer_loading` grows request bytes, the
reduction is provider-side), and #3533 is **mechanical fault-in recall** (`deterministic-faultin-sim`,
`live_accuracy_claim_allowed:false`) — it witnesses that a deferred tool is never *silently lost*,
not that a live model *completes the task*. Both explicitly point at #3536 for the OBSERVED number.

## Why #3536 is not captured here

#3536 requires the one thing the synthetic gates cannot give: a **real dispatched fable worker**
end-to-end. Two constraints kept it from being run in this session:

1. **It is a live, billed, outward-facing, multi-hour run** — a `claude-fable-5` worker resolving
   real issues, whose requests flow through the `--defer-cold-tools` gateway so the #3531 exit
   summary can measure the effect. That is an operator decision, not an unattended one.
2. **`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`** advises against standing up local
   `fak serve`/guard sessions on this shared Windows box (it hosts the live dispatch fleet; stray
   serves pile up GB of memory). The dogfood should run where the fleet already dispatches workers.

So this gate is reported **`not yet`**, with the runbook below as the next checkable step.

## The #3536 runbook (operator)

Pick 2-3 surgical `--target-issue` tasks of known difficulty. For each, run twice — deferral ON,
then default-off — same model and same issue:

```
# ARM (deferral on)
fak manage --defer-cold-tools --model claude-fable-5 --debug-stats -- claude -p "<resolve target issue N>"
# ABLATE (default off — the current shipping behavior)
fak manage --model claude-fable-5 --debug-stats -- claude -p "<resolve target issue N>"
```

Capture, per run, from the guard exit summary (the #3531 surface) and `/metrics`:

- **turn-1 floor** (`fak_context_value` footprint: `system + tools`, built-in vs MCP split);
- **per-turn `defer` count + Δtools** (the "cold-tool deferral" exit-summary section /
  `fak_gateway_tool_defer_{cold,turns}_total`);
- **peak ctx** and **compaction fire count** (the compaction section);
- **outcome**: did the worker commit real, self-authored, audited work? (attribute by
  run-window + authorship, per prior fleet-attribution lessons — never by timeline).

### Before/after table to fill in

| issue | arm | turn-1 floor | defer count | Δtools | peak ctx | compaction fires | committed audited work? |
|-------|-----|--------------|-------------|--------|----------|------------------|-------------------------|
| #___  | ON  |              |             |        |          |                  |                         |
| #___  | OFF |              |             |        |          |                  |                         |

### PASS/FAIL rubric (feeds #3537)

- **PASS** iff, on the sampled issues: (a) turn-1 floor drops materially on the ARM (target ≈ the
  built-in tool slice, ~26k), AND (b) **resolution parity** — deferral-on completes the task at
  least as well as off (no capability regression the #3533 gate would not already have caught).
- **FAIL** iff the ARM loses resolution (a task the OFF arm completes that the ON arm cannot),
  or the floor does not drop — in which case the offending cold tool likely needs promotion to
  `defaultHotToolSet` (a #3533 follow-up), and #3537 stays blocked.

## #3537 (the flip) — ready, gated

When #3536 reports **PASS**, #3537 is a one-line default change plus the four verdicts linked:

- flip the guard flag default: `deferColdTools := fs.Bool("defer-cold-tools", false, …)` →
  default `true` in `cmd/fak/guard.go` (and `cmd/fak/serve.go`), **or** the embedded
  `gateway.Config` default;
- keep the explicit opt-out (`--defer-cold-tools=false` / `FAK_ABLATE_DEFER_TOOLS=1`);
- link the four gate verdicts (this note + the three green scorecards) in the PR.

Until #3536 is PASS, #3537 **stays blocked** — the flip is deliberately empty of mechanism; it is
the gate, not the build.

## Cross-links

- **#3229** epic · **#3232** the lever · **#3530** the flag · **#3531** the exit-summary floor/defer
  surface · **#3532** token-delta A/B · **#3533** held-accuracy · **#3534** poison/quarantine ·
  **#3536** this note's open gate · **#3537** the payoff flip · **#3200** the pin/quarantine guard
  the fault-in leans on.
