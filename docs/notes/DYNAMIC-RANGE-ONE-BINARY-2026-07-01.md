---
title: "One binary, eleven orders of magnitude: fak's dynamic range from super loops to nanosecond decides"
description: "The doctrine note for fak's temporal dynamic range: one Go binary adjudicates a tool call in ~0.4–2.4 µs and orients a days-long fleet intent, with the same closed refusal vocabulary, the same witness discipline, and the same admission structure at every band. Names the five wins that only exist because the bands share one binary, the risk register each win drags in (monoculture blast radius, version skew, fail-open budget coverage, missing bottom-up backpressure, cross-band causality, runtime-tail coupling), which guards already stand, and the gaps — each gap a filed ticket."
date: 2026-07-01
---

# One binary, eleven orders of magnitude: fak's dynamic range from super loops to nanosecond decides

> A `Decide` on the canonical allow costs **362 ns** (`BENCHMARK-AUTHORITY.md:285`). A
> `fak superloop walk tend` orients an operator intent whose members tick on 10-minute
> to daily cadences and settle over days. That is ~11 orders of magnitude of timescale
> served by **one Go binary**, and the span is not an accident of packaging — the wins
> below exist *because* the bands share one module, and each win drags a specific risk
> in with it. This note names both sides and files the gaps as tickets.

This is a positioning/doctrine note in the house style of the
[L3](L3-DISAGGREGATED-CACHE-REIMAGINED.md)/[L4](L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md)
studies: every "exists today" claim cites the file that proves it; every projection is
marked **unbuilt**; numbers are labeled witnessed/observed/modeled per the
[net-true-value standard](../standards/net-true-value.md). It deliberately does NOT
re-derive what three adjacent notes already own — the **what-gets-mutated** taxonomy
([three layers](RESEARCH-three-layers-of-agent-optimization-2026-06-24.md)), the
**gate-depth ladder** ([gate down the stack](EXPLAINER-gate-down-the-stack-2026-06-22.md)),
and the **KPI inventory** over the loop ladder
([agentic loop KPIs](AGENTIC-LOOP-KPIS-2026-06-25.md)). This note owns the axis none of
them state: what *one binary spanning all the clocks at once* buys, and what it costs.

## 1. The band ladder (what actually runs at each clock)

The loop ladder syscall → turn → session → fleet → rsi is already named
(`AGENTIC-LOOP-KPIS-2026-06-25.md:22`). Here it is with the measured anchors and the
guard that keeps each band honest. Every number is **witnessed** in-repo unless marked
otherwise; hardware and provenance live in `BENCHMARK-AUTHORITY.md`.

| Band | Clock | Representative decision | Measured anchor | Standing guard |
|---|---|---|---|---|
| **B0 decide** | ~0.4–2.4 µs | one in-process `Decide` fold on a proposed tool call | 362 ns/op allow (`BENCHMARK-AUTHORITY.md:285`); 2,427 ns in-process vs 6.913 ms spawned hook → **~2,849×** boundary tax (`:295-296`, `CLAIMS.md:23`) | no-exec structure (`internal/architest` `TestHotPathHasNoExec` + 3 sibling gates); 100 µs latency gate w/ ~40× headroom (`internal/gateway/adjudication_latency_test.go:29-43`); 5 µs turntaxmeter rung ceilings (`internal/turntaxmeter/overheadbudget.go:86-102`) |
| **B1 local serve** | ~3–90 µs | vDSO hit, result admission, L3 page verify | vDSO 1-shot serve p50 3,459 ns wall-clock (`examples/turntax/EXAMPLE-OUTPUT.md:31`); admit chain 29–87 µs (`BENCHMARK-AUTHORITY.md:292-293`); L3 read budget ~1–5 µs preserved by one-decision digest check (`internal/gateway/l3referee.go:31-37`) | ctxmmu/normgate 10 µs ceilings; **vDSO has no budget row — see gap G1** |
| **B2 turn** | seconds | one model round-trip + adjudicate the proposed call set | dominant cost is the model; planner work is Θ(c·N) op-counts, 8,000-token resident view default-on (`docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md:186`) | session gate at the turn boundary (`internal/agent/loop.go:315`); tool bytes held off-wire until the kernel sees the complete call set (`TOKEN-STREAMING-TTFT-2026-06-25.md`) |
| **B3 session** | minutes–hours | admit/throttle/drain/stop a run at turn quanta | Running→Throttled→Paused→Draining→Stopped (`internal/session/session.go:53-69`) | token/turn debit; stop only at a clean boundary; closed stop reasons |
| **B4 loop tick** | 5–30 min | dispatch one worker, close one witnessed issue | dispatch 10 min / progress 15 / status 30 / resume watchdog ~5 (`docs/dispatch-loop.md:215-216`, `cmd/fak/resume.go:1073`) | preflight cap = min(max-workers, dos target, host, seats) with `live = max(kernel lease count, OS scan)` (`internal/dispatchtick/preflight.go:157-183`); `COLLISION_RISK` on tree overlap (`cmd/fak/dispatch_tick.go:585-595`) |
| **B5 fleet wave** | hours | price a collision-free launch plan, bound the 529 burst | wave `LAUNCH_SAFE_SET`/`REPARTITION` pricing + prelaunch dry audit (`cmd/fak/dispatch_wave.go:269,580,649`); `fak resume admit` per-source gate (`cmd/fak/resume.go:154`) | fenced leases ride git refs cross-machine (`internal/leaseref/leaseref.go:1-13`); every spawn passes the same B4 tick |
| **B6 intent** | days | walk an operator intent worst-first over its member loops | `fak superloop walk` — WALK/SELECT/DESCEND/FOLD, all pure folds (`internal/superloop/superloop.go:418-530`) | fold refuses to read clean on any unmeasured or dark member (`superloop.go:483`); the walk mutates nothing by construction |
| **B7 self-change** | days–weeks | keep-or-revert a kernel candidate; cut a release | RSI loop isolated-worktree keep-or-revert (`internal/rsiloop`); `fak release ship --execute` | CI witness before tag; `CORE_SELF_MODIFY` refuses a commit that would judge its own edit; `FRONTIERSWE_SCORE_PARITY_FAILED` refuses speed-by-losing-quality |

The honest fence from the gate-ladder note carries over unchanged: **these are
different clocks, and they never multiply into one end-to-end number**
(`EXPLAINER-gate-down-the-stack-2026-06-22.md`, "What this picture is not"). The only
hard cross-band ratios are same-fold-two-transports measurements (the ~2,849× syscall
A/B; the ~10.7s → ~0.3s commit-gate fusion, `AGENTS.md:97-98`).

## 2. What one binary buys (the wins, each with its mechanism)

**W1 — One closed refusal vocabulary spans every band.** `STALE_LEASE` at a fence
(`internal/leaseref/fence.go:49-69`), `COLLISION_RISK` at a spawn,
`LOOP_DONE_UNWITNESSED` at a loop turn, `CI_BASE_RED` at a release: all tokens in one
declared set (`dos.toml [reasons]`, lookup via `dos check-reason`). A refusal minted at
any clock is machine-readable at every other clock — the B4 sweep terminates *because*
a B0-shaped refusal is a first-class value ("the loop never needs to know the cap, so
it can never exceed it", `internal/dispatchsweep/dispatchsweep.go:18-21`). No
cross-service schema negotiation exists because there is no second service.

**W2 — The same admission structure at every band.** The µs fence, the lane lease, the
spawn cap, and the wave plan are one structure (a lease census + a disjointness rule)
consulted at four clocks: `live = max(kernel lease count, OS worker scan)` folds the
kernel's own lease state into the fleet cap in-process
(`internal/dispatchtick/preflight.go:178-183`); `dos_arbitrate` is the same decision as
a pure function. There is no "orchestrator's view" vs "kernel's view" to reconcile.

**W3 — Transport collapse is a repeatable move, not a one-off.** Because every gate is
a Go fold in the same module, any boundary can be pulled in-process when its spawn cost
is measured to dominate: the syscall A/B (~2,849×), the commit-gate fusion (~10.7s →
~0.3s), and next the routing scout — classify on a local 135M model in tens of ms
instead of a 500–2000 ms remote call, *without pre-adjudication egress of the prompt*
(`MICRO-SCOUT-NATIVE-ROUTING-2026-07-01.md`). "Push the gate down the stack" is only a
standing option because the logic is importable at every altitude.

**W4 — One witness discipline at every clock.** The vDSO re-checks purity instead of
trusting a hint (`internal/vdso/vdso.go`); the L3 referee re-hashes returned bytes
(`L3_PAGE_DIGEST_MISMATCH`); a turn's "done" needs a loop witness
(`LOOP_DONE_UNWITNESSED`); a fleet close re-verifies per-SHA (`docs/dispatch-loop.md:152`);
a release tags only after the CI witness. The rule — *a claim never travels up a band
without its evidence* — is enforceable because every band emits into the same ledgers
and the same verifier reads them (`dos verify`, `fak commit-audit`).

**W5 — Budget envelopes have one shape everywhere.** A declared ceiling plus a closed
breach token: 5 µs rung ceilings → `OVERHEAD_BUDGET_EXCEEDED`
(`internal/turntaxmeter/overheadbudget.go`), 250 ms hook p99 → `GATE_LATENCY_REGRESSION`
(`internal/turntaxmeter/hooklat.go:53`), per-turn `budget.max_tokens` → `ActionStopBudget`
(`cmd/fak/loop_drive.go:453-468`), the spawn cap → `REFUSE_AT_CAP`, the four
generation-budget dimensions at B6 (`docs/generation-super-loop-budgets.md`). The
*shape* is uniform; the cascade is not yet plumbed (gap G4).

One more property is worth naming because it is easy to get wrong: **one binary, many
processes**. The same executable runs as the in-process gate inside `fak guard`, the
gateway, the dispatch tick, and the watchdog — but cross-band state moves only through
witnessable substrates (git refs for leases, durable ledgers, the run registry), never
through shared memory between processes. That is why any single process at any band can
be killed and restarted without corrupting another band's state, and why the fleet's
lease truth survives a host (`internal/leaseref/leaseref.go:1-25`).

## 3. Where spanning the gap bites (risks → standing guards → gaps)

**R1 — Monoculture blast radius.** One bad release lands at every band simultaneously:
a defect in a shared primitive (say, the lease census) is *correlated* across the µs
fence, the spawn cap, and the wave plan — the bands cannot fail independently because
they share code. Standing: `release_decide` hold tokens, the CI witness before tag,
`docs/ROLLBACK.md` + stable-release anchors, RSI keep-or-revert for kernel candidates.
Residual and accepted: correlation itself is the price of W1/W2; the mitigation is
rollback speed, not diversity.

**R2 — Version skew across running copies (the inverse of R1).** The binary is one;
the running copies are many and long-lived. A detached B4/B5 worker spawned yesterday
executes yesterday's admission logic against today's shared tree and registry.
**Witnessed today, on this host:** `fak version` prints `build: (no VCS stamp — built
without module/VCS provenance; cannot confirm the commit)` — the binary cannot even
attest which commit it is, so no gate *could* refuse a mixed-version wave right now.
Gap **G2** (ticket below). Prior art in-tree: the garden already flags "stale @latest"
(`cmd/fak/watchdog_autoheal.go:242-245`), but nothing folds binary identity into spawn
or wave admission.

**R3 — A µs regression multiplies up the ladder, and part of B1 is fail-open.**
A B0/B1 cost rides every tool call of every turn of every worker. The envelope system
exists (W5) but its coverage has a hole: the turntaxmeter budget table has **no `vdso`
row**, and `CheckSpan` is fail-open on undeclared rungs — "an undeclared rung can never
breach" (`internal/turntaxmeter/overheadbudget.go:138-147`). The gateway latency gate
covers the adjudication hop but explicitly not the vDSO fold path
(`internal/gateway/adjudication_latency_test.go:14-18`). So the single hottest serve
path is guarded structurally (no-exec) but not against a pure-Go latency regression.
Gap **G1** (ticket below).

**R4 — Backpressure flows down, not up.** Caps flow down the ladder (B6 floor → B4 cap
→ B3 budget → B0 ceiling), but health does not flow up: the spawn preflight folds the
lease census, host, seats, and accounts — and consults **no gate-latency signal**. A
fleet can keep admitting workers onto a kernel whose hook p99 has breached its 250 ms
budget; nothing turns `GATE_LATENCY_REGRESSION` into spawn reluctance. Gap **G3**
(ticket below). This is the classic control-theory failure of wide-dynamic-range
systems: the fast inner loop saturates silently while the slow outer loop keeps
commanding more load.

**R5 — Cross-band causality: a µs decision can zero a day of work.** The SWE-bench
Verified case is the canonical witness: the same model on the same instance resolves
through raw SGLang and is driven to an empty patch through the gateway, because a
deny the model never saw looped the run to its step limit
(`docs/benchmarks/SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md`, analyzed in
`RESEARCH-three-layers-of-agent-optimization-2026-06-24.md`). That is the trust floor
*working* — and it is also a B0 decision whose B4-scale cost was invisible at decide
time. The dynamic-range discipline this implies: deny verdicts need macro-cost
attribution in the full-span trace (part of G5) so a policy that is structurally
correct and economically ruinous is visible as such.

**R6 — Runtime-tail coupling.** All ns/µs anchors are Go benchmark numbers; in
production the same process runs the µs gate, SSE streaming, session bookkeeping, and
watchdog-autoheal probes, sharing one Go GC and scheduler. Whether B0's p99 holds while
the process does B3-band work is **unmeasured** — the 100 µs gate has ~40× headroom but
is not exercised under concurrent fleet load. Gap **G5** (ticket below).

**R7 — Self-reference at the top band.** B7 edits the binary that gates B0–B6.
Standing and adequate for now: `CORE_SELF_MODIFY` refuses a pathspec commit that
touches the machinery that would judge it; the RSI loop verifies candidates in an
isolated worktree and keeps-or-reverts on measurement; parity tokens refuse
speed-bought-with-quality. Held risk, no new ticket: the guards are recent — the right
next check is standing usage evidence, not new machinery.

## 4. The doctrine (six laws for an 11-order system)

1. **Every band declares its envelope, and coverage is part of the claim.** A budget
   table with a fail-open hole (R3) is debt with the same standing as a failing test.
2. **A claim never crosses a band without its witness.** Self-reports do not travel up;
   `dos status` structurally has no `claimed` field. This is W4 stated as an obligation.
3. **Push a gate down a band only with a same-fold-two-transports witness.** The
   ~2,849× pattern is the template; a descent without an A/B is a story, not a win.
4. **Refuse with the shared vocabulary at every clock.** A new band-specific prose
   refusal is a bug (`UNCLASSIFIED`); declare the token or use a standing one.
5. **Backpressure must flow both ways.** Admission caps down, health signals up. Half
   of this law is unbuilt (G3) — that is what makes it a law and not a description.
6. **One binary, many processes; state crosses bands only through witnessable
   substrates.** Never shared memory across band-processes; the restartability of
   every band depends on it.

## 5. Gaps → tickets (filed 2026-07-01)

Epic: **#2218** — the dynamic-range contract.

| Gap | Ticket | One line |
|---|---|---|
| G1 | #2219 | declare the vDSO budget row + a distribution gate on the syscall fold path (close the fail-open hole in R3) |
| G2 | #2220 | binary provenance + version-skew witness: VCS-stamp the build, surface per-worker binary identity, flag mixed-version waves (R2) |
| G3 | #2221 | bottom-up backpressure: fold gate-latency health into spawn preflight as a fifth cap term (R4) |
| G4 | #2222 | budget cascade: plumb the four generation-budget dimensions super loop → member → turn over the existing `FAK_GOAL_MAX_TOKENS` seam (W5 completion) |
| G5 | #2223 | full-span witnessed trace + tail-under-load: one run traced B6→B4→B2→B0 with per-band attribution (three-clocks fence), deny macro-cost attribution (R5), and B0 p99 under concurrent in-process fleet load (R6) |
| — | #2224 | superloop drive rung: enter the worst-first member through the same B4 admission gates, one member per walk, witnessed (the named next rung of `docs/super-loops.md:171-175`) |

## 6. Honest scope

- The band anchors are witnessed on the specific hardware named in
  `BENCHMARK-AUTHORITY.md`; the ladder's *shape* is portable, the numbers are not.
- Nothing here claims an end-to-end speedup across bands; per law 3 and the
  gate-ladder fence, cross-clock multiplication is refused.
- The wins W1–W5 are descriptions of shipped structure; the doctrine's laws 1, 5 and
  the R5 attribution idea are **unbuilt** where marked — the tickets are the work, and
  until they land the honest status of the dynamic-range contract is `not yet`.
