# Context-safety liveness-contradiction light (#1230, C12 of epic #1217)

_Research / design only. This is the **C12 spec** for visual primitive **(6)**,
the liveness-contradiction light, under the context-safety epic
[#1217](https://github.com/anthony-chaudhary/fak/issues/1217). It specs the one
panel that catches the **#1143 "healthy but serving nothing"** failure (the GDN
evict panic: `/healthz` still answers `200` but every decode hangs) — cataloged
as **F1** in the C2 failure-mode catalog
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
It reuses the **#750 two-heartbeat oracle** pattern, obeys the C1 doctrine
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
cross-checks **A** (paired-axis: up vs throughput = 0) and **D** (re-derive both
signals from their metrics), and conforms to the primitive-6 schema row sketched
in the C6 note
([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md)).
The re-derivation contract it leans on is C9
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md)).
No code ships here — the deliverable is this committed spec._

---

## The failure this light exists to catch

The other five context-safety primitives answer "did fak destroy *value* when it
shed context?" This one answers a sibling question one rung lower: **"did fak's
shed leave the server alive-looking but dead?"** A liveness probe is the cheapest
self-report in the stack, and the #1143 GDN evict panic is the canonical case
where it lies.

The sequence (F1, from the C2 catalog):

1. A budget eviction calls `Evict` on a hybrid Gated-DeltaNet (recurrent) KV
   cache, which cannot evict a middle span — the typed boundary
   `*model.RecurrentEvictUnsupportedError`
   (`internal/model/kvcache.go:54`).
2. On the *un-recovered* path the panic kills the decode goroutine. The HTTP
   liveness surface — `/healthz`, and the `fak_bgloop_up` heartbeat pulse — keep
   answering, because they live on a different goroutine that never touched the
   cache.
3. Result: **`/healthz` = `200`, `fak_bgloop_up` = `1`, and decode throughput =
   `0`.** Looks healthy, serves nothing. No journal row is written on the
   un-recovered path (the panic kills the handler before any `CAP_EVICT` /
   `CAP_FAULT` row is emitted), so the *event* half is invisible — the only
   observable is the **contradiction between the two heartbeats**.

The recovered path exists — `recoverRecurrentEvictUnsupported`
(`internal/gateway/gateway.go:1266`), wired into the `complete` panic-recover
deferred at `internal/gateway/gateway.go:1216` — and turns the panic into a
returned typed error. But the light must not assume recovery: it must surface the
contradiction *whether or not* the handler recovered, because a recovery that
silently zeroes throughput is still "serving nothing." The light watches the two
heartbeats; it does not trust either one alone.

---

## The two heartbeats — what the light reads

The doctrine's paired-axis rule (cross-check **A**) is the whole design here: a
liveness panel that shows *one* signal is a lie by construction, because the
failure is precisely a healthy first signal hiding a dead second one. The light
reads **two independent signals from two independent code paths** and reds when
they disagree.

### Heartbeat 1 — the liveness pulse (`fak_bgloop_up`)

The always-on kernel heartbeat loop is registered by the gateway's in-kernel
background-loop supervisor (`internal/gateway/bgloops.go:32-43`): a pure liveness
pulse whose `Tick` returns `nil` and whose only job is to prove "the loop runtime
is alive without an external scheduler firing it" (the doctrine comment,
`bgloops.go:18-22`). Its gauge is emitted at
`internal/gateway/bgloops.go:227-234` —
`fak_bgloop_up{loop,state} = 1` unless `state == bgloop.StateStopped`. This is the
**self-report** signal: it says "the process and its loop runtime are up." It is
exactly the signal that stays green during the F1 panic, because a stalled decode
does not stop the heartbeat loop.

### Heartbeat 2 — decode throughput (the work signal)

The orthogonal signal is **actual model work completing**:
`fak_gateway_inference_decode_tokens_per_second`
(`internal/gateway/metrics.go:1970-1975`) — completion tokens per second of decode
wall-clock, computed as `measuredComplTok / measuredDecodeSecs`, and explicitly
**falling back to `0` (not the blended rate) when no turn measured TTFT**
(`metrics.go:1968`) so the two rates never silently coincide. Its underlying
accumulators are `inferMeasuredDecodeSecs` (`metrics.go:79-85`) and the total
`inferDecodeSecs` (`metrics.go:66`), surfaced cumulatively as
`fak_gateway_inference_duration_seconds_total`
(`internal/gateway/metrics.go:1937-1938`). This is the **witnessed-work** signal:
it only advances when the planner actually generated tokens. During the F1 panic
it sits at `0` while heartbeat 1 reads `1`.

The two are independent by construction — one is a timer-driven nil-tick on the
supervisor goroutine, the other is the planner's measured decode accumulator on
the request path. A single bug cannot move both the same way, which is what makes
their *disagreement* a witness rather than a coincidence.

---

## The #750 two-heartbeat oracle pattern, reused

The #750 oracle is the repo's standing pattern for liveness that does not trust a
single "I'm up" bit: **register-says-alive is one heartbeat; actually-made-progress
is the second, and the verdict is the disagreement between them, never either one
alone.** Its in-tree analog is the loop-health `deriveState` fold
(`internal/loopmgr/health.go:281-312`), which classifies a loop **DARK** precisely
when the registry says the loop *exists / is up* but its last-tick timestamp has
not advanced past its cadence — a registered-but-not-ticking contradiction. The
loop-health fold renders that as the `fak_bgloop_last_tick_timestamp_seconds`
gauge (`internal/gateway/bgloops.go:219-225`) read *against* `fak_bgloop_up`
(`bgloops.go:227-234`): up=1 plus a last-tick that has gone stale is the same
disagree shape.

C12 reuses that shape, but swaps the second heartbeat from "loop ticked" to
"decode produced tokens," because the F1 failure is specifically a *serving*
stall, not a *loop* stall:

| Oracle role | #750 / loop-health analog | C12 liveness-contradiction light |
|---|---|---|
| Heartbeat 1 (self-report "up") | `fak_bgloop_up` (`bgloops.go:227-234`) | `fak_bgloop_up` (`bgloops.go:227-234`) / `/healthz` |
| Heartbeat 2 (witnessed progress) | last-tick advanced (`fak_bgloop_last_tick_timestamp_seconds`, `bgloops.go:219-225`) vs cadence (`health.go:281-312`) | decode rate advanced (`fak_gateway_inference_decode_tokens_per_second`, `metrics.go:1970-1975`) |
| Verdict | DARK when up=1 but last-tick stale | **RED** when up=1 but decode-rate=0 over a window with admitted turns |
| Honesty | the contradiction is the witness, not either bit | identical — neither heartbeat alone may turn the light green |

The point of reusing #750 rather than inventing a new probe: the disagree
predicate is already the repo's accepted definition of "alive-looking but dead,"
and the C2 catalog (F1) already names this light as its reuse.

---

## The `LivenessContradiction` datum

The schema the panel consumes — the two heartbeats, the disagree predicate, the
RED state, each field re-derivable from a metric so the light is a *witness*, not
an assertion:

```
LivenessContradiction {
  // Heartbeat 1 — the self-report "up" pulse.
  Up            bool      // fak_bgloop_up == 1 (heartbeat loop not StateStopped) AND /healthz==200
  UpSource      string    // "metrics:fak_bgloop_up" — tamper-evident accumulator, not a cached render

  // Heartbeat 2 — the witnessed-work pulse.
  DecodeRate    float64   // fak_gateway_inference_decode_tokens_per_second, over the window
  DecodeSource  string    // "metrics:fak_gateway_inference_decode_tokens_per_second"

  // The admitted-demand guard — so an idle server is not flagged (see predicate below).
  AdmittedTurns int64     // turns admitted in the window (a quiet server has 0 and is GREEN)

  // The verdict.
  State         string    // GREEN | RED | UNKNOWN
  Disagree      bool      // the predicate below
  Window        Duration  // the evaluation window (a single tick is never a verdict)
}
```

### The disagree predicate (the RED-on-disagree rule)

```
Disagree  ==  Up == true
          &&  AdmittedTurns > 0          // demand existed in the window
          &&  DecodeRate == 0            // …but no tokens were produced
          &&  sustained over Window      // not a single scrape (false-red fence)

State = RED      when Disagree
        GREEN    when Up && (DecodeRate > 0 || AdmittedTurns == 0)
        UNKNOWN  when !Up                 // a down pulse is the EXISTING liveness alarm, not this light's job
```

Three deliberate guards, each closing a false-positive:

- **`AdmittedTurns > 0`** — an idle server legitimately has `DecodeRate == 0`. The
  light reds only when *demand existed and was not served*. The admitted-turn
  count is itself a witnessed metric — `fak_sched_admitted_total`
  (`internal/gateway/admission.go:507`) over the served-admission path
  (`completeServed` → `beginServedAdmission`,
  `internal/gateway/gateway.go:1279-1291` / `admission.go:544`) — not an assumption.
- **`sustained over Window`** — a single zero-rate scrape between turns is normal.
  The verdict is a change-point/persistence judgment over a window, never a single
  tick — the same false-red fence C10 (#1228) puts on the CI gate.
- **`State = UNKNOWN when !Up`** — if the up-pulse itself is down, the *ordinary*
  liveness alarm already fires; this light's distinct job is the **up-but-not-serving**
  contradiction, so it declines to judge rather than double-alarm.

The rendered shape is a single binary light (the C6 primitive-6 shape), but it is
**driven by the paired time-series** of the two heartbeats so the contradiction is
re-derivable from the underlying metrics, not asserted by the renderer.

---

## How D makes it cross-checkable — re-derive both signals

Per C9 (#1227), the light is **GREEN only if both heartbeats re-derive from a
tamper-evident source and the predicate over them re-derives to the same verdict
the light shows.** Both signals are WITNESSED `/metrics` accumulators (monotonic
counters / gauges; a re-scrape re-reads the same accumulator, not a cached render
— the C9 tamper-evidence rule for the metric source):

| Number on the light | Re-derived from | Source kind (C9) | Tolerance |
|---|---|---|---|
| `Up` | `fak_bgloop_up` gauge (`bgloops.go:227-234`) | `metrics` (WITNESSED) | exact (0/1) |
| `DecodeRate` | `fak_gateway_inference_decode_tokens_per_second` (`metrics.go:1970-1975`), itself re-derivable from `measuredComplTok / measuredDecodeSecs` | `metrics` (WITNESSED) | exact `0` for the `==0` test |
| `AdmittedTurns` | the scheduler admitted-total counter (`fak_sched_admitted_total`, `internal/gateway/admission.go:507`) over the served-admission path (`completeServed` → `beginServedAdmission`, `gateway.go:1279-1291` / `admission.go:544`) | `metrics` (WITNESSED) | exact integer |
| `Disagree` / `State` | recompute the predicate over the three re-derived numbers | derived | must equal rendered `State` |

The light's `NumberCheck` set (C9 `RederivationVerdict`) therefore re-derives the
*verdict itself*: the renderer cannot show GREEN over a window in which the
re-derived predicate is RED, and vice-versa. If the decode-rate metric is
unavailable (no turn ever measured TTFT, so the rate falls back to `0`), the light
cannot distinguish "stalled" from "never warmed up" — that number is **not
re-derivable as a disagree witness**, so the panel refuses green with the C13
(#1231) reason **`SAFETY_UNWITNESSED`** rather than rendering an unearned GREEN.
A re-derivation that *cannot* be performed is REFUSED, never silently trusted (C9
acceptance point 4).

This is the doctrine's paired-axis (**A**) and re-derivation (**D**) composed: A
guarantees the visual *shows* both axes (up AND throughput); D guarantees a reader
can *falsify the light from the metrics themselves* — re-scrape the two
accumulators, recompute the predicate, and confirm the light's color. Neither the
operator nor the renderer is trusted; the two metric taps are.

---

## Honest `not yet` gaps (per fence c)

- **No panel re-derives this predicate today — `not yet`.** Both heartbeats exist
  as WITNESSED metrics (`fak_bgloop_up`, `fak_gateway_inference_decode_tokens_per_second`)
  and the #750 disagree shape exists for *loops* (`loopmgr/health.go:281-312`), but
  **no rendered light watches `fak_bgloop_up` against decode-rate and reds on the
  contradiction.** The signals are witnessed; the contradiction surface is unbuilt.
  This note is the contract; the build is the gated follow-on.
- **The F1 *event* half is invisible on the un-recovered path — `not yet`.** When
  the panic kills the handler before any `CAP_EVICT` / `CAP_FAULT` journal row is
  written, the only observable is the throughput contradiction — there is **no
  tamper-evident event row** the C9 checker can anchor to. The light's verdict
  therefore re-derives from the metric stream (WITNESSED) but is **not
  journal-anchored**; under C9 this is GREEN-eligible but lower-assurance than a
  chain-anchored row, and the missing row is the C13 **`SAFETY_UNRECOVERED`**
  condition (the event happened but no tamper-evident witness records it). Pinning
  an F1 stall to a journal row is unfiled work.
- **The C13 refusal tokens (`SAFETY_UNWITNESSED` / `SAFETY_UNRECOVERED`) do not
  exist yet.** They are specified by C13 (#1231) as new `dos.toml [reasons]`; until
  they are filed and pass `dos_check_reason`, the light cannot emit a *verifiable*
  refusal — it can only render UNKNOWN. The doctrine requires the tokens; they are
  the C13 deliverable, not this note's.
- **Window / persistence thresholds are unfixed.** The false-red fence (sustained
  over `Window`, not a single tick) is stated as a rule; the concrete window length
  and persistence count are deferred to C10 (#1228), which owns when a RED verdict
  fails `make ci`.

---

## Acceptance check (against #1230)

The issue's acceptance: *the two signals (`/healthz` / liveness pulse vs
decode-throughput) and the RED-on-disagree rule, reusing the #750 two-heartbeat
oracle, with D making it cross-checkable.*

- **The two signals** are named and bound to real seams: heartbeat 1 =
  `fak_bgloop_up` (`internal/gateway/bgloops.go:18-22,32-43`, emitted at
  `internal/gateway/bgloops.go:227-234`) / `/healthz`; heartbeat 2 = decode
  throughput `fak_gateway_inference_decode_tokens_per_second`
  (`internal/gateway/metrics.go:1970-1975`) over the
  `fak_gateway_inference_duration_seconds_total` accumulator (`metrics.go:1937-1938`).
- **The RED-on-disagree rule** is stated as the `Disagree` predicate — `Up &&
  AdmittedTurns>0 && DecodeRate==0`, sustained over a window — with the three
  false-positive guards (idle-server, single-scrape, down-pulse → UNKNOWN).
- **The #750 two-heartbeat oracle reuse** is shown explicitly: the
  self-report-vs-witnessed-progress shape, mapped to its in-tree analog
  `loopmgr/health.go:281-312` (DARK = up but last-tick stale) and reused with
  decode-progress as the second heartbeat.
- **The F1 binding** is grounded: `*model.RecurrentEvictUnsupportedError`
  (`internal/model/kvcache.go:54`) + the recovery
  `recoverRecurrentEvictUnsupported` (`internal/gateway/gateway.go:1266`) wired at
  `gateway.go:1216` — the light surfaces the contradiction whether or not the
  handler recovered.
- **D makes it cross-checkable**: every number (`Up`, `DecodeRate`,
  `AdmittedTurns`, and the `Disagree`/`State` verdict) re-derives from a
  tamper-evident WITNESSED metric (C9 #1227), and a number that cannot be
  re-derived as a disagree witness refuses green with a C13 reason rather than
  rendering an unearned OK.
- **Cross-check A + D** are the two the issue names; both are specified and bound
  to the C1 doctrine and the C6 primitive-6 schema row.

**Honesty fences carried (from #1217).** (a) This liveness light is the
*serving-alive* dual sibling of the value-preservation guarantees, kept distinct
from the security floor and from the #1147 self-tax / mediation-overhead plane —
cross-linked, never blended into one "fak is up" number. (b) Both heartbeats are
WITNESSED metrics; nothing is summed across a WITNESSED and an OBSERVED lane (the
light never reads a provider liveness bit). (c) The un-built surface, the
journal-anchored event, and the C13 tokens are named as `not yet` gaps, not
rendered as results. (d) Design-only.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Renders C2
failure F1 (#1219); reuses the #750 two-heartbeat oracle; conforms to the C6
primitive-6 schema (#1223); leans on the C9 re-derivation contract (#1227) and the
C13 refusal tokens (#1231); shares its window/persistence fence with the C10 CI
gate (#1228). Design-only — no implementation ships under #1217 until the notes
are reviewed._
