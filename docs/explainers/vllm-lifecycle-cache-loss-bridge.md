# The vLLM lifecycle cache-loss bridge

*Bridging fak session dormancy to an upstream vLLM worker's sleep/pause/wake
controls, with explicit evidence of what happened to the KV cache. Issue
[#1730](https://github.com/anthony-chaudhary/fak/issues/1730); parents #1178,
#1193, #1203, #1204.*

## The failure this prevents

An external vLLM worker can go dormant in ways that silently throw the KV cache
away. Per vLLM's own `docs/features/sleep_mode.md`:

- **sleep level 1** offloads weights to CPU memory but *forgets the KV cache*;
- **sleep level 2** forgets *both* weights and the KV cache;
- a **prefix-cache reset** forgets the KV while the engine stays up;
- a **pause** (scheduler stop) keeps the KV resident in place.

fak's reuse layer tracks which prefixes are "warm" — resident KV it can route a
resume onto for a cache discount. If the engine underneath sleeps or resets and
fak keeps reporting a prefix warm, a resume routes as a **cache hit onto memory
the engine no longer holds** — a false warm hit — and a fleet view can imply the
worker is healthily serving while it is actually asleep. That is a correctness
hole on the default serving path, not a cosmetic one.

## The contract

Every session-driven dormancy action lowers to one immutable **cache-loss
witness** row that answers three questions honestly:

1. **What happened to the KV?** A closed three-value vocabulary —
   `preserved` / `forgotten` / `unknown`. `unknown` is a first-class answer,
   never silently collapsed to `preserved`; when fak cannot prove the KV
   survived it fails closed (treats it as potentially lost).
2. **Must warm-prefix beliefs be demoted cold?** True on *any* KV loss and on
   `unknown`. This is what stops a stale warm belief from surviving dormancy.
3. **Can the engine actually serve?** `Serving` is false whenever the engine
   cannot serve (paused, sleeping, error), so a metrics or `session ls` surface
   never reads "healthy" for an asleep worker.

| Action  | Phase      | KV          | Warm demoted | Serving |
|---------|------------|-------------|--------------|---------|
| `pause` | `paused`   | `preserved` | no           | no      |
| `sleep` (level 1/2) | `sleeping` | `forgotten` | yes | no |
| `sleep` (no level)  | `error`    | `unknown`   | yes | no |
| `reset` | `serving`  | `forgotten` | yes          | yes     |
| `wake`  | `serving`  | `unknown`   | yes          | yes     |

`wake` is deliberately *not* a re-warm: the KV forgotten during sleep is not
restored, so warm beliefs stay cold until a fresh cache signal proves residency.

### The revalidation rule

A resume may only claim a warm prefix hit against a possibly-slept worker after a
**fresh `BlockStored`/cache signal** re-proves the KV is resident. The gate
starts unproven, any KV-loss (or `unknown`) witness drives it back to unproven,
and only a fresh cache signal re-warms it — the exact acceptance rule "resume
refuses a warm hit after a vLLM sleep/reset unless a fresh BlockStored/cache
signal revalidates it."

## Where it lives

The pure, wall-clock-free witness kernel is shipped in
[`internal/cachemeta/sleep_witness.go`](../../internal/cachemeta/sleep_witness.go)
(`WitnessDormancy`, the `EngineSleepLevel`/`EngineDormancyAction`/`KVDisposition`/
`EngineLifecyclePhase` vocabularies, `SleepWitness`, and the `WarmHitGate`
revalidation state machine), with the decision tests in
`internal/cachemeta/sleep_witness_test.go`. The kernel owns *only* belief/witness
logic: by design it never calls a vLLM control endpoint (that is the engine
adapter's job — the same boundary `external_invalidation.go` keeps) and emits no
metrics itself; it exposes the honest `Serving` bit a metrics / `session ls`
layer reads.

Keep this closed cache-loss vocabulary (`preserved` / `forgotten` / `unknown`)
identical to the NIXL disaggregated-lease demotion terms
([#1732](https://github.com/anthony-chaudhary/fak/issues/1732)) so the two warmth
sources never drift.

## State of the bridge (2026-07-03)

**Shipped:** the witness kernel and its decision tests
(`6c8ec8a4`, `internal/cachemeta`; `go test ./internal/cachemeta/` green).

**Not yet wired** — the remaining acceptance is engine/CLI wiring, which is out
of the `docs` lane and blocked for a guarded self-source worker
(`internal/**` / `cmd/**`):

- the `internal/engine` lifecycle adapter driving vLLM `/sleep`·`/wake_up`
  *only* through a fak session lifecycle action, and folding each transition into
  a `SleepWitness`;
- the resume warm-hit path consulting `WarmHitGate` so a post-sleep resume
  returns a miss until revalidated;
- the `sleeping` / `waking` / `error` state surfaced in `fak session ls --fleet`
  and metrics, distinct from `serving`;
- the integration test against a *fake vLLM control endpoint* (credential-free,
  no live GPU) proving sleep(level=1/2) records `kv_cache=forgotten`, demotes
  warm prefixes, and that resume refuses a warm hit post-sleep until revalidated.

**Generation classification:** `gen/next` (Generation G1 — Next Gen). This is
near-term foundation whose vocabulary and belief kernel have landed but which
still needs a gate + default-exposure proof (the `session ls --fleet` / metrics
surface and the fake-endpoint integration test) before agents can rely on it.

- *Promotion evidence:* the closed cache-loss vocabulary and the pure witness /
  `WarmHitGate` kernel landed with six passing decision tests, retiring the "is
  the cache-loss contract decided?" blocker.
- *Demotion / retirement evidence:* none — the issue is neither obsolete nor
  duplicated; the wiring is genuinely pending, so it stays open, not retired.
- *Invalidating assumption:* the pending integration test assumes the fake
  control endpoint mirrors vLLM's public `/sleep`·`/wake_up` + prefix-cache-reset
  surface (`docs/usage/security.md`). If a different control-API shape is
  intended, the adapter wiring and that test are built against the wrong seam —
  operator input needed before the engine-lane leaf lands.
