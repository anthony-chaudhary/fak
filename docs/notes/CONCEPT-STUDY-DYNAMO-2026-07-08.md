# study-repo: NVIDIA dynamo → fak (2026-07-08)

A `study-repo` pass over **[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)**, the
open-source datacenter-scale inference-orchestration stack. Full clone (whole repo, complete
history), pinned at **`be36b5239636edb1956c243899fc9a099ebafd0e`** (HEAD 2026-07-08).

## Why dynamo is a direct reference

fak's own milestones map almost 1:1 onto dynamo's architecture — M2 *KV cache owned/observed/2x*,
M3 *Disaggregated serving at throughput parity*, M9 *Dispatch fleet to 100 workers*, M10 *Model
support & routing parity* — plus the `agentic-serving` area. dynamo is a fleet-of-inference-workers
orchestrator; fak is a fleet-of-agent-sessions orchestrator. The transferable surface is the
**orchestration pattern** (routing cost functions, autoscaler control laws, fault-tolerant
migration, queue-lane admission, load-signal aggregation) — **not** the GPU/CUDA/NIXL/KV-tensor
code, which fak has no use for. The structural mapping used throughout: worker = agent session,
KV-cache overlap = warm-context / prompt-cache overlap, GPU load = worker in-flight load.

## License gate

dynamo is **Apache-2.0** (with an MIT test-data carve-out under `lib/llm/tests/data/deepseek-v3.2`
we did not touch); fak is **Apache-2.0**. Integrate would be license-legal, but every candidate is
Rust/Python → Go, so **all are `inspire`** (clean-room reimplementation, source cited). No bytes
vendored.

## What was read (code, not the pitch)

Six parallel readers over the load-bearing modules, each grounding candidates at a real
`path:line`, then a per-candidate witness against fak (`fak_feature_query`/`fak capabilities`/`fak
index` + raw grep):

- **Routing policy** — `lib/kv-router/src/scheduling/{selector.rs,policy_queue.rs}` (the real
  scheduling crate: cost function, DRR queue, exact-worker lanes).
- **Session affinity** — `lib/llm/src/session_affinity/{coordinator.rs,push_router.rs}`.
- **Planner / autoscaler** — `components/src/dynamo/planner/core/{throughput_scaling,load_scaling,state_machine}.py`.
- **Fault tolerance / migration** — `lib/llm/src/migration.rs`, `lib/runtime/src/health_check.rs`,
  `lib/runtime/src/pipeline/network/egress/push_router.rs`.
- **Discovery / registry** — `lib/runtime/src/{component/client.rs,discovery/mod.rs,storage/kv/etcd.rs}`.
- **Load telemetry** — `lib/llm/src/kv_router/publisher/worker_metrics.rs`,
  `lib/llm/src/discovery/worker_monitor.rs`, `lib/kv-router/src/sequences/prompt_registry.rs`.

**29 candidates, all grounded at a verified `path:line@sha`; witness: 2 PRESENT, 27 PARTIAL, 0
ABSENT.** The top-3 citations were spot-read to confirm the code implements the technique (not a
README claim): the `1/(1+decay·excess_load)` overlap decay, the `cost ≤ deficit` DRR ring, and the
`track_response` replay capture all verified verbatim.

## The decisive finding — fak already owns the biggest clusters

The witness stage checked *code*; a follow-up backlog dedup found fak already tracks the largest
borrows as open issues. So those candidates became **code-grounded enrichment comments**, not new
tickets (the anti-monolith / anti-re-file discipline):

- **#2238** *fleet-level KV-aware routing* (explicitly names dynamo's router) ← the worker cost
  function, rational overlap-credit decay, and load-floor tie-break. fak's `CacheAwarePolicy`
  (`internal/gateway/residency_router.go:489`) already does `overlap/(1+load)`; the delta is decay
  by **excess-above-floor** + a continuous yield instead of the hard `SkewThreshold` cutover.
- **#1913** *DAG-aware co-scheduling* (SAGA "Agent Fair Share") ← the Deficit-Round-Robin policy
  queue and exact-worker HOL-free lanes, as a concrete impl of the fairness lever.
- **#2037** *prompt-cache-affinity routing* ← dynamo's five session-affinity coordinator mechanisms
  (single-flight pin, lease + idle-TTL, opportunistic recording, rebind-on-death, idle-reaper GC).

Two candidates witnessed **PRESENT** and were dropped: three-tier worker state and reported+in-flight
load blending both already live in `internal/dispatchtick/accounts.go`.

## Filed this pass (Focused scope) — the genuinely-absent track

Backlog searches confirmed no existing issue for in-flight migration / hung-worker detection.
Filed **epic #3352** *fault-tolerant in-flight session migration* (milestone M1) + 4 leaves:

| # | leaf | source `path:line@sha` | fak seam |
|---|---|---|---|
| #3353 | warm-continue a live turn (replay-as-context) | `lib/llm/src/migration.rs:342@be36b52396` | `internal/gateway/messages_stream_passthrough.go:329` |
| #3354 | migratable vs non-migratable relaunch classification | `lib/llm/src/migration.rs:59@be36b52396` | `internal/resume/outcome.go:70` |
| #3355 | replay-safety precondition guard (fail-closed) | `lib/llm/src/migration.rs:214@be36b52396` | `internal/resume/stopped/stopped.go:245` |
| #3356 | silence-is-failure hung-worker detection | `lib/runtime/src/health_check.rs:99` + `.../egress/push_router.rs:1547@be36b52396` | `cmd/fak/watchdog_autoheal.go:464` + `internal/gateway/messages_stream_passthrough.go:341` |

## Deferred (available for a Broad follow-on pass, not filed)

Real PARTIAL borrows held back to keep this pass focused; each is a small ship-alone leaf if wanted:

- **Autoscaler control laws (→ M9):** two-timescale predictive-floor + reactive-clamp (#11), consolidation-aware scale-down veto (#12), settle-before-redecide gate (#13), dual-cadence lazy pull (#14).
- **Registry / lease liveness:** lease-keyed registration cascade-delete on disconnect (#20), soft-evict + reconciler re-admission (#21), coalesced single-watch discovery (#23), register-verify-rollback TOCTOU guard (#24).
- **Telemetry hygiene:** change-gated + debounced load publish (#25), hysteretic anti-flap overload latch (#29).

## Full candidate table

| # | subsystem | borrow (technique) | source `path:line@sha` | witness | route | disposition |
|---|---|---|---|---|---|---|
| 1 | kv-router-policy | Worker-selection cost function (load − warm-credit, argmin) | `lib/kv-router/src/scheduling/selector.rs:205@be36b52396` | PARTIAL | inspire | enriched #2238 |
| 2 | kv-router-policy | Rational overlap-credit decay by excess backlog | `lib/kv-router/src/scheduling/selector.rs:196@be36b52396` | PARTIAL | inspire | enriched #2238 |
| 3 | kv-router-policy | Deficit Round Robin weighted fair-share across classes | `lib/kv-router/src/scheduling/policy_queue.rs:559@be36b52396` | PARTIAL | inspire | enriched #1913 |
| 4 | kv-router-policy | Exact-worker lane queues (HOL-free, blocked-workers set) | `lib/kv-router/src/scheduling/policy_queue.rs:296@be36b52396` | PARTIAL | inspire | enriched #1913 |
| 5 | kv-router-policy | Reservoir-sampling tie-break at the load floor | `lib/kv-router/src/scheduling/selector.rs:386@be36b52396` | PARTIAL | inspire | enriched #2238 |
| 6 | session-affinity | Lazy first-touch pin with single-flight coordination | `lib/llm/src/session_affinity/coordinator.rs:193-215@be36b52396` | PARTIAL | inspire | enriched #2037 |
| 7 | session-affinity | Ref-counted lease with idle-TTL refreshed on release | `lib/llm/src/session_affinity/coordinator.rs:530-534@be36b52396` | PARTIAL | inspire | enriched #2037 |
| 8 | session-affinity | Opportunistic affinity (record the LB choice) | `lib/llm/src/session_affinity/push_router.rs:326-333@be36b52396` | PARTIAL | inspire | enriched #2037 |
| 9 | session-affinity | Availability re-check + one-shot invalidate-and-rebind | `lib/llm/src/session_affinity/push_router.rs:112-124@be36b52396` | PARTIAL | inspire | enriched #2037 |
| 10 | session-affinity | Idle-reaper GC of pin table + bounded reservation cap | `lib/llm/src/session_affinity/coordinator.rs:137-149@be36b52396` | PARTIAL | inspire | enriched #2037 |
| 11 | planner-autoscaler | Two-timescale control: slow loop sets FLOOR, fast clamps up | `components/src/dynamo/planner/core/throughput_scaling.py:52@be36b52396` | PARTIAL | inspire | deferred (M9 autoscaler) |
| 12 | planner-autoscaler | Consolidation-aware scale-down veto (re-predict survivors) | `components/src/dynamo/planner/core/load_scaling.py:549@be36b52396` | PARTIAL | inspire | deferred (M9 autoscaler) |
| 13 | planner-autoscaler | Settle-before-redecide gate (expected≠ready ⇒ hold) | `components/src/dynamo/planner/core/state_machine.py:334@be36b52396` | PARTIAL | inspire | deferred (M9 autoscaler) |
| 14 | planner-autoscaler | Dual-cadence pipeline with lazy expensive-observation pull | `components/src/dynamo/planner/plugins/orchestrator/engine_adapter.py:822@be36b52396` | PARTIAL | inspire | deferred (M9 autoscaler) |
| 15 | fault-tolerance-migration | Transparent in-flight migration (replay delivered output) | `lib/llm/src/migration.rs:342@be36b52396` | PARTIAL | inspire | filed #3353 |
| 16 | fault-tolerance-migration | Closed migratable/non-migratable error classification | `lib/llm/src/migration.rs:59@be36b52396` | PARTIAL | inspire | filed #3354 |
| 17 | fault-tolerance-migration | Resume-safety precondition guard (fail-closed) | `lib/llm/src/migration.rs:214@be36b52396` | PARTIAL | inspire | filed #3355 |
| 18 | fault-tolerance-migration | Passive-first liveness with activity-reset canary | `lib/runtime/src/health_check.rs:99@be36b52396` | PARTIAL | inspire | filed #3356 |
| 19 | fault-tolerance-migration | Silence-is-failure inactivity timeout ⇒ quarantine + retry | `lib/runtime/src/pipeline/network/egress/push_router.rs:1547@be36b52396` | PARTIAL | inspire | filed #3356 |
| 20 | runtime-discovery-registry | Lease-keyed registration, cascade-delete on disconnect | `lib/runtime/src/storage/kv/etcd.rs:51-52@be36b52396` | PARTIAL | inspire | deferred (registry liveness) |
| 21 | runtime-discovery-registry | Soft-eviction + periodic reconciler re-admission | `lib/runtime/src/component/client.rs:611-659@be36b52396` | PARTIAL | inspire | deferred (registry liveness) |
| 22 | runtime-discovery-registry | Three-tier worker state (fault vs backpressure separate) | `lib/runtime/src/component/client.rs:193-331@be36b52396` | PRESENT | inspire | DROP (PRESENT) |
| 23 | runtime-discovery-registry | Coalesced single-watch discovery source (ref-counted) | `lib/runtime/src/component/client.rs:672-752@be36b52396` | PARTIAL | inspire | deferred (registry liveness) |
| 24 | runtime-discovery-registry | Register-verify-rollback optimistic conflict detection | `lib/runtime/src/discovery/mod.rs:789-844@be36b52396` | PARTIAL | inspire | deferred (registry liveness) |
| 25 | load-telemetry | Change-gated + debounced per-worker load publish | `lib/llm/src/kv_router/publisher/worker_metrics.rs:92@be36b52396` | PARTIAL | inspire | deferred (telemetry hygiene) |
| 26 | load-telemetry | Blend reported load + own in-flight dispatched count | `lib/kv-router/src/sequences/prompt_registry.rs:41@be36b52396` | PRESENT | inspire | DROP (PRESENT) |
| 27 | load-telemetry | Routing cost = load·(queued − warm_credit) + in_flight (=#1+#2) | `lib/kv-router/src/scheduling/selector.rs:196@be36b52396` | PARTIAL | inspire | enriched #2238 |
| 28 | load-telemetry | Softmax-temperature spread across near-equal workers | `lib/kv-router/src/scheduling/selector.rs:65@be36b52396` | PARTIAL | inspire | enriched #2238 |
| 29 | load-telemetry | Hysteretic, change-gated anti-flap overload latch | `lib/llm/src/discovery/worker_monitor.rs:264@be36b52396` | PARTIAL | inspire | deferred (telemetry hygiene) |

## Honest limits

- The witness is **lexical + a snapshot**: `fak_feature_query`/`fak index` is substring ranking
  (guarded here with raw grep against a false-ABSENT), true only as of 2026-07-08. Re-witness before
  acting on an old row.
- The two PRESENT drops mean fak already ships the mechanism; the enrichment comments describe the
  *specific delta* dynamo has, not a from-scratch gap.
- License reading is a good-faith Apache-2.0-vs-Apache-2.0 compatibility check, not legal advice.

## Companions

- Filed: epic **#3352** → leaves **#3353–#3356** (M1 durable sessions).
- Enriched: **#2238** (KV-aware routing), **#1913** (agentic-first scheduling, epic #1911), **#2037**
  (prompt-cache-affinity routing).
- Skills: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass),
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) (the witness+file back-half).
- Sibling study notes: [MinIO MemKV](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md),
  [headroom](CONCEPT-STUDY-HEADROOM-2026-07-08.md),
  [field-borrow landscape](CONCEPT-FIELD-BORROW-LANDSCAPE-2026-07-08.md).
