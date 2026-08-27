# Concept study — ai-dynamo/dynamo exhaustive refresh (2026-08-26)

Issue: #9271. This is the construction receipt for the machine inventory and
upstream corpus, not a claim that Dynamo's design should replace fak's. The
high-value result is narrower: six implementation seams are supported by both
current upstream evidence and current FAK code evidence; the rest are routed to
existing work, watch, benchmark-only, or excluded.

## Source identity and cutoff

| Field | Value |
|---|---|
| Repository | `ai-dynamo/dynamo` |
| URL | <https://github.com/ai-dynamo/dynamo> |
| Pinned revision | `f494601ef16b2a17b35ca340c656287b2787971a` |
| Commit time | `2026-08-26T21:38:45Z` |
| Observation/API cutoff | `2026-08-26T22:01:04Z` |
| Default branch | `main` |
| Latest release at cutoff | `v1.4.1`, published `2026-08-22T17:07:43Z` |
| Repository push observed | `2026-08-26T21:58:09Z` |
| License | Apache-2.0 at the root; exceptions below |
| Machine tree map | `docs/research/inventory/ai-dynamo-dynamo.json` |
| Machine corpus | `docs/research/inventory/ai-dynamo-dynamo-corpus.jsonl` |

All source anchors below end in `@f494601e` and mean the full pinned SHA above.
The corpus records the full SHA and cutoff on every record. Current `main` was
752 commits ahead of the July 18 pin. GitHub's compare response exposed only
250 commits and 300 changed files, so no completeness claim rests on that
truncated comparison; the exhaustive tree walk is against the fetched pinned
commit.

## Method and coverage ledger

The local tree was generated with `fak study-inventory`, then corrected after
representative source reads. It contains 5,401 inventoried files in 1,438
directories, 121,058,969 bytes, 3,026 runtime files, 1,164 test/fixture files,
1,114 docs files, and 1,670,419 counted text lines across all 20 top-level
subsystems. Git reports 5,415 tracked paths; the 14-path difference is the
inventory tool's deliberate file-type/vendor exclusions, not an unobserved
subtree.

| Source class | Coverage at cutoff | Durable receipt |
|---|---|---|
| README/docs/architecture | All tree paths classified; decision-bearing pages read manually | inventory plus anchors below |
| Runtime/tests | All tree paths classified; load-bearing routing, replay, planner, KVBM, GMS, frontend, and ThunderAgent seams read | inventory plus anchors below |
| Issues | 1,867 of 1,867, GraphQL cursor exhaustion | corpus coverage record and one metadata record per issue |
| Pull requests | 11,902 of 11,902, GraphQL cursor exhaustion | corpus coverage record and one metadata record per PR |
| Discussions | 19 of 19 top-level discussions | corpus; replies are aggregate-only |
| Releases | 45 of 45 releases | corpus metadata |
| Tags | 85 of 85 tags | corpus metadata |
| Issue comments | 30,000 observed by aggregate counters | compact coverage manifest; bodies intentionally omitted |
| Review comments | 60,100 observed by aggregate counters | compact coverage manifest; bodies intentionally omitted |
| Reviews | PR review totals retained as aggregate metadata | compact coverage manifest; individual review bodies omitted |
| Changelog/TODO | release metadata plus complete path scan and classified TODO/deprecation records | inventory/corpus source-class records |
| License/provenance | root license, SPDX scan, exceptions, submodule check | inventory and provenance section below |
| FAK issue join | 102 open/closed FAK issues returned by broad Dynamo search, plus mechanism-specific searches | reconciliation section below |

The REST `/issues` enumeration hit GitHub's page-number ceiling at page 100.
The committed issue and PR coverage therefore uses GraphQL cursor traversal,
not the incomplete REST result. Full comment bodies and review bodies are not
committed: their roughly 90,000 records are a poor copyright/size trade, while
the counts, cutoff, API coordinates, and decision-changing top-level objects
remain reproducible. Titles and compact generated classifications are metadata,
not copied prose bodies.

## Reconstructed worldview

Dynamo is an orchestration layer above vLLM, SGLang, TensorRT-LLM, and related
backends for multi-GPU and multi-node datacenter serving
(`README.md:35-38@f494601e`). Its operator owns workers, cache placement, and
often the fabric, and optimizes goodput and TCO under TTFT/ITL constraints.

FAK's overlap is real but bounded. FAK owns a governed agent/tool gateway and a
fak-native inference path; it must retain model execution, kernels, memory,
scheduling, cache, adaptation, and operations. Dynamo is therefore a source of
mechanisms and adversarial witnesses, not an execution fallback. Every native
candidate below is constrained to Qwen3.8 execution inside FAK. llama.cpp is
permitted only as an explicitly selected parity/benchmark reference and may
never become an automatic recovery or convenience path.

## Subsystem and mechanism map

| Dynamo area | Decision-bearing mechanism | FAK axis | Disposition |
|---|---|---|---|
| `lib/kv-router` | tier-weighted overlap, output-footprint load, keyed tracking identities, radix invalidation | residency routing, namespace isolation | ADAPT two live integrations; WATCH tracking identity; BENCH radix removal |
| `lib/runtime` | atomic occupancy reservation, retarget-on-fallback, release-on-drop | route booking and fleet membership | PRESENT for generic RAII occupancy; ADAPT atomic cache-aware selection/booking |
| `lib/kvbm-consolidator` | publish STORE on first real source, REMOVE on last source | cache event visibility | ADAPT cross-source consolidation |
| `lib/gpu_memory_service` | explicit commit/publish boundary | consumer-ready transfer admission | DEDUP #8259/#8261/#8993 |
| `components/.../thunderagent_router` | pause/resume whole agent programs at tool boundaries | agent work scheduling | WATCH; compare under #8772 before product work |
| `components/.../replay` | versioned runner capabilities and deterministic topology replay | replay/trajectory evaluation | ADAPT evaluation contract under #8772; simulation is not live proof |
| `components/.../global_planner` | advisory plans, partial-apply limitations | autoscaling/control | DEDUP #5276/#5277; negative evidence against execution-equivalence claims |
| multi-DC KV routing | exact local match plus lossy remote projection | cachemeta/coherence | WATCH/benchmark only; projection must not become admission authority |
| frontend/preprocessor | advisory tool-choice normalization with divergent Python/Rust behavior | structural capability gate | EXCLUDE as a policy mechanism; FAK's structural gate is stronger |
| tracing/model taints | agent span joins; mutable scheduling taints | trajectory/provenance/policy | PRESENT tracing; EXCLUDE taints as authorization |

## What changed since the July studies

The July 8 study pinned `be36b523…`; the July 18 deep study pinned
`ea89e8bdfcc8b9c95514fa9beabc5bd8296ec546`. The current pin is 752 commits
ahead of the latter. The earlier notes remain correct on their witnessed axes,
but this refresh changes the decision map in five ways:

1. ThunderAgent now exposes an experimental whole-program working-set scheduler.
   It pauses only after a request reaches a tool boundary and resumes before new
   pauses (`components/src/dynamo/thunderagent_router/router.py:245-300@f494601e`).
   This is a useful experiment for #8772, not evidence to import an agent runtime.
2. The replay layer now advertises a runner-owned spec version and supported
   hooks so an older wheel cannot self-certify against a newer consumer
   (`components/src/dynamo/replay/simulation.py:53-70@f494601e`). This is a strong
   contract pattern, but offline goodput remains simulation evidence.
3. Occupancy is now an explicit reservation that can retarget atomically on
   fallback and releases on `Drop`
   (`lib/runtime/src/routing_policy/occupancy.rs:188-235@f494601e`). FAK already
   has RAII occupancy, but cache-aware selection and admission are still split.
4. Cross-source cache-event consolidation is now concrete enough to borrow:
   the first publishable source emits STORE and only the last source emits REMOVE
   (`lib/kvbm-consolidator/src/tracker.rs:240-333@f494601e`). FAK has no equivalent
   source/event identity in its live event stream.
5. Two July borrow tickets are closed while their helpers remain test-only. The
   tier-overlap and anticipated-output implementations exist in FAK, but current
   live routing still scores raw overlap and request-count load. Their state is
   implementation debt, not ABSENT capability and not complete integration.

## Reconciliation with NIXL and existing FAK work

The NIXL inventory at `b0cbb237354d72b83500d5214d2b4484f9866fa3`
established the lower transfer substrate. This study does not reopen that seam:
NIXL descriptors and registration are transport inputs; FAK still owns
consumer-ready route admission, pricing, policy, and receipts above them.

| Axis | Current FAK evidence | Join/disposition |
|---|---|---|
| Tier-weighted cache overlap | helper in `internal/gateway/tier_overlap_credit.go:98-139`; live scoring remains raw in `internal/gateway/residency_router.go:466-482` | #5272 CLOSED; reopen or successor ticket |
| Anticipated decode footprint | helper in `internal/gateway/decode_footprint_load.go:104-114`; no production caller | #5274 CLOSED; reopen or successor ticket |
| Request trigger stamping | classifier in `internal/modelroute/inputtrigger/inputtrigger.go:38-56,110-168`; `AdmitTurn` has tests but no live ingress call | #6419 CLOSED; reopen or successor ticket |
| Cross-source cache events | `internal/engine/cacheevents.go:32-66,377-392` lacks source/event identity and fans out every normalized event | no exact issue found; new ticket |
| Frequency/value admission | helpers in `internal/cacheprice/readmission.go:93-170` and `internal/compute/kvadmission.go:99-143`; no live radixkv caller | no exact issue found; new ticket |
| Generic occupancy lifetime | admission permit in `internal/gateway/admission.go:211-234`; fleet acquire/release brackets dispatch | PRESENT; do not file generic RAII ticket |
| KV transfer rendezvous | synchronous stage transport exists, but no two-source readiness join | enrich #3259/#3310/#3413/#8259; no new ticket by default |
| Byte-sized transfer readiness | shipped implementation under #8261 | keep closed; remaining consumer readiness belongs to #8259 |
| Autoscaler shadow/forecast | open #5276 and #5277 | DEDUP |
| Provisional cache warmth | open #3888 | DEDUP |
| Multi-dimensional/SLO admission | open #2242 | enrich with three-axis evidence; DEDUP |
| Native hardware witness | open #6409 | DEDUP |
| Provider ablation index | open #8772 | attach ThunderAgent/replay evaluation; DEDUP |

The broad repository search returned 102 FAK issues containing Dynamo-related
terms at the cutoff (25 open, 77 closed). Mechanism-specific searches were also
run because repository-name search alone misses source-neutral tickets. Closed
July clusters #3353-#3356, #3368-#3377, #5278, #2238, #1913, and #2037 retain
their status; this pass found no evidence that requires duplicating them.

`fak capabilities` was queried for routing, transfer, cache consolidation,
tracking identity, replay, and admission axes. Raw Go reference searches guarded
against false “PRESENT” results. `fak-dev index` was unavailable because the
committed arm64 model sources currently redeclare `q4kSDOTEnabled`; no candidate
depends on an index result.

## Negative knowledge

- Dynamo's global planner constructs plans but does not prove execution-equivalent
  application; partial apply and process-local state remain explicit limits
  (`components/src/dynamo/global_planner/orchestrator.py:303-373@f494601e`).
- Multi-DC cache summaries are lossy projections. They are useful as a ranking
  hint, never as policy, admission, or exact-residency proof
  (`docs/fern/pages/features/multi-dc-kv-routing.md:71-89@f494601e`).
- Frontend tool-choice handling is advisory and divergent: forced choice without
  tools becomes unconstrained, and Python/Rust paths differ. It cannot replace
  FAK's structural default-deny gate
  (`components/src/dynamo/frontend/prepost.py:160-175,302-325@f494601e`).
- Replica event time uses approximate receive time and a bounded queue can drop
  newest events (`lib/kv-router/src/sequences/multi_worker.rs:553-579@f494601e`).
  Those streams are not exact causal receipts.
- Mutable model taints influence scheduling but are not authorization
  (`components/src/dynamo/common/model_taints.py:13-41@f494601e`).
- Replay reports and AIConfigurator/aisimulate outputs are advisory model evidence,
  not live native-performance proof.

## License and provenance boundary

The root is Apache-2.0. A scan found 4,621 files with Apache SPDX markers, 19
CC-BY-4.0 agent-skill/document files, NVIDIA proprietary notices in Fern assets,
separately licensed vendored JavaScript, and MIT DeepSeek-V3.2 test data credited
to a specific Hugging Face commit. There are no gitlinks/submodules. Candidate
work is therefore clean-room Go adaptation from mechanisms and tests. Do not copy
CC-BY prose, proprietary assets, vendored bundles, generated artifacts, or test
data without carrying the applicable obligations.

## Related-repository adjacency

| Repository | Why it changes a decision | Treatment |
|---|---|---|
| `ai-dynamo/nixl` | Concrete transfer descriptors, registration, lifecycle, and backends beneath Dynamo | Existing exhaustive inventory; substrate only |
| `ai-dynamo/aisimulate` + `AIConfigurator` | Replay and configuration-model context | Advisory/benchmark evidence, never live proof |
| `ai-dynamo/aiperf` | Live serving measurement vocabulary | Benchmark harness input |
| `ai-dynamo/agent-plugins` + ThunderAgent sources | Provenance for whole-program scheduling | Watch/A-B under #8772 |
| `ai-dynamo/ModelExpress` + snapshot paths | Weight/snapshot cold-start context for GMS | Adjacent to #8259, no new generic cache claim |
| `ai-dynamo/grove` | Kubernetes topology and gang scheduling | Deployment adjacency; outside native-kernel ownership |
| `ai-dynamo/enhancements` | RFC/design rationale | Read when a selected mechanism needs history; not runtime evidence |
| Mooncake trace/tooling | External workload evidence | Benchmark-only; no copied corpus |

The current repository manifests these adjacencies in `Cargo.toml:55-93` and
`pyproject.toml:50-122@f494601e`; they are included because they alter how a
Dynamo mechanism should be interpreted, not to inflate the study scope.

## Deterministic candidate priority map

Scores are triage values, not performance claims. Formula:

`score = centrality + native impact + evidence + readiness + operating value + reversibility - duplicate penalty`

Centrality is fixed at Core 30, Enabling 20, Stewardship 10, Peripheral 0;
other columns are bounded at 20, 15, 15, 10, 10, and 20 respectively. Ties sort
by candidate ID. “Existing” rows route evidence to an open ticket rather than
authorize a new issue.

| Rank | ID | Candidate | C | N | E | R | O | V | D | Score | Disposition |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | C01 | Consolidate cross-source cache events | 30 | 18 | 15 | 14 | 10 | 8 | 0 | 95 | #9307 |
| 2 | C02 | Wire anticipated decode footprint into live routing | 30 | 20 | 15 | 15 | 8 | 9 | 5 | 92 | #9308 (successor to #5274) |
| 3 | C03 | Wire tier-weighted overlap into live routing | 30 | 20 | 15 | 15 | 7 | 9 | 5 | 91 | #9309 (successor to #5272) |
| 4 | C04 | Stamp `InputTrigger` at real gateway ingress | 30 | 16 | 15 | 15 | 8 | 9 | 5 | 88 | #9310 (successor to #6419) |
| 5 | C05 | Wire value/frequency admission into native radixkv | 30 | 18 | 14 | 11 | 7 | 8 | 2 | 86 | #9311 |
| 6 | E01 | Versioned replay/ThunderAgent A/B contract | 20 | 17 | 15 | 15 | 10 | 8 | 0 | 85 | existing #8772 |
| 7 | C06 | Make cache-aware selection and occupancy booking atomic | 30 | 15 | 9 | 10 | 10 | 8 | 0 | 82 | #9312 |
| 8 | E02 | Consumer-ready two-source KV rendezvous | 30 | 18 | 15 | 10 | 10 | 6 | 18 | 71 | existing #3259/#3310/#3413/#8259 |

## Construction-ready ticket packets

No child issue was created by this worker. The coordinator should adjudicate
reopen-versus-successor choices and file only the packets that survive current
ownership checks.

### C01 — Consolidate cache events across producers before cache visibility

- Centrality: Core. Priority: 95/100, highest because it closes a correctness
  ambiguity in an already-live coherence stream with a direct upstream state
  machine and no matching FAK issue.
- For: native FAK routing and cache-coherence consumers receiving engine,
  offloader, restore, and transfer events for the same logical KV block.
- Problem: a block can be announced by more than one producer, so forwarding
  every STORE/REMOVE can remove still-live residency or double-count visibility.
- Today: `internal/engine/cacheevents.go:32-66,377-392` carries no stable source
  or event identity and normalizes/fans out each event independently.
- Better because: a source-aware consolidator emits the first publishable STORE,
  suppresses duplicate sources, and emits REMOVE only when the last source leaves.
- Witness: a behavior test with two sources for one logical block: one STORE,
  first REMOVE suppressed, final REMOVE emitted; reorder/duplicate replay is
  deterministic. Native Qwen3.8 route selection sees the block until final remove.
- P1 Context: advanced — preserves reusable cache context until its final source
  leaves and prevents duplicate visibility work.
- P2 Net value: advanced — avoids stale-negative routing and visibility flaps;
  witness the gain net of one bounded source set per live block.
- P3 Adaptation: preserved — add explicit producer identity and a stable logical
  block key behind a versioned schema; do not copy Rust structures or owned-GPU
  assumptions, and retain a rollback reader.
- P4 Operations: advanced — bound state, expose suppressed/unknown event counters,
  make replay idempotent, and witness restart/bootstrap behavior end to end.
- Dependencies: cache event schema, cachemeta consumers, #3320 where ownership
  overlaps. Source: `lib/kvbm-consolidator/src/tracker.rs:240-333@f494601e`.
- Native-engine constraint: exercise FAK's Qwen3.8 cache/routing path; no llama.cpp
  fallback or non-native receipt can satisfy the witness.

### C02 — Integrate anticipated decode footprint into the live residency scorer

- Centrality: Core. Priority: 92/100; implementation and tests exist, so the
  remaining production wiring is small and high leverage. Reopen #5274 or create
  a narrowly named successor rather than duplicate the concept.
- For: native Qwen3.8 requests whose prompt overlap is high but expected decode
  footprint would overload the selected worker.
- Problem: request-count load treats a short decode and a long decode equally.
- Today: `internal/gateway/decode_footprint_load.go:104-114` computes the signal,
  while `internal/gateway/residency_router.go:466-482` uses occupancy/count load.
- Better because: the live score includes bounded anticipated output blocks and
  decays or reconciles them against observed output without retry inflation.
- Witness: fail-before/pass-after routing test where equal overlap/count routes
  away from the worker with materially larger booked output; cancellation and
  stream completion release exactly once.
- P1 Context: preserved — no prompt/context duplication is introduced; booked
  output metadata travels with the existing route decision.
- P2 Net value: advanced — better overload avoidance must survive the bounded
  accounting cost and a matched quality/latency comparison.
- P3 Adaptation: advanced — use FAK metadata, a bounded configurable term, and a
  conservative default when expected output is unknown.
- P4 Operations: advanced — meter prediction error, cap contribution, journal the
  effective score, and prove release on cancellation and stream completion.
- Dependencies: #5274 history, scorer configuration, occupancy lifecycle.
  Sources: `lib/llm/src/kv_router/routing_host/request_guard.rs:543-557@f494601e`
  and selection logic around `:780-839`.
- Native-engine constraint: matched Qwen3.8 route/quality envelope; no silent
  backend substitution.

### C03 — Integrate tier-weighted overlap into live cache-aware routing

- Centrality: Core. Priority: 91/100. Reopen #5272 or create a successor because
  the helper is present but production routing does not call it.
- For: native Qwen3.8 traffic with the same prefix represented at device, host,
  and slower storage tiers.
- Problem: equal token overlap does not imply equal ready time or reuse value.
- Today: `internal/gateway/tier_overlap_credit.go:98-139` implements weights;
  `internal/gateway/residency_router.go:466-482` still scores raw overlap.
- Better because: route credit reflects the tier actually reached while keeping
  weights explicit, bounded, and observable.
- Witness: live-policy test where identical overlaps at different tiers produce
  the configured order, zero-weight tiers add no credit, and unknown tier falls
  back conservatively.
- P1 Context: advanced — preserves useful prefix context while distinguishing the
  amount of work required to make each tier's hit usable.
- P2 Net value: advanced — remote/slow hits stop dominating fast reuse; the
  matched witness includes lookup and scoring overhead.
- P3 Adaptation: advanced — weights are bounded configuration derived from native
  FAK measurements, not copied upstream constants, with raw-overlap rollback.
- P4 Operations: advanced — export per-tier score contributions, unknown-tier
  behavior, and a kill switch through the real routing receipt.
- Dependencies: #5272, cachemeta tier provenance, C02 scorer composition.
  Source: `lib/kv-router/src/scheduling/overlap.rs:201-223@f494601e`.
- Native-engine constraint: weights must be derived or validated on Qwen3.8 in
  FAK; llama.cpp may only be an explicitly labeled reference.

### C04 — Stamp `InputTrigger` at the real gateway ingress

- Centrality: Core. Priority: 88/100. Reopen #6419 or file an integration-only
  successor; the classifier is implemented but its production contract is not.
- For: routing, replay, and caching decisions that distinguish user input,
  tool-result continuation, retry, and synthetic/resumed work.
- Problem: without ingress stamping, downstream helpers cannot reliably join a
  turn to the event that caused it.
- Today: `internal/modelroute/inputtrigger/inputtrigger.go:38-56,110-168` and
  `inputtrigger_subject.go:23-38` exist; `AdmitTurn` is called only in tests.
- Better because: one immutable, validated trigger is created before routing and
  carried through receipts, enabling honest tool-boundary and replay comparisons.
- Witness: end-to-end gateway test for user, tool-result, retry, and missing/
  invalid trigger; receipt and replay retain the same classification.
- P1 Context: advanced — one immutable trigger prevents downstream consumers from
  reconstructing the cause repeatedly and preserves the join through compaction.
- P2 Net value: preserved — this ticket makes no runtime speed claim; it enables
  honest evaluation without adding a scheduler or an extra inference call.
- P3 Adaptation: advanced — keep FAK's versioned trigger taxonomy and capability
  gate; do not import ThunderAgent runtime semantics.
- P4 Operations: advanced — the real ingress/receipt/replay path witnesses a
  compatibility default, cardinality bounds, provenance, and redaction.
- Dependencies: #6419, receipt schema, replay/trajectory consumers, #8772.
  Source: ThunderAgent boundary behavior at
  `components/src/dynamo/thunderagent_router/router.py:245-300@f494601e`.
- Native-engine constraint: must preserve FAK execution ownership and Qwen3.8
  receipts; trigger metadata cannot choose a fallback engine.

### C05 — Wire value/frequency admission into native radixkv

- Centrality: Core. Priority: 86/100; the algorithms exist, but policy wiring and
  operating-envelope evidence are missing.
- For: native Qwen3.8 cache tiers under scans or low-reuse write pressure.
- Problem: admitting every candidate can evict more valuable prefixes and amplify
  host/disk writes.
- Today: `internal/cacheprice/readmission.go:93-170` and
  `internal/compute/kvadmission.go:99-143` are pure/tested helpers with no live
  radixkv caller.
- Better because: bounded frequency/value evidence gates admission and
  readmission while preserving deterministic fallback when telemetry is absent.
- Witness: fail-before/pass-after scan workload plus representative agent-prefix
  trace showing lower churn without quality loss; include memory, CPU, and write
  accounting end to end.
- P1 Context: advanced — preserves reusable prefixes under scans and avoids
  admitting low-value context that displaces hotter agent state.
- P2 Net value: advanced — any hit/write benefit must exceed sketch, decision,
  verification, and miss-recovery overhead in the captured workload.
- P3 Adaptation: advanced — use FAK namespaces, prices, and tiers behind bounded
  policy inputs rather than Dynamo's owned-GPU assumptions.
- P4 Operations: advanced — witness bounded sketch size, decay, cold-start policy,
  journaled rejects, recovery behavior, and a rollback switch on the live path.
- Dependencies: radixkv admission seam, cacheprice signals, #4296/#3419 ownership.
  Related upstream mechanism: KVBM frequency-gated offload and TinyLFU paths.
- Native-engine constraint: Qwen3.8 native traces and cache receipts are required;
  a simulator-only improvement is insufficient.

### C06 — Atomically bind cache-aware selection to occupancy admission

- Centrality: Core. Priority: 82/100; failure semantics are important, but the
  exact FAK race needs a captured reproduction before implementation.
- For: native requests selected by cache overlap while workers change, reject,
  disappear, or require fallback.
- Problem: query-only selection followed by separate booking can overbook the
  winner, leak occupancy, or forward to an unbooked fallback.
- Today: FAK has safe permit lifetime and fleet acquire/release, but cache-aware
  selection in `residency_router.go:485-514` and admission in
  `admission.go:300-352` are separate decisions.
- Better because: selection returns one reservation; fallback retargets it under
  the same admission lock; endpoint/cancel failure releases exactly once.
- Witness: deterministic concurrent test capturing winner loss, booking failure,
  cancellation, and fallback; no negative/leaked occupancy and no unbooked send.
- P1 Context: preserved — the change binds existing context-aware selection to
  its booking without growing or duplicating prompt state.
- P2 Net value: advanced — the correctness gain and retry reduction must survive
  lock contention and added route-path cost against the current split operation.
- P3 Adaptation: advanced — adapt the reservation invariant behind the existing
  interfaces, not Dynamo's transport/API shape, with split-operation rollback.
- P4 Operations: advanced — contention, cancellation ordering, fallback,
  observability, and recovery are witnessed through the real dispatch path before
  default-on.
- Dependencies: fleet admission, C02/C03 score composition, route retry semantics.
  Sources: shipped reservation at
  `lib/runtime/src/routing_policy/occupancy.rs:188-235@f494601e`; upstream's own
  remaining non-atomic EPP TODO at
  `deploy/inference-gateway/ext-proc/src/epp.rs:980-1018@f494601e` is negative
  evidence against copying that path.
- Native-engine constraint: the booked worker must execute Qwen3.8 inside FAK;
  fallback may change worker, never engine ownership.

## Excluded, watch, and benchmark-only queue

| Candidate | Disposition | Reason/join |
|---|---|---|
| Import Dynamo as native execution backend | EXCLUDED | violates FAK-native ownership; Dynamo is orchestration above other engines |
| Use frontend tool-choice handling as policy | EXCLUDED | advisory/inconsistent; weaker than FAK structural capability gate |
| Treat model taints as authorization | EXCLUDED | mutable scheduling metadata is not a capability grant |
| Copy global planner state machine | EXCLUDED | partial application and process-local assumptions; #5276/#5277 own narrow useful pieces |
| ThunderAgent pause/resume product feature | WATCH | experimental/source-only; evaluate via #8772 after C04 receipts |
| Keyed provider-local tracking hash | WATCH | strong key handling, but domain derivation is still upstream TODO #11971; check FAK namespace threat model first |
| Multi-DC lossy cache projection | WATCH/BENCH | useful ranking hint only; join #4296/#3320, never exact admission |
| Middle-segment radix invalidation | BENCHMARK-ONLY | add adversarial fixture if FAK's compressed removal differs; source `lib/kv-router/src/indexer/radix_tree.rs:634-645@f494601e` |
| Bounded-queue replica event timestamps | BENCHMARK-ONLY | approximate receive time/drop behavior is a negative fixture, not a design to adopt |
| Replay goodput results | BENCHMARK-ONLY | deterministic simulation validates policy logic, not live latency/quality |
| GMS commit publication | DEDUP | #8259/#8261/#8993 |
| Two-source transfer rendezvous | DEDUP | #3259/#3310/#3413/#8259 |
| Advisory/predictive autoscaling | DEDUP | #5276/#5277 |
| Three-axis admission | DEDUP | #2242 |

## Completeness critic and remaining boundaries

The inventory classified every discovered top-level directory and file at the
pinned revision. Deep manual reads covered the decision-bearing paths named in
this note plus their nearby tests and docs. It did not semantically read every
generated asset, vendored dependency, container SBOM, fixture, example, binding,
or backend-specific recipe. `.agents/skills/visual-review/assets/vendor` and
`.git` were intentionally skipped by the tree walker. LFS was unavailable in the
worker environment; pointer metadata was observed, but no conclusion depends on
unfetched large objects.

GitHub cursor exhaustion establishes top-level issue and PR completeness at the
cutoff, not future completeness. Review, issue-comment, and discussion-reply
bodies are represented by aggregate receipts rather than copied content. The
compare API's 250-commit/300-file cap and the REST `/issues` page-100 limit are
explicit boundaries. Related repositories were adjacency-checked, not given new
exhaustive inventories in this pass; NIXL reuses its existing exhaustive map.

The construction pass filed #9307-#9312. C02-C04 are integration-only successors to completed helper tickets #5274, #5272, and #6419; C01, C05, and C06 own new live seams. E01/E02 remain deduplicated to existing work. Every ticket retains the native Qwen3.8 witness and the source cutoff above.
