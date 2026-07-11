---
title: "Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)"
description: "Deeper re-study of the Laminar (lmnr) monorepo (Apache-2.0; pinned SHA 5f14c5c86f1bfc18c852dbc7c712609e44b8d69c, cloned read-only to scratch), widening the 2026-07-09 pass from app-server to the frontend/signals surfaces. 12 subsystems fan-read; 85 candidate borrows witnessed against fak (31 PRESENT, 54 PARTIAL); 46 marked fileable, curated to 10 filed (epic #4062 + #4063–#4071); 37 deferred, tabled below so nothing is silently dropped."
---

# Borrow scout: lmnr (Laminar) → fak — deep re-study (2026-07-10)

Deeper re-study of **Laminar (lmnr)** (Rust `app-server` + TypeScript `frontend`, **Apache-2.0**; pinned SHA
`5f14c5c86f1bfc18c852dbc7c712609e44b8d69c`, cloned read-only into scratch — never the tree), following the
[2026-07-09 scout](study-lmnr-borrow-scout-2026-07-09.md) which read six `app-server` subsystems and filed
#3549 / #3550 / #3551. This pass **widened coverage to the `frontend/` signals + dashboard surfaces** and
re-witnessed on-axis (not at the capability name), which is why several techniques the prior pass marked
"PRESENT (primitive)" resolve here to a **PARTIAL** with a concrete missing sub-axis.

Every borrow is **INSPIRE**, not INTEGRATE: lmnr is Rust/TS, fak is Go, so any port is a clean-room
reimplementation — no source is copied. Apache-2.0 would permit a copy with attribution; the attribution here
is provenance, and every anchor is the exact `path:line @ 5f14c5c` the borrow was read at.

## Method & honesty boundary

- **Read the code, not the pitch.** 12 subsystems fan-read (app-server query engine / trace compression /
  signals+alerting / PII redaction / LLM-provider abstraction / limits+notifications / input-extraction /
  checkpoints; frontend signals / dashboards / SQL-editor / evaluation).
- **Witnessed each candidate against fak on-axis** via `fak_feature_query` + `fak_capabilities` + raw grep,
  with a false-ABSENT guard (grep for fak's own prior borrow of the same family) and a false-PRESENT guard
  (confirm the cited fak seam actually covers the *axis*, not just the capability name).
- **Partial-witness disclosure:** the witness fleet was rate-limited mid-run — of the candidates dispatched,
  **85 returned a verdict**; ~60 witness agents were throttled (429 / no structured output) and did **not**
  complete. So the scorecard is "of 85 witnessed," not "of all candidates extracted." A future pass re-running
  the dropped witnesses may surface more.
- **All 10 filed seams were re-verified to exist** (file + line) before filing; witness-claimed line numbers
  are not trusted blind.

## Scorecard

- **85 candidates witnessed** → **31 PRESENT** (fak already ships it, often more rigorously — no borrow) /
  **54 PARTIAL** (a real missing sub-axis).
- **46 marked fileable**; curated to **10 filed** (1 epic + 9 leaves) on the axes that fit a headless in-process
  kernel. **37 deferred** — tabled in full below (§Full candidate table) so the drop is auditable, not silent.
- Curation dropped mostly **frontend UI-polish** borrows (sparklines, heatmaps, autocomplete, diff renderers,
  hidden nav columns) — low fit for fak, which is a CLI/kernel, not a dashboard SaaS.

## Filed this pass (INSPIRE from lmnr Apache-2.0 @ 5f14c5c)

**Epic [#4062](https://github.com/anthony-chaudhary/fak/issues/4062)** — `epic(signals)`: grow
`internal/signals` from a judge-runner into a behavioral-detector product. fak already ships lmnr's signal
*primitive* (`Signal{Prompt, Schema, SampleRate}` + schema-validated `Run`); the product around it is not built.
Three children:

- **[#4063](https://github.com/anthony-chaudhary/fak/issues/4063)** `feat(signals)` — prebuilt run-failure
  detector **catalog** with a closed category enum (`internal/signals/catalog.go` + `--builtin` source).
- **[#4064](https://github.com/anthony-chaudhary/fak/issues/4064)** `feat(signals)` — structured
  **trigger-filter gate** so the judge fires only on items whose `Item.Meta` matches a condition (cheap
  prefilter before the model call). Sampling half is already PRESENT.
- **[#4065](https://github.com/anthony-chaudhary/fak/issues/4065)** `feat(signals)` — **`fak signals test`**:
  dry-run one signal's judge against one real trace item, print the schema-validated verdict, persist nothing.

Standalone leaves:

- **[#4066](https://github.com/anthony-chaudhary/fak/issues/4066)** `feat(promptshape)` — shape-invariant
  `Skeleton()` prompt hash (strip the `x-anthropic-billing-header`, hash first-sentence + sorted tag-set) so
  Claude Code template-siblings group across SDK-version churn. Distinct axis from cache-break #2847.
- **[#4067](https://github.com/anthony-chaudhary/fak/issues/4067)** `test(guardtrace)` — programmable fault
  schedule (`Retryable429/NonRetryable/Timeout`) on the reusable `FakeUpstream`, replacing ~30 ad-hoc inline
  fail-N httptest counters.
- **[#4068](https://github.com/anthony-chaudhary/fak/issues/4068)** `test(gateway)` — pin the **exact** exposed
  MCP tool inventory in a golden test (today's tests check only subset/count, so a rename or net-zero add+drop
  passes silently).
- **[#4069](https://github.com/anthony-chaudhary/fak/issues/4069)** `feat(modelroute)` — **saturate** an
  unsupported effort tier to the nearest supported one instead of hard-refusing the request.
- **[#4070](https://github.com/anthony-chaudhary/fak/issues/4070)** `feat(boundarylint)` — surface an
  **unparseable source** file as a recorded soft skip (sibling to `SKIP_DEBT` #3840), not a silent
  zero-findings "clean" pass. fak's *security* floor already fails closed here; the *detector* family does not.
- **[#4071](https://github.com/anthony-chaudhary/fak/issues/4071)** `feat(blobfs)` — report byte-denominated
  dedup savings (novel vs avoided bytes) in `Stats()`, the storage-plane mirror of `cachemeta.pool`'s
  `BytesDeduplicated`. Relates to dedup epic #2503.

## Strong deferred leads (revisit next pass, not UI-polish)

These are on-axis PARTIALs deliberately not filed this pass (kept the filing set to ~10). A future scout should
witness-and-file rather than re-derive:

- **Refuse a lane taxonomy where two lanes share a tree** (`regionadmit.go:108`) — CTE-shadow → lane-collision
  analogy; relates to the admission-kernel epic #3269.
- **Trust-tiered resource ceilings by request Surface** (`laneadmit.go` / `policy/isolation.go`) — cap
  autonomous surfaces, leave operator uncapped (two near-duplicate candidates; merge).
- **Per-probe timeout on `FleetMembership` health tick** (`fleet_membership.go:294`) so a hung worker can't
  wedge the tick.
- **boundarylint `RELAXATION_DEBT` soft rule** for undocumented protocol/policy overrides (`rules_skipdebt.go`).
- **field-list ⇄ JSON-Schema round-trip builder** for `internal/signals` verdict shapes (`schema.go:34`).
- **Deterministic tool-error sanitizer** stripping echoed argv + engine-version fingerprint (`wirescreen`) —
  overlaps the JSON-leaf redaction #3551; witness before filing to avoid a dup.

## Full candidate table (46 fileable; 9 filed, 37 deferred)

`V` = verdict (P = PARTIAL). "deferred" rows were witnessed fileable but not filed this pass; nothing here is
dropped silently. The 31 PRESENT verdicts (fak already ships it — no borrow) are omitted from this table.

| Status | V | Technique (proposed leaf) | fak seam |
|---|---|---|---|
| #4063 | P | feat(signals): ship a prebuilt run-failure detector catalog with a closed category enum | `internal/signals/signals.go:26` |
| #4064 | P | Add a structured trigger-filter gate to signals so the judge fires only on matching items | `internal/signals/signals.go:144` |
| #4065 | P | Add `fak signals test` to run a signal's judge against one real trace item and print the structured verdict, nothing saved | `cmd/fak/signals.go:24` |
| #4066 | P | Add internal/promptshape skeleton hash to group Claude Code agents by prompt shape | `internal/promptaudit/provenance.go:214` |
| #4067 | P | Add a programmable fault schedule (Retryable429/NonRetryable/Timeout) to guardtrace.FakeUpstream for retry-path tests | `internal/guardtrace/upstream.go:46` |
| #4068 | P | Pin the exact exposed MCP tool inventory in a golden test to catch add/drop/rename drift | `internal/gateway/mcp_expose_test.go:149` |
| #4069 | P | Saturate an unsupported effort tier to the nearest supported one instead of refusing | `internal/gateway/deepseek_anthropic.go:158` |
| #4070 | P | boundarylint: surface unparseable source files as a recorded skip instead of a silent zero-findings pass | `internal/boundarylint/boundarylint.go:113` |
| #4071 | P | Report byte-denominated dedup savings (novel vs avoided bytes) in blobfs Stats | `internal/blobfs/store.go:192` |
| deferred | P | Validate judge-emitted rubric criterion ids against the cached rubric id set | `internal/trajctl/judgescorer.go:160` |
| deferred | P | Add per-period notify-once suppression key to slackoutbox drain | `internal/slackoutbox/drain.go:279` |
| deferred | P | feat(callavoid): add apply-failure self-heal eviction and uncached passthrough fallback to the memo freshness model | `internal/callavoid/avoid.go:186` |
| deferred | P | Preflight outbound provider request bytes against a max-request-size ceiling and classify oversize as non-retryable before send | `internal/agent/retry.go:95` |
| deferred | P | Codify never-cache-the-fire-negative invariant on the cooldown fire-once gate | `internal/accounts/cooldown.go:160` |
| deferred | P | Nudge dropped-tool-call slips back within the turn budget instead of aborting the owned loop | `internal/agent/loop.go:492 — the `if comp.ToolCallsDropped &` |
| deferred | P | Add edge-aware collapsed line-diff renderer to internal/headroom | `internal/headroom/native.go:98` |
| deferred | P | vdso: make coherence invalidation change-triggered, not occurrence-triggered — suppress no-op writes from bumping the epoch and publishing a Mutation | `internal/vdso/vdso.go:507-518` |
| deferred | P | Republish provably-dominated stale-older leases in leaseref Fence instead of failing closed | `internal/leaseref/fence.go:134` |
| deferred | P | Tier laneadmit.Decide admission ceilings by request Surface: cap autonomous surfaces, leave operator uncapped, opt-in (0=unlimited) | `internal/laneadmit/laneadmit.go:32` |
| deferred | P | Refuse a lane taxonomy where two lanes share a tree instead of lexically-first tie-breaking in LaneOf | `C:\work\fak\internal\regionadmit\regionadmit.go:108` |
| deferred | P | Add a deterministic tool-error sanitizer stripping echoed argv and version fingerprint | `internal/wirescreen/redactor.go:296` |
| deferred | P | Resolve per-trust resource ceilings from the isolation dial (CeilingFor) | `internal/policy/isolation.go:56` |
| deferred | P | Strip echoed-input and engine-version fingerprints from tool-error text before it re-enters context | `internal/wirescreen/redactor.go:297` |
| deferred | P | Add boundarylint SOFT RELAXATION_DEBT rule for undocumented protocol/policy overrides | `internal/boundarylint/rules_skipdebt.go:32` |
| deferred | P | Add a shared whitespace-normalizing test helper for formatter-divergent Python-to-Go ports | `AGENTS.md:305` |
| deferred | P | agentshape: immutable agent-shape id = hash(stable prompt + tool-set digest + model) for fleet drift/dedup, distinct from cache-coherence keys | `Derive a small pure identity from primitives fak already own` |
| deferred | P | doomloop drain --deliver: drop a superseded re-anchor nudge at delivery time (late-binding, fail-open) so a stale correction can't clobber a recovered worker | `cmd/fak/doomloop.go — runDoomloopDeliver, immediately before` |
| deferred | P | Add order-independent prompt-shape signature from top-level scaffolding tags in taskidentity | `internal/taskidentity/taskidentity.go:14` |
| deferred | P | Bound each FleetMembership probe with a per-probe timeout so a hung worker can't wedge the health tick | `internal/gateway/fleet_membership.go:294` |
| deferred | P | Add churn-invariant turn-identity key that drops all system messages before hashing | `internal/sessionreset/text.go:38` |
| deferred | P | Guard gatewayusageledger Cut's LEAST/GREATEST fold against placeholder-timestamp stub rows | `internal/gatewayusageledger/cut.go:112` |
| deferred | P | Gate Anthropic temperature/thinking shape by model-id quirk predicates with a region-prefix test matrix | `internal/agent/wireprofile.go:68` |
| deferred | P | Price prompt tokens above a per-model long-context threshold in CachePricing | `internal/gateway/cache_pricing.go:179` |
| deferred | P | Generalize modelreg.Resolve to an ordered candidate-key normalization list | `internal/modelreg/modelreg.go:227` |
| deferred | P | Add field-list to/from JSON Schema round-trip builder for internal/signals verdict shapes | `internal/signals/schema.go:34` |
| deferred | P | Offer slot-scoped legal-verb completion on unknown `fak session` verbs instead of a full usage dump | `cmd/fak/session_cmd.go:108` |
| deferred | P | Roll up doomloop verdict incidents into recurring failure families | `cmd/fak/doomloop.go:483` |
| deferred | P | Scope the worker-prompt orientation surface per lane via a mode registry in RenderIssuePrompt | `internal/dispatchtick/prompt.go:82` |
| deferred | P | Gate fak_capabilities cards on a live fulfillability probe before advertising | `internal/selfquery/selfquery.go:44` |
| deferred | P | Gate sparkline normalization on a significance floor so a near-flat series reads flat, not full-ramp | `internal/cachevaluereport/markdown.go:33` |
| deferred | P | Give sessionsearch raw exact-match filter fields distinct from its tokenized text lane | `internal/sessionsearch/sessionsearch.go:287` |
| deferred | P | Preserve schema-undeclared memq cell attrs on round-trip and surface a drift notice | `internal/memq/memq.go:58` |
| deferred | P | Gap-fill fleettrend sparklines over a contiguous time axis | `internal/fleettrend/fleettrend.go:132` |
| deferred | P | Auto-carry a hidden identity column in trajquery projections to preserve each row's link to its source | `internal/trajquery/query.go:111` |
| deferred | P | Append a durable per-session model-usage span (tokens/latency/outcome) to fak's ledger on session exit | `cmd/fak/serve.go:398` |
| deferred | P | Make the self-telemetry vs agent-transcript separation a checked invariant (borrow lmnr's per-layer disjoint-tree discipline) | `Present half: internal/abi/registry.go — EventSubscriber` |
