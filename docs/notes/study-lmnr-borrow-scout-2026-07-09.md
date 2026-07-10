---
title: "Borrow scout: lmnr (Laminar) → fak (2026-07-09)"
description: "Study of the Laminar (lmnr) LLM-observability app-server (Apache-2.0; pinned SHA 5f14c5c86f1bfc18c852dbc7c712609e44b8d69c, cloned read-only to scratch) for techniques worth porting. 6 subsystems read by fan-out; 3 borrows filed (#3549/#3550/#3551), the rest witnessed PRESENT."
---

# Borrow scout: lmnr (Laminar) → fak (2026-07-09)

Study of **Laminar (lmnr)** — an LLM-tracing / evals / observability platform (Rust `app-server`,
Apache-2.0; pinned SHA `5f14c5c86f1bfc18c852dbc7c712609e44b8d69c`, cloned read-only to scratch) — for
techniques worth porting into fak. Every borrow is **INSPIRE**, not INTEGRATE: lmnr is Rust, fak is Go,
so any port is a clean-room reimplementation — no source is copied. lmnr is Apache-2.0, so even a copy
would be permitted with attribution; the attribution here is provenance, and the anchors below are the
exact `path:line @ 5f14c5c` each borrow was read at.

Read by fan-out across six subsystems (one reader each, full-file reads): **SQL/MCP query engine**,
**trace compression / storage**, **signals + alerting**, **PII redaction**, **realtime streaming +
evals**, **LLM-provider abstraction + checkpoints + debugger replay + scheduler**.

Scope note: lmnr's core product (a hosted multi-tenant ClickHouse/Postgres trace store with a web UI)
is **out of scope** for fak — fak is an in-process kernel, not a telemetry SaaS. The candidates below are
lmnr's *mechanism* techniques (structured-output grounding, scoped-query ergonomics, structure-aware
redaction, dedup, fingerprinting) that map onto fak's existing `trajquery` / `wirescreen` / `signals` /
`ctxplan` / `cachemeta` / `dos` machinery.

## Scorecard (candidate techniques, one technique each)

| # | Technique (lmnr anchor @ 5f14c5c) | fak witness | Verdict |
|---|---|---|---|
| 1 | **Content-addressed structural dedup** of repeated LLM messages/tool-defs (BLAKE3 over canonical JSON; store once, hash-ref, read-time reconstruction) — `checkpoints/version.rs`, `input_dedup.rs`, `tool_dedup.rs` | `internal/ctxplan` already collapses DUPLICATE-DIGEST spans to one representative (`plan.go:242`, `ElideDuplicate`), content-addressed `Digest` page-back-in handle; **open ticket #3340** for cross-turn verbatim-span→pointer dedup | PRESENT / already-ticketed — no borrow |
| 2 | **Structural-skeleton prompt fingerprint** stable under dynamic values (LLM-generated regex strips volatile spans → stable hash to group templates) — `checkpoints/system_prompt.rs`, `version.rs` | `internal/cachemeta/sysprompt_fingerprint.go` `ComputePrefixFingerprint` (#1563 "O(1) prefix compare") | PRESENT — no borrow |
| 3 | **Out-of-band oversize payload → inline reference marker** (`<lmnr_payload_url>`, S3) — `spans.rs` | ctxmmu opaque oversize pointer `{_paged,ref,len}`; wirescreen digester (`internal/wirescreen/digester.go`) | PRESENT — no borrow |
| 4 | **AST-scoped SQL**: parse → inject tenant predicate → re-emit (never concat); post-rewrite fail-closed re-verify; column allowlist hides the scope column — lmnr query engine + `search/signal_events.rs` | `internal/trajquery` — own SELECT parser, `View.Rewrite` RLS, static **and** dynamic per-row scope confirmation, allowlist hides scope column | PRESENT (fak ahead: dynamic row belt) — no borrow |
| 5 | **Per-project distributed lock** around read→classify→write (drop-on-held → re-triggered later, guaranteed release, double-check under lock) — `checkpoints/consumer.rs:142` | dos lease/arbiter kernel (`dos_arbitrate`), lane leases with lock modes | PRESENT — no borrow |
| 6 | **Signal = (NL prompt, JSON schema, severity)** LLM structured-output classifier with enum allowlists — `signals/*`, `search/signal_events.rs` | `internal/signals/schema.go` `ValidateAgainstSchema` + `enumContains` | PRESENT (primitive) — no borrow |
| 7 | **Prompt-cache breakpoint placement** (`cache_control` on last system block / last tool / last block of first user msg) — `llm/bedrock/mod.rs:144` | `internal/cachemeta` prefix stability/coherence/score | PRESENT — no borrow |
| 8 | **Tiered model resolution** (caller requests Small/Med/Large tier → env override → fallback table); **centralized retry ownership** (disable per-SDK retries so providers behave identically) — `llm/models.rs`, `bedrock/mod.rs:44` | `dispatch_model_policy`, accounts/gateway routing (rich model routing already) | PRESENT-ish — no borrow (no concrete gap witnessed) |
| 9 | **Model-quirk gating by substring + tests-as-spec** (adaptive-thinking Opus 4.7/4.8 & Claude 5.x reject legacy `{type:enabled,budget_tokens}` and `temperature`; `effort` a sibling `output_config`) — `bedrock/mod.rs:498-584` | fak *is* a Claude harness; gateway builds requests. Not witnessed to a gap this pass | LIKELY PRESENT — no ticket (unwitnessed) |
| 10 | **Debugger replay cache**: 3-outcome Hit/Miss/**Live** where a Redis error degrades to Live-never-Miss; single-flight warmup w/ lock heartbeat renewed at TTL/3 — `debugger/mod.rs:390,418,449` | dos reason taxonomy (abstain vs refuse; fail-closed), leases w/ TTL | PRESENT-in-spirit — no borrow |
| 11 | **SSE (not WS) live transport** + two-tier Redis-pubsub→per-pod-map fan-out + heartbeat-send-failure-as-GC + `(tenant, resource-string)` sub keys — `realtime/mod.rs` | fak TUI panes (`fleetpane`) + Monitor read state; no multi-viewer stream server | ABSENT but **LOW FIT** (fak is single-node kernel/CLI, not a dashboard server) — no ticket (latent) |
| 12 | **Eval mechanics**: scores as open JSON map + **merge-patch incremental accumulation** (each judge lands idempotently) + shared group-label run comparison + v7-UUID time-sortable PKs — `evaluations/mod.rs:155,231`, `datasets/service.rs:243` | lab/bench/ablate exist; bench-storage internals not read this pass | PARTIAL / **unwitnessed** — no ticket (open lead, see below) |
| 13 | **Scheduler**: leader-elected poll + **catch-up watermark** (enumerate missed hour-boundaries) + **deterministic period from trigger timestamp**, not `now()` — `reports/scheduler.rs`, `generator.rs:85` | watchdog / `dispatch_tick`; watermark/period-determinism not read this pass | PARTIAL / **unwitnessed** — no ticket (open lead, see below) |
| A4 | **ID-grounded LLM triage** — tag candidates `[Event ID: <id>]`, force a required tool returning a **subset** `{summary, event_ids[]}`, **re-resolve IDs from the store** so the model triages but cannot fabricate — `reports/generator.rs:297,474` | signals schema validates a verdict but nothing re-resolves selected IDs; digest feeders (#2564) are ungrounded | **PARTIAL — FILED #3549** |
| C6 | **Query MCP tool w/ schema-in-description** — tables/columns/**enum literals** injected into the tool description from one source so the agent writes valid scoped SQL first-try; field-name identifier-regex guard + adversarial tests — `search/signal_events.rs:26,229` | `fak trajquery` is CLI-only (`cmd/fak/main.go:231`); MCP server (`internal/gateway/mcp_index.go`) exposes only index/feature/capabilities tools | **ABSENT — FILED #3550** |
| PII | **JSON-leaf-aware redaction** — walk nested JSON, render only string **leaves** as prose, scan, map spans back via offset + JSON Pointer; skip structural keys; expand depth-capped stringified-JSON — `pii-redactor/src/json_walker.rs:104,40,186,243` | wirescreen `Propose`/`Flag(body []byte)` scan a **flat** byte buffer (`internal/wirescreen/wirescreen.go:46`) | **PARTIAL — FILED #3551** |

**Outcome: 3 borrows filed of ~16 candidates examined across 6 subsystems.** Witnessing prevented ~10
duplicate / N-A / low-fit tickets against machinery fak already ships — often more rigorous than lmnr's
(ctxplan content-dedup + CAS page-back, `sysprompt_fingerprint`, trajquery static+dynamic RLS, dos lease
kernel, cachemeta breakpoint scoring, signals schema+enum validation). Two open leads (#12 eval
merge-patch scoring, #13 scheduler catch-up watermark) were **deliberately not filed** — they are
plausible gaps but were not witnessed against fak's actual bench/watchdog internals this pass; a future
scout should read `internal/lab`/bench storage and the watchdog tick before ticketing.

## Filed tickets (one borrow each, INSPIRE from lmnr Apache-2.0 @ 5f14c5c)

- **[#3549](https://github.com/anthony-chaudhary/fak/issues/3549)** — `feat(signals)`: ID-grounded LLM
  triage. The model may only *select* real event IDs (forced `{summary, event_ids[]}` tool call,
  re-resolved from the store), never invent one — `dos_verify`'s "re-resolve from artifacts" applied to
  the reporting path. Builds on the existing `internal/signals` schema validator. Consumer: digest feeder #2564.
- **[#3550](https://github.com/anthony-chaudhary/fak/issues/3550)** — `feat(mcp)`: expose `trajquery`
  over MCP with the scoped `View` schema (columns + enum literals) baked into the tool description +
  a contract test pinning the injection. Surfaces an *existing* capability via MCP per the
  MCP-catalog-over-new-core-tool doctrine (#2926).
- **[#3551](https://github.com/anthony-chaudhary/fak/issues/3551)** — `feat(wirescreen)`: JSON-leaf-aware
  redaction. Scan string leaves of nested tool-result JSON (skip structural keys like `tool_use_id`;
  reach stringified-JSON envelopes; map spans back via JSON Pointer), wrapping the existing Redactor;
  flat-byte path stays as the non-JSON fallback. Consumer: #1983; adjacent: #3280, #3340.
