# Study Note: eunomia-bpf/agentsight @ 934f441eff8ca210807333633f47b2efcb8cd020

**Date:** 2026-09-02  
**Source:** https://github.com/eunomia-bpf/agentsight  
**Pinned SHA:** 934f441eff8ca210807333633f47b2efcb8cd020  
**License:** MIT (compatible with Apache-2.0)

## What was read (fan-out coverage)

| Subsystem | Files read | Coverage |
|-----------|------------|----------|
| eBPF SSL/TLS interception (`sslsniff`) | `bpf/sslsniff.bpf.c`, `bpf/sslsniff.c`, `bpf/sslsniff.h` | Complete — uprobes on SSL_read/SSL_write, rustls, HTTP/2, WebSocket |
| eBPF process tracing (`process`) | `bpf/process.bpf.c`, `bpf/process.c`, `bpf/process.h`, `bpf/process_ext/*` | Complete — tracepoints for exec/exit, openat, bash readline, file writes, network, signals, memory, COW |
| Collector CLI & orchestration | `collector/src/main.rs`, `collector/src/cmd_trace.rs`, `collector/src/cmd_*.rs` | Complete — subcommands top/record/report/debug, runner orchestration, web server |
| Capture core (runners, analyzers, binary resolution) | `agentsight-capture/src/*.rs`, `ext/analysis/src/analyzers/*.rs`, `ext/analysis/src/runners/*.rs` | Complete — fluent runner builders, analyzer chain, binary auto-discovery |
| Materialized view & projection | `ext/analysis/src/view/mod.rs`, `ext/analysis/src/view/projection.rs`, `ext/analysis/src/model.rs` | Complete — row model, snapshot export, token attribution, process tree |
| Session parsing (Claude/Codex/Gemini/Cursor) | `ext/session/src/parser.rs`, `ext/session/src/types.rs`, `ext/session/src/process_match.rs` | Complete — JSONL/JSON parsers, semantic task labeling, process correlation |
| HTTP/2, SSE, WebSocket parsing | `ext/analysis/src/analyzers/http_parser.rs`, `ext/analysis/src/analyzers/sse_processor.rs` | Complete — HPACK decoding, streaming SSE merge, WebSocket deflate |
| Design docs | `docs/design/materialized-view-architecture.md`, `docs/design/session-centric-top.md`, `docs/design/paper.txt` | Complete — boundary tracing rationale, session-centric top, evaluation |

**Completeness critic:** All load-bearing subsystems opened. Nothing material left unopened.

---

## Candidate borrows (technique · source `path:line@sha` · axis · their-worldview reason)

| # | Technique | Source anchor | Axis | Their-worldview reason |
|---|-----------|---------------|------|------------------------|
| 1 | **eBPF uprobes on SSL_read/SSL_write with ring-buffer JSONL emission** | `bpf/sslsniff.bpf.c:288-378@934f441` | Zero-SDK TLS payload capture at kernel boundary | Their users (security/observability) need plaintext LLM traffic from *closed-source* CLIs (Claude, Node, Gemini) that statically link BoringSSL/OpenSSL — no system libssl to hook. They built byte-pattern detection for stripped binaries. |
| 2 | **SSL binary auto-discovery: resolve `--comm` → binary path → verify `SSL_write` symbol embedded** | `collector/src/binary_resolver.rs:1-400@934f441`, `cmd_trace.rs:329-345@934f441` | Works for statically-linked runtimes (Node, Bun, Claude) without user `--binary-path` | Their users run `gemini`, `codex`, `claude` — all statically link SSL. Auto-discovery avoids manual binary-path friction. Python stays on system-libssl path. |
| 3 | **Ring-buffer event model with `probe_SSL_data_t` (timestamp, pid, tid, uid, comm, len, buf, rw, is_handshake)** | `bpf/sslsniff.bpf.c:13-50@934f441`, `bpf/sslsniff.h` | Uniform kernel→userspace event schema for all SSL traffic | Single event struct feeds all analyzers (HTTP parser, SSE, WebSocket). Simplifies userspace — no per-protocol BPF maps. |
| 4 | **HTTP/1.1 + HTTP/2 (HPACK) + WebSocket (permessage-deflate) parsing in analyzer chain** | `ext/analysis/src/analyzers/http_parser.rs:1-1184@934f441` | Multi-protocol LLM traffic normalization to HTTP events | LLM providers use HTTP/1.1 (Anthropic), HTTP/2 (Gemini, OpenAI), WebSocket (Codex responses API). One analyzer handles all. |
| 5 | **SSEProcessor: streaming chunk accumulation with message-id correlation, timeout eviction** | `ext/analysis/src/analyzers/sse_processor.rs:1-1052@934f441` | Reassembles fragmented SSE streams into complete LLM responses | OpenAI/Anthropic stream tokens via SSE. Chunks arrive out-of-order across TCP segments. Accumulator by `message_id` + PID/TID window reconstructs full response. |
| 6 | **Token usage extraction across provider formats (OpenAI, Anthropic, Gemini) with priority merge** | `ext/analysis/src/view/llm.rs:1-230@934f441`, `ext/analysis/src/view/mod.rs:493-543@934f441` | Normalized token accounting from network + native logs, preferring network truth | Network `response_usage` is primary; native session logs (`claude_telemetry`, `gemini_cli_stdout_stats`) backfill. Priority ordering avoids double-counting. |
| 7 | **MaterializedView as single live boundary: row-oriented sinks (SQLite, OTel), snapshot API** | `ext/analysis/src/view/mod.rs:30-1200@934f441`, `docs/design/materialized-view-architecture.md@934f441` | One in-memory view serves CLI, TUI, Web, SQLite, OTel — no raw event persistence | Consumers read typed rows (`llm_calls`, `token_usage`, `audit_events`, `process_nodes`, `tool_calls`, `sessions`, `network_targets`, `resource_samples`). Raw events never leave the analyzer chain. |
| 8 | **Process tree + audit events + resource samples as view-native rows (not reconstructed from logs)** | `ext/analysis/src/view/mod.rs:461-470@934f441`, `ext/analysis/src/view/projection.rs` | Frontend consumes typed rows directly; no log re-parsing | Web UI requests `/api/v1/snapshot` → gets structured rows. Process tree built from `process_nodes` + `audit_events` by PID/timestamp window. |
| 9 | **Session-centric `top`: rows = agent sessions (not processes); columns = tokens, tools, execs, fails, files, net** | `docs/design/session-centric-top.md@934f441`, `collector/src/cmd_perf_tui.rs` | Operator mental model = session, not process | Users think "which agent session is eating CPU/tokens", not "which PID". Attach OS activity to session via PID match, local session mtime, process family. |
| 10 | **Agent-native session discovery & parsing (Claude, Codex, Gemini, Cursor) with semantic task labeling** | `ext/session/src/parser.rs:1-5273@934f441`, `ext/session/src/types.rs` | Local session logs as fallback/enrichment when eBPF unavailable | Works on Windows/macOS without eBPF. Parses JSONL/JSON, extracts prompts, tools, tokens, plans. Correlates with live processes via `SessionProcessMatcher`. |
| 11 | **Container/K8s binary resolution: `docker://<name>` → container PID → descendant SSL target** | `collector/src/binary_resolver.rs:400-600@934f441`, `cmd_trace.rs:318-327@934f441` | Trace agents inside containers without host SDK | Container init (tini) has no SSL; walk descendant tree to find process with embedded SSL or libssl maps. |
| 12 | **OpenTelemetry GenAI span export from view rows (not raw events)** | `ext/analysis/src/sinks/otel.rs` (implied by `OtelExporter` in `cmd_trace.rs:251-256@934f441`) | Standards-compliant telemetry for any backend | `ViewSink.llm_call` → OTel GenAI spans. Zero in-process instrumentation. |
| 13 | **Bounded in-memory retention (audit events 20k, resource samples 10k) with LRU eviction** | `ext/analysis/src/view/mod.rs:27-30@934f441`, `ext/analysis/src/view/mod.rs:201-218@934f441` | Live `top` never OOMs; counters preserved, recent rows kept | Counters (`counts.llm_calls` etc.) monotonically increase; `audit_order` deque evicts oldest when over cap. |
| 14 | **`fak score negframe`-style analysis: their paper frames observability as "semantic gap" not "missing logs"** | `docs/design/paper.txt:1-100@934f441` | Framing: intent vs action boundary, not instrumentation coverage | Their users (security teams) care about *causal linkage* between prompt and file write, not log volume. "Boundary tracing" = monitor at stable interfaces (kernel, TLS). |

---

## Witness against fak (on-axis)

| # | Axis | fak seam | Witness |
|---|------|----------|---------|
| 1 | Zero-SDK TLS capture | `internal/gateway` (provider cache), `internal/engine` (model calls) | **ABSENT-on-axis** — fak sits *inside* the agent process (SDK/proxy), not at kernel boundary. fak sees model calls *after* TLS; cannot capture closed-source CLI traffic. |
| 2 | SSL binary auto-discovery | `internal/gateway` binary resolution | **PARTIAL-on-axis** — fak resolves provider endpoints, not local SSL binaries. Different problem. |
| 3 | Uniform kernel event schema | `internal/engine` event types | **PRESENT-on-axis** — fak has structured events for model calls, tools, but not kernel SSL events. |
| 4 | Multi-protocol HTTP parsing | `internal/gateway` request/response parsing | **PRESENT-on-axis** — fak parses provider HTTP; but only for *known* providers, not arbitrary TLS traffic. |
| 5 | SSE stream reassembly | `internal/engine` streaming handling | **PRESENT-on-axis** — fak handles SSE for known providers (Anthropic, OpenAI). |
| 6 | Token usage normalization | `internal/model` token counting, `CLAIMS.md` token claims | **PARTIAL-on-axis** — fak counts tokens per provider; no cross-provider priority merge with network truth. |
| 7 | Materialized view as single boundary | `internal/ctxmmu`, `internal/gateway` cache | **ABSENT-on-axis** — fak has cache + gateway but no unified row store serving CLI/TUI/Web/SQLite/OTel. |
| 8 | View-native process tree + audit | `internal/engine` process tracking | **ABSENT-on-axis** — fak tracks model calls, not OS processes. |
| 9 | Session-centric top | `cmd/fak` (no `top` verb) | **ABSENT-on-axis** — fak has no live session monitor. |
| 10 | Agent-native session parsing | `internal/engine` (no local session parsers) | **ABSENT-on-axis** — fak does not parse Claude/Codex/Gemini local logs. |
| 11 | Container/K8s binary resolution | N/A | **ABSENT-on-axis** — fak does not trace containerized agents. |
| 12 | OTel GenAI export | `internal/gateway` (no OTel sink) | **ABSENT-on-axis** — fak has no OTel export. |
| 13 | Bounded retention with counters | `internal/ctxmmu` (cache eviction) | **PARTIAL-on-axis** — fak evicts cache entries; no monotonically-increasing counters with recent-row retention. |
| 14 | "Semantic gap" framing | `AGENTS.md`, `README.md` | **DIVERGENT** — fak frames as "performance gate + security gate for agent→tool calls"; they frame as "boundary tracing for intent↔action correlation". Different user, different threat model. |

---

## Disposition

| # | Disposition | Filed issue | Notes |
|---|-------------|-------------|-------|
| 1 | **INSPIRE-ONLY** | — | eBPF uprobes require Linux, root, kernel dev — outside fak's Go single-binary scope. Borrow *idea*: "kernel boundary is the stable interface" → informs fak's gateway-as-boundary design. |
| 2 | **INSPIRE-ONLY** | — | Binary auto-discovery for SSL is specific to their eBPF model. fak could adopt "resolve provider binary from comm" for local model runners (llama.cpp, ollama). |
| 3 | **INSPIRE-ONLY** | — | Ring-buffer event model → fak's internal event bus could use similar structured schema. |
| 4 | **INSPIRE-ONLY** | — | HTTP/2 + WebSocket parsing → fak's gateway already handles HTTP/1.1; HTTP/2 support would be new. |
| 5 | **INSPIRE-ONLY** | — | SSE accumulation with message-id correlation → fak's streaming handlers could adopt message-id based reassembly for fragmented provider streams. |
| 6 | **INTEGRATE** | #10737 | Token usage priority merge (network > native) → fak's token counting adds source priority when multiple sources report same call. |
| 7 | **INTEGRATE** | #10739 | MaterializedView pattern → bounded in-memory read model with typed rows in `internal/sessionview`. |
| 8 | **INTEGRATE** | #10740 | View-native process tree & session activity counters → enriched `fak ps` / `fak top` columns. |
| 9 | **INTEGRATE** | #10740 | Session-centric live monitor → `fak ps` / `fak top` enriched with tool, execution, failure, and effect activity. |
| 10 | **INTEGRATE** | #10738 | Agent-native Gemini CLI session parser → `fak session-audit` discovers and audits `~/.gemini/tmp/**/chats/*.json`. |
| 11 | **WATCH** | — | Container resolution → relevant if fak targets containerized agent workloads. |
| 12 | **INTEGRATE** | #9559 | OTel GenAI export → already tracked under existing issue #9559. |
| 13 | **INTEGRATE** | #10739 | Bounded retention with monotonic counters → combined with #10739 materialized view row model. |
| 14 | **WORLDVIEW-FINDING** | `docs/roadmap` | Their "semantic gap = intent vs action at kernel/TLS boundary" reframes fak's "security gate" — fak's gate is at *tool call*, not *kernel syscall*. Different layer. Documented as design consideration. |

---

## Filed issues

- **#10736** (Parent study tracker): `research: deeply study eunomia-bpf/agentsight`
- **#10737** (Child issue 1): `feat(session): add multi-source token usage priority merge to prevent double-counting`
- **#10738** (Child issue 2): `feat(sessionaudit): add Gemini CLI session parser and discovery to session-audit`
- **#10739** (Child issue 3): `feat(sessionview): add bounded materialized view row model for session observability`
- **#10740** (Child issue 4): `feat(top): enrich session ps/top table with tool call and effect activity counters`
- Prior existing tracking: **#9559** (`feat(trajectory): export normalized attribution events as JSONL and OpenTelemetry`)

**Durable study receipts:**
- Discovery & candidate witness: `study_f51c1d2130e4c1b521ac9c689718677da6f0faba1bfb185cfd8cb53b4b7d1d13`
- Mapped issue inventory: `study_8e02e83cb64d8494cf8fe3d320cf13be15d6d1b9f9a296f4f52eaba07f124f0d`
- Inventory map: `docs/research/inventory/eunomia-bpf-agentsight.json`

## Companions

- `field-borrow` — for each INTEGRATE candidate above
- `sota-check` — if any candidate is a compute kernel (none here)
- Parent tracker: **#10736**