# Borrow study: trajectory query / replay / signals — agent-lens + Laminar (2026-07-09)

Studied two external repos for how they **capture, store, query, search, replay, and monitor agent
trajectories**, then witnessed each borrowable idea against fak's existing data plane and filed the
net-new ones. Goal framing: fak wants sessions/trajectories to be *queryable* — this survey finds the
specific mechanisms neither fak's capture layer nor the already-scoped session-analytics epic (#2822)
covers.

Sources (deep-read, cited at `path:line@sha`):
- **agent-lens** — `github.com/dreadnode/agent-lens` @ `9eab2b0ca` (0.2.0). A multi-session agent
  interpretability harness: runs Claude-Code/Codex sessions, normalizes both onto the **ATIF**
  trajectory schema, tracks every FS write via a **shadow git**, captures raw API traffic through a
  reverse proxy, and offers **resample / replay / intervention** counterfactual tools + an auto-judge.
- **Laminar (lmnr)** — `github.com/lmnr-ai/lmnr` @ `39a1e595b`. An OTLP trace-observability backend
  (Rust `app-server` + ClickHouse + Quickwit): wire-side **content-addressed dedup**, a hardened
  **SQL-over-traces** sandbox shared by MCP/CLI/agent/UI, and **Signals** (plain-English LLM evaluators
  → events → clusters → alerts).

## fak baseline (what already exists — witnessed)

fak has a **mature capture + analysis data plane** but three deliberate holes that are exactly the
"queryable trajectory" gap:

- `internal/trajectory` folds the kernel ABI event stream into typed per-turn `Turn` rows (tool,
  verdict, `ArgsDigest`/`ResultDigest`, token/byte cost, simhash) — exported via `fak traj export`,
  scored via `internal/trajhook` (`duplicate_query`, `cost_outlier`, `high_deny_rate`), similarity via
  `internal/simhash`. **Off by default** (`FAK_TRAJECTORY=1`). Spec: `docs/observability/trajectory.md`.
- `fak trajctl` — append-only objective/score ledger with `curve` (HEALTHY|STALL|DRIFT signal) and
  `scorers` (calibration leaderboard).
- Guard session logs — two hash-chained JSONL streams: the audit journal (kernel verdicts) and the
  toolproc journal (`.fak/toolproc/journal.jsonl`, spawn/exit/session_end), plus a `guard_sessions.jsonl`
  registry. Replayable via `fak audit verify`; live view via `fak toolproc ps`.
- `internal/memq` — a JSON pipeline query language over memory (filter/rank/limit/dedup + wikilink
  edges). `internal/recall/journal_index` — trust-ranked recall. `internal/sessionsearch` — pure-Go
  TF-IDF FTS over the toolproc journal. `internal/sessionobs` / `fak sessions` — git-witnessed
  value-vs-waste session corpus. `internal/eveimport` — **import-only** OTel span ingest.

**The three holes** — (1) fak journals carry *verdicts + metadata, not prose bodies* (redaction is
structural) → no full-fidelity replay or portable export; (2) scorers are *hand-coded Go*, not
NL-defined LLM evaluators; (3) there is *no query surface a worker can point at its own run* (no SQL,
no MCP/CLI "query your trajectory"), and OTel is inbound-only. There is also no central runs DB — the
`.dispatch-runs/` / `.goal-runs/` trees are loose per-run files.

Confirmed ABSENT via grep: no SQLite/FTS5 C dep, no `database/sql`, no OTLP export.

## Dedup against prior fak work

- **#2822 session-analytics** already scopes tool-call rollups (tool-mix, timing, **input/output SHAPE**,
  cross-session trends) over this same data plane, and `docs/notes/RESEARCH-SESSION-TOOL-ANALYTICS-SOTA-2026-07-06.md`
  surveyed the SOTA (OTel/OpenInference/PM4Py/BFCL). **This study steers away from shape-analytics** —
  everything below is the *query / replay / signal / interchange / attribution* layer that note did not
  cover (it surveyed neither agent-lens nor Laminar in depth).
- **#3340** (`feat(agent): cross-turn verbatim-span dedup — fold a repeated tool-output span to a
  prefix-monotonic pointer`) already scopes trajectory-body dedup. Laminar's content-addressed dedup is
  **concrete prior art for #3340, not a new issue** — routed there as a comment, not re-filed.
- Homes are existing epics under the harness-native program **#2387**: sessionledger **#2392**
  (content-addressed witness-stamped event log), checkpoint **#2394** (git-witnessed rewind),
  devindex **#1287** (agents query fak's own facts), session-analytics **#2822**.

## Borrows (witnessed, filed)

| # | Borrow | Source @sha | fak witness | Angle | Home / issue |
|---|--------|-------------|-------------|-------|--------------|
| 1 | **ATIF as fak's portable trajectory interchange/export format** — one schema normalizes Claude-Code + Codex; subagent refs by `parent_tool_use_id`; consumable by SWE-bench/LiveCodeBench/Terminal-Bench eval + SFT/RL pipelines | `atif_adapter.py` (`ATIFAdapter.process_event` → `build_trajectory`/`build_subagent_trajectories`/`attach_subagent_refs`); schema = external `harbor.models.trajectories` @`9eab2b0ca` | `fak traj export` emits fak's own redacted analysis-JSONL, not a portable/replayable trajectory; no ATIF. fak already runs LiveCodeBench (raw-vs-fak arm). **ABSENT** | both (dev interop/eval + product interchange) | #2392 + M5 → **#3440** |
| 2 | **Shadow-git per-step write attribution** — a git dir separate from the repo's real `.git` (`--git-dir`/`--work-tree`) snapshots the worktree non-invasively; the git index is the diff cache that attributes each file change to the exact step → `state_changelog.jsonl` | `shadow_git.py` (`ShadowGit`, `commit_baseline`/`commit_snapshot`, worktree ops) + `state.py` (`StateManager.check_for_writes`) @`9eab2b0ca` | fak has commit-level `dos_commit_audit` (claim-vs-diff) but **no within-session step→bytes binding**; guard journals carry verdicts, not diffs. **ABSENT at step granularity** | dev (witness/debug) | #2394 → **#3441** |
| 3 | **Plain-English behavioral Signals** — a signal = `{name, prompt (plain English), structuredOutputSchema (JSON Schema the evaluating LLM fills), sampleRate}`; BATCH backfill vs REALTIME on-ingest; events → clusters → alerts. Ships templates: Failure/Looping/Friction/Hallucination detectors | `frontend/lib/actions/signals/index.ts`; templates `frontend/components/signals/prompts.ts`; `signal_runs` modes in `app-server/src/query_engine/schema.rs` @`39a1e595b` | `internal/trajhook` scorers are hand-coded Go; no NL-defined + LLM-evaluated + sampled signal. Answers the 07-06 note's open problem (behavioral/phase segmentation). **PARTIAL** | product (fleet obs) + dev | #2822/M8 (behavioral, not shape) → **#3442** |
| 4 | **SQL-over-trajectories + hardened AST-rewrite validator, one-schema-many-consumers, exposed as an MCP/CLI tool the agent points at its own runs** — SELECT-only; blocked-function allowlist; CTE-shadow reject; physical table→tenant-scoped `_v0(project_id=…)` **view rewrite** (scoping at the *table* level, not a droppable WHERE); fail-closed post-rewrite recheck. One `schema.rs` renders into agent prompt + MCP tool desc + validator + UI | `app-server/src/query_engine/validator.rs` (`validate_and_secure_query`); `schema.rs` (`build_schema_prompt`); `api/v1/sql.rs` + `api/v1/cli/sql.rs` (one handler, MCP+CLI); NL→SQL structured-refusal `frontend/lib/actions/sql/generate.ts` @`39a1e595b` | `memq` (pipeline query over memory) + `sessionsearch` (TF-IDF) exist, but no SQL-over-trajectory and no agent-facing "query your own run" tool; #1287 devindex is adjacent. **ABSENT** | dev (agent-facing) + product | #1287 → **#3443** |

## Borrows documented, not filed (routed or secondary)

- **Content-addressed trajectory dedup / ~20x compression** — blake3 over canonicalized sorted-key
  JSON → 32-byte hash; ship hashes on the wire (`preprocess_for_queue` nulls repeated payloads except
  root spans); rehydrate at read via a `ReplacingMergeTree` dedup store + dictionary + `spans_v0`
  `arrayMap(dictGetOrDefault(...))` view. Source: `app-server/src/traces/input_dedup.rs`,
  `frontend/lib/clickhouse/migrations/43_llm_messages.sql` @`39a1e595b`. → **routed to #3340** as concrete
  implementation prior art (fak already owns a CAS via `internal/abi` RefBlob; #2929 gateway hibernation
  is adjacent). Not re-filed.
- **Auto-judge every N turns with early-exit** — `judge.py` (`Judge`/`JudgeVerdict`, `render_trajectory`)
  @`9eab2b0ca`. Pairs with Signals (#3 above); folds into the M6 agentic-loop / guard-lifecycle #1193
  surface rather than a standalone issue.
- **Counterfactual resample / replay / intervention** — re-fire a captured request N times (`resample.py`),
  branch a whole session from turn N with worktree-based FS reset via `uuid_map` ↔ shadow-git tags
  (`replay.py`), and JSON-path edits to a deep-copied request for prompt variants
  (`ui/src/routes/api/resample/+server.ts:62`) @`9eab2b0ca`. Research-flavored; natural extension of
  checkpoint #2394 (which plans rewind but not counterfactual intervention). Documented for when #2394
  lands; not filed now.
- **UI patterns** (agent-lens `ui/`, SvelteKit, directory-convention-as-API): flat-steps→turn-card
  grouping, per-step `#step-N` deep-link anchors, turn↔request-index alignment, cross-sample variance
  stats (avg tokens / distinct tool-sets / distinct stop-reasons). Relevant if/when fak builds a
  trajectory viewer; not filed.

## The strategic read

agent-lens fills the **dev/research** side (ATIF interchange, shadow-git step→diff attribution,
resample/replay, auto-judge); Laminar fills the **product/fleet** side (Signals, SQL-over-traces,
MCP-for-the-agent, content-addressed compression). fak already owns the hard part — the witnessed,
hash-chained capture plane — so these borrows are the *read side*: turning a tamper-evident event log
into something a human or a fleet worker can query, replay, and get alerted on. ATIF export (#1) is the
highest-leverage single borrow: it makes every fak session consumable by the eval + training pipelines
fak already runs.
