---
title: "fak CLI Reference — Verbs of the Agent Kernel"
description: "The fak CLI reference: every verb (serve, run, preflight, bench, turntax, agent, recall, dream, debug, policy, hook) for the in-process agent kernel."
---

# fak — the agent kernel

`fak` is the runnable implementation of the Fused Agent Kernel: one Go binary
that puts a policy and quarantine boundary between an AI agent and its tools — with
an external dependency set of two `golang.org/x` modules, so the gateway, the
capability gate, the quarantine, the
audit log, and the metrics all live in that single process (no Python/CUDA stack, no
sidecars). The practical use is gateway-first: keep your existing model or agent host,
front it with `fak serve`, and make tool calls cross a reviewable capability floor
before anything dangerous executes.

Under the hood, every tool call becomes a syscall-like request: adjudicated
in-process, served from a local **tool vDSO** when possible, screened by a
**pre-flight + grammar ladder** before it fires, and admitted through a
**context-MMU** before tool results enter model context.

## `goal`: canonical intent across harnesses

`fak goal` keeps durable user intent separate from an execution root or session. Bind only stable identities explicitly supplied by a harness; never derive identity from a title, prompt, task, issue text, or registration ID.

```bash
# Create one canonical goal, then bind harness-native identities to it.
fak goal create --title "Improve fleet observability" --actor operator --authority user
fak goal bind --id goal_<hex> --namespace claude:goal --external-id <claude-goal-id> --actor claude --authority harness
fak goal bind --id goal_<hex> --namespace codex:goal  --external-id <codex-goal-id>  --actor codex  --authority harness

# Resolve at a launch boundary. Read env.FAK_GOAL_ID from the versioned JSON and
# export it for the child; an unknown or ambiguous binding exits nonzero.
fak goal resolve --namespace claude:goal --external-id <claude-goal-id>
```

Historical roots stay unbound unless an operator supplies an independent witness. Preview the exact root records first, then apply; the command also records a typed `fak:session-root` binding in the canonical registry.

```bash
fak goal backfill-root --id goal_<hex> --root-registration-id <root-id> --witness <artifact-ref> --actor operator --authority operator-declared
fak goal backfill-root --id goal_<hex> --root-registration-id <root-id> --witness <artifact-ref> --actor operator --authority operator-declared --apply
```

The same lookup contract supports explicitly witnessed `fak:trajctl`, `github:issue`, and `dos:unit` bindings. An optional `--revision` selects one external revision; omitting it fails closed if several revisions match. Fak does not introspect Claude or Codex state automatically. If a harness supplies no stable identity, launch without `FAK_GOAL_ID` and the execution root remains explicitly unbound.

## `microcontextdemo`: bounded logical-context fabric

`microcontextdemo` is the research/demo CLI for one immutable agent base serving bounded physical model slots across 100, 1,000, and 10,000 logical contexts. It is a separate demo binary, so invoke it with `go run ./cmd/microcontextdemo` rather than as a `fak` subcommand.

```bash
# Offline LCD floor: deterministic scheduler/shared-base semantics, not model tokens/s.
go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64

# Verify captured controlled witnesses.
go run ./cmd/microcontextdemo -verify-kernel-prefix-ab experiments/microcontext/s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json
go run ./cmd/microcontextdemo -verify-quality experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json
go run ./cmd/microcontextdemo -verify-health-scorecard experiments/microcontext/s5-gcp-1000-cuda-health-scorecard-2026-08-07.json
```

Core live flags match `microcontextdemo -h`: `-endpoint`, `-model`, `-provider`, and `-hardware` declare endpoint provenance; `-contexts` and `-workers` separate logical orchestration from bounded physical execution; `-request-timeout` and `-run-timeout` bound calls and runs. The API-only admission envelope uses `-api-concurrency`, `-api-rpm`, `-api-tpm`, and `-api-spend-micros`. Real in-kernel compatibility batches use `-gguf`, `-compat-batch-hardware`, `-compat-batch-size`, and `-compat-batch-execution`.

All artifact modes have explicit `-verify-*` counterparts in `-h`. The research contract and claim boundaries are in [`docs/research/micro-context-fabrics.md`](research/micro-context-fabrics.md); captured artifacts are indexed in [`docs/research/README.md`](research/README.md).

## Use fak with your coding agent (Claude Code, Cursor, …)

If you drive a coding agent, fak fits in two ways:

- **In front of the model** — point your agent's `ANTHROPIC_BASE_URL` (or OpenAI
  base URL) at `fak serve`, and every tool call the model proposes is
  denied / repaired / quarantined by the kernel before your agent runs it. No
  agent-side changes; the Anthropic `/v1/messages` and OpenAI
  `/v1/chat/completions` wires are both adjudicated, and a dropped or repaired
  call comes back with an in-band `[fak]` note so the agent adapts instead of
  looping. Witnessed live on macOS + Windows with the real Claude Code CLI:
  [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) (one command — `scripts/dogfood-claude.sh`,
  or `scripts/dogfood-claude.ps1` on Windows).
- **As an MCP server** — `fak serve --stdio` exposes the kernel's verbs
  (`fak_adjudicate`, `fak_syscall`, `fak_admit`, …) as MCP tools, so your agent
  can ask the kernel for a verdict before it runs a tool, or screen a result it
  executed itself through the exfil floor. Copy-paste config + the tool catalog:
  [`examples/mcp/`](https://github.com/anthony-chaudhary/fak/tree/main/examples/mcp) (drop [`examples/mcp/.mcp.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/.mcp.json)
  in your project root for Claude Code).

Both put the **same reviewable capability floor** (`--policy floor.json`) on every
tool call. New here? The fastest "feel it" is the no-credential boundary below;
the fastest "use it for real" is the dogfood path above.

On the in-front-of-the-model path, an upstream rate limit **never sleeps past your
client**. fak absorbs short waits in-handler — the transient backoff schedule caps
at 30s, and a provider-named wait that fits under the ceiling is simply slept, which
is the right UX for a throttle. But when the honored wait is longer than a wrapped
client can hold a request open, sleeping is structurally uncompletable: your agent's
own request timeout (~300s for Claude Code) fires first, so the wait burns futile
client-side retries and then dies with an opaque timeout that names none of the
rate-limit truth fak already holds. Past the ceiling fak therefore stops retrying and
hands the truthful response downstream instead — the real upstream status (429, or
529 for an overload), the provider's own `Retry-After` header verbatim, and the
distinct `upstream_retry_ceiling` error code, so a Retry-After-honoring harness backs
off correctly and a supervisor can park the turn across the reset rather than
hammering it. `FAK_INHANDLER_WAIT_CEILING` sets the boundary (any Go duration,
default `90s` — safely under that ~300s client timeout — clamped to `[0, 1h]`; `0`
disables the ceiling and restores absorb-everything). The total retry budget
(`FAK_PLANNER_RETRY_BUDGET`, default 4h) still bounds the loop overall, but on this
proxy path it is deliberately **not** reachable in-handler: riding out a longer
window is a supervisor's job (`fak manage`), not a sleeping request handler's.

## Try the boundary first

Run the no-credential proof before reading the architecture:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

This shows the adoption posture in one screen: a dangerous support action is
denied, a benign search is allowed, and the offline injection A/B blocks the
poisoned instruction while the task still completes.

## Try the live gateway

The boundary above is offline. To put the gate **in front of a real model** over
OpenAI-compatible HTTP, run `fak serve`: it fronts any OpenAI-compatible upstream
(Ollama / vLLM / llama.cpp / a cloud provider), and on every `/v1/chat/completions`
it denies / repairs / quarantines the tool calls the model proposes before returning
the survivors.

```bash
ollama serve &                         # any OpenAI-compatible server on :11434
ollama pull qwen2.5:1.5b
go run ./cmd/fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
# from a second terminal:
curl -s http://127.0.0.1:8080/healthz                       # liveness
curl -s http://127.0.0.1:8080/metrics | head                # Prometheus scrape
```

Point any OpenAI client at `http://127.0.0.1:8080/v1`. `fak serve` also speaks the
Anthropic `/v1/messages` wire, exposes `/metrics` + `/debug/vars`, and takes
`--policy floor.json` + `--require-key-env VAR` to harden it. One caveat worth
knowing up front: `stream:true` SSE is **synthesized from the finished,
already-adjudicated turn** — well-formed, but not true token-by-token streaming. (The
client's `max_tokens`/`temperature`/`top_p`/`stop` are now forwarded per request, so
long completions are no longer truncated.) Full walkthrough (Tiers 0–2):
[`GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md).

> **Status: shipped & benchmarked** (current release `v0.30.0` — single source of truth is
> the root [`VERSION`](https://github.com/anthony-chaudhary/fak/blob/main/VERSION) file). `go build`/`go vet`/`go test ./...` green across the
> internal packages, CI green, and the A/B benchmark gate passes. Confirmed not by self-report
> but by the DOS truth syscall (`dos_verify`): the shipped line runs from **`v0.1.0`** (the
> syscall skeleton, sha `c72ddf1`) through **`v0.2.0`** (model fusion + security substrate +
> gateway) and **`v0.2.1`** (gateway hardening + the turn-tax bench). Note: `dos verify` on
> the v0.2.0/v0.2.1 release-bump tags returns `shipped:false` because those commits touch only
> `VERSION` + release notes; the actual code ships across the commits each tag caps (see
> `../STATUS.md` §1 for the caveat). The truth syscall confirms the *code* ships, not the
> version bump — see `../docs/releases/`.
>
> **What v0.2 added on top of the v0.1 syscall skeleton** (each its own witnessed lane):
> a **real model fused into the kernel** (a pure-Go SmolLM2-135M forward pass, every rung
> proven bit-for-bit vs HuggingFace, then made parity-fast — `MODEL-BASELINE-RESULTS.md`);
> a **kernel-authored security substrate** (information-flow control, a trust/provenance
> classifier, plan-CFI, an effect-verifying witness gate, and a normalize-and-rescan
> admission driver — the kernel stops believing the model's self-reports); a **gateway**
> (`fak serve`) fronting the syscall boundary over OpenAI-compatible HTTP + MCP; a durable
> **session core-dump + context debugger + dream cleanup** (`recall` / `fak debug` /
> `fak dream` — a quarantine that survives the process boundary, plus an offline
> re-screen/prune pass, `RECALL-RESULTS.md` / `CDB-RESULTS.md` /
> `MEMORY-DREAM-CLEANUP-RESULTS.md`); and a dedicated
> **turn-tax benchmark** (`fak turntax`) pricing the extra error-code model turn the
> 1-shot kernel deletes. The `fak agent` verb still drives the **live** turn-count A/B
> (real Gemini + local Qwen2.5, transcript-hashed) — see `LIVE-RESULTS.md`.

## Syscall subsystem check — useful, not the headline KPI (call-mix-independent)

`fak bench --suite tau2-smoke` replays a frozen tool-call trace through the one
binary and times the **in-process tool-call adjudication boundary** against a
**spawned-hook baseline** measured on the same machine (the same `Fold` decide, two
transports — apples-to-apples):

```
in-process adjudication p50 : 2,427 ns
spawned-hook        p50     : 6.913 ms  (process-per-decide, this machine, n=100)
SUBSYSTEM CHECK (gate_primary): pass   (~2,849x fusion speedup)
```

This is a **subsystem regression sentinel**, deliberately **not** the headline
product KPI: it is independent of the tool mix — it times the adjudication fold,
not the workload, so a low vDSO hit-rate can never turn it red — but it does not
prove production readiness, model quality, serving throughput, or the 45x fleet
claim. (For speedup figures: 45× = Phase-0 batched-decode gate currently failing at
40.98×; ~60× = headline session wall-time vs naive stateless; ~1.5–4× = realistic
gain vs tuned warm-cache stack.) The vDSO hit-rate and token savings are reported
as **soft UPSIDE secondaries**, never the gate. See `STATUS.md` §2 and `CLAIMS.md`
(unit 82).

## `fak orchestration status`

`fak orchestration status` joins the newest local orchestration launch receipt, worker process liveness, and JSONL turn progress into one read-only fleet view. Select an older launch with `--session ID`; use `--home DIR` when runtime artifacts live outside the current directory.

```bash
fak orchestration status
fak orchestration status --session 01a015da-c527-7ed0-b8e8-3e0d9b7da5ed
fak orchestration status --json
```

Human output leads with the run verdict and running/completed/exited counts. `--json` emits `fak.codex_orchestration_status.v1` with the same per-worker PID, process, log freshness, last-event, and turn-count evidence. The command observes workers; it does not stop, replace, or mutate them.

## `fak agent` response profiles

`fak agent --output-style <selection>` opts the owned agent loop into a named response shape. The default is `full` (no shaping). Run `fak agent profiles` or `fak agent profiles --json` to list shipped, reserved, and not-yet selections. The recommended concise compatibility setting is `caveman:medium`; it canonicalizes to the fak-authored `caveman:native:medium`. See [Response profiles](response-profiles.md) for intensity guidance, native-versus-original provenance, preservation rules, and the current harness boundary.

```bash
fak agent profiles
fak agent --output-style caveman:medium --task "Explain this failure"
fak agent --output-style full
```

Unknown or not-yet selections fail before the run. Response profiles do not change work policy or tool authorization.

`fak agent --work-profile standard|ponytail:{low|medium|high}` independently selects implementation-policy pressure. `standard` is the off/default state; `ponytail:*` expands to the fak-native canonical form. Mix it with `--output-style` without either axis implying the other. See [Work profiles](work-profiles.md).

## Verbs

The command surface has two executable boundaries. The shipped **`fak` runtime** is
the product — what an adopter or operator touches (`manage`, `serve`, `agent`,
`run`, `preflight`, `policy`, `attest`, `audit`, `egress`, `info`, `session`,
`ps`/`top`, `signal`, `resume`, `doctor`, `recover`, `model`, `codex`,
`self-update`, `version`, `help`). `fak help` and `fak help --all` describe that
runtime. Repository workflows, scorecards, benches, and issue/docs tooling live
in the separate **`fak-dev`** executable; `fak-dev help` lists them. The legacy
`fak dev <verb>` spelling is a compatibility handoff to a sibling or
`PATH`-installed `fak-dev`, not a dev namespace linked into the runtime.

### `fak architecture`

`architecture` is the read-only query surface for the same tier declarations and Go import graph enforced by `internal/architest`; it does not maintain a second package taxonomy.

```bash
# Whole graph: tier counts, direct fan-in hotspots, violations, and diagnostics.
fak architecture

# One leaf: declared tier, import-derived floor, direct plus transitive dependency reach/depth, and reverse blast radius.
fak architecture --leaf archreport

# Stable machine-readable form for automation.
fak architecture --leaf archreport --json

# Compare two supplied workspace snapshots (no implicit Git execution).
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --json
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-violations
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-diagnostics
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on increased-tier-gap
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on increased-violation-distance
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on increased-blast-radius
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-blast-impacts
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on increased-blast-path-length
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-lateral-edges
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-lateral-couplings
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-or-increased-lateral-bridges
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on introduced-or-increased-lateral-articulation-points
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on resolved-lateral-resilient-pairs
fak architecture --baseline-workspace /path/to/before --workspace /path/to/after --fail-on decreased-lateral-edge-connectivity

# Privacy-safe adoption fold (ISO-week counts; no paths, hostnames, or leaf names).
fak architecture --usage
fak architecture --usage --json
```

The JSON schema is `fak-architecture/1`. A full report includes `tiers`, `leaves`, a typed `edges` inventory, weakly connected same-tier `lateral_components`, direct-fan-in `hotspots`, transitive `blast_hotspots`, `diagnostics`, and the upward `violations` count. Lateral components expose tier evidence, sorted members, member count, and internal lateral-edge count; singletons are omitted, rankings prefer larger/denser components, and scoped reports retain only the selected leaf's component. Typed `lateral_bridges` identify articulation dependencies whose removal splits a component, retain directed import orientation plus canonical endpoints, list both sorted sides, and price the induced co-memberships as `coupling_pairs = len(left_side) * len(right_side)`; diamond edges are correctly omitted. Typed `lateral_articulation_points` identify package seams whose removal fragments a component, retain every sorted fragment, and price cross-fragment coupling pairs while excluding the removed package; chains/stars expose their internal/center packages and cycles omit all members. Complementary `lateral_biconnected_blocks` expose maximal same-tier regions of three or more packages that remain connected after removing any one member, with sorted membership and internal edge count; articulation packages may belong to multiple resilient blocks and bridge-only pairs are omitted. Each block quantifies package-failure resilience with `min_vertex_cut` and one canonical sorted `critical_separator` (complete K_n reports n-1 removals), separately from unit-capacity undirected `min_edge_cut` and every canonical `critical_pair` witnessing that edge minimum by pairwise max-flow; duplicate import orientation cannot inflate either capacity. `vertex_pair_cuts` add canonical local cuts and endpoint-excluding separators for every non-adjacent package pair; adjacent pairs are omitted because no finite separator excluding endpoints disconnects their direct edge, and text surfaces pair rows at the block minimum. Diffs project blocks into canonical `introduced_lateral_resilient_pairs` and `resolved_lateral_resilient_pairs`; `--fail-on resolved-lateral-resilient-pairs` gates only lost single-package-resilient connectivity and recommends restoring a cycle or redundant path. Blocks also retain `pair_cuts` for every member pair; each row carries one canonical, sorted `cut_edges` minimum-cut witness derived from the final residual graph, with witness cardinality equal to `cut`, plus sorted `source_side` and `sink_side` package partitions. The queried left package is source-side, the right package is sink-side, and every witness edge crosses the disjoint partition whose union is the block. JSON retains every pair witness and partition, while text expands critical pairs as actionable edge sets and failure domains. Stable pairs emit `lateral_edge_connectivity_changes` with additive before/after cut edges and source/sink sides; text and the failure policy name both the bottleneck and failure-domain transition directly. `--fail-on decreased-lateral-edge-connectivity` gates only reductions in edge-disjoint path capacity. Stable tier+membership blocks also emit `lateral_vertex_connectivity_changes` with before/after package separators; `--fail-on decreased-lateral-vertex-connectivity` gates only reduced package-failure tolerance and recommends diversifying paths around the critical separator. Stable non-adjacent pairs emit `lateral_vertex_pair_cut_changes` with before/after local separators; `--fail-on decreased-lateral-vertex-pair-cuts` gates only reduced internally vertex-disjoint package paths, while pair rows that appear or disappear due to adjacency changes are not treated as numeric drift. Diffs preserve point identity by tier and package, expose introductions/resolutions plus stable fragment and coupling-impact drift, and `--fail-on introduced-or-increased-lateral-articulation-points` gates new package seams or positive impact while resolutions/decreases stay clean. Diffs preserve bridge identity by tier and canonical endpoints, expose introduced/resolved bridges plus stable `lateral_bridge_changes`, and `--fail-on introduced-or-increased-lateral-bridges` gates new articulation points or positive induced-coupling drift while resolutions/decreases remain clean. Diffs expand each component into canonical unordered co-membership pairs, exposing `introduced_lateral_couplings` and `resolved_lateral_couplings`; merges/growth therefore name every newly coupled pair, splits name every resolved pair, and `--fail-on introduced-lateral-couplings` gates only new reachability. Every live declared dependency edge carries endpoint tier numbers/names, signed `tier_delta`, and closed `direction` (`rootward`, `lateral`, or `upward`); scoped reports retain only edges originating at the selected leaf, and text summarizes direction counts while JSON retains the inventory. Legal rootward edges that descend more than one tier appear in `rootward_layer_skips` with positive distance and skipped-tier count, ranked by bypass size; scoped reports retain only skips originating at the selected leaf. Diffs expose introduced/resolved skips and stable-endpoint distance changes; `--fail-on introduced-or-increased-rootward-layer-skips` gates only new bypasses or positive skipped-tier drift, leaving ordinary one-layer descent, resolutions, and contractions clean. Diffs expose full after/before evidence as `introduced_typed_edges` and `resolved_typed_edges` while retaining legacy `added_edges`/`removed_edges`; `--fail-on introduced-lateral-edges` gates only newly added same-tier coupling and recommends moving or extracting the seam downward. Forward hotspot tables complement reverse risk: `fan_out_hotspots` rank direct dependency count, while `dependency_hotspots` rank transitive reach, depth, fan-out, then name. Stable leaves emit `fan_out_changes`; `--fail-on increased-fan-out` independently gates growth in immediate dependency count without conflating fan-in, transitive reach/depth, or edge direction. Blast hotspots carry leaf, blast radius, and maximum shortest-path hops, ranked by radius descending, depth descending, then name; scoped reports omit all whole-graph hotspot tables while retaining selected-leaf evidence. A leaf distinguishes `dependencies` (what it imports) from `dependents` (what imports it directly). Sorted `transitive_dependencies`, `dependency_reach`, and `dependency_depth` expose the complete forward footprint; typed `dependency_paths` retain one canonical shortest source-to-dependency path, choosing the lexically smallest full path on equal lengths and omitting the source when cycles return to it. `dependency_dominators` distinguish optional shortest-path intermediaries from mandatory seams present on every directed path to a dependency; rows carry the strict sourceward-to-destinationward dominator chain plus shortest-path context, and scoped text renders them as extraction boundaries. `redundant_dependencies` identify direct imports whose destination remains reachable after removing that edge; each row carries the deterministic shortest alternate path, and scoped text renders these as transitive-reduction opportunities rather than mandatory seams. Directed `dependency_cycles` are strongly connected components: multi-package SCCs and self-import loops are retained with sorted members and internal edges, while acyclic singletons are omitted. Each selected leaf carries its `dependency_cycle` membership. These directed cycles are distinct from `lateral_components`, which treat same-tier coupling as undirected for resilience analysis. Stable leaves emit `dependency_reach_changes` and `dependency_depth_changes`; `--fail-on increased-dependency-reach` gates footprint growth, while `--fail-on increased-dependency-depth` independently gates a deeper shortest dependency stack. Contractions remain clean for both policies. Sorted duplicate-free `transitive_dependents` and `blast_radius` expose the complete reverse closure and its size. Typed `blast_paths` retain one deterministic shortest source-to-dependent path for every closure member (lexical tie-break for equal lengths), so remediation does not require another graph traversal. Scoped text prints those paths; whole-graph text stays concise and JSON retains all evidence. `tier_gap` measures declared tier minus import-derived floor. Upward imports are machine-readable in `violation_edges` with typed `from`/`to` leaves, tier numbers/names, and derived `tier_distance`; edges sort by distance descending then endpoint names, and `max_violation_distance` exposes the deepest inversion for full or scoped reports; the legacy `violations` display-string array remains as an additive compatibility projection for `fak-architecture/1` consumers. Full reports rank `sink_candidates` whose gap is at least two, largest gap first then leaf name, so the old verbose-test mis-tier advisory is queryable by operators. Hotspots are sorted by direct fan-in descending, then leaf name.

With `--baseline-workspace`, the command emits `fak-architecture-diff/1`: added/removed leaves, old→new tier changes, added/removed direct edges, derived direct fan-in and tier-gap changes, introduced/resolved typed upward-violation edges, stable-edge violation-distance changes (plus their legacy string projections), introduced/resolved typed diagnostics, and a typed `clean`/`regression` verdict. `fan_in_changes` records each affected leaf’s before/after count and signed delta; growth precedes shrinkage, then sorts by absolute magnitude descending and leaf name, and this derived view is not double-counted in `changes`. `tier_gap_changes` records comparable leaves’ before/after import floors and gaps; worsening precedes improvement with the same magnitude/name ordering, while added/removed leaves are excluded. Both are derived views and are not double-counted in `changes`. Diagnostics are matched by stable `(kind, leaf)` identity, so workspace-specific paths in diagnostic messages cannot fabricate a delta; output retains the relevant side’s message and recovery. The caller supplies both snapshots; an empty diff is `0 change(s)` and exits successfully. Add `--fail-on introduced-violations` for CI/pre-push use when a newly introduced upward edge should exit `3`, or `--fail-on introduced-diagnostics` when a newly stale declaration or other typed diagnostic should exit `3`, or `--fail-on increased-tier-gap` when a comparable leaf drifts farther above its import-derived floor, or `--fail-on increased-violation-distance` when an existing upward edge becomes a deeper inversion, or `--fail-on increased-blast-radius` when a stable leaf's transitive dependent closure grows. Positive distance or blast-radius drift is a regression; improvement remains clean absent another regression. Typed `introduced_blast_impacts` and `resolved_blast_impacts` preserve `(source, dependent)` identity plus the corresponding shortest path, so equal-count closure replacement cannot disappear; `--fail-on introduced-blast-impacts` gates only newly impacted package membership. For stable impact pairs, typed `blast_path_changes` retain both paths, hop counts, and signed hop delta; `--fail-on increased-blast-path-length` gates only positive depth drift while equal-length reroutes and contractions remain visible but clean. The blast-impact remediation names removing/inverting each introduced path or moving the shared seam down. Each policy names its remediation; resolved findings and unrelated architecture changes remain exit `0`.

The command does not run Git, execute package code, or mutate the workspace. It parses `internal/architest/architest_test.go` plus non-test Go import blocks. Malformed contracts or source files refuse with a recovery action. A stale tier declaration is a typed `stale-tier-declaration` diagnostic—not a global report outage—so healthy full and scoped queries remain usable; the committed `internal/architest` gate still fails until its recovery action (create the package or remove the tier row) is applied.

Each report invocation appends a `fak-architecture-usage/1` row under the user cache (`$XDG_CACHE_HOME/fak/architecture-usage.jsonl`, or the platform equivalent). Rows contain only timestamp, full/scoped mode, text/JSON format, outcome, and aggregate diagnostic/violation counts—never workspace paths, hostnames, usernames, leaf names, or error text. `FAK_ARCHITECTURE_USAGE_FILE=PATH` overrides the location; `off` disables recording. `--usage` folds rows into ISO-week counts, with JSON schema `fak-architecture-usage-summary/1`.

Exit codes: `0` report/fold or clean comparison emitted; `1` workspace/contract/source/ledger inspection failed; `2` flag or positional-argument misuse; `3` comparison gate found an introduced upward violation.

The `session`, `signal`, and `ps` verbs are the front door to out-of-band control
of a session that is **already running** — steer, redirect, pause, resume, cancel,
terminate, throttle, budget, priority. That closed vocabulary, what each op may
touch, the witness that proves it applied, and the closed refusal tokens are
specified in [`docs/operator-control-plane.md`](operator-control-plane.md).

```
fak dev       [<verb> ...]                             # compatibility handoff to the separate fak-dev executable
fak-dev help                                              # canonical maintainer-tool catalog (separate artifact)
fak run       --trace testdata/tau2/tau2-smoke.json    # replay a trace through the kernel
fak preflight --tool create_user --args '{"_positional":["alice"]}'   # rung-only check
fak architecture [--leaf NAME] [--json]              # inspect the enforced internal tier/import graph
fak bench     --suite tau2-smoke --out report.json     # A/B vDSO ablation -> report.json (the ns gate)
fak ablate    --sweep vdso                             # N-arm self-ablation: one frozen trace, feature on/off, deltas off the kernel counters
fak ablate    --rungs --trace TRACE.json               # per-rung attribution: replay a frozen turnbench trace, mask one adjudicator rung per arm, diff the kernel counters (--rungs=grammar,ifc-sink restricts; default suite turntax-airline)
fak turntax   --suite turntax-airline                  # price the extra error-code MODEL turn the 1-shot kernel deletes
fak agent     --offline | --base-url URL --model M --api-key-env VAR  # LIVE turn-count A/B (see LIVE-RESULTS.md)
fak manage    [--session-pressure-gate high,report=pressure.json] [--] <agent command>  # primary managed-agent front door; short alias: fak m; legacy guard spelling remains compatible during sunset
fak session   ls | status <id> | stop|pause|resume|throttle <id> | budget <id> [--turns N] [--addr URL]   # operator control of a served session's live drive state, over /v1/fak/session(s)
fak ps        [--json] [--watch] [--interval D] [--frames N] [--addr URL] [--key K]   # the read-only process table: one aligned row per live served session (`fak top` is `--watch`)
fak signal    <id> pause | resume | stop [--reason R] | steer --text "..."   # job control for a running session over the control plane: the OS process-model names, one running session at a time
fak info      [--gateway-url URL] [--interval DUR] [--once] [--json]   # the live fak-info overlay: poll a gateway's /debug/vars and print one plain-words line per tick
fak resume    plan [--resident-tokens N] [--idle-seconds S] [--ttl 5m|1h] [--horizon N] [--shed-budget N] [--seed-tokens N] [--image DIR] [--json]   # the deterministic resume-cache decision: project the cache POSTURE (cold past the TTL, warm inside it), price RESUME_FULL / CUT / RESET, recommend a cut-by-default re-entry
fak recover   <REASON> [--dry-run|--execute] [--json]   # closed-vocabulary refusal recovery: map a guard/DOS reason token to the concrete commands that clear it
fak relay     resume (--baton FILE|- | FILE) [--json]   # inspect a fak.relay.baton.v1 leg handoff OFFLINE: exactly what a successor leg would receive (pointer-only, no reload re-verification); --json emits the canonical byte-stable wire form
fak task      sample [--json] [--done N --total N]     # process-local task-manager snapshot: hardware/runtime sample + task/step/concept progress and ETA
fak task      handoff --file HANDOFF.json [--json] [--live] [--repo owner/repo]  # verified completion handoff: require StateDone + VerifiedDone + current state, then plan/sync 1-2 follow-up issues
fak test      [fast|full|race|<pkg>] [-n] [-- go test args]   # host-aware test runner; Windows routes go test through WSL/test.ps1
fak commit    --path P... (-m STR | -F FILE/-) [--push] [--json]   # path-scoped shared-trunk commit; refuses unsafe states and emits score/grade/score_notes for every outcome
fak sweep     [--json] | --clean-junk [--json] | --apply --lane L -m "SUBJECT" [--path P...] [--push]   # dirty-tree lane planner; guarded junk cleanup; lane groups carry score + score_notes before apply
fak sync      [check|apply|push] [--fetch] [--json]   # safe shared-trunk sync/push; preserves unrelated dirty work and reports the sweep next action
fak profile   <pkg> [--bench RE] [--cpuprofile F] [--memprofile F] [--top] [-n]   # host-aware Go benchmark profiler; captures pprof CPU + allocation profiles
fak console agent --account claude-seat --dry-run -- -p "task"  # native launch-plan for real Claude Code through fak manage, using a selected Claude config home
fak codex     [--dry-run] [--freshness-gate on|off] [--split off] [--managed-cache on] -- exec --json "task"  # checkout-local launchers prove freshness against origin/main, guarded self-update/re-exec once when stale, and refuse unknown/update failure; --freshness-gate off is the explicit escape; release installs without a checkout stay offline
fak c <target>|--target NAME|--auto|--list-targets      # pick a named compute backend (mac/gcp/local/anthropic + ~/.fak/targets.json); --auto ranks by health then cheapest/most-local (cost local<mac<gcp<anthropic), fails over past a DOWN target. quota is a [stub] (not a live fak accounts read) and never excludes
fak snapshot  kinds | demo | info | dump-fleet | restore-fleet   # dump/restore any primitive (turn|tool|session|fleet|RSI loop) to a portable sha256-integrity bundle
fak serve     --addr :8080 [--require-key-env VAR] [--fleet-bus [--fleet-bus-dir DIR]] [--session-registry PATH|off]   # OpenAI-compatible HTTP + MCP gateway (any-language agents). `--session-registry` scopes WHICH SESSIONS this serve can see and write (#5825): unset keeps today's shared per-user default (`FAK_SESSION_REGISTRY`, else `<UserConfigDir>/fak/session-registry.json` — the single file EVERY serve on the box shares), which is the right reach for a real fleet but means a serve started only to drive its own sessions still adopts every live session on the host, so a fanned `fak fleet control send --op pause --all` writes to peers' work. `--fleet-bus-dir` does NOT narrow this: it scopes the BUS (announcements, directives, claims, acks) and nothing about which sessions get written, so a private bus over the shared registry reads like a sandbox and is not one. Pass a path to hydrate from and mirror to that file alone, or `off` for a pure in-memory table that adopts nothing and persists nothing. An armed `--fleet-bus` serve prints its own reach — session count and registry — before it drains a single directive
fak recall    --dir DIR                                # persist/inspect a finished session as a durable core image
fak dream     --dir DIR --out-dir DIR                   # offline cleanup pass over a sleeping core image
fak debug     --session DIR --cmd report|info|bt|x|ws|grep|tombstone|context-query|context-diff   # attach to a session core image; demand-page its working set
fak answer-shape --text - --max-repeat 0.5 [--max-chars N]   # degeneration/verbosity witness over a text; exit 1 when it loops/runs away
fak doctor    --text - [--max-repeat 0.5] [--max-chars N]   # run the answer-shape witness + the kernel admit cross-check, then recommend
fak index     lane <path>... | leaf [<query>] | docs <query> | refs <pkg>.<Sym> [--json] [--limit N]   # query the devindex self-index for lane ownership, leaf search, docs, and Go refs; alias: fak devindex
fak codelint  PATH...                                  # lint agent-written code (Go/JSON in-process, Python/CUDA via toolchain); exit 1 on a hard parse/compile error
fak policy    --dump | --check FILE                        # author/validate the deployable capability floor
fak route     --aspect tool_call --tool refund_payment [--manifest FILE] [--simulate "a,b,b"]   # which model/ensemble routes this aspect; --dump/--check author the routing manifest
fak routebench [--corpus FILE] [--routed F] [--single F] [--json]            # offline routing benchmark: per-aspect+ensemble vs single-model on cost/latency/quality (no model in the loop)
fak vcache    status | prove | prove-telemetry | actions | apply-actions | score   # virtual provider-cache status, planned/applied action ledgers, and token-savings proof/refutation scorecard
fak cachevalue report|review|feed [--since DATE] [--usage-ledger FILE] [--context-budget-tokens N] [--json] [--append-ledger FILE] [--markdown-out FILE] # cache-effectiveness P&L, cumulative fleet savings/session-extension aggregate, plus generated cache-frontier review artifacts
fak cachevalue shapes [--since DATE] [--json] [--trend] [--ledger FILE]   # cluster the WITNESSED Track-1 kernel ledger by session SHAPE — length band (single/short/long) × realized-reuse outcome (n/a/cold/partial/warm) — so a reader sees WHICH KINDS of sessions earn KV-prefix reuse, a fact the week×session_type trend hides (#3115); --trend swaps the static snapshot for each shape's week-over-week reuse-share drift. #1066 fence: outcome bands cut on WITNESSED realized reuse only, never the vs-naive re-prefill multiple. The default `--ledger` (`docs/nightrun/cache-value.jsonl`) is a **gitignored local nightrun artifact**, so a fresh checkout renders `INSUFFICIENT` (0 sessions) until a nightrun writes it — that is the empty-ledger read, not a broken verb; point `--ledger` at your own Track-1 JSONL to fold rows meanwhile
fak kvbm      replay|trace [--artifact/--trace FILE] [--json] [--check]   # KVBM eviction validation: replay proves pin/restore safety; trace proves cost-aware>=LRU, oracle score, and no-thrash stability
fak callavoid prove-memo | account [--in FILE] [--json] [--gate]   # avoided-call economics: break-even memo proof + per-window amplification scorecard (JSON in/out)
fak cadence   [--json] [--check] [--append-history] [--window N]   # consolidated regular-cadence report: folds scores + maturity + work-done + releases into one control-pane envelope, including the top public `fak maturity route` seed; --append-history writes the durable ledger with standing_score + difficulty fields (docs/cadence/history.jsonl)
fak milestone report|post [--json] [--check] [--append-history]   # milestone report: the maturity CLIMB (model x backend M0-M7 grid) + the epic ROADMAP, split by WORK CLASS — DISCRETE epics on a completion % vs ONGOING optimization programs (kernel-opt, cache-opt) shown as frontier activity with NO % (they have no 100%). Trended in docs/milestones/history.jsonl
fak program   report [--json] [--check] [--append-history] [--window N]   # ongoing-program report (the milestone sibling for never-'done' work): kernel-optimization, cache-optimization, and human-operator-effectiveness by FRONTIER + TREND, never a completion %. Trended in docs/programs/history.jsonl
fak operator  brief [--cadence FILE] [--program FILE] [--milestone FILE] [--heaviness FILE] [--previous FILE] [--collect] [--json] [--check]   # human pacing brief: folds cadence/program/milestone plus optional operator-heaviness and previous-brief JSON into source coherence, since_previous delta, attention timebox/read-order, human-use guidance, strengths, choices, challenges, learning, and human/agent/watch/background buckets
fak operator  heaviness [--json] [--markdown] [--compare FILE]   # operator-surface pressure scorecard: verb surface, guard flag burden, refusal vocabulary, doc-map discoverability, appeal channel, heaviness_debt, and heaviness_pressure
fak score     <name> [--json] [--markdown] [--compare FILE]   # parent verb (#1505) grouping the meta-scorecards / RSI loops so the top-level surface stays operator daily-drivers. `fak score list` names them: conflation, dogfood, dojo-rsi, guard-rsi, guard-verdict-rsi, product, skill-effectiveness, support-maturity, token-defaults, ui-quality. Each forwards to the same handler its legacy top-level verb ran (behavior-preserving; the legacy verbs remain as thin aliases)
fak maturity  [next] [--json|--markdown] [--compare base.json]   # feature-maturity lifecycle scorecard: places every declared capability on a closed ladder and emits `fak maturity next`, the ranked backlog. `fak maturity anatomy [package-path] [--json]` adds a static package readout; `--all [--limit N]` compares every declared internal leaf and ranks structural hotspots: see [`docs/maturity-anatomy.md`](maturity-anatomy.md) for definitions and interpretation; shape, control-flow complexity, success/error/ambiguous exits, guards and assumption comments, documentation coverage, and internal dependency position. `fak maturity route [--limit N] [--fetch-existing|--live]` turns the top public-routeable backlog rows into stable, deduped GitHub issue plans so the issue-dispatch loop can work them.
fak idea-scout [--json] [--max-issues N] [--min-score N] [--config FILE] [--candidates FILE --issues FILE --scout-issues FILE] [--live]   # research-to-issue feeder (docs/idea-scout.md): score arXiv/GitHub/Hacker News/Reddit hits for relatedness, dedupe four ways (seen-cache, the label-targeted filed-stamp index, existing issue bodies, near-duplicate titles) and plan triage-ready issues. DRY-RUN BY DEFAULT — it mutates nothing. `--live` is the blast radius: it runs `gh issue create` against the current repo's REAL tracker, up to `--max-issues` public issues, and records each filed source id in `.idea-scout/seen.json`. The `--candidates`/`--issues`/`--scout-issues` fixtures replay a run offline with no network and no `gh`. Exit 2 when the filed-stamp index cannot be built completely — it REFUSES rather than risk re-filing.
fak scoreboard post [--from card.json --debt-key K | --kpi NAME --value V --grade A --verdict OK --detail ...] [--dry-run]   # post a scorecard result/score to the Slack scoreboard channel (its own FAK_SCOREBOARD_* workspace, separate from the lab bridge); CI + local agents publish a number the moment it changes
fak scorecard control-pane [--json|--check|--pin] [--post]   # LOCAL producer for the same #scoreboard feed (#998, local side of 52ed934b): --post (or FAK_SCOREBOARD_AUTOPOST=1) auto-posts the freshly-regenerated portfolio number tagged --source <hostname> the moment the scorecard is folded — off by default, reuses internal/scoreboard (no second manual `scoreboard post`), deduped via .fak/scoreboard-autopost-state.json so an unchanged rerun is silent
fak bench-loop status|next|walk|run [--json]   # benchmark super-loop manager: folds registry, recorded runs, nightrun ledger, local next selection, and authority gap; run delegates to fak nightrun run
fak bench post    --rollup latest|regression [--n N] [--catalog PATH] [--baseline PATH] [--dry-run]   # post a bench-channel rollup: the latest catalog runs (WITNESSED/OBSERVED-labeled) or tok/s drops vs the pinned baseline. FAK_BENCH_* workspace (token falls back to the scoreboard token)
fak bench request [--now STAMP | --plan-json FILE] [--top N] [--dry-run]   # post a bench RUN-REQUEST (the bench_plan next-test-per-machine) to the bench channel. A request is a POST, not a dispatch — no inbound listener; the bench-nodes act on it out-of-band
fak blockers post [--severity status|operator|clear] --title ... [--detail ... --owner "<@U>" --action ... --action-url URL --ref ...] [--dry-run]   # post a BLOCKER to the central Slack #blockers channel: a background `status` line records quietly, an `operator` one is SURFACED (pages <!here>/owner, red, with a do-this-next). FAK_BLOCKERS_* (token falls back to the scoreboard token; #blockers is the built-in default)
fak blockers feed --issues FILE [--label blocked --repo-url URL] [--dry-run]   # CI roll-up: fold a `gh issue list --json number,title,url,assignees,labels` payload into one card — clear when empty, operator (paged) when a blocker is UNOWNED, background status when all are assigned
fak chatrelay --endpoint URL --channel C0X [--model M --mention <@U> --system S --prime=false --once --interval 3s --dry-run]   # bridge ONE Slack channel to a `fak serve` /v1/chat/completions endpoint: poll history, forward each human message, post the reply in-thread. Generic chatbot front end — no shell, no command router; channel text is chatbot input, never a command. FAK_CHATRELAY_* (token falls back to the scoreboard token; channel has NO fallback). See docs/fak/slack-sessions.md
fak chatops   --channel C0X --admins U07A,U07B [--bot-user <@U07BOT> --audit FILE --prime=false --once --interval 3s --dry-run]   # inbound read-only control door: poll ONE control channel, parse each admin mention as ONE verb from a closed grammar (help/ping/status/fleet answer; dispatch/resume/halt are declined until the guarded act path lands). Fail-closed admin allowlist on the immutable user id; refuses to start with no admins; every decision journaled to --audit. FAK_CHATOPS_* (token falls back to the scoreboard token; channel/admins have NO fallback). See docs/fak/slack-sessions.md
fak fleet control send --op OP [--payload|--text TEXT] (--all | --instance I,I | --machine M | --role R) [--lane L --wave W --label X] [--ttl 5m] [--wait 10s] [--reason R] [--bus DIR] [--json] | status --directive ID | instances [--ttl D]   # the centralized control point (#5600): fan ONE op to every announced fleet instance over a shared bus directory (`fak serve --fleet-bus` arms an instance), then fold the ACKS back — `send` exits 0 only when every addressed instance witnessed the apply, 1 when it published but nobody has answered (including `--wait 0`), and 2 when the selector addresses nobody (`FLEETBUS_NO_TARGET`) rather than accepting a directive that can never apply. Instance axes (`--all`/`--instance`/`--machine`/`--role`) pick WHICH processes; session axes (`--lane`/`--wave`/`--label`) narrow WITHIN each one. An instance that matched nothing acks REFUSED, never a hollow "applied". The selectors pick which INSTANCES and which sessions within them, but the set an instance can write at all is its own `fak serve --session-registry` scope — neither `--bus` here nor `--fleet-bus-dir` there narrows it (#5825), so `--all` against the default shared per-user registry reaches every session on each addressed host, including peers'
fak leaseref  live [--dir DIR] | liveness [--session ME] [--dir DIR] | session-publish --session S [--ttl SEC] [--dir DIR] | list [--json] [--dir DIR] | audit [--dir DIR] | reap [--dir DIR] | sync [--remote R] [--push-only|--fetch-only] [--dir DIR]   # cross-machine lease visibility: read refs/fak/locks/* into the dos_arbitrate live_leases shape (#825); `liveness` classifies each live lease self|peer-live|peer-dead|peer-unknown by the owning session's heartbeat (#2164); `session-publish` refreshes that heartbeat as a side ref; `audit` is the read-only staleness report; `sync` converges the namespace with a remote (push-then-fetch, side refs only)
fak attest    --policy FILE [--probes FILE] [--json]        # compliance attestation: prove the capability floor from preflight (exit 0 PROVEN / 1 drift / 2 usage)
fak audit     verify <journal.jsonl> | export <journal.jsonl>   # audit-trail consumer: re-verify a fak manage decision journal's hash chain, or export it
fak egress    check (--url URL | --command CMD | --host HOST | --tool T --args JSON)   # prove the network-egress floor on one destination — the cloud-metadata / SSRF class
fak self-update [--check] [--force] [--root DIR] [--target PATH]   # converge a built-from-source fak binary on origin/main; --check reports staleness vs HEAD and exits without building
fak stopfailure plan | reset-stale [--apply]                # inspect and settle stale .dos/stop-failures breaker markers
fak hook      < call.json                              # spawned-hook decide (the A/B baseline)
```

`answer-shape` is the consumer-facing, GRADED dual of the context-MMU's write-time
repeat-admit rung: the kernel quarantines only the most blatant byte-repeat
pollution, while `fak answer-shape` judges the SHAPE of any candidate answer —
word-n-gram repeat, repeated-line blocks, and short-period tiling, headlined as one
`repeat` fraction — against thresholds you pick, off the hot path. It reads stdin on
`-` (or with no source), exits `1` when degenerate, so it gates a pipeline. `fak
doctor` wraps that witness into operator recommendations and additionally reports the
real verdict the context-MMU would reach on the same bytes (`ctxmmu.ScreenBytes`) —
the fak analogue of `dos doctor`.

`fak codelint PATH...` is the write-/definition-time CODE check at the kernel
boundary — the code-content dual of `fak preflight`'s tool-registry check. It routes
each file (or every file under a directory) to the owning language pack: Go and JSON
parse in-process via the stdlib, Python and CUDA shell out to their toolchains
(degrading to no-opinion when the toolchain is absent). It reports only HARD
parse/compile errors (the zero-false-positive tier — semantic checks are out of
scope) and exits `1` so it gates a pipeline. Because the input is untrusted model
output, it honors no in-content ignore comment, and it runs off the hot path.

`fak callavoid` is the operator-facing surface over the avoided-call economics leaf —
no Go required. Both subcommands are JSON-first (read input from stdin or `--in FILE`,
emit JSON), and the arithmetic is `internal/callavoid`'s, verbatim and deterministic:

```bash
# is memoizing this exact pure call net-positive? (k accesses, validate/mutation/capture costs)
echo '{"accesses":20,"validate_cost":0.02,"mutation_rate":0.05,"capture_cost":0.1}' \
  | fak callavoid prove-memo            # -> {"status":"PROVEN","decision":"memoize",...}

# how much amplification did a window of work get? (a Tally of the kernel's counters)
echo '{"execute":4,"memo_hit":6}' \
  | fak callavoid account               # -> {"status":"amplifying","grade":"B","amplification":2.46,...}

# gate a pipeline: exit 1 when avoidance was a NET LOSS this window
echo '{"stale_miss":5}' | fak callavoid account --gate    # exit 1 (regressing)
```

It exits `0` on a valid decision, `2` on malformed input (an unknown field or non-JSON,
caught loudly — never a silent zero-value decision), and `1` only under `--gate` on a
regressing window. Field names are the snake_case struct tags shown above.

`fak leaseref` is the operator-facing READ side of the cross-machine lease
substrate (`#825`). `internal/leaseref` persists a lease record (tree globs,
holder, acquire time, TTL) under the `refs/fak/locks/<id>` ref namespace, so the
lease rides ordinary `git fetch` / `git push` between clones — the same mechanism
`grite` uses with `refs/grite/locks`. This verb projects that ref store into an
admission decision:

```bash
# make a peer's lease (held on another machine) visible locally, then feed an arbiter
fak leaseref sync           # converge refs/fak/locks/* with origin: push local records, then fetch peers'
dos arbitrate --lane docs --tree 'docs/**' --leases "$(fak leaseref live)"

fak leaseref sync --fetch-only --remote origin   # import peers' leases without publishing (the old manual `git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'`)
fak leaseref list           # every record under refs/fak/locks/*, marked LIVE / EXPIRED
fak leaseref liveness --session $ME   # classify each LIVE lease self|peer-live|peer-dead|peer-unknown by session heartbeat (#2164)
fak leaseref session-publish --session $ME --ttl 2400   # publish/refresh refs/fak/locks/session-$ME, a side-ref heartbeat used by liveness
fak leaseref audit          # READ-ONLY staleness report (control-pane envelope); reaps nothing
fak leaseref reap           # delete the expired (reapable) records — a crashed holder is bounded
fak leaseref release --id L --holder $ME   # the release twin of acquire: hand the lease back NOW instead of waiting out the TTL (holder-checked; exit 3 on a refusal)
```

`audit` is the read-only counterpart of `reap`: it classifies every lease live-vs-expired and
emits the `fak garden` control-pane envelope (`ok`/`verdict`/`reason`, `verdict ACTION` when an
expired lease lingers) **without deleting anything**. Keeping the report (`audit`) and the
mutation (`reap`) as separate verbs is deliberate — a read-only garden tick can fold the audit
member without ever mutating the cross-machine lock state.

`live` emits the **non-expired** records as the `dos_arbitrate` `live_leases`
array `[{lane,lane_kind,tree}, …]` (each ref-stored lease is a tree-scoped
`cluster` lane), so an arbiter on machine B can *see* a lease machine A pushed.
The write side is `internal/safecommit` (opt-in `FAK_LEASEREF=1`), which publishes
its commit lock here alongside the same-host `flock`. The honest boundary, kept in
the code and these docs: this is **distribution / visibility, not atomic
acquisition** — it lets the arbiter *see* a cross-machine conflict, it does not
arbitrate a same-fetch-window race; a signature envelope over the record is
deferred follow-up. Exits `0` ok, `2` on a usage/parse error, `1` on a git/store
failure.

`liveness` (#2164) answers the question a lease's pid cannot: **is the lane's
owner actually alive?** A record's pid names the *acquiring* process, which dies
almost immediately, so a dead pid never means a free lane. Instead, a lease
acquired with `--session S` is bound to the guard-session descriptor at
`refs/fak/locks/session-<S>` — the ref a live session *heartbeats* on every PCB
transition — and `liveness` classifies each live lease `self` (yours),
`peer-live` (heartbeating — never steal it), `peer-dead` (heartbeat lapsed or
terminal `STOPPED` — the only *reclaimable* class), or `peer-unknown` (no
binding / no descriptor — publishing is best-effort, so absence is not death;
fails closed to not-reclaimable). Each row carries the `evidence` comparison
that decided it. Reclaiming still goes through the fenced `acquire` — this view
only tells an agent which refusals are worth contesting.

### Exact-model rollout gate: `fak model canary-gate`

```text
fak model canary-gate --input <path|->
```

`canary-gate` reads one strict JSON `modelops.Input` value from `--input` (`-` means
stdin), checks the candidate's exact model ID against its checked-in SLO policy, and emits
one indented JSON decision. The action and process exit status are intentionally paired:

| Action | Exit | Meaning |
|---|---:|---|
| `PROMOTE` | `0` | The candidate has enough samples and meets every declared threshold. |
| `ROLLBACK` | `3` | The candidate failed a threshold and an ordered fallback satisfies the required capability tier. |
| `HOLD` | `4` | Evidence is insufficient or no capability-safe fallback is healthy; do not silently downgrade. |

Malformed input, unknown JSON fields, trailing JSON values, unreadable files, and unexpected
arguments exit `2`. Run `fak model canary-gate --help` for the live usage text. The canonical
top-three policy and rollback witness are
[`examples/modelops-top3-canary.json`](../examples/modelops-top3-canary.json) and the
dogfood readout [`docs/notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md`](notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md),
which records a witnessed `ROLLBACK` decision against that policy.
### Region admission: `fak loop region` + the loop drive's region hold

The lease fabric above answers *"who holds what"*; **region admission**
([`docs/region-admission.md`](region-admission.md), `internal/regionadmit`)
answers the question every surface should ask before mutating a tree: *"may
THIS actor act on THIS (lane, tree) right now?"* — tree geometry plus the
`dos.toml` lane semantics (a named lane serializes; an exclusive lane runs
alone), refusing `COLLISION_RISK` with the conflicting lease named.

```bash
fak loop region --lane gateway --actor session:$ME     # decision only: exit 0 admit / 3 refuse
fak loop region --tree 'internal/gateway/**' --json    # {"admit":false,"reason":"COLLISION_RISK","rung":"tree_overlap","conflict":{...}}
```

The dispatch tick's lane-lease acquire runs this same decision, and a GOAL.md
loop that declares `lane:`/`region:` (or `fak loop drive --lane/--tree`) both
checks it and **holds** a fenced region lease for the whole drive — so loops,
dispatch workers, and manual sessions that consult the fabric are mutually
visible instead of racing one tree blind.

### Gardening stale work + the watchdog cadence

`fak garden` is the one composed, read-only fold over the repo's self-maintenance passes.
Three of its members watch **stale work** specifically:

- `orphaned_runs` — `fak loop recover --control-pane`: dispatched runs that started but were
  never finished or witnessed. It reads the loop ledger **tolerantly** — a forked seq chain (a
  concurrent double-append to the append-only, hash-chained audit log) no longer takes the
  detector down; it recovers the valid prefix, surfaces the integrity break as a finding, and
  plans the worklist from the prefix. The audit log is never rewritten. Advisory (non-gating).
- `release_staleness` — `fak release-staleness --json`: a **gating** member that turns a stale
  `@latest` (the trunk moving far past the last release tag) into a loud red.
- `stale_leases` — `fak leaseref audit`: expired cross-machine leases under `refs/fak/locks/*`,
  reported read-only (it reaps nothing). Advisory; the remedy is the explicit `fak leaseref reap`.
- `growthgate` — `fak growthgate --json`: the standing-bloat twin of the stale-work rungs —
  unbounded append-only ledger/log growth over its per-class byte budgets. Advisory (non-gating);
  its ACTION verdict is wired to the acting tick's growth-collect edge below (#5349).

```bash
fak garden            # human snapshot of every member, stale-work members included
fak garden --check    # CI/watchdog gate: non-zero when a gating member regressed
```

To run the pass **unattended**, install an OS-scheduler unit whose command is `fak garden
--check` on a cadence, named `FleetStaleWorkGarden` (Windows Scheduled Task) /
`com.fleet.stale-work-garden` (launchd) / `fleet-stale-work-garden.timer` (systemd). `fak start`
(via `serve`/`guard`) then **auto-heals** that unit: the watchdog-autoheal pass probes it and
restarts it if it has stopped — the same probe/restart/lease/debounce machinery that keeps the
fleet supervisors alive (`FAK_WATCHDOG_AUTOHEAL=off` disables it; `=warn` logs without
restarting). `FAK_GARDEN=off` is the env-side brake on the garden pass itself.

`fak garden tick` is the **acting** counterpart of the read-only fold: on an hourly cadence it
takes the documented, idempotent remediation for each surfaced condition — reap expired leases,
surface the orphan-run worklist, sweep orphan `*.lock` residue — and, for the `growthgate`
member, runs the **growth-collect** act-edge (`ActGrowthReap`, #5349) that binds growthgate's
reported-never-acted ACTION verdict to its collector. Once per non-`--dry-run` tick it censuses
the repo root **plus** the Fleet tree (`FAK_FLEET_DIR`, else `%LOCALAPPDATA%/Fleet`; skipped when
neither resolves), partitions it with `growthgate.ReapPlan` — COLD, over its class budget, **and**
a disposable class (HOT files within the 300 s heat window and non-disposable WALs/chained ledgers
are never in the set) — and **always appends the would-reap set to the reap ledger**
(`FAK_GARDEN_GROWTH_LEDGER`, else `%LOCALAPPDATA%/Fleet/growthgate-reap.jsonl`) as the soak
evidence. It is **delete-safe by default**: nothing is removed unless the apply opt-in is set
(`FAK_GARDEN_GROWTH_COLLECT=apply` or `fak garden tick --growth-apply`) — the #5079 grace-prune
precedent, delete-on-schedule stays ledger-only until the ledger has shown a correct reap set over
a soak window; flipping apply on is a separate follow-on. `--dry-run` performs no side effect at
all. The `reaped_growth_logs` counter is threaded into the tick's JSON envelope and witness metrics.

### Walking the item backlog — `fak garden walk`

Where `fak garden` folds the **~8 orchestrator members** and `fak garden tick` acts at the
member level, `fak garden walk` zooms **in** to the hundreds of *individual* garden items a
member surfaces — today the open-issue backlog (300+ live items), classified by the same issue
gardener the console uses. It is the answer to "one command that walks sets of 100s of garden
items, aware of what to do or not, to save resources/time": it loads the set once (no per-item
network), and folds it through a **resource-aware** policy:

- **skip-active** (on by default) — an item already `in-progress` is being handled, so it is
  dropped before the worklist. The cheap pre-filter that fires even when update timestamps are
  unreliable (on a bot-churned tracker `--skip-fresh` over-skips, so it is **off** by default).
- **budget** — at most `--budget N` items (default 20) earn a worklist row, picked **worst-first**
  by the gardener's score; the rest are **deferred** to the next pass. So the output — and the
  follow-up work it implies — is bounded no matter how large the set, and a recurring walk drains
  the backlog worst-first over passes.
- **propose, don't execute** — each row carries the exact `gh` command (close a dormant question,
  mark a stale issue) but the walk never runs it; auto-apply is a later, witness-gated rung. The
  same propose-don't-mutate discipline as the garden tick and `trajectory-garden`.

```bash
fak garden walk                       # worst-20 worklist over the open-issue backlog
fak garden walk --budget 50 --json    # bounded machine-readable worklist
fak garden walk --skip-fresh 7        # also skip items touched in the last week
fak garden walk --register            # arm the durable 6h walk loop (loopmgr; survives restart)
```

Every run appends a witnessed run-end to the loop ledger (walked / attention / acted / deferred /
skipped), so `fak loop health` shows the walk living. `--register` installs the durable
`garden-item-walk` loop unit (6 h cadence) the same way the stale-work tick registers itself.

### Fan a shipped spine out into its follow-on backlog — `fak issue fanout`

`fak issue fanout` ([#2510](https://github.com/anthony-chaudhary/fak/issues/2510),
spine `5b8f0bd1`, `internal/issuefanout`) is the **spine-first** producer that *fills* the
backlog the garden → dispatch boundary below *drains*: it expands one shipped working spine
into its contract-ready follow-on issues across the fan-out area taxonomy
(`qa,dogfood,product,observability,integration,docs,release`). It **plans by default** —
printing the candidate set and touching nothing — and only files when asked. The synopsis is
the one `fak issue` prints, so the reference and `--help` read the same truth:

```
fak issue fanout   --title T --leaf L --spine REF [--parent REF]
                   [--paths p1,p2] [--areas a1,a2] [--max N] [--json]
```

The planning flags, exactly as `fak issue fanout --help` describes them:

- `--title` — human name of the shipped spine.
- `--leaf` — owning leaf/lane (stamps keys, lane, default paths).
- `--spine` — spine witness: commit SHA, demo command, or doc path.
- `--parent` — epic/issue ref the fan-out hangs off (default: `--spine`).
- `--paths` — comma-separated file trees (default `internal/<leaf>/`).
- `--areas` — comma-separated area filter (`qa,dogfood,product,observability,integration,docs,release`).
- `--max` — cap candidates (`0` = full taxonomy; floor `3`).
- `--json` — emit the machine-readable fan-out plan (feed to `fak issue cohort --from-plan`).

The project-work sizing flags stamp the epic-rollup fields on every generated candidate.
`--parent-issue` and `--parent-baseline-points` together switch sizing on: supply both and each
candidate carries a parent ref, an `Estimate: N points` line, and a `Contribution: N/M points`
line; omit either and the candidates carry no rollup denominator at all.

- `--parent-issue` — parent issue number for project-work denominator binding.
- `--parent-baseline-points` — declared parent production-scope baseline points.
- `--completion-standard` — generated child maturity (default `production`).
- `--target-envelope` — production target operating envelope (stamped only under the
  `production` completion standard).
- `--witnessed-envelope` — currently witnessed operating envelope (same `production`-only rule).

Two further modes ride the same verb. `--live` files the planned candidates as GitHub issues
via `gh`, after a bounded marker-key (`fanout-<leaf>-<slug>`) dedupe against existing issues so
a rerun files zero — `--repo owner/repo` targets a non-current repo, `--dedupe-cap N` bounds the
existing-issue scan (default `300`), and `--existing-json FILE` swaps a fixture in for the live
`gh` query. `--live` also *requires* `--parent-issue` and `--parent-baseline-points`: filing
refuses outright when either is missing, so a plan that renders fine can still be unfilable —
set both before reaching for `--live`. `--adoption` measures the default instead of planning: given `--leaves` (shipped
leaves to audit) and `--markers` (the filed `fanout-<leaf>-<slug>` keys), it reports which leaves
cleared the fan-out floor versus which are gaps, exiting `1` on any gap.

### The garden → dispatch boundary

Garden and dispatch are deliberately separate authorities, and it is easy to expect the wrong
one to spawn workers. The actual contract:

- `fak garden` / `fak garden --check` — **read-only** fold over the ~8 orchestrator members.
  Mutates nothing.
- `fak garden tick` — **narrow mechanical cleanup only** (stale-work remediation at the member
  level), not issue-level work and not a worker spawn.
- `fak garden walk` — **propose-only**. It classifies the open-issue backlog and emits a
  budget-bounded, worst-first worklist with the exact `gh`/dispatch command per item, but it
  never runs any of them — no worker is spawned by `garden walk` itself.
- **Dispatch loops** (`fak dispatch auto` / `fak dispatch tick`, the `issue-resolve-dispatch/<backend>[/<goal-token>]`
  Task Scheduler arm — see [`docs/dispatch-loop.md`](dispatch-loop.md)) are the only path that
  actually spawns workers, and they own the safety machinery that has to guard that: seat/weekly-cap
  checks, lane lease / DOS arbitration, and the issue worker prompt + picker semantics. `--goal
  throughput` and `--goal high-priority` give background loops separate ledger and lease-holder
  identities while keeping overlapping path trees serialized by the same lease fabric. The matching
  super-loop intents are `drain-throughput`, `drain-high-priority`, and the aggregate `drain-issues`.

Today there is no bridge command wired between `garden walk`'s proposed worklist and dispatch's
spawn path — an operator (or another script) has to carry the `--json` worklist over by hand.
That gap, and the shape of the bridge that should close it (`fak garden dispatch`, or a
dispatch-loop source mode consuming `garden walk --json` — running through the same admission
gates as ordinary issue dispatch rather than bypassing them), is tracked in
[#1791](https://github.com/anthony-chaudhary/fak/issues/1791), which also names the two dispatch
issues it builds on: [#1404](https://github.com/anthony-chaudhary/fak/issues/1404) (moving issue
worker prompt/picker semantics into the Go dispatch tick) and
[#1462](https://github.com/anthony-chaudhary/fak/issues/1462) (proving the handoff-to-issue-to-close
path end to end). Until #1791 lands, "make garden's findings turn into worker runs" means going
through dispatch directly, not `garden walk --apply` (no such flag exists, by design).

`fak slack health` is the watchdog **dual of the Slack feeders**. The cadence feeders
(`scoreboard-feed.yml`, `bench-feed.yml`, …) POST a card on a schedule and fail OPEN — a
missing token or channel renders to the step summary and exits 0 — so a misconfigured or
broken feeder is SILENT (green CI, nothing in the channel). `fak slack health` CONFIRMS the
other half: per surface it folds resolution + `auth.test` + a real `conversations.history`
read into one closed verdict — `OK | INCOMPLETE | AUTH_FAIL | STALE` — and exits non-zero on
any non-OK so a scheduled job can gate on it. Staleness is judged against the feeder's own
cadence (a daily feed is STALE after ~36h, a weekly feed after ~8d); a surface with no
scheduled feeder is never graded STALE. `--json` emits `{surface, ready, auth_ok,
last_post_age_s, budget_s, verdict}` per surface. The unattended arm is
`.github/workflows/slack-watchdog.yml` (daily 10:19 UTC, fails open without the token): on a
non-OK verdict it files ONE deduped GitHub issue so the dispatch loop picks it up.

`fak slack beat` is the **liveness pulse** — the third leg. The feeders post on change; the
health verb folds a verdict but only exits a code (or files a once-a-day issue). The gap: a
QUIET channel looks identical to a DEAD feeder. `fak slack beat` runs the same health fold and
posts ONE compact line to a status channel UNCONDITIONALLY on its cadence — `✅ slack surfaces
alive — 7/12 OK · freshest feeder post 1h ago` on a healthy day, `⚠️/🔴 … _down:_ <surface
MODE …>` when one is broken. A green beat means alive; a missing beat means the scheduler
itself died. It posts through the same transport as `fak slack send` and resolves the channel
(default `$FAK_DISPATCH_CHANNEL`, then `$FAK_SCOREBOARD_CHANNEL`) the same env-then-file way;
`--dry-run` renders fork-safe, exit-coded so a scheduled tick flags a misconfiguration.

`fak slack outbox` is the **durable outbox** — the recovery leg ([#2262](https://github.com/anthony-chaudhary/fak/issues/2262),
epic [#2259](https://github.com/anthony-chaudhary/fak/issues/2259)). check/health/beat *detect*
a dropped post; the outbox is what stops the drop: producers (`fak slack send --durable`, the
scoreboard feeder) ENQUEUE a row into an append-only local JSONL spool
(`$FAK_SLACK_OUTBOX_DIR`, default `.dispatch-runs/slack-outbox`) and return once it is on
disk — never a network call. One lock-serialized drainer then posts per-channel FIFO through
the shared transport: it honors `Retry-After`, paces ≤1 msg/s per channel, coalesces queued
`chat.update` rows for the same card to the newest state, and closes the crash-between-post-
and-record window with a nonce probe (the nonce rides in message `metadata`; a half-succeeded
post is recovered, never re-sent — at-least-once with idempotent posts is the honest
contract). A row that exhausts its retry budget goes `dead` — kept, listed, operator-retryable
— never silently dropped; every outgoing body passes the PUBLIC_LEAK/SECRET_SHAPE needle scan
first, and a hit refuses the row terminally with the finding as its structured reason.
`fak slack health` gains an `outbox` rung: dead rows or a pending backlog older than 2h grade
the surface non-OK, so a wedged drain pages instead of rotting. To keep the noisy dgx-bridge
channels clean, the drainer also **reaps** — an ephemeral channel on the `FAK_SLACK_EPHEMERAL_CHANNELS`
allowlist (or a row that opted in via its own delete-after) has its posted messages
`chat.delete`d once they go idle past a TTL (default 30m, measured from the message's last
activity so a live card is not culled mid-run). Reaping rides the tail of every drain — no
separate scheduler — and `fak slack outbox reap` runs (or `--dry-run` previews) one pass on
demand; an already-gone message is recorded reaped idempotently, a transient delete retries
next drain, and a channel nobody opted in is never touched.

`fak slack alert` is the **Prometheus-alerts→Slack receiver** — the inbound producer that
turns a firing alert into a durable outbox row. Prometheus (`tools/grafana/prometheus.yml`
`alerting:` block) hands firing rules to Alertmanager, which groups/routes them and POSTs a v4
webhook to this verb; it folds the payload through `internal/promalert` (severity emoji,
firing/resolved status, per-alert summary/description/labels/runbook link) and ENQUEUES the
card on the alerts channel (`FAK_ALERTS_CHANNEL`, else the public `#grafana` default). Reusing
the outbox — not Alertmanager's built-in `slack_config` — is deliberate: the built-in Slack
receiver needs an *incoming-webhook URL*, but every fak surface authenticates with the shared
**bot token**, and the outbox makes each alert crash-durable and witnessable
(`fak slack outbox status | calls`). Run it one-shot (`< payload.json`, or `--dry-run` to
render only) or as the long-running HTTP receiver Alertmanager targets
(`--serve --addr 127.0.0.1:9096`); see [`tools/grafana/README.md`](../tools/grafana/README.md)
for the compose wiring.

**Triaging a `slack-watchdog` issue.** When the watchdog files its once-a-day deduped issue
(labelled `slack-watchdog`, `triage-only` — e.g. [#1855](https://github.com/anthony-chaudhary/fak/issues/1855)),
the fix is *operational* (host token / channel env / feeder cron), not a repo code change — that
is what `triage-only` means. One issue covers ALL currently non-OK surfaces, so it closes only when
`fak slack health` returns all-OK (exit 0), not per surface. Remediate by the verdict the report
names:

| Verdict | What it means | Remediation |
|---|---|---|
| `AUTH_FAIL` | the surface's token resolved but `auth.test` rejected it (bot rotated/revoked) | replace the surface's bot token on the host, then re-run `fak slack health` |
| `INCOMPLETE` | token or channel unresolved — the feeder would post nowhere (config drift) | wire the surface's `FAK_*_CHANNEL` (or a `ChannelDefault`) **and** a valid token so it resolves; an *optional* surface with no channel is `DEFERRED`, not `INCOMPLETE`, and needs no action |
| `STALE` | ready + auth OK but no post witnessed inside the cadence budget | check the feeder workflow's (`<surface>-feed.yml`) last scheduled run — a single missed/failed run self-heals on the next successful post; a persistent stale usually means the feeder is failing or the bot was removed from the channel (so `conversations.history` reads nothing) |

Before closing, re-run `fak slack health` (or `--json`) and confirm it exits 0 — a self-authored
"resolved" without that witness is not resolution. A surface whose condition is transient (a `STALE`
feeder that has since posted) verifies clean on the next run; a config or token fix must be applied on
the host first. Nothing here is fixed by a commit to this repo.

```bash
fak slack check --auth   # token auth + one bounded history read per configured channel; inaccessible => ready=false
fak slack check --auth --json  # adds channel_access {ok, reason, remediation, error}; ready now includes live access
fak slack health         # + did a post actually land inside each feeder's cadence?
fak slack health --json  # machine-readable verdict for the watchdog / a dashboard
fak slack beat           # post a one-line liveness pulse even when nothing else posted
fak slack beat --dry-run # render the pulse, resolve channel/token, post nothing (fork-safe)
fak slack send --durable --channel C… --text "…"   # enqueue-then-drain: survives a dead wire
fak slack alert --file payload.json    # fold an Alertmanager webhook → durable outbox row → Slack
fak slack alert --dry-run < payload.json           # render the alert card only; touch nothing
fak slack alert --serve --addr 127.0.0.1:9096      # HTTP receiver Alertmanager POSTs webhooks to
fak slack outbox status [--json]   # pending/posted/dead/refused counts + ages (watchdog food)
fak slack outbox drain [--dry-run] # run one serialized drain pass now (dry-run: plan only)
fak slack outbox retry --all|--nonce N [--dry-run]  # re-arm dead rows for the next drain
fak slack outbox dead [--json]     # list dead rows with their structured reasons
fak slack outbox compact [--dry-run] [--json] [--retain D] [--retain-dead D]  # fold old settled/dead rows + heartbeats out of the spool
fak slack outbox reap [--dry-run] [--json] [--ttl D] [--channels C1,C2]  # delete ephemeral (bridge-channel) messages idle past their TTL (FAK_SLACK_EPHEMERAL_CHANNELS)
fak slack outbox limits [--json]   # effective retention windows + live occupancy (terminal/droppable rows, pass-due)
fak slack outbox calls [--json]    # per-source Slack API-call spend vs saved (rate-limit gauge; before/after noise baseline)
```

`run`, `preflight`, and `agent` take `--policy FILE` to load the capability floor
from a declarative JSON **manifest** instead of the compiled-in default — so WHICH
tools the agent may call is a reviewable file, not a Go edit. See `POLICY.md`.

`fak route` is the same idea applied to model selection: which MODEL — or which
ENSEMBLE of models + a reduction — serves a given **aspect** of a request. Where a
SOTA router picks one model for the whole request, fak routes at every level — a
single tool call, a sub-query, a reasoning step — each to a different model, with
first-class ensembles (`vote` / `best_of` / `all_reduce` / `concat`), all from one
reviewable JSON manifest (`--dump` → edit → `--check` → `--manifest`). The routing
**decision** + the ensemble **reduce** are shipped and pure (witnessed by
`go test`); executing a decision on live engines is the wiring tracked in the
model-routing epic. `--simulate` folds stand-in member outputs through the chosen
plan's reduction so the ensemble half runs end to end with no model in the loop. See
[`docs/model-routing.md`](model-routing.md).

`fak routebench` is the offline measuring instrument: it runs a corpus of recorded
cases through TWO manifests — a per-aspect + ensemble policy vs a single-model
baseline (the SOTA shape) — and prints the delta on **cost / latency / quality**.
Each case carries the stand-in OUTPUT every candidate model produces (like `fak
route --simulate`), so it reuses the pure `Route` + `Combine` halves and is
deterministic end to end — no key, no GPU. Default (no args): the built-in 8-case
demo corpus + `DefaultManifest` vs a one-frontier-model baseline; `--corpus` /
`--routed` / `--single` load your own, `--dump-corpus` emits the starter corpus to
edit. Every figure is a ROUGH lens, never a bill or a measured SLA. See
[`docs/model-routing.md`](model-routing.md#the-offline-routing-benchmark-fak-routebench).

`fak vcache calibration-record --provider NAME [--model NAME] --telemetry usage.jsonl` folds provider cache-feedback JSONL into the dated calibration ledger. `fak guard` and `fak serve` also append automatically when their gateway observes provider cache feedback. `fak vcache calibrate --samples PROBES --ledger FILE` appends fitted, per-provider/model TTL, minimum-prefix, and cached-read constants with an independent `*_measured` bit for every field. Guard and serve load only fresh matching measured constants: a measured minimum prefix suppresses cache-breakpoint authoring below the provider floor, and a measured read multiplier overrides cache-read accounting; stale, mismatched, and observation-only rows preserve static defaults. `fak vcache calibration-status [--providers anthropic,openai] [--max-age 168h] [--json]` returns `fresh`, `stale`, or `missing` per provider; stale and missing rows name the required live-session refresh. Prediction-error rates remain visible during the session at `/metrics` (`fak_vcache_warmth_false_warm_rate` and `fak_vcache_warmth_false_cold_rate`).

`fak vcache status|prove|prove-telemetry` is the proof surface for the virtual
provider-cache work. `status` reports the honest current state: the M5 Governor is up
as a local, off-path policy engine, while provider calibration/warming/recall remain
open in #716-#718 and the Codex/OpenAI cached-token probe is proven by #727. Add
`--sessions` to attach the compact current-workspace session summary (`fak
session-audit summary --here`): recent Fable/Opus output mix, total context,
cache-read share, and top long-context sessions. With `--sessions`, the JSON also
includes `recent_session_actions`, the advisory action ledger from that same
summary; `--session-action-gate high|medium|none` selects the embedded gate
threshold without changing the `vcache status` exit code. `prove` runs
the deterministic star-anchor savings arithmetic without a provider or model; the
default Codex-like workload (4096-token anchor, 7 sibling requests, 10-token suffixes,
0.1 read / 1.25 write multipliers) proves 21,094.4 token-equivalents saved, 73.4%,
and exits 1 for refuted workloads such as an anchor below the provider minimum.
`prove-telemetry --file experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl`
replays observed provider counters and proves 13,141.5 token-equivalents saved
(4.73%) on the four-turn Claude Code prefix probe, with the first positive request at
4; the same verifier refutes the first three turns because the cache reads have not
repaid the 1h cache-write cost yet. The same JSONL reader accepts raw OpenAI
Responses usage (`usage.input_tokens_details.cached_tokens`), Chat Completions
usage (`usage.prompt_tokens_details.cached_tokens`), Codex CLI `token_count` rows,
and `codex exec --json` `turn.completed` usage rows. The replayable Codex artifact
at `experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl` proves
9,147,340.8 token-equivalents saved (85.98%) over 68 token-count events. `status`
reports the verifier as ready, includes a cached-token sample proof and zero-cache
refutation, and keeps the raw OpenAI API probe as an optional no-credential skip path.
These are cost proofs only: correctness never depends on a provider cache hit.

`fak vcache actions --json` renders the provider-cache action plan for the current
observed-window snapshot, mapping each prefix family to `ride_natural`,
`heartbeat_pin`, `lazy_rebuild`, `evict_manifest`, `no_cache`, or `explicit_cache`.
Spendful rows are `gated` until transport witnesses are supplied. `fak vcache
apply-actions --manifest FILE` applies only local, no-provider-call effects to a
fak-owned manifest: `evict_manifest` removes a warm row, `no_cache` marks a family
uncached, and heartbeat/explicit-cache rows remain pending unless a later provider
executor supplies an independent execution witness.

`fak session-audit summary --here --since-days 7 --max 40 --json` emits the compact
machine-readable shape behind that `vcache status --sessions` block. `fak
session-audit actions --here --since-days 7 --max 40 --json` lowers its Fable/Opus
and long-context recommendations into a stable advisory action ledger with witness
commands; add `--fail-on high` to make that ledger a guard gate that exits 1 when
recent cost/context pressure should block more high-cost turns. `fak manage
--session-pressure-gate high --model claude-fable-5` treats the explicit Fable
route as satisfying those current high-pressure actions while explicit Opus or
unknown routes still refuse; append `,justify=...` to the same spec with an
explicit Opus model to allow a justified high-cost launch without disabling the
gate. The gate is ONE flag carrying a spec —
`THRESHOLD[,days=N][,max=N][,report=PATH][,justify=TEXT]`, defaults `days=7`
and `max=40` — so a bare `--session-pressure-gate high` still reads exactly as
before; `justify=` consumes the rest of the spec (prose has commas) and so
comes last. `GET /v1/fak/session-audit/actions` serves the same read-only action
ledger for gateway/control clients. Both are scoped by
the current workspace's Claude transcript namespace by default, label clipped
`--max` windows, and keep exact token counts separate from assumed-cost estimates.

`fak vcache score` also reports per-plane evidence and a separate
`default_usefulness` score. Provider counters populate `planes.provider_observed`
only; they do not count as fak-owned activation. Pass witnessed local activity
with `--kernel-kv-events`, `--context-events`, `--provider-vcache-decisions`, or
`--external-engine-events`; pass pure-fak KV value with
`--kernel-kv-prompt-tokens` and `--kernel-kv-reused-tokens`; pass O(1)
context/query value with `--context-shed-tokens` and
`--context-resident-tokens`; pass SGLang/vLLM/llama prefix-cache evidence with
`--external-engine-hit-rate`.

`scripts/ci.ps1` (or `make ci`) runs build + vet + test + the CLAIMS lint as one gate.

> It is *designed for extension*: other ideas bake in as a new package + one
> registration, never a core edit (see `ARCHITECTURE.md`).

## What's here

| File | What it is |
|---|---|
| `internal/abi/types.go` | The **frozen, additive-only** ABI spine: the syscall envelope, the discriminated-union Verdict, the addressable `Ref`, the async + provisional-lifecycle seams, and the core interfaces. No subsystem is named in it. |
| `internal/abi/registry.go` | The **extension mechanism**: `Register*` from a driver's `init()`, reserved number ranges for link-time disjointness, the driver interfaces (engine / region / page-out / witness / steward). |
| `ARCHITECTURE.md` | The extension model — how a new idea bakes in, the 4 seams that had to be frozen now, the bake-in walkthrough (speculative exec, async, zero-copy, the syscall-tuned model, an unforeseen idea). |
| `DIRECTION.md` | The **strongly-typed-core direction** — the request/enforcement path is Go (illegal states unrepresentable, the same thesis applied to the source); non-Go is permitted only at rare named *seams/interconnects* that sit off the path behind a typed, serialized boundary (the ML-ecosystem oracle, build/CI glue, out-of-band analysis). With the reviewer's three greps that prove it holds. |
| `DISAGGREGATED-AGENT-MEMORY.md` | The **strategy note** — fak as the MMU + reference monitor for shared agent memory: six memory semantics (S1–S6) mapped to shipped primitives, the cross-agent / cross-tenant / cross-node axes, and §2.5's four-layer distinction (routing vs addressing vs fusion vs semantics) that keeps a routing win from being mistold as a fak win. |
| `MEMORY-LAYERS-EXPLAINER.md` | The **four-layer explainer** (teaching artifact for the above §2.5): *routing* (where a cell lives), *addressing* (its name), *fusion* (zero-copy co-residence), *semantics* (coherent mutation / isolation / provenance / capability — fak's actual change), with an ASCII stack diagram, the Docker(defines the object) vs Kubernetes(routes the object) analogy, and the one-line "is this a routing claim or a fak claim" test. |
| `docs/SKILL-CONTEXT-MEMORY.md` | The **skill-context memory** note - treats `.claude/skills/` as the procedural twin of `.claude/memory/`: named, versioned, load-on-demand context capsules that can emit witnessed, cacheable `SkillContextRecord`s instead of replaying long context. |
| `POLICY.md` | The **deployable capability floor** — the dump→edit→check→load workflow, the `fak-policy/v1` manifest schema, the closed refusal vocabulary, and the honest scope of what the floor does and does not bound. The adopter's front door. |
| `PARTITION.md` | The `dos-plan-price` fleet partition: wave-0 gate, the 4 wave-1 leaves, the 3 wave-2 workers, the serial tail, and the **growth slots** for post-v0.1 ideas. |
| `WORKER-PACKET.md` | The per-worker dispatch packet the fleet consumes (goal · leased tree · `dos hook stop` witness · `dos verify` done-proof). |
| `LIVE-RESULTS.md` | The **live** turn-count A/B: a real model (Gemini OpenAI-compat + local Qwen2.5) drives the `fak agent` loop twice over one task; each run carries a transcript hash. The honest read of what fusion does and does not buy live. |
| `TICKETS.md` | Issues the live `fak agent` lane surfaced (FIXED / OPEN / NOTE). |
| `RECALL-RESULTS.md` | The **session-recall** lane: a quarantine that survives the session boundary (a finished session as a *core dump*; benign pages demand-paged byte-identical, sealed pages refused across the boundary + re-screened on page-in). Witness-grounded; 5/5 adversarially confirmed. |
| `CDB-RESULTS.md` | The **context-debugger** lane (`fak debug`): attach to a finished session as a core dump and answer a follow-up by *demand-paging only the working set* (Denning) — never replaying the address space. Ingests a REAL Claude Code transcript; measured on a 2.8 MB session, an 18 KB page table over a 1.2 MB swap device, follow-ups paging in ~1.8–6.2% of the resident image. |
| `MEMORY-DREAM-CLEANUP-RESULTS.md` | The **memory-dream cleanup** lane (`fak dream`): an offline sleep pass over a core image that re-screens resident pages, pre-seals refuted witnesses, repairs sealed descriptors from metadata only, surfaces duplicate aliases, and prunes unreferenced CAS bytes. |
| `IN-KERNEL-MODEL-RESULTS.md` | The **model fused into the kernel**: a pure-Go SmolLM2-135M forward pass (134.5M params / 272 tensors), every rung proven against HuggingFace (0/134.5M decode bit-mismatches, layer cos=1.000000, KV-decode + KV-quarantine-evict token-for-token identical). The kernel owns the KV cache; the design is in `IN-KERNEL-MODEL-DESIGN.md`. |
| `MODEL-BASELINE-RESULTS.md` | The fused forward pass **measured** against the next-best CPU baselines (HF transformers, llama.cpp): naive tax → parity lane (decodes *faster* than same-precision HF f32) → an int8/Q8 SIMD lane at near-parity with llama.cpp Q8_0 (~7.7 ms/tok, the in-flight Act 3). Every number recomputed from raw JSON, survived a 4-skeptic adversarial pass. |
| `MODEL-ARCH-SUPPORT-AUDIT-2026-06-18.md` | Current top-10 architecture support audit: what is witnessed today, what is only loader/shape-compatible, and which GitHub issues cover the remaining families. |
| `FAK-NATIVE-QWEN35-RESULTS.md` | The Qwen3.5/Qwen3.6 Gated-DeltaNet lane: 0.8B coherent f32 chat plus the 2026-06-19 pure-fak Qwen3.6-27B GGUF->Q8 smoke with first-token llama.cpp parity on the M3 Pro. |
| `KV-QUARANTINE-BRIDGE-RESULTS.md` | The deepest "model is secondary" rung: the byte-gate drives the **KV-gate** — a `Quarantine` verdict evicts the poison's K/V span, leaving the attention cache bit-identical (max\|Δ\|=0) to never-having-seen it. |
| `TURN-TAX-RESULTS.md` | The **turn-tax** benchmark (`fak turntax`): prices the extra error-code MODEL turn the 1-shot kernel deletes (forced vs elision, with a consistency guard and a happy-path=0 control), keeping the safety floor on its own axis. |
| `FLEET-SWEEP-RESULTS.md` | The 2-D **turns x agents** sweep: shared-cache fleet vs isolated agents, exact-zero no-share controls, and the scoped-invalidation eraser that fixes the write-rate crossover. |
| `FANOUT-BENCH-RESULTS.md` | The one-master-goal **fan-out** benchmark: N=1..1024 sub-agents, real cross-agent tool-result dedup, transparent prefix-cache economics, and the fold-bound latency knee. |
| `VISUALS-benchmarking-status-2026-06-18.md` | The refreshed benchmark visual/status dashboard tying the current plots, headline numbers, and caveats together. |
| `EXPLAINER-trust-floor-two-lenses-2026-06-17.md` | **fak explained twice** — once for *security researchers*, once for *agent-optimization*, with a Rosetta table mapping each primitive to both vocabularies. |

## The contract in one breath

`Kernel.Submit` adjudicates (folds the LSM-style `Adjudicator` chain by `FoldRank`)
and enqueues; `Reap` returns the typed completion; `Syscall` is the sync convenience
over the two. Payloads are addressable `Ref`s (zero-copy is a backend swap).
`Verdict` is a closed, trainable, discriminated union with an open registered range
(the syscall-tuned model's clean target). Speculation/transaction effects are
provisional until `Promote`/`Rollback`. Everything else — engines, vDSO tiers,
rungs, the MMU codec, stewards, KPIs, witnesses, and ideas nobody's had yet —
attaches through a registry.

## What shipped (witness-closed)

| Subsystem | Tree | What it does | Witness |
|---|---|---|---|
| in-process adjudicator | `internal/adjudicator` | DOS reference monitor: provable-deny / unprovable-defer, structured 12-reason refusal, bounded-disclosure SELF_MODIFY witness, redact-transform | `go test`, `BenchmarkDecide` |
| deployable policy | `internal/policy` | the capability floor as a declarative, version-tagged JSON manifest (`--policy FILE`); closed-vocab deny validation; fail-loud load; `--dump`↔`--check` round-trip; `fak policy` verb | `go test` (9, incl. `TestRoundTrip`) |
| compliance attestation | `cmd/fak/attest.go` | `fak attest --policy FILE` proves the floor from preflight: runs the real adjudication fold over a probe set (derived from the manifest, or `--probes FILE`) and emits a re-checkable attestation — every deny enforced with its cited reason, every allow admitted, default-deny holds; exit 0 PROVEN / 1 drift | `go test ./cmd/fak` (attest_test.go) |
| tool vDSO | `internal/vdso` | 3-tier local fast path (pure / content-cache / static), world-versioned, LRU, canonical keys | `go test` |
| engine | `internal/engine` | OpenAI-compatible client, base_url-swappable, cassette replay, usage extraction, mock | `go test` |
| pre-flight + grammar | `internal/preflight`,`internal/grammar` | rung ladder + JSON-schema; positional→named auto-repair; fail-open; grammar dedup; hard-negative harvest | `go test` |
| context-MMU | `internal/ctxmmu` | write-time quarantine (secret/injection/poison/repeat), page-out to <2KB pointer, witness-gated page-in | `go test` + `poison.json` |
| security substrate | `internal/ifc`,`provenance`,`plancfi`,`witness`,`canon`,`normgate`,`agentdojo`,`harvest` | the kernel stops believing the model: source-stamped taint + sink-gated flows (ifc), kernel-authored trust/provenance, plan-CFI (`RequireApproval`), an effect-verifying `dos_verify` witness gate, a normalize-and-rescan admission driver (normgate), and a dynamic ASR-gated attack battery (agentdojo) | `go test` (per pkg); `cmd/ctxbench -chain` |
| KV-quarantine bridge | `internal/kvmmu` | the same ctxmmu `Quarantine` verdict mechanically evicts the poison's K/V span, leaving the kernel-owned attention cache bit-identical (max\|Δ\|=0) to never-having-seen it | `go test` (5); `KV-QUARANTINE-BRIDGE-RESULTS.md` |
| session core-dump + debugger + dream cleanup | `internal/recall`,`internal/cdb` | persist a finished session as a page-table-over-CAS core image; `fak debug` attaches to it (incl. a REAL transcript) and demand-pages only the working set a question touches; agent/requester tombstones suppress unwanted memories from future context without deleting audit bytes; `fak dream` auto-cleans the sleeping image by re-screening, pre-sealing refuted witnesses, and pruning dead CAS bytes | `go test` + `recall-report.json`,`cdb-report.json`,`dream-report.json` |
 | in-kernel model | `internal/model` | a pure-Go forward pass the kernel owns (KV cache as a Go structure), with proven bit-for-bit correctness for SmolLM2-135M (Llama family) and first-token parity for Qwen3.6-27B (Gated-DeltaNet); an int8/Q8 SIMD lane at near-parity with llama.cpp Q8_0 is the active **in-flight** extension | `go test` (oracle argmax-exact); `MODEL-BASELINE-RESULTS.md`; `FAK-NATIVE-QWEN35-RESULTS.md` |
| gateway | `internal/gateway` | `fak serve`: OpenAI-compatible HTTP (`/v1/chat/completions` adjudication proxy, `/v1/fak/*`) + MCP over stdio/HTTP, so any-language agents route tool calls through the syscall boundary; mints a tainted agent-scoped `Ref` from raw bytes (IFC/secret/self-modify rungs stay armed) | `go test`; v0.2.1 adversarial-review hardening |
| model routing + account binding | `internal/modelroute` | `fak route`: per-aspect + ensemble model routing as a pure, deterministic policy — `Route(Subject)→Decision` (a tool call / sub-query / step routes to its own model or ensemble) + `Combine(reduction,votes)→Result` (`first`/`vote`/`best_of`/`all_reduce`/`concat`); version-tagged JSON manifest, `--dump`↔`--check`; `--accounts` / `--accounts-dump` / `--accounts-check` bind abstract route model ids to provider accounts, upstream model names, and residency-honest engine routes. The served gateway path dispatches single picks and ensembles with `--route-manifest`; standalone `fak agent` route-manifest wiring remains the labeled follow-on | `go test` (`internal/modelroute`, `cmd/fak` route tests, gateway route-manifest tests); `docs/model-routing.md`; `docs/model-accounts.md` |
| dispatch fusion | `internal/kernel` | one in-process chain; no `os/exec` on the hot path | `go test` (ABSENCE proof) |
| KPI + A/B bench | `internal/metrics`,`internal/bench` | vDSO ablation; the primary gate; provenance + identical-workload guard | `report.json`, `baseline.json` |
| turn-tax bench | `internal/turnbench` | `fak turntax`: prices the extra error-code MODEL turn (malformed/duplicate/poison) a SOTA loop fires vs the 1-shot kernel, per lever, safety floor on its own axis | `go test` (incl. happy-path=0 control); `TURN-TAX-RESULTS.md` |
| stewards + RSI gate | `internal/steward`,`internal/shipgate` | single-invariant stewards + meta-prune; keep-or-revert on a non-forgeable keep-bit, worktree isolation, escalation breaker | `go test` |
| version-everything | `internal/modver` | `fak version modules`: a per-module version report over the module tree — content-addressed rev + date (+ optional joined `-scores`), with `-only <prefix>`, `-sort name\|rev\|date`, `-top N`, and `-json` views; `-stamp` appends changed-module rows to the `fak-module-versions/1` ledger (`docs/nightrun/module-versions.jsonl`, seeded with 410 modules) so a module's version is git-witnessed, not asserted | `go test` (`internal/modver`, `cmd/fak/version_modules_test.go`); `docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md` |

## What this is NOT (labeled, not hidden — see `CLAIMS.md`)

- **The in-kernel model is a reference forward pass; GPU throughput is real, but it is not
   yet a full production serving engine.** The kernel ships proven bit-for-bit correctness for
   SmolLM2-135M (Llama family) and first-token parity for Qwen3.6-27B (Gated-DeltaNet). The CUDA backend takes
  it onto the GPU and reaches **decode parity with llama.cpp Q8_0 (≈120 tok/s on an RTX 4070;
  `GPU.md` §3b)** — so GPU throughput is **not** out of scope. What *is* still future work is
  production *serving* beyond the native in-kernel lifecycle scheduler (paged attention,
  multi-tenant SLA scheduling); the
  live `fak agent` / `fak serve` lanes drive an external OpenAI-compatible engine for that
  today.
- **NOT** zero-copy KV co-residence with an external engine: that remains the
  addressable-`Ref` seam wired to a **copy** backend (a backend swap later, behind
  capability `zerocopy`). The in-kernel model owns *its own* KV cache; sharing one KV
  arena with a separate serving process is the unbuilt stub.
- **NOT** GPU-dependent for the shipped pure-Go binary: token-per-watt / metrics-service
  KV-residency are read-only **SIMULATED** telemetry (no watt source on the box).
- **NOT** a fine-tuned *syscall/adjudication* model: the typed `LabelRow`/`VerdictKind`
  training targets exist (and `internal/harvest` now folds the live verdict stream into a
  corpus of them), but the model that would emit Verdicts from them is not trained — the
  fused model is a stock reference, not a tuned adjudicator.
- The vDSO real-world hit-rate is low (~0.7% addressable on real tau2-airline) — the
  demo trace is deliberately cache-favorable, which is why the headline is the
  call-mix-independent adjudication gate, not the vDSO win.

Every claim in `CLAIMS.md` carries exactly one of `[SHIPPED]` / `[SIMULATED]` /
`[STUB]` (lint-enforced). See `../BUILD-72h-fused-agent-kernel.md` for the original
scope and `../PLAN-fak-mvp-100-units-2026-06-16.md` for the 100-unit plan.

## Build history (how wave 0 was landed)

The following work was completed as the initial wave-0 build:

1. Land + freeze wave 0 (this artifact): `go build ./... && go vet ./...`, commit the
    golden conformance test, author the operator fixtures, run the vDSO purity gate.
2. `dos-plan-price PARTITION.md` → confirm collision-free; `dos-arbitrate` the leases.
3. `dos-goal-fleet` the wave-1 packets; gate wave 2 on `dos-witness-claim`; fold; tag.

See [PARTITION.md](https://github.com/anthony-chaudhary/fak/blob/main/PARTITION.md) for the current partition manifest and wave plan.

License: Apache-2.0 (matches the Microsoft Agent Governance Toolkit dep).

## `fak launch`

`fak launch doctor [--json] [--repair]` diagnoses shim/provider posture; `--repair` refreshes the managed upgrade-stable fak target and owned shims.


`fak doctor launch-posture [--entrypoint agent|guard|serve] [--harness NAME] [--provider NAME] [--base-url URL] [--workspace DIR] [--json]` derives the default-on launch posture for the selected repository and wire. It reports eight mechanisms — bounded repository tools, Caveman, Ponytail, compact history, stale-read elision, cold-tool deferral, vCache anchoring, and dated provider-cache calibration — as `active`, `inert`, `disabled`, or `unsupported`, and names an action for every configured-but-inactive state. A configured vCache path is `inert` when its provider calibration is missing, stale, or fresh-but-observational; `active` now means a fresh measured constant is wired to steering. Defaults come from the same gateway/profile constants used by launch code. This is a preflight, not savings telemetry: `active` means the launch reaches the mechanism's runtime seam; use gateway/session metrics to prove realized savings.

`fak launch install [--provider claude|codex|all] [--default NAME] [--no-path]`
installs managed shims and, unless `--no-path` is set, an idempotent fak-owned PATH block
for supported PowerShell/POSIX startup files. Uninstall removes only that block.

`fak launch add NAME --command PATH [--arg ARG ...] [--default] [--shim]` persists a
custom provider as an argv template. `fak launch remove NAME` removes the binding and
owned shim; `fak launch list [--json]` lists bindings without exposing local command paths
or argument values. Names are lowercase command aliases, cannot be path-like, and cannot
shadow reserved fak verbs. See [Zero-adoption provider launch](zero-adoption-launch.md).
## `fak workpattern` — named coding-workload catalog and miners

`fak workpattern` is the offline front door for the versioned coding-workload vocabulary. It separates goal-shaped **patterns** from reusable ordered **subpatterns**, and reports only evidence supported by explicit detectors.

```bash
fak workpattern list --json
fak workpattern source --source . --json
fak workpattern trajectory --trajectory turns.jsonl --json
fak workpattern trajectory --chat scrubbed-chat.json --json
fak workpattern report --source . --trajectory turns.jsonl --json
```

The JSON schema is `fak.workpattern-report/1`. It records catalog and detector versions, input digests, findings, and abstentions. Source findings include paths and source ranges. Trajectory findings include trace/turn ranges and detector reasons. Default trajectory output contains tool names, ranges, hashes, counts, and reasons—not prompt, message, or tool-argument bodies. `--include-excerpts` is an explicit opt-in and excerpts remain redacted/truncated by the trajectory miner.

Supported chat input is the deliberately content-free `fak.scrubbed-chat/1` format documented by `internal/trajectory.ImportScrubbedChat`; unsupported or malformed formats fail closed. Findings are evidence-backed candidates, not semantic intent judgments, and no match means abstention rather than proof that a pattern is absent.

Research basis and vocabulary proposal: [`research/coding-workload-vocabulary.md`](research/coding-workload-vocabulary.md). Machine companion: [`research/coding-workload-vocabulary.json`](research/coding-workload-vocabulary.json).


## `fak workpattern` — named coding-workload catalog and miners

`fak workpattern` is the offline front door for the versioned coding-workload vocabulary. It separates goal-shaped **patterns** from reusable ordered **subpatterns**, and reports only evidence supported by explicit detectors.

```bash
fak workpattern list --json
fak workpattern source --source . --json
fak workpattern trajectory --trajectory turns.jsonl --json
fak workpattern trajectory --chat scrubbed-chat.json --json
fak workpattern report --source . --trajectory turns.jsonl --json
```

The JSON schema is `fak.workpattern-report/1`; it includes catalog/detector versions, input digests, findings, and abstentions. Source findings carry source ranges. Trajectory findings carry trace/turn ranges and reasons. Default output excludes prompt, message, and tool-argument bodies. `--include-excerpts` is explicit opt-in and remains redacted/truncated by the miner. The content-free chat adapter accepts only `fak.scrubbed-chat/1` and fails closed on malformed/unsupported formats.

Research basis: [`research/coding-workload-vocabulary.md`](research/coding-workload-vocabulary.md); machine companion: [`research/coding-workload-vocabulary.json`](research/coding-workload-vocabulary.json). Findings are evidence-backed candidates, not autonomous intent judgments.
# `fak stale-work`

Ranks tracked documentation into bounded, issue-ready, provenance-bearing review packets without
mutating candidates. See [`docs/stale-work.md`](stale-work.md). `--selfcheck` proves dependency
drift outranks age-only history.

`fak stale-work loop` consumes that packet and renders one contract-valid dedicated issue unit per
candidate. It is dry-run by default, deduplicates against an `--issues` snapshot, serializes
overlapping paths into waves, and refuses dispatch until an existing issue passes the shared issue
contract. `--state`/`--state-out` read and explicitly persist evidence-digest adjudications;
`--witnesses` reconciles only independent issue/git/test evidence. GitHub creation and worker
launch are separately armed by `--live-issues` and `--live-launch`.

## `fak skill compile` — explicit skill programs

```text
fak skill compile [--json] [--dialect <name>] [--expose <canonical-name>]... <SKILL.md>
```

Compiles exactly one fenced `fak-program` JSON block from a skill file into a
content-addressed host registration and an independently content-addressed
model-visible snapshot. Natural-language skill prose is never inferred as
executable control flow.

Registration is hidden by default. `--expose` selects canonical names for the
current snapshot; `--dialect` applies declared provider/harness aliases after
selection. The model learns current availability only from the `tools` carried
in its provider request—not from installation, a skill being present, a builtin
name, or a model's training prior.

JSON output fields are stable at version `fak.skill-compile/v1`:

- `registration`: canonical program, source identity, and registration digest;
  host-only executor argv/adapter data lives here.
- `model_view.digest`: identity of the exact selected surface.
- `model_view.dialect`: requested alias dialect.
- `model_view.tools`: selected provider-visible names, descriptions, input
  schemas, canonical names, and registration digests; executor data is absent.
- `model_view.omitted`: installed registrations omitted from this snapshot with
  a reason such as `NOT_SELECTED`.

Exit status is `0` on success and `2` for usage, read, compile, unknown
selection, invalid dialect alias, collision, or JSON-encoding failure. Without
`--json`, the command prints a concise registration/exposure summary.

A deterministic command adapter is optional but must be explicit:
`fak.command-adapter/v1` declares every JSON field mapped to an argv entry,
stdin, or environment value and declares `result: "json"`. Execution uses an
argv vector, never shell-string interpolation, and refuses undeclared adapters,
missing fields, nonzero exits, and non-JSON output.

Runnable hidden/exposed example and self-check:
[`examples/skill-program/`](../examples/skill-program/).

## `fak study-monitor`

Render and validate the durable external-repository queue used by the `study-repo` and `scout-loop` skills:

```bash
fak study-monitor
fak study-monitor --due-days 7 --json
fak study-monitor --registry docs/research/monitored-repositories.json --as-of 2026-08-14
```

The command reads `docs/research/monitored-repositories.json` by default, sorts by priority, and reports each source's status, pinned checked revision, `last_checked` age, and whether it is due for refresh. `--as-of` exists for deterministic witnesses and tests. The command does not contact GitHub or mutate the registry; scouts update all check fields together after inspecting the source.