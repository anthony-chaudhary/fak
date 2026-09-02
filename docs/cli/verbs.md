---
title: "fak CLI reference — Verb details — the shipped `fak` runtime catalog"
description: "Per-verb deep documentation for the fak runtime, split out of docs/cli-reference.md; the verb index lives there."
---

# Verb details — the shipped `fak` runtime catalog

Split out of [the CLI reference](../cli-reference.md); the verb overview and flag-level help stay there (`fak help --all` is the generated surface).

### `fak idempotency`

`fak idempotency` guards a non-idempotent command with a durable key. `run`
fsyncs a `PENDING` intent before starting the command. A successful command
becomes `APPLIED` and replays its captured stdout within the dedup window. A
command error, response loss, or failure to persist the success becomes
`UNKNOWN_APPLIED`; that state never expires into automatic re-execution.

```bash
fak idempotency run --op issue-create --token "$TOKEN" --ledger .idem.jsonl -- gh issue create ...
fak idempotency status --op issue-create --token "$TOKEN" --ledger .idem.jsonl --json

# After operation-specific read-back, record exactly one explicit verdict.
fak idempotency resolve --op issue-create --token "$TOKEN" --ledger .idem.jsonl --applied-result "created issue #42"
fak idempotency resolve --op issue-create --token "$TOKEN" --ledger .idem.jsonl --absent
fak idempotency resolve --op issue-create --token "$TOKEN" --ledger .idem.jsonl --unknown
```

`--applied-result` records the original result and makes later calls replay it.
`--absent` permits one fresh execution. `--unknown` records an inconclusive
read-back and leaves the key blocked. Existing success-only ledger rows without a
`state` field remain compatible and load as `APPLIED`. This is a fail-closed
ambiguity gate, not a universal exactly-once transaction protocol.

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
specified in [`docs/operator-control-plane.md`](../operator-control-plane.md).

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
fak manage disable [--reason TEXT] [--] [agent command]          # one-child BREAK-GLASS raw repair session (default child: codex); strips inherited guard routing, persists no disabled state
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
fak codex     [--dry-run] [--freshness-gate on|off] [--freshness-max-age D] [--freshness-force] [--native-permissions] [--split off] [--managed-cache on] -- exec --json "task"  # checkout-local launchers prove freshness against origin/main, guarded self-update/re-exec once when stale, and refuse unknown/update failure; --freshness-check-now bypasses the six-hour successful-check lease; overlapping launches immediately use the current launcher while one owner checks or updates; --freshness-gate off is the explicit escape; release installs without a checkout stay offline
`fak codex` managed launches default to Codex's non-interactive `--dangerously-bypass-approvals-and-sandbox` mode because fak still enforces its independent routing, capacity, policy, hook, and loop gates. This disables Codex's native approval prompts and sandbox, including for Codex subagents that inherit the parent permission mode. Use `--native-permissions` to restore Codex's native approvals and sandbox; legacy `--skip-permissions` remains accepted as an explicit bypass opt-in.

Freshness precedence is CLI over environment over `%UserConfigDir%/fak/codex-freshness.json` (`{"max_age":"6h","force":false}`) over the six-hour default: --freshness-max-age D overrides FAK_CODEX_FRESHNESS_MAX_AGE; --freshness-force (and compatibility spelling --freshness-check-now) always performs a real check. A fresh ak.codex-freshness.v1 receipt binds its timestamp to the full running and target SHAs; missing, corrupt, expired, future-skewed, or SHA-mismatched receipts fall through to inspection rather than authorizing launch.

fak c <target>|--target NAME|--auto|--list-targets      # pick a named compute backend (mac/gcp/local/anthropic + ~/.fak/targets.json); --auto ranks by health then cheapest/most-local (cost local<mac<gcp<anthropic), fails over past a DOWN target. quota is a [stub] (not a live fak accounts read) and never excludes
fak snapshot  kinds | demo | info | dump-fleet | restore-fleet   # dump/restore any primitive (turn|tool|session|fleet|RSI loop) to a portable sha256-integrity bundle
fak serve     --addr :8080 [--require-key-env VAR] [--fleet-bus [--fleet-bus-dir DIR]] [--session-registry PATH|off]   # OpenAI-compatible HTTP + MCP gateway (any-language agents). `--session-registry` scopes WHICH SESSIONS this serve can see and write (#5825): unset keeps today's shared per-user default (`FAK_SESSION_REGISTRY`, else `<UserConfigDir>/fak/session-registry.json` — the single file EVERY serve on the box shares), which is the right reach for a real fleet but means a serve started only to drive its own sessions still adopts every live session on the host, so a fanned `fak fleet control send --op pause --all` writes to peers' work. `--fleet-bus-dir` does NOT narrow this: it scopes the BUS (announcements, directives, claims, acks) and nothing about which sessions get written, so a private bus over the shared registry reads like a sandbox and is not one. Pass a path to hydrate from and mirror to that file alone, or `off` for a pure in-memory table that adopts nothing and persists nothing. An armed `--fleet-bus` serve prints its own reach — session count and registry — before it drains a single directive
fak recall    --dir DIR                                # persist/inspect a finished session as a durable core image
fak dream     --dir DIR --out-dir DIR                   # offline cleanup pass over a sleeping core image
fak debug     --session DIR --cmd report|info|bt|x|ws|grep|tombstone|context-query|context-diff   # attach to a session core image; demand-page its working set
fak answer-shape --text - --max-repeat 0.5 [--max-chars N]   # degeneration/verbosity witness over a text; exit 1 when it loops/runs away
fak doctor    --text - [--max-repeat 0.5] [--max-chars N]   # run the answer-shape witness + the kernel admit cross-check, then recommend
fak-dev index lane <path>... | leaf [<query>] | docs <query> | refs <pkg>.<Sym> [--json] [--limit N]   # query the devindex self-index for lane ownership, leaf search, docs, and Go refs; alias: fak devindex
fak codelint  PATH...                                  # lint agent-written code (Go/JSON in-process, Python/CUDA via toolchain); exit 1 on a hard parse/compile error
fak policy    --dump | --check FILE                        # author/validate the deployable capability floor
fak route     --aspect tool_call --tool refund_payment [--manifest FILE] [--simulate "a,b,b"]   # which model/ensemble routes this aspect; --dump/--check author the routing manifest
fak routebench [--corpus FILE] [--routed F] [--single F] [--json]            # offline routing benchmark: per-aspect+ensemble vs single-model on cost/latency/quality (no model in the loop)
fak vcache    status | prove | prove-telemetry | actions | apply-actions | score   # virtual provider-cache status, planned/applied action ledgers, and token-savings proof/refutation scorecard
fak cachevalue report|review|feed [--since DATE] [--usage-ledger FILE] [--context-budget-tokens N] [--json] [--append-ledger FILE] [--markdown-out FILE] # cache-effectiveness P&L, cumulative fleet savings/session-extension aggregate, plus generated cache-frontier review artifacts
fak cachevalue shapes [--since DATE] [--json] [--trend] [--ledger FILE]   # cluster the WITNESSED Track-1 kernel ledger by session SHAPE — length band (single/short/long) × realized-reuse outcome (n/a/cold/partial/warm) — so a reader sees WHICH KINDS of sessions earn KV-prefix reuse, a fact the week×session_type trend hides (#3115); --trend swaps the static snapshot for each shape's week-over-week reuse-share drift. #1066 fence: outcome bands cut on WITNESSED realized reuse only, never the vs-naive re-prefill multiple. The default `--ledger` (`docs/nightrun/cache-value.jsonl`) is a **gitignored local nightrun artifact**, so a fresh checkout renders `INSUFFICIENT` (0 sessions) until a nightrun writes it — that is the empty-ledger read, not a broken verb; point `--ledger` at your own Track-1 JSONL to fold rows meanwhile
fak kvbm      replay|trace [--artifact/--trace FILE] [--json] [--check]   # KVBM eviction validation: replay proves pin/restore safety; trace proves cost-aware>=LRU, oracle score, and no-thrash stability
fak callavoid prove-memo | account [--in FILE] [--json] [--gate]   # avoided-call economics: break-even memo proof + per-window amplification scorecard (JSON in/out)
fak turnavoid replay --in TRACE.jsonl [--json]             # offline whole-model-turn avoidance replay; strict JSONL, effect-witnessed credit
fak cadence   [--json] [--check] [--append-history] [--window N]   # consolidated regular-cadence report: folds scores + maturity + work-done + releases into one control-pane envelope, including the top public `fak maturity route` seed; --append-history writes the durable ledger with standing_score + difficulty fields (docs/cadence/history.jsonl)
fak milestone report|post [--json] [--check] [--append-history]   # milestone report: the maturity CLIMB (model x backend M0-M7 grid) + the epic ROADMAP, split by WORK CLASS — DISCRETE epics on a completion % vs ONGOING optimization programs (kernel-opt, cache-opt) shown as frontier activity with NO % (they have no 100%). Trended in docs/milestones/history.jsonl
fak program   report [--json] [--check] [--append-history] [--window N]   # ongoing-program report (the milestone sibling for never-'done' work): kernel-optimization, cache-optimization, and human-operator-effectiveness by FRONTIER + TREND, never a completion %. Trended in docs/programs/history.jsonl
fak operator  brief [--cadence FILE] [--program FILE] [--milestone FILE] [--heaviness FILE] [--previous FILE] [--collect] [--json] [--check]   # human pacing brief: folds cadence/program/milestone plus optional operator-heaviness and previous-brief JSON into source coherence, since_previous delta, attention timebox/read-order, human-use guidance, strengths, choices, challenges, learning, and human/agent/watch/background buckets
fak operator  heaviness [--json] [--markdown] [--compare FILE]   # operator-surface pressure scorecard: verb surface, guard flag burden, refusal vocabulary, doc-map discoverability, appeal channel, heaviness_debt, and heaviness_pressure
fak score     <name> [--json] [--markdown] [--compare FILE]   # parent verb (#1505) grouping the meta-scorecards / RSI loops so the top-level surface stays operator daily-drivers. `fak score list` names them: conflation, dogfood, dojo-rsi, guard-rsi, guard-verdict-rsi, product, skill-effectiveness, support-maturity, token-defaults, ui-quality. Each forwards to the same handler its legacy top-level verb ran (behavior-preserving; the legacy verbs remain as thin aliases)
fak maturity  [next] [--json|--markdown] [--compare base.json]   # feature-maturity lifecycle scorecard: places every declared capability on a closed ladder and emits `fak maturity next`, the ranked backlog. `fak maturity anatomy [package-path] [--json]` adds a static package readout; `--all [--limit N]` compares every declared internal leaf and ranks structural hotspots: see [`docs/maturity-anatomy.md`](../maturity-anatomy.md) for definitions and interpretation; shape, control-flow complexity, success/error/ambiguous exits, guards and assumption comments, documentation coverage, and internal dependency position. `fak maturity route [--limit N] [--fetch-existing|--live]` turns the top public-routeable backlog rows into stable, deduped GitHub issue plans so the issue-dispatch loop can work them.
fak idea-scout [--json] [--max-issues N] [--min-score N] [--config FILE] [--candidates FILE --issues FILE --scout-issues FILE] [--live]   # research-to-issue feeder (docs/idea-scout.md): score arXiv/GitHub/Hacker News/Reddit hits for relatedness, dedupe four ways (seen-cache, the label-targeted filed-stamp index, existing issue bodies, near-duplicate titles) and plan triage-ready issues. DRY-RUN BY DEFAULT — it mutates nothing. `--live` is the blast radius: it runs `gh issue create` against the current repo's REAL tracker, up to `--max-issues` public issues, and records each filed source id in `.idea-scout/seen.json`. The `--candidates`/`--issues`/`--scout-issues` fixtures replay a run offline with no network and no `gh`. Exit 2 when the filed-stamp index cannot be built completely — it REFUSES rather than risk re-filing.
fak scoreboard post [--from card.json --debt-key K | --kpi NAME --value V --grade A --verdict OK --detail ...] [--dry-run]   # post a scorecard result/score to the Slack scoreboard channel (its own FAK_SCOREBOARD_* workspace, separate from the lab bridge); CI + local agents publish a number the moment it changes
fak scorecard control-pane [--json|--check|--pin] [--post]   # LOCAL producer for the same #scoreboard feed (#998, local side of 52ed934b): --post (or FAK_SCOREBOARD_AUTOPOST=1) auto-posts the freshly-regenerated portfolio number tagged --source <hostname> the moment the scorecard is folded — off by default, reuses internal/scoreboard (no second manual `scoreboard post`), deduped via .fak/scoreboard-autopost-state.json so an unchanged rerun is silent
fak bench-loop status|next|walk|run [--json]   # benchmark super-loop manager: folds registry, recorded runs, nightrun ledger, local next selection, and authority gap; run delegates to fak nightrun run
fak bench post    --rollup latest|regression [--n N] [--catalog PATH] [--baseline PATH] [--dry-run]   # post a bench-channel rollup: the latest catalog runs (WITNESSED/OBSERVED-labeled) or tok/s drops vs the pinned baseline. FAK_BENCH_* workspace (token falls back to the scoreboard token)
fak bench request [--now STAMP | --plan-json FILE] [--top N] [--dry-run]   # post a bench RUN-REQUEST (the bench_plan next-test-per-machine) to the bench channel. A request is a POST, not a dispatch — no inbound listener; the bench-nodes act on it out-of-band
fak bench system-baseline [--baseline-duration 1s --interval 250ms --max-sampler-duty-percent 10 --out FILE] [--top-consumers] -- COMMAND... | --verify FILE   # attest ambient host and sampled process-tree load for exactly one benchmark repetition
fak blockers post [--severity status|operator|clear] --title ... [--detail ... --owner "<@U>" --action ... --action-url URL --ref ...] [--dry-run]   # post a BLOCKER to the central Slack #blockers channel: a background `status` line records quietly, an `operator` one is SURFACED (pages <!here>/owner, red, with a do-this-next). FAK_BLOCKERS_* (token falls back to the scoreboard token; #blockers is the built-in default)
fak blockers source --repo OWNER/REPO --label LABEL --issues-out FILE --status-out FILE   # fail-closed CI acquisition: write UNKNOWN first, require the exact label before+after `gh issue list`, validate the JSON array, then write issues and flip the marker to OK
fak blockers feed --issues FILE [--source-status FILE] [--label blocked --repo-url URL] [--dry-run]   # CI roll-up: fold an explicit `gh issue list --json number,title,url,assignees,labels` array into one card — a successful [] is clear; missing/blank/null/malformed input or a supplied source marker other than OK fails closed; UNOWNED blockers page, while all-assigned blockers report ownership without inferring progress
fak chatrelay --endpoint URL --channel C0X [--model M --mention <@U> --system S --prime=false --once --interval 3s --dry-run]   # bridge ONE Slack channel to a `fak serve` /v1/chat/completions endpoint: poll history, forward each human message, post the reply in-thread. Generic chatbot front end — no shell, no command router; channel text is chatbot input, never a command. FAK_CHATRELAY_* (token falls back to the scoreboard token; channel has NO fallback). See docs/fak/slack-sessions.md
fak chatops   --channel C0X --admins U07A,U07B [--bot-user <@U07BOT> --audit FILE --prime=false --once --interval 3s --dry-run]   # inbound read-only control door: poll ONE control channel, parse each admin mention as ONE verb from a closed grammar (help/ping/status/fleet answer; dispatch/resume/halt are declined until the guarded act path lands). Fail-closed admin allowlist on the immutable user id; refuses to start with no admins; every decision journaled to --audit. FAK_CHATOPS_* (token falls back to the scoreboard token; channel/admins have NO fallback). See docs/fak/slack-sessions.md
fak fleet control send --op OP [--payload|--text TEXT] (--all | --instance I,I | --machine M | --role R) [--lane L --wave W --label X] [--ttl 5m] [--wait 10s] [--reason R] [--bus DIR] [--json] | status --directive ID | instances [--ttl D]   # the centralized control point (#5600): fan ONE op to every announced fleet instance over a shared bus directory (`fak serve --fleet-bus` arms an instance), then fold the ACKS back — `send` exits 0 only when every addressed instance witnessed the apply, 1 when it published but nobody has answered (including `--wait 0`), and 2 when the selector addresses nobody (`FLEETBUS_NO_TARGET`) rather than accepting a directive that can never apply. Instance axes (`--all`/`--instance`/`--machine`/`--role`) pick WHICH processes; session axes (`--lane`/`--wave`/`--label`) narrow WITHIN each one. An instance that matched nothing acks REFUSED, never a hollow "applied". The selectors pick which INSTANCES and which sessions within them, but the set an instance can write at all is its own `fak serve --session-registry` scope — neither `--bus` here nor `--fleet-bus-dir` there narrows it (#5825), so `--all` against the default shared per-user registry reaches every session on each addressed host, including peers'
fak leaseref  live [--dir DIR] | liveness [--session ME] [--dir DIR] | session-publish --session S [--ttl SEC] [--dir DIR] | list [--json] [--dir DIR] | audit [--dir DIR] | reap [--dir DIR] | sync [--remote R] [--push-only|--fetch-only] [--dir DIR]   # cross-machine lease visibility: read refs/fak/locks/* into the dos_arbitrate live_leases shape (#825); `liveness` classifies each live lease self|peer-live|peer-dead|peer-unknown by the owning session's heartbeat (#2164); `session-publish` refreshes that heartbeat as a side ref; `audit` is the read-only staleness report; `sync` converges the namespace with a remote (push-then-fetch, side refs only)
fak attest    --policy FILE [--probes FILE] [--json]        # compliance attestation: prove the capability floor from preflight (exit 0 PROVEN / 1 drift / 2 usage)
fak audit     verify <journal.jsonl> | export <journal.jsonl>   # audit-trail consumer: re-verify a fak manage decision journal's hash chain, or export it
fak egress    check (--url URL | --command CMD | --host HOST | --tool T --args JSON)   # prove the network-egress floor on one destination — the cloud-metadata / SSRF class
fak self-update [--check] [--force] [--root DIR] [--target PATH]   # converge a built-from-source fak binary on origin/main; --check reports staleness vs HEAD and exits without building
fak self-update --build-gc [--root DIR]                              # emit a JSON dry-run plan for stale self-update build worktrees; no fetch/build/gate/swap
fak self-update --build-gc --apply [--root DIR]                      # revalidate and remove only eligible stale self-update build worktrees, then emit the JSON receipt
fak-selfupdate [same flags]                                      # thin standalone bootstrap; shares the exact updater, receipt schema, cache, transaction, rollback, and source-selection implementation
fak self-update --manifest-url HTTPS_URL [--manifest-channel stable] [--manifest-cohort NAME] [--manifest-cache PATH] [--offline]
fak self-update --installer msix --msix-appinstaller-uri HTTPS_URL --msix-package PACKAGE --msix-publisher SUBJECT --msix-artifact-digest SHA256 --msix-source-revision REV [--msix-full-fallback-uri HTTPS_URL --msix-full-artifact-digest SHA256] [--msix-repair|--msix-uninstall]

Windows MSIX/App Installer delivery is explicit opt-in. `--installer` has precedence over
`FAK_SELF_UPDATE_INSTALLER`; when neither is set, `native` remains the default. The adapter
requires the package identity `Name` from `AppxManifest.xml`, separate app (`fak`),
artifact-digest, and source-revision identities plus
the expected signing-certificate subject. It validates the downloaded package signature and
publisher before invoking App Installer. Online `.appinstaller` delivery uses Windows block-map
differential updates; `--offline` selects `--msix-full-fallback-uri` (or the primary URI when no
separate full bundle is supplied). A distinct full fallback requires its own `--msix-full-artifact-digest`; the receipt reports the selected delivery and whether fallback ran. Downgrades are refused unless `--msix-allow-downgrade` is
explicitly set. Repair and uninstall are explicit, mutually exclusive operations. PowerShell
runs non-interactively in a hidden process; selecting `msix` on a non-Windows host is refused.

Source self-update normally removes its detached pristine build worktree when the run finishes. If a run is killed before deferred cleanup, the next mutating source update invokes the same owner-aware collector automatically. Operators can inspect that lifecycle without starting an update using `fak self-update --build-gc`; it is always JSON and dry-run by default. Add `--apply` to mutate. Both modes enforce the collector's 30-minute grace floor plus owner, process-reference, clean-tree, ancestry, and exact-path gates. `--check` remains a separate strictly read-only freshness check and cannot be combined with `--build-gc`.

Exact `fak-selfupdate-build-<pid>` worktrees belong to this owner-aware collector. Generic `fak tree-doctor` cleanup intentionally defers them rather than applying a weaker generic age rule. The receipt schema is `fak.self-update.build-gc/v1`; its top-level `mode` is `plan` or `apply`, and `report` retains every `selfinstall.BuildGCReport` field, including per-worktree reasons and apply failures.

Source self-update derives a deterministic digest from Go's complete non-standard dependency graph for `./cmd/fak`, including generated/runtime source files, `go:embed` assets, native inputs, module metadata, toolchain/platform/CGO architecture knobs, tags, and build/link flags. A docs-only or test-only source revision can therefore reuse the prior digest-verified artifact without compiling again; any graph or build-envelope uncertainty falls back to the full build, vet, and smoke gates. JSON update receipts expose `build_provenance` with the selected source commit, the source commit that originally built reused bytes, build-input and artifact digests, artifact size, build envelope, and reuse decision.

Self-update keeps these identities separate: Git owns the selected/source commits (exact
40-hex comparison), the build-input walker owns its SHA-256, the verified file bytes own the
artifact SHA-256 and slot ID, `VERSION`/linker provenance owns app version, and selfinstall
owns the monotonic metadata generation. A digest-equal candidate refreshes selected-source
metadata but does not copy, swap, activate, or change app version. Persisted rollback state
selects a verified artifact slot by digest, never by an ambiguous app-version string.

Self-update also emits a deterministic per-component plan. Every `targets[]` row names the desired and installed artifact digests, acquisition (`reuse` or `transfer_or_build`), activation (`no_op` or `activate`), compatibility group, and rollback action. Already-current components are excluded from activation; compatibility-coupled stale components remain one staged transaction, so an activation failure restores every component moved by that transaction.
`--manifest-url` opts into conditional selection before any git fetch, build, or install. The
server returns a canonical JSON payload signed by the public Ed25519 key pinned in the binary.
The signature is verified before schema, manifest ID, channel, cohort, OS, architecture,
installed revision identity, expiry, disposition, target version/revision, or optional retry
time is trusted. `no-update` and `cohort-hold` stop without changing the installed binary.
Authenticated envelopes and ETags are cached; a `304` or `--offline` may reuse only an
unexpired identity-matched cache and never extends signed expiry. `429`/`503` Retry-After is
bounded to 24 hours and cached. `--force` contacts the server despite cache freshness/backoff,
but never bypasses signature, identity, or expiry checks. Without `--manifest-url`, behavior
is unchanged.

Manifest v2 may also carry a full executable target for the selected OS/architecture. The
signature binds the artifact URL, platform, architecture, SHA-256, byte size, app version,
source revision, expiry, and monotonic metadata generation as one identity. Self-update
rejects corrupt signatures or bytes, expired metadata, generation or app-version rollback,
same-generation changed payloads (freeze/mirror equivocation), duplicate usable targets, and
version/revision mix-and-match. A usable target is downloaded with a strict size bound and
SHA-256 check, then must pass `version --json` smoke and attest the signed app version plus
exact source commit before activation. Verified bytes are copied into immutable
generation+digest slots; activation reads from that slot and never deletes older verified
slots, so the prior artifact remains recoverable. When the authenticated catalog has no
complete target for this host/component set, self-update preserves the full pristine-source
build, vet, provenance-smoke, and transactional activation fallback. A present matching target
that fails authentication or verification is a hard refusal, never a reason to build different
bytes from source.

Signed targets may also advertise per-source zstd deltas. Self-update selects one only when the installed artifact SHA-256 exactly matches the delta source and the patch is below the transfer-ratio threshold. Patch size/SHA-256 and patched target size/SHA-256 are verified before the existing smoke/activation gate. Wrong source, corrupt patch, unavailable zstd, patch failure, or poor ratio falls back to the independently signed full artifact. The receipt's `transfer` object records chosen path, delta/full bytes, total time, verification, fallback reason, and fallback bytes/time cost.
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

`fak turnavoid replay` is the offline, whole-model-turn counterpart. It reads strict
`fak.turnavoid.trace/v1` JSONL, pairs every candidate with its immutable control row,
and credits a realized avoided model turn only when an independent required-effect
observation is equivalent. Retained-turn token/latency reductions, avoided tool calls,
invalidated opportunities, and counterfactual-only rows remain separate:

```bash
fak turnavoid replay --in trace.jsonl          # concise, arm-by-arm text
fak turnavoid replay --in trace.jsonl --json   # fak.turnavoid.report/v1
cat trace.jsonl | fak turnavoid replay --in - --json
```

Each arm reports committed turns, realized and withheld turn deltas, preserved and
suppressed required effects, gross model/tool latency and cost, and net values after
validation, speculation, retry, and recovery overhead. Overhead can make net savings
negative without erasing a witnessed realized-turn count.

The command exits `0` on a valid replay and `2` on usage, schema, validation, or output
errors. See [`docs/notes/TURN-AVOIDANCE-FIRST-CLASS-2026-08-24.md`](../notes/TURN-AVOIDANCE-FIRST-CLASS-2026-08-24.md)
for the taxonomy, accounting boundary, research provenance, and rollback contract.

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

# public-repository backup plane: participating machines share this key out of band
fak leaseref announce --issue 123 --id L --holder "$ME" --tree 'docs/**' --ttl 900 --action acquire --public-safe-key-file ~/.config/fak/lease-announce.key
```

`announce --public-safe-key-file` projects the raw lease ID, holder, and each exact tree
entry into domain-separated HMAC-SHA256 fingerprints before posting. The issue comment
therefore carries a versioned, machine-readable advisory record without publishing machine
or session names, repository paths, or the key. Nodes that share the same private key derive
the same fingerprints and can recognize exact-scope duplicates in `announce-view`.

Keep the key outside the repository and distribute it through an existing secret channel;
passing it by file avoids command-line exposure. Fingerprints hide raw values but do not hide
transition timing, generation, TTL, action, or the number of tree entries. This plane is
advisory visibility only: it neither grants a lease nor detects overlap between different glob
spellings. Omit `--public-safe-key-file` only for a coordination issue whose readers may see
the raw holder and tree values.

Successful `acquire`, `renew`, and `release` operations can post this public-safe record
as an explicit lifecycle option. The non-secret destination and mode travel as flags; only
the key-file locator remains in the secret-bearing environment:

```sh
export FAK_LEASEREF_ANNOUNCE_KEY_FILE=~/.config/fak/lease-announce.key
fak leaseref acquire --id L --holder "$ME" --tree 'docs/**' \
  --announce on --announce-issue 123 --announce-repo OWNER/REPO
# renew and release accept the same three --announce* flags
```

The key-file variable names a file; the key itself must not be placed in argv, environment,
JSON, logs, issue comments, or the repository. The default/unset state is explicitly
**disabled**. Pass `--announce offline` to explicitly suppress the network edge;
a missing/unreadable/empty key is reported explicitly and no comment is attempted. A `gh`
post failure emits only a sanitized warning and can never reverse, mask, or change the exit
status of the already-successful local lease operation. Comments expose transition timing,
generation, TTL, action, and scope count as described above, but no raw IDs, holder names,
paths, key, repository target, or key-file path.

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
[`examples/modelops-top3-canary.json`](../../examples/modelops-top3-canary.json) and the
dogfood readout [`docs/notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md`](../notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md),
which records a witnessed `ROLLBACK` decision against that policy.
### Region admission: `fak loop region` + the loop drive's region hold

The lease fabric above answers *"who holds what"*; **region admission**
([`docs/region-admission.md`](../region-admission.md), `internal/regionadmit`)
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

### Fan a shipped spine out into its follow-on backlog — `fak-dev issue fanout`

`fak-dev issue fanout` ([#2510](https://github.com/anthony-chaudhary/fak/issues/2510),
spine `5b8f0bd1`, `internal/issuefanout`) is the **spine-first** producer that *fills* the
backlog the garden → dispatch boundary below *drains*: it expands one shipped working spine
into its contract-ready follow-on issues across the fan-out area taxonomy
(`qa,dogfood,product,observability,integration,docs,release`). It **plans by default** —
printing the candidate set and touching nothing — and only files when asked. The synopsis is
the one `fak issue` prints, so the reference and `--help` read the same truth:

```
fak-dev issue fanout --title T --leaf L --spine REF [--parent REF]
                   [--paths p1,p2] [--areas a1,a2] [--max N] [--json]
```

The planning flags, exactly as `fak-dev issue fanout --help` describes them:

- `--title` — human name of the shipped spine.
- `--leaf` — owning leaf/lane (stamps keys, lane, default paths).
- `--spine` — spine witness: commit SHA, demo command, or doc path.
- `--parent` — epic/issue ref the fan-out hangs off (default: `--spine`).
- `--paths` — comma-separated file trees (default `internal/<leaf>/`).
- `--areas` — comma-separated area filter (`qa,dogfood,product,observability,integration,docs,release`).
- `--max` — cap candidates (`0` = full taxonomy; floor `3`).
- `--json` — emit the machine-readable fan-out plan (feed to `fak-dev issue cohort --from-plan`).

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
  Task Scheduler arm — see [`docs/dispatch-loop.md`](../dispatch-loop.md)) are the only path that
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
(`--serve --addr 127.0.0.1:9096`); see [`tools/grafana/README.md`](../../tools/grafana/README.md)
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
[`docs/model-routing.md`](../model-routing.md).

`fak routebench` is the offline measuring instrument: it runs a corpus of recorded
cases through TWO manifests — a per-aspect + ensemble policy vs a single-model
baseline (the SOTA shape) — and prints the delta on **cost / latency / quality**.
Each case carries the stand-in OUTPUT every candidate model produces (like `fak
route --simulate`), so it reuses the pure `Route` + `Combine` halves and is
deterministic end to end — no key, no GPU. Default (no args): the built-in 8-case
demo corpus + `DefaultManifest` vs a one-frontier-model baseline; `--corpus` /
`--routed` / `--single` load your own, `--dump-corpus` emits the starter corpus to
edit. Every figure is a ROUGH lens, never a bill or a measured SLA. See
[`docs/model-routing.md`](../model-routing.md#the-offline-routing-benchmark-fak-routebench).

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
