package devindex

// C3 of epic #1287 (#1290): the structured CLI-verb catalog behind `fak index verbs`
// (CLI + MCP). The committed `fak` verb list used to live only as freeform raw strings
// in cmd/fak/usage.go — unparseable, and drifting from the main.go dispatch.
//
// Design (post-#1293): the catalog is a LIVE VIEW. COVERAGE is DERIVED from the
// cmd/fak/main.go dispatch switch (see Verbs()), so it can never fall behind the
// binary — a newly dispatched verb appears automatically, no hand-maintenance, no
// drift gate needed. The committed verbManifest below is only a curated QUALITY
// OVERLAY (synopsis -> owning lane -> aliases -> doc link); a dispatched verb with no
// overlay entry still appears with a fallback synopsis, and an overlay entry for a
// verb not (yet) dispatched is simply not emitted. UndeclaredVerbs (freshness.go) is
// retained as an advisory CURATION-drift signal (which live verbs lack a curated
// entry), not a coverage gate.
//
// Lane note: this overlay + the query/derive functions live INSIDE internal/devindex;
// the `fak index verbs` cmd shell is the cmd/ half — out of this package's lane.

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Verb is one entry of the structured CLI-verb catalog: the verb name as typed
// (`fak <Name>`), the one-line synopsis shown in usage, optional command aliases
// that route to the same handler (the extra strings in a `case "leaf", "leaves":`),
// the owning lane/leaf the handler's code lives under, and an optional doc-map path
// for the deeper reference. It is the parseable replacement for a raw usage string.
type Verb struct {
	Name     string   `json:"name"`
	Synopsis string   `json:"synopsis"`
	Aliases  []string `json:"aliases,omitempty"`
	Lane     string   `json:"lane,omitempty"`
	Doc      string   `json:"doc,omitempty"`
	// Tier is the epic-#2228 CLI concept tier (frontdoor|dev|hidden), stamped
	// from the verbTiers table (tiers.go) — never authored per manifest entry,
	// so the classification keeps its one home. Empty only for a curated entry
	// whose verb is not (yet) dispatched.
	Tier VerbTier `json:"tier,omitempty"`
}

// Spellings returns the verb's canonical name plus every alias — the full set of
// argv[1] tokens that route to this verb. The freshness gate joins on this set so a
// main.go `case "a", "b":` with one manifest entry covering both does not red.
func (v Verb) Spellings() []string {
	out := make([]string, 0, 1+len(v.Aliases))
	out = append(out, v.Name)
	out = append(out, v.Aliases...)
	return out
}

// verbManifest is the curated QUALITY OVERLAY for the CLI-verb catalog — NOT its
// coverage source. Coverage is a live VIEW: Verbs() derives the verb SET from the
// cmd/fak/main.go dispatch switch (so the catalog can never fall behind the binary —
// a newly dispatched verb appears automatically, with no hand-maintenance), and this
// table supplies the synopsis, owning lane, alias grouping, and doc link for the verbs
// it names. A dispatched verb with no entry here still appears, carrying a fallback
// synopsis; adding an entry here only upgrades its quality. (#1290 gave the structured
// shape; deriving coverage from the live switch is what keeps it fresh — #1293.)
//
// The Lane is the leaf the verb's handler code lives under (almost always `cmd`, the
// cmd/** shell; `gateway` for the serve/guard front doors, `devindex` for the
// self-index), resolvable by LeafByName so the C3->taxonomy join cannot drift
// off-taxonomy. A synopsis is a faithful one-line condensation of the verb's usage.go
// help; verbs with no usage.go block carry a tight handler-derived line. Keep it sorted
// by Name so a diff is stable. An entry for a verb not (yet) in the dispatch switch —
// e.g. a peer's in-flight verb committed later — is simply not emitted until it goes
// live, so this table and the binary never have to be reconciled in lockstep.
var verbManifest = []Verb{
	{Name: "ablate", Synopsis: "self-ablation: replay one trace under N cache-lever configs; live savings dashboard by default", Lane: "cmd"},
	{Name: "ablate-arm", Synopsis: "internal: the re-exec child of fak ablate — reads one arm request on stdin, writes one AblationRun", Lane: "cmd"},
	{Name: "accounts", Synopsis: "config-home registry: every CLAUDE_CONFIG_DIR seat with its disk-true identity + tombstone rehome", Lane: "cmd"},
	{Name: "affected", Synopsis: "fast inner loop: run go test only for the packages your working-tree change can affect", Lane: "cmd"},
	{Name: "agent", Synopsis: "run a live agent task and A/B the turn count (offline or against a provider)", Lane: "cmd"},
	{Name: "agent-queue", Synopsis: "reconcile an agent-pool snapshot into start/hold queue actions", Lane: "cmd"},
	{Name: "agentic", Synopsis: "compile a declared objective into a deterministic agentic execution contract", Lane: "cmd"},
	{Name: "agents", Synopsis: "query live and historical agent sessions with constrained SQL, grouping, counts, and JSON", Lane: "cmd"},
	{Name: "ailuminate", Synopsis: "AILuminate safety-benchmark runner (describe/eval/compare)", Lane: "cmd"},
	{Name: "amd-gpu-facts", Synopsis: "AMD GPU facts probe: per-device utilization / VRAM / busiest-engine as JSON, optional --watch sampling", Lane: "cmd"},
	{Name: "answer-shape", Synopsis: "degeneration/verbosity witness: grade how repetitive and how long a candidate answer is", Lane: "cmd"},
	{Name: "antipattern-scorecard", Synopsis: "the unified work-loss card: fold REDUNDANT_REWORK + UNWIRED_PKG + ORPHAN_FUNC into one antipattern_debt", Lane: "cmd", Doc: "docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md"},
	{Name: "api-host", Synopsis: "probe an API-host target (base_url/api_key/model) for readiness + acceptance, JSON/Markdown reports", Lane: "cmd"},
	{Name: "architecture", Synopsis: "analyze dependency tiers, violations, fan-out, depth, and blast radius for a leaf or workspace", Lane: "cmd"},
	{Name: "armbench", Synopsis: "run provenance-locked paired benchmark arms from one immutable manifest", Lane: "cmd", Doc: "docs/armbench.md"},
	{Name: "assume", Synopsis: "check a declared assumption against its witness: seat-launchable (doctor claim vs accounts-next authority)", Lane: "cmd"},
	{Name: "attest", Synopsis: "compliance attestation generator: prove the policy capability floor from preflight", Lane: "cmd"},
	{Name: "audit", Synopsis: "audit-trail consumer: verify/export a fak guard decision journal's hash chain", Lane: "cmd"},
	{Name: "backend", Synopsis: "scaffold a new compute backend: arch/device taxonomy + build-tagged Register stub + parity-test skeleton", Lane: "cmd"},
	{Name: "balance", Synopsis: "the night-balance readout: resume recovery-vs-stranding and gardening-vs-throughput, folded side by side", Lane: "cmd"},
	{Name: "bench", Synopsis: "transport A/B: in-process adjudication p50 vs spawned-hook p50", Lane: "cmd"},
	{Name: "bench-ingest", Synopsis: "fold benchmark snapshot fixtures into provenanced modelscore rows; offline, --check validates only", Lane: "cmd"},
	{Name: "bench-loop", Synopsis: "benchmark super-loop manager: status, next action, surface walk, and local collection entrypoint", Aliases: []string{"benchloop"}, Lane: "cmd"},
	{Name: "bench-runs", Synopsis: "query + compare recorded benchmark runs (list/show/compare/best/table/summary)", Lane: "cmd"},
	{Name: "benchmarks", Synopsis: "the index of every benchmark fak ships (list/describe/run)", Lane: "cmd"},
	{Name: "bgloop", Synopsis: "background runner for the durable long-running-loop ledger", Lane: "cmd"},
	{Name: "blast", Synopsis: "estimate the dependency blast radius of a broken package against live leases and queued issues", Lane: "cmd"},
	{Name: "blockers", Synopsis: "the blockers report/Slack surface: what is gating the fleet right now", Lane: "cmd"},
	{Name: "borrow-provenance", Synopsis: "pin borrowed source provenance and verify its bytes against the recorded SHA-256", Lane: "cmd"},
	{Name: "boundary", Synopsis: "boundary-tell linter: unexpanded paths, hardcoded URLs, no-timeout HTTP, change-detector tests", Lane: "cmd"},
	{Name: "breath", Synopsis: "advisory counted-ratchet gate for the measurable half of the In-one-breath documentation contract", Lane: "cmd", Doc: "docs/ONE-BREATH-CONTRACT.md"},
	{Name: "budget", Synopsis: "per-task budget readout: tokens/turns spent vs a soft target with a per-category breakdown", Lane: "cmd"},
	{Name: "buildcheck", Synopsis: "collision-free compile check: discard output (never in-tree), -overlay-masking untracked sibling .go files", Lane: "cmd"},
	{Name: "c", Synopsis: "shorthand for 'fak console agent': launch a fak-guard-wrapped interactive Claude Code session", Lane: "cmd"},
	{Name: "cache", Synopsis: "read the live cache spine: `cache tiers` reports the probed tier ladder, per-tier pressure, and a demote/spill/restore rollup", Lane: "cmd"},
	{Name: "cachesweep", Synopsis: "sweep a prefix-access trace across cache budgets: reuse curve, infinite-cache ceiling, 99% ROI knee", Lane: "cmd"},
	{Name: "cachevalue", Synopsis: "the cache-value rollup: realized agent-memory / KV-reuse savings", Lane: "cmd"},
	{Name: "cadence", Synopsis: "consolidated regular-cadence report: scores, maturity, work-done, releases in one envelope", Lane: "cmd"},
	{Name: "callavoid", Synopsis: "the call-avoidance report: identical-call dedup (vDSO) round-trips saved", Lane: "cmd"},
	{Name: "capabilities", Synopsis: "query token, turn, cache, routing, session-control, and supporting-floor outcomes by intent", Lane: "cmd"},
	{Name: "chat", Synopsis: "minimal chat client against a fak serve/guard gateway", Lane: "cmd"},
	{Name: "chatops", Synopsis: "the inbound read-only Slack control door: answers help/ping/status/fleet behind a fail-closed admin allowlist", Lane: "cmd"},
	{Name: "chatrelay", Synopsis: "the chat-relay Slack surface bridging a channel to a gateway", Lane: "cmd"},
	{Name: "check-tool-failure", Synopsis: "lookup the closed non-guard tool-failure vocabulary (summary/fix/retryable)", Lane: "cmd"},
	{Name: "checkpoint-debt-dispatch", Synopsis: "checkpoint-scorecard -> sink: fan one deduped finding per (subsystem, axis) gap to stdout/localdb/github", Lane: "cmd"},
	{Name: "checkpoint-scorecard", Synopsis: "score which long-running process subsystems persist resumable WIP state and expose witnessed status", Lane: "cmd"},
	{Name: "ci-preflight", Synopsis: "is the COMMITTED trunk tip CI-buildable and green? (ci-spec-change migration gate; reads committed tip)", Lane: "cmd"},
	{Name: "validate", Synopsis: "check committed tip plus only explicit --mine paths with full build/vet and affected tests", Lane: "cmd"},
	{Name: "llms-full", Synopsis: "generate or check llms-full.txt from committed tip plus explicit owned paths", Lane: "cmd"},
	{Name: "claim-check", Synopsis: "grade an efficiency/perf claim against the six-question net-true-value rubric", Lane: "cmd"},
	{Name: "claude-mac-fak", Synopsis: "one-command Mac gateway dogfood through the node-macos-a fak serve gateway", Aliases: []string{"mac"}, Lane: "cmd"},
	{Name: "clean-bins", Synopsis: "prune stray gitignored go-build binaries at the module root; keeps the live fak (git-maint's build twin)", Lane: "cmd"},
	{Name: "cluster", Synopsis: "multi-node compute: run a real cross-node collective over fak's process group", Lane: "cmd"},
	{Name: "codelint", Synopsis: "language-server-pack code linter: route each file to its pack and report parse/compile errors", Lane: "cmd"},
	{Name: "codex", Synopsis: "OpenAI Codex integration launcher (codex exec routed through the kernel)", Lane: "cmd"},
	{Name: "codex-mcp", Synopsis: "install + verify fak's Codex-facing MCP server bridge (install/status, initialize-handshake checked)", Lane: "cmd"},
	{Name: "codex-mcp-health", Synopsis: "health check for the Codex MCP integration", Lane: "cmd"},
	{Name: "codex-memory", Synopsis: "the Codex agent-memory bridge", Lane: "cmd"},
	{Name: "codex-resume", Synopsis: "resume one or more Codex sessions with bounded deadlines, idle detection, and rollout reporting", Lane: "cmd"},
	{Name: "sessiondiag", Synopsis: "read-only Codex/fak abrupt-session evidence and log-store pressure diagnosis", Lane: "cmd"},
	{Name: "commit", Synopsis: "commit staged paths with the lane ship-stamp trailer enforced (safe shared-trunk commit)", Lane: "cmd"},
	{Name: "commit-subject-coverage", Synopsis: "commit-subject grammar coverage: scan the last N commits for stamp-conformant subjects, advisory floor", Lane: "cmd"},
	{Name: "complain", Synopsis: "file a dogfood complaint about an agent-experience friction", Lane: "cmd"},
	{Name: "component", Synopsis: "check component contracts and workload coverage from a declared root", Lane: "cmd"},
	{Name: "compute-trace", Synopsis: "capture or summarize bounded compute-event trace artifacts", Lane: "cmd"},
	{Name: "concept", Synopsis: "position, classify, or validate entries in the concept catalog", Lane: "cmd"},
	{Name: "concept-usage-score", Synopsis: "native concept-usage scorecard control-pane payload", Lane: "cmd"},
	{Name: "config", Synopsis: "show configuration guides and audit deployed fak posture against them", Lane: "cmd"},
	{Name: "coordinate", Synopsis: "render a deterministic multi-agent coordination plan or demo", Lane: "cmd"},
	{Name: "conflation-scorecard", Synopsis: "native provenance-honesty control-pane payload (WITNESSED vs OBSERVED), folded into conflation_debt", Lane: "cmd"},
	{Name: "conformance", Synopsis: "safety-conformance suite: re-adjudicate the ABI wire contract + dogfood verdict matrix vs the compiled kernel", Lane: "cmd"},
	{Name: "conpty", Synopsis: "ConPTY version preflight: resolve the OpenConsole/conpty.dll pair on PATH, refuse one below the 0xE9 floor", Lane: "cmd"},
	{Name: "console", Synopsis: "the native terminal control-pane spine: issue/loop/session lanes + garden/guard/overview", Lane: "cmd"},
	{Name: "coverage-matrix", Synopsis: "the generated model x backend coverage matrix", Lane: "cmd"},
	{Name: "cron", Synopsis: "project the in-kernel loop schedule down to a real OS scheduler unit (launchd/systemd/taskscheduler)", Lane: "cmd"},
	{Name: "customization-index", Synopsis: "audit the agent customization index for coverage, freshness, and due follow-up", Lane: "cmd"},
	{Name: "debug", Synopsis: "the context debugger: attach to a finished session as a core dump and demand-page its working set", Lane: "cmd"},
	{Name: "deepseekbench", Synopsis: "DeepSeek V4 Pro/Flash TTFT/TPOT scorecard: no-key dry-run fixture, live run gated by key + --spend", Lane: "cmd"},
	{Name: "demo", Synopsis: "zero-flag 60-second proof: run fak's offline scenario through the real kernel, one verdict per call class", Lane: "cmd"},
	{Name: "disambiguation", Synopsis: "query, audit, and generate the canonical concept disambiguation index", Lane: "cmd"},
	{Name: "dispatch", Synopsis: "the witness-gated issue-dispatch loop: spawn, ship #N, witness, close", Lane: "cmd"},
	{Name: "dispatch-aging", Synopsis: "anti-starvation dispatch order: effective weight = base priority + aging boost + hard starve deadline", Lane: "cmd"},
	{Name: "dispatch-conservation", Synopsis: "worker-unit conservation ledger: units_spent = accounted + leaked, so an ungraded death is a LEAK not silence", Lane: "cmd"},
	{Name: "dispatchlat", Synopsis: "dispatch-tick timing percentiles: fold tick_total phase latencies into p50/p90/p99 from the loop ledger", Lane: "cmd"},
	{Name: "execution-route", Synopsis: "compose harness, model, tool-policy, and context-continuity choices into one inspectable execution decision", Lane: "cmd"},
	{Name: "doctor", Synopsis: "operator diagnostic: run the answer-shape witness + the real kernel admit verdict and recommend", Lane: "cmd"},
	{Name: "done", Synopsis: "pre-claim self-check: tests, claims-lint, and loopgate/DOS witness in one GREEN/RED verdict", Lane: "cmd"},
	{Name: "dogfood-issues", Synopsis: "file dogfood issues from observed agent-experience defects", Lane: "cmd"},
	{Name: "dogfood-score", Synopsis: "scores the launched-session dogfooding loop: wired to run honestly + truthful self-report", Lane: "cmd"},
	{Name: "dojo", Synopsis: "the prediction-vs-reality gym: score each calibration lever's claimed vs realized behavior", Lane: "cmd"},
	{Name: "dojo-rsi", Synopsis: "the self-pacing dojo RSI loop (fold/propose/run/loop/trend)", Lane: "cmd"},
	{Name: "doomloop", Synopsis: "the two-axis doom-loop guard: classify workers on effort-spent vs verified progress and wire the correction", Lane: "cmd"},
	{Name: "dormancy", Synopsis: "classify dormant loop work from the loop ledger and emit metrics or JSON", Lane: "cmd"},
	{Name: "dream", Synopsis: "offline sleep pass over a core image: re-screen, pre-seal refuted witnesses, prune", Lane: "cmd"},
	{Name: "dup", Synopsis: "authoring-time dedup query: which tracked sites hold a block token-similar to a candidate (--file/--stdin)", Lane: "cmd"},
	{Name: "edit-tx", Synopsis: "transactional multi-file edit: apply a JSON batch of writes/deletes, run checks, roll back on failure", Lane: "cmd"},
	{Name: "egress", Synopsis: "prove the network-egress floor on one destination (the cloud-metadata / SSRF class)", Lane: "cmd"},
	{Name: "egresslist", Synopsis: "refresh bundled egress filter lists from pinned upstream sources", Lane: "cmd"},
	{Name: "enroll", Synopsis: "pin, show, or revoke this box's org trust anchor (the opt-in door to the org-policy plane)", Lane: "cmd"},
	{Name: "eve", Synopsis: "the impure shell over internal/evebridge: the Eve integration program's operator verbs (security preflight)", Lane: "cmd"},
	{Name: "experiments", Synopsis: "the experiments registry/runner over experiments/", Lane: "cmd"},
	{Name: "fanout", Synopsis: "fold the nightrun fan-out reuse ledger into a trend report", Lane: "cmd"},
	{Name: "feature", Synopsis: "query the unified self-feature catalog (dev facts, live tools, memory drivers, capability cards)", Lane: "cmd"},
	{Name: "fleet", Synopsis: "headless-worker fleet control surface: monitor/janitor/fold/replace/capacity/control (read-only by default)", Lane: "cmd"},
	{Name: "fleet-accounts", Synopsis: "fleet-wide account management across config-home seats", Lane: "cmd"},
	{Name: "fleet-trend", Synopsis: "fleet-status trend ledger: append a fleet_top snapshot as a tick, fold the trailing window (append/show)", Lane: "cmd"},
	{Name: "fleetcap", Synopsis: "dry-run fleet-capacity lens: workers required for a target issue-resolution rate via Little's law", Lane: "cmd"},
	{Name: "footprint", Synopsis: "the always-sent MCP tool-schema floor scorecard: price the tools/list floor offline via internal/mcpfootprint", Lane: "cmd"},
	{Name: "frontierswe", Synopsis: "FrontierSWE long-horizon time-to-solution benchmark surface (describe/eval/compare)", Lane: "cmd"},
	{Name: "fused", Synopsis: "the fused-kernel thesis in one turn: classify/explain/run a classical + a weight op through one deny floor", Lane: "cmd"},
	{Name: "garden", Synopsis: "the issue-garden: triage and groom GitHub issues (kind/priority/area)", Lane: "cmd"},
	{Name: "git-daily", Synopsis: "the scheduled daily git-hygiene tick: reap orphaned git locks, then consolidate the object DB; deduped per day", Lane: "gitgate"},
	{Name: "git-maint", Synopsis: "lock-aware safe object-DB consolidation: build midx + commit-graph, fold loose objects, never prune", Lane: "gitgate"},
	{Name: "gitd", Synopsis: "resident per-repo git broker over a unix socket: cached content-addressed reads, provenance on every answer", Lane: "gitbroker"},
	{Name: "glm52-prefill-sweep", Synopsis: "GLM-5.2 prefill-latency sweep: --dry-run prints the plan, --endpoint runs it and lands ledger artifacts", Lane: "cmd"},
	{Name: "go", Synopsis: "go build/vet/test passthrough that masks peers' untracked-sibling poison via a buildcheck -overlay", Lane: "cmd"},
	{Name: "goal", Synopsis: "manage canonical goals, lifecycle transitions, bindings, evidence, and execution topology", Lane: "cmd"},
	{Name: "goal-park", Synopsis: "park a supervisor goal claim: status|claim a goal identity to a single owner via the .fak/goal-park store", Lane: "cmd"},
	{Name: "godsplit-plan", Synopsis: "read-only boundary + hazard planner for a behavior-preserving Go god-file split (the /modularize --json plan)", Lane: "cmd"},
	{Name: "grafana", Synopsis: "export fak fleet metrics as a Grafana dashboard/datasource", Lane: "cmd"},
	{Name: "growthgate", Synopsis: "the unbounded-growth ratchet: census append-only ledger/log bloat vs byte budgets; --reap COLD logs", Lane: "cmd"},
	{Name: "manage", Synopsis: "wrap an agent harness: deny, repair, or quarantine tool calls through the managed-agent front door", Aliases: []string{"m", "guard"}, Lane: "gateway", Doc: "docs/manage.md"},
	{Name: "guard-accuracy", Synopsis: "score the guard reversibility classifier's accuracy against a labeled corpus: false-positive (benign escalated) + false-negative (dangerous let through) rates; the accuracy dual of guard-verdict-rsi, folded worst-first as a control-pane scorecard", Lane: "cmd"},
	{Name: "guard-audit", Synopsis: "prune mirrored guard journals from the vault by age/count (dry-run unless --apply)", Lane: "cmd"},
	{Name: "guard-commit-gate", Synopsis: "internal: Claude Code PreToolUse commit-gate actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-goal-question", Synopsis: "internal: enforce the active-goal question boundary before AskUserQuestion runs", Lane: "cmd"},
	{Name: "guard-precompact", Synopsis: "internal: Claude Code PreCompact hook actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-sessionstart", Synopsis: "internal: Claude Code SessionStart hook actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-rsi-scorecard", Synopsis: "native control-pane payload for guard RSI loop maturity and realized value", Lane: "cmd"},
	{Name: "guard-stophook", Synopsis: "internal: Claude Code Stop hook actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-stops", Synopsis: "operator tally of the typed Stop-hook decision ledger: clean stops, bounded stand-downs, fail-open stops", Lane: "cmd"},
	{Name: "guard-stops-slack", Synopsis: "durable update-in-place Slack scoreboard feeder for the guard Stop-decision ledger", Lane: "cmd"},
	{Name: "guard-verdict-rsi", Synopsis: "the guard verdict RSI loop: fold the decision journal, score verdict-quality, keep on gain", Lane: "cmd"},
	{Name: "harness", Synopsis: "compose, inspect, resolve, and verify reusable agent harness stacks", Lane: "cmd"},
	{Name: "harness-debt-dispatch", Synopsis: "harness-strength verdict -> backlog: file one deduped deletion issue per REDUNDANT/HOBBLING HARD scaffold", Lane: "cmd"},
	{Name: "headless-lint", Synopsis: "scan an agent's final output for operator-directed notes (push?/TODO?) and type each to a remediation class", Lane: "cmd"},
	{Name: "headroom", Synopsis: "the context-compression seam: shrink tool outputs/logs/files before they reach the model", Lane: "cmd"},
	{Name: "help", Synopsis: "the compact help: task-grouped overview, per-verb depth (help <verb>), --all index, --full wall", Aliases: []string{"-h", "--help"}, Lane: "cmd"},
	{Name: "hook", Synopsis: "spawned-hook decide (the A/B baseline transport; reads call.json on stdin)", Lane: "cmd"},
	{Name: "hooklat", Synopsis: "guard-hook latency rollup: fold the DOS hook-observation streams into percentiles, gate the p99 tail", Lane: "cmd"},
	{Name: "hooks", Synopsis: "the commit-boundary git-hook gates in one process (pre-commit / commit-msg)", Lane: "cmd"},
	{Name: "horizon-recovery", Synopsis: "recover stranded / stalled long-horizon work", Lane: "cmd"},
	{Name: "host-crash", Synopsis: "watch the host for crash/reboot events and append a durable JSONL signal (--once scans once)", Lane: "cmd"},
	{Name: "hostdiag", Synopsis: "census, correlate, or trend host diagnostic evidence", Lane: "cmd"},
	{Name: "host-relaunch-broker", Synopsis: "drain the control-plane->desktop relaunch spool and launch the queued Windows Terminal sessions", Lane: "cmd"},
	{Name: "hwgate-lint", Synopsis: "scan agent output for local-hardware stops (no GPU here) and redirect each to a sanctioned compute node", Lane: "cmd"},
	{Name: "hygiene", Synopsis: "the whole-tree hygiene gates in one process (the --audit-tree twin of fak hooks)", Lane: "cmd"},
	{Name: "idea-scout", Synopsis: "research-to-issue feeder: score arXiv/GitHub/HN/Reddit hits, dedupe, plan issues; --live FILES them for real", Lane: "cmd", Doc: "docs/idea-scout.md"},
	{Name: "idempotency", Synopsis: "durable intent for non-idempotent ops: run/selfcheck; UNKNOWN_APPLIED recovery via status/resolve", Lane: "cmd", Doc: "docs/cli-reference.md"},
	{Name: "index", Synopsis: "queryable self-index: lane/leaf/docs/claims/verbs/refs (query, don't survey)", Aliases: []string{"devindex"}, Lane: "devindex", Doc: "AGENTS.md"},
	{Name: "info", Synopsis: "the live fak-info overlay: poll a gateway's /debug/vars and print one plain-words line per tick", Lane: "cmd"},
	{Name: "init", Synopsis: "scaffold a minimal, valid fak.toml deployment manifest for a new workspace", Lane: "cmd"},
	{Name: "intent", Synopsis: "task-claim collision check: intent leases (claim/release/list) so two agents don't both fix the same issue", Lane: "cmd"},
	{Name: "issue", Synopsis: "the generated-issue contract: review machine-created GitHub issue candidates before sync", Lane: "cmd"},
	{Name: "issue-contract-repair", Synopsis: "audit machine-created GitHub issues against the generated-issue contract, emit repair actions (json/markdown)", Lane: "cmd"},
	{Name: "knownbad", Synopsis: "record, match, claim, resolve, revoke, and report live known-bad signatures", Lane: "cmd"},
	{Name: "kvbm", Synopsis: "cost-aware KV eviction validation: replay/trace a policy corpus vs the offline oracle (cost-aware >= LRU)", Lane: "cmd"},
	{Name: "lab", Synopsis: "the GPU-lab status surface: per-state / per-class node counts + readiness", Lane: "cmd"},
	{Name: "launch", Synopsis: "the reversible provider launch shim: route claude/codex through fak (install/uninstall/default/enable/status)", Lane: "cmd"},
	{Name: "leaseref", Synopsis: "cross-machine lease visibility: read the refs/fak/locks/* lease ref namespace", Lane: "cmd"},
	{Name: "learning-debt-dispatch", Synopsis: "learning-scorecard -> backlog: file capped triage issues for HARD learning-debt defects", Lane: "cmd"},
	{Name: "lifecycle", Synopsis: "inspect and control phase-aware capability lifecycle state", Lane: "cmd"},
	{Name: "learning-observation", Synopsis: "store and trace observation -> candidate -> witness -> verdict learning records", Lane: "cmd"},
	{Name: "learning-mesh", Synopsis: "compile learning records into deterministic cross-envelope transfer candidates", Lane: "cmd"},
	{Name: "lint", Synopsis: "the static tool linter: the definition-time dual of the kernel's call-time re-checks", Lane: "cmd"},
	{Name: "llmd-smoke", Synopsis: "smoke-test an llm-d OpenAI-compatible engine: /v1/models, one chat completion, engine-tagged metrics", Aliases: []string{"llm-d-smoke"}, Lane: "cmd"},
	{Name: "logvault", Synopsis: "central chain-aware backup of every durable fak log store into one vault (plan/capture/verify/sources)", Lane: "cmd"},
	{Name: "loop", Synopsis: "the durable long-running-loop ledger (append/run/status/admit)", Lane: "cmd"},
	{Name: "loop-index-scorecard", Synopsis: "fold the six agentic-loop stages (orient->plan->act->verify->ship->learn) into one loop-index + loopindex_debt", Lane: "cmd"},
	{Name: "loop-map", Synopsis: "the agentic-loop map: the stages, levers, and child issues of the dev-experience epic", Lane: "cmd"},
	{Name: "loop-score", Synopsis: "score a single loop run's outcome", Lane: "cmd"},
	{Name: "ls", Synopsis: "alias for 'fak model ls': list known model aliases + cache status", Lane: "cmd"},
	{Name: "macbench", Synopsis: "the Mac local-inference benchmark suite (run/watch/watch-status/recover)", Lane: "cmd"},
	{Name: "macfit", Synopsis: "compute whether a model fits in Mac unified memory (weights + KV/context vs capacity minus reserve)", Lane: "cmd"},
	{Name: "marketing", Synopsis: "the marketing Slack surface", Lane: "cmd"},
	{Name: "maturity", Synopsis: "the feature-lifecycle maturity ladder report", Lane: "cmd"},
	{Name: "mcp-filter-proof", Synopsis: "prove MCP tool filtering with an offline contract or a model-driven live A/B", Lane: "cmd"},
	{Name: "memgate", Synopsis: "host-memory admission gate: snapshot free RAM, admit/deny a --require-gb load, --wait to poll until it frees", Lane: "cmd"},
	{Name: "memory", Synopsis: "the memory-operation algebra: author a render/clean/compact/dream Op pipeline (drivers/explain/run)", Lane: "cmd"},
	{Name: "memory-read", Synopsis: "render an agent-memory store's MEMORY.md + per-fact bodies as one bounded digest (the read side of the store)", Lane: "cmd"},
	{Name: "memory-stability-governor", Synopsis: "grade agent-memory reload stability over a replay trajectory against a per-point tau and drift budget", Lane: "cmd"},
	{Name: "merge", Synopsis: "shared-trunk merge dry-run: predict empty-net-diff vs clean changed files vs conflicts", Lane: "cmd"},
	{Name: "micro", Synopsis: "the in-process Go microagent runtime front door: run one agent on Mock (host boots a fleet, --dry-run plans)", Lane: "cmd"},
	{Name: "microbench", Synopsis: "per-agent RSS/CPU density for the in-process microagent runtime: in-process cells vs a guarded-CLI baseline", Lane: "cmd"},
	{Name: "milestone", Synopsis: "the roadmap/milestone report (discrete-deliverable epics, completion %)", Lane: "cmd"},
	{Name: "milestone-scorecard", Synopsis: "native milestone-roadmap scorecard control-pane payload", Lane: "cmd"},
	{Name: "mlp-score", Synopsis: "MLP (first-lovable-cut) grade: pure internal/mlpscore fold over a committed HEAD snapshot (--json/--markdown)", Lane: "cmd"},
	{Name: "mode-debt-dispatch", Synopsis: "permission-regime fan-out: one deduped issue per HARD un-lifted dial from the mode-debt scorecard", Lane: "cmd"},
	{Name: "model", Synopsis: "resolve an hf:// URI to a locally cached file path (Hub download + SHA256 verify)", Lane: "cmd"},
	{Name: "model-default", Synopsis: "report the default model identity and fold dated Qwen3.8 evidence", Lane: "cmd"},
	{Name: "model-observe", Synopsis: "proxy and summarize model performance observations or verify cache-state transitions", Lane: "cmd"},
	{Name: "multisubmit", Synopsis: "multi-submission planner: lay out N profiles for a resolved issue, seat round-robin over the rotation pool", Lane: "cmd"},
	{Name: "native-benchmarks", Synopsis: "report required native benchmark witnesses and fail when obligations are missing", Lane: "cmd"},
	{Name: "native-first-lint", Synopsis: "lint prose for native-inference stops that should route to sanctioned compute instead", Lane: "cmd"},
	{Name: "native-performance", Synopsis: "query, compare, and profile-next the committed native Qwen3.8 optimization graph", Lane: "cmd"},
	{Name: "negate", Synopsis: "negation operator: detect/resolve/reframe a negative over internal/negframe (positive-complement L2 registry)", Lane: "cmd"},
	{Name: "new-leaf", Synopsis: "scaffold a new leaf package at a layering tier, optionally registering its defconfig blank-import", Lane: "cmd"},
	{Name: "new-model", Synopsis: "scaffold a new model adapter/leaf", Lane: "cmd"},
	{Name: "news", Synopsis: "the news Slack surface for source-linked external industry/SOTA/OSS research updates", Lane: "cmd"},
	{Name: "nightrun", Synopsis: "run it all night: the local-capability-aware data-collection door (next/plan/run/ledger/caps)", Lane: "cmd"},
	{Name: "node", Synopsis: "the compute-node registry (register/list nodes)", Lane: "cmd"},
	{Name: "node-compare", Synopsis: "compare fleet compute-node result files side by side into one cross-node report", Lane: "cmd"},
	{Name: "nodeusage", Synopsis: "the compute-node-usage Slack surface for #node-usage", Lane: "cmd"},
	{Name: "operator", Synopsis: "the human pacing brief: fold cadence/program/milestone into human/agent/watch/background buckets", Lane: "cmd"},
	{Name: "org", Synopsis: "inspect org-policy posture and which control channel owns each capability", Lane: "cmd"},
	{Name: "orient", Synopsis: "task-scoped convention orientation for path globs: lane, arch tier, owning tests, stamp, and live lease", Lane: "devindex"},
	{Name: "opt", Synopsis: "the optimization-fuser / RSI opt-target loop", Lane: "cmd"},
	{Name: "plan-audit", Synopsis: "audit plan docs for drift over a glob (json/markdown, --check exits nonzero on drift)", Lane: "cmd"},
	{Name: "policy", Synopsis: "the deployable capability floor: --dump | --check a policy manifest", Lane: "cmd"},
	{Name: "popularization-tickets", Synopsis: "emit the concept-popularization ticket set as JSON, lane TSV, or issue-body files", Lane: "cmd"},
	{Name: "preflight", Synopsis: "adjudicate one tool call against a policy (ALLOW/DENY by structure, no model in the loop)", Lane: "cmd"},
	{Name: "process-guard", Synopsis: "the host process-resource guard: detect / reap runaway or leaking processes", Lane: "cmd"},
	{Name: "product", Synopsis: "the product-direction Slack surface for #product", Lane: "cmd"},
	{Name: "product-scorecard", Synopsis: "native product scorecard control-pane payload: standing chart, most-critical areas, coverage gaps", Lane: "cmd"},
	{Name: "profile", Synopsis: "host-aware profiler: capture CPU + allocation profiles of a package's benchmarks (Windows->WSL)", Lane: "cmd"},
	{Name: "program", Synopsis: "the ongoing-optimization program report (a frontier + a trend, never a completion %)", Lane: "cmd"},
	{Name: "progress", Synopsis: "detect fleet progress stalls from recent and baseline windows", Lane: "cmd"},
	{Name: "project", Synopsis: "the ProjectsV2 board control-pane fold: report/verdict/Slack-ready board dimension, not a write-only sync", Lane: "cmd"},
	{Name: "propagation-debt-dispatch", Synopsis: "propagation-scorecard -> backlog: file one deduped issue per HARD un-propagated convention gap", Lane: "cmd"},
	{Name: "propagation-scorecard", Synopsis: "score how far each proven scorecard convention has fanned out across the family (propagation debt)", Lane: "cmd"},
	{Name: "provider-cost", Synopsis: "import, report, and reconcile provider cost ledgers against registered sessions", Lane: "cmd"},
	{Name: "ps", Synopsis: "the read-only process table: one aligned row per live served session", Lane: "cmd", Doc: "docs/operator-control-plane.md"},
	{Name: "public-scrub", Synopsis: "public-release safety audit over staged/range/tree/message content (the PUBLIC_LEAK gate family)", Lane: "cmd"},
	{Name: "performance-rsi-scorecard", Synopsis: "render the native performance RSI scorecard and evidence debt", Lane: "cmd"},
	{Name: "pull", Synopsis: "alias for 'fak model pull': the Ollama-style run-by-name model download", Lane: "cmd"},
	{Name: "qa-process-debt-dispatch", Synopsis: "qa-process scorecard -> backlog: file one deduped issue per HARD qa_process_debt gap (revert/coverage)", Lane: "cmd"},
	{Name: "quality", Synopsis: "the missing-middle quality ladder (run|explain): score a case reference-vs-engine with a replay bundle", Lane: "cmd"},
	{Name: "question-ledger", Synopsis: "deterministic labeling authority for docs/questions/asked.jsonl that the /question-loop skill defers to", Lane: "cmd"},
	{Name: "quantbench", Synopsis: "run the quantization benchmark contract from JSON input or emit its self-test matrix", Lane: "cmd"},
	{Name: "quantwatch", Synopsis: "collect bounded arXiv and GitHub quantization updates, offline or live", Lane: "cmd"},
	{Name: "qwen36-node-reports", Synopsis: "import Qwen-3.6 node-report archives from the Taildrop inbox and extract them into the per-node report tree", Lane: "cmd"},
	{Name: "qwen36-parity-witness-gate", Synopsis: "gate Qwen-3.6 speed parity against a witness JSON (--require-witness fails closed; --min-ratio sets the bar)", Lane: "cmd"},
	{Name: "readme-visual-audit", Synopsis: "README visual-quality audit: score the repo README's visual/asset health (json)", Lane: "cmd"},
	{Name: "recall", Synopsis: "persist a finished session as a core dump and reload it in a fresh store (quarantine survives)", Lane: "cmd"},
	{Name: "recover", Synopsis: "closed-vocabulary refusal recovery: print or run the safe commands for a reason token", Lane: "cmd"},
	{Name: "refactor-verify", Synopsis: "read-only proof that a god-split / code-motion refactor dropped no top-level declaration (--expect-motion)", Lane: "cmd"},
	{Name: "relay", Synopsis: "the perpetual-session baton, read half: print what a fresh relay leg would receive from a baton file", Lane: "cmd"},
	{Name: "release", Synopsis: "the release front door over the tools/release_*.py helpers (status/cut/tag/publish/...)", Lane: "cmd"},
	{Name: "release-lock", Synopsis: "the process-safe release lock every VERSION/tag mutation must take (acquire/release/status)", Lane: "cmd"},
	{Name: "release-staleness", Synopsis: "the publish-freshness signal: how far the latest @latest tag lags HEAD (commits + days)", Lane: "cmd"},
	{Name: "rename-concept", Synopsis: "plan/apply a tree-wide concept rename: case-form variants, mechanical vs holdout triage, residual rescan", Lane: "cmd"},
	{Name: "replay", Synopsis: "explicit spelling of the trace-replay path (fak run --trace)", Lane: "cmd"},
	{Name: "repo-hygiene-scorecard", Synopsis: "native repo-hygiene scorecard control-pane payload (hygiene_debt)", Lane: "cmd"},
	{Name: "resume", Synopsis: "the deterministic resume-cache decision: what happens to the prompt cache on resume, and what to do", Lane: "cmd"},
	{Name: "rollup", Synopsis: "the executive roll-up snapshot regenerated from the tree", Lane: "cmd"},
	{Name: "route", Synopsis: "the model-routing oracle: per-aspect + ensemble model routing for one classified subject", Lane: "cmd"},
	{Name: "routebench", Synopsis: "the offline routing benchmark: a per-aspect+ensemble policy vs a single-model baseline", Lane: "cmd"},
	{Name: "run", Synopsis: "run an agent turn (or a recorded trace via --trace) through the kernel", Lane: "cmd"},
	{Name: "runtime-capabilities", Synopsis: "report the capabilities exposed by the current runtime build", Lane: "cmd"},
	{Name: "rungstats", Synopsis: "stats over the verification-ladder rungs", Lane: "cmd"},
	{Name: "savings", Synopsis: "audit, gate, and lint the Track-2 OBSERVED-$ cache-savings ledger", Lane: "cmd"},
	{Name: "savings-vector", Synopsis: "the savings-vector report: realized token/cost savings by lever", Lane: "cmd"},
	{Name: "schedscan", Synopsis: "fleet scheduled-task health: decode each Windows LastTaskResult to meaning + fix; surface failing tasks first", Lane: "cmd"},
	{Name: "schedule-held", Synopsis: "evaluate held hardware-job schedules and measure admission-policy overhead", Lane: "cmd"},
	{Name: "score", Synopsis: "parent verb over the meta-scorecards / RSI loops: `fak score <name>` routes to each legacy scorecard handler", Lane: "cmd"},
	{Name: "scoreboard", Synopsis: "the scoreboard Slack surface for #scoreboard", Lane: "cmd"},
	{Name: "scorecard", Synopsis: "the scorecard control pane: every metric's debt + grade + trend", Lane: "cmd"},
	{Name: "search", Synopsis: "search the repository text corpus with bounded results and JSON output", Lane: "cmd"},
	{Name: "scratch-janitor", Synopsis: "plan or remove abandoned session scratch directories with age and resume guards", Lane: "cmd"},
	{Name: "self-update", Synopsis: "converge a built-from-source fak binary on origin/main", Lane: "cmd"},
	{Name: "serve", Synopsis: "run the OpenAI-compatible gateway in front of a local or remote model", Lane: "gateway"},
	{Name: "serve-wiring", Synopsis: "audit fak serve flag -> gateway.Config -> runtime-read wiring", Lane: "gateway"},
	{Name: "service", Synopsis: "run or install fak as a long-lived OS service (Windows service dispatcher + install/status)", Lane: "cmd"},
	{Name: "session", Synopsis: "the operator control surface for a served session: read live DRIVE state, cancel/update in flight", Lane: "cmd", Doc: "docs/operator-control-plane.md"},
	{Name: "session-audit", Synopsis: "audit agent-session JSONL transcripts (discover/audit/summary/deep) into model-mix + long-context reports", Lane: "cmd"},
	{Name: "sessionjournal", Synopsis: "crash-survivable session journal: boot-epoch fold to LIVE/CRASHED/STALE/CLOSED (open/beat/close/report)", Lane: "cmd", Doc: "docs/notes/CONCEPT-SESSION-CRASH-JOURNAL-BOOT-EPOCH-2026-07-09.md"},
	{Name: "sessions", Synopsis: "ingest + score this host's agent transcripts (the session->outcome learn loop)", Lane: "cmd"},
	{Name: "shadowgit", Synopsis: "per-step write ledger via a SEPARATE git dir; the repo's .git is never touched (baseline/snapshot/status)", Lane: "cmd"},
	{Name: "shellprov", Synopsis: "record a privacy-safe shell-launch provenance receipt", Lane: "cmd"},
	{Name: "sidecar", Synopsis: "the sidecar pane: read-only fold over sessions / accounts / lanes / context posture, on terminal and Slack", Lane: "cmd"},
	{Name: "signal", Synopsis: "job control for a running session (pause/resume/stop/steer) over the control plane", Lane: "cmd", Doc: "docs/operator-control-plane.md"},
	{Name: "signals", Synopsis: "plain-English behavioral signals (NL prompt + verdict schema) judged over an agent's turns (validate/plan)", Lane: "cmd"},
	{Name: "skill", Synopsis: "the queried skill-loader operator surface over .claude/skills (+ MCP resolver): query/residency/swap", Lane: "cmd"},
	{Name: "skill-effectiveness-scorecard", Synopsis: "native skill-pack effectiveness control-pane payload", Lane: "cmd"},
	{Name: "slack", Synopsis: "debug + use the whole Slack surface from one place (check/health/beat/walk/refresh/send/outbox)", Lane: "cmd"},
	{Name: "snapshot", Synopsis: "dump/restore any primitive on the loops ladder to a portable sha256-integrity bundle", Lane: "cmd"},
	{Name: "sota", Synopsis: "prior-art lookup before writing a kernel: the SOTA stack, link, route, oracle, and papers for an op or file", Lane: "cmd"},
	{Name: "sota-coverage-scorecard", Synopsis: "native SOTA-coverage scorecard: prior-art coverage as sota-debt, --check exits nonzero on HARD debt", Lane: "cmd"},
	{Name: "speed-ab", Synopsis: "run a captured speed A/B manifest and emit its benchmark witness", Lane: "cmd"},
	{Name: "spend", Synopsis: "cross-account spend rollup with provenance labels; --check gates unlabeled spend figures", Lane: "cmd"},
	{Name: "stale-work", Synopsis: "discover and rank stale repository work against open issues within a bounded scan", Lane: "cmd"},
	{Name: "stallscan", Synopsis: "host stall self-monitor: read low-usage churn signals to fingerprint a lockup; snapshot or --watch loop", Lane: "cmd"},
	{Name: "steer", Synopsis: "steer open pull requests through guarded commit-audit and merge readiness checks", Lane: "cmd"},
	{Name: "steering", Synopsis: "the steerability Slack surface for #steering-guard (status/report/alert/pin)", Lane: "cmd"},
	{Name: "stopfailure", Synopsis: "operator surface for .dos/stop-failures breaker markers (plan/reset-stale/clear-reviewed)", Lane: "cmd"},
	{Name: "study", Synopsis: "add, retrieve, or search local content-addressed study receipts", Lane: "cmd"},
	{Name: "study-forge", Synopsis: "capture and validate deterministic paginated GitHub forge corpora", Lane: "cmd"},
	{Name: "study-adjacency", Synopsis: "validate and render the bounded related-system runtime study manifest", Lane: "cmd"},
	{Name: "study-classify", Synopsis: "classify a validated forge corpus into deterministic dispositions and evidence-backed mechanism clusters", Lane: "cmd", Doc: "docs/cli-reference.md"},
	{Name: "study-inventory", Synopsis: "render a deterministic local checkout map for exhaustive study-repo passes", Lane: "cmd"},
	{Name: "study-link", Synopsis: "build and validate a deterministic evidence ledger joining study clusters to witnessed FAK work", Lane: "studylink"},
	{Name: "study-monitor", Synopsis: "report recurring research sources that are due for another study pass", Lane: "cmd"},
	{Name: "study-priority", Synopsis: "build and validate a versioned hard-gated dependency queue over uncovered actionable study joins", Lane: "studyprio", Doc: "docs/cli-reference.md"},
	{Name: "study-tickets", Synopsis: "construct and validate a zero-leftover ticket closure over the selected study queue", Lane: "studytickets", Doc: "docs/cli-reference.md"},
	{Name: "superloop", Synopsis: "operator-intent meta-loop: walk a set of member loops/scorecards/gardens worst-first (list/explain/walk)", Lane: "cmd"},
	{Name: "support", Synopsis: "per-cell support read-out: one line per model x backend cell (rung.regime.target.next-action)", Lane: "cmd"},
	{Name: "support-maturity-scorecard", Synopsis: "native support-maturity payload: fold the model x backend coverage matrix into a grade", Lane: "cmd"},
	{Name: "sweep", Synopsis: "drive a dirty multi-session tree toward zero: report by lane, then --apply one lane group", Lane: "cmd"},
	{Name: "swebench", Synopsis: "SWE-bench Verified benchmarking (describe/eval/compare)", Lane: "cmd"},
	{Name: "sync", Synopsis: "safe sync/push for a dirty shared worktree; never pull/stash/reset", Lane: "cmd"},
	{Name: "task", Synopsis: "the process-local task manager snapshot (hardware/runtime + task/step/concept progress + ETA)", Lane: "cmd"},
	{Name: "tasks", Synopsis: "the shared task list folded to a typed table with lease-gated claims (table | sample)", Lane: "cmd"},
	{Name: "temp-artifacts", Synopsis: "preview or reap aged temporary artifacts through quarantine and post-move rechecks", Lane: "cmd"},
	{Name: "terminal-relief", Synopsis: "measure terminal-host pressure and safely relaunch restorable dashboards when armed", Lane: "cmd"},
	{Name: "test", Synopsis: "host-aware test runner: resolve the right go test invocation (Windows->WSL via test.ps1)", Lane: "cmd"},
	{Name: "test-quality", Synopsis: "score test-source defects against a shrinks-only baseline and emit repair candidates", Lane: "cmd"},
	{Name: "tier-calibrate", Synopsis: "outcome-calibration fold over recorded tier decisions: propose threshold moves from witnessed outcomes", Lane: "cmd"},
	{Name: "token-defaults-scorecard", Synopsis: "native token-saving-defaults control-pane payload", Lane: "cmd"},
	{Name: "token-profile", Synopsis: "price a forecast of uncached/cached input and reserved output tokens into USD + scheduler weight units", Lane: "tokenprofile"},
	{Name: "tool-coverage-audit", Synopsis: "audit load-bearing tool coverage for a workspace against the advisory minimum floor", Lane: "cmd"},
	{Name: "tool-width", Synopsis: "fold tool-width observations and ratchet batched-turn rate against a baseline", Lane: "cmd"},
	{Name: "toolproc", Synopsis: "the kernel's process table for tool calls: fold a lifecycle journal into deadline/stall/orphan/kill verdicts", Lane: "cmd", Doc: "docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md"},
	{Name: "top", Synopsis: "= fak ps --watch (the live process-table top mode)", Lane: "cmd", Doc: "docs/operator-control-plane.md"},
	{Name: "traj", Synopsis: "the trajectory-corpus toolkit (similar/cluster/score/gc/export) over recorded turns", Lane: "cmd"},
	{Name: "trajctl", Synopsis: "trajectory-control objective lifecycle (declare/close/list/curve/score/scorers) over the trajctl ledger", Lane: "cmd"},
	{Name: "trajectory", Synopsis: "audit Claude and Codex trajectory logs into scrubbed JSONL and Markdown summaries", Lane: "cmd"},
	{Name: "trajquery", Synopsis: "scoped SQL SELECT over your own trajectory corpus; the validator refuses out-of-scope queries (run/validate)", Lane: "cmd"},
	{Name: "tree-doctor", Synopsis: "the worktree doctor: detect and prune stray / dead git worktrees", Lane: "cmd"},
	{Name: "trunk-build-probe", Synopsis: "read-only diagnosis of whether the release gate's red trunk is a forgotten `git add` vs a real break", Lane: "cmd"},
	{Name: "trunk-red", Synopsis: "fold the trunk-red witness ledger into distinct shared build breaks, worst (most clones stuck) first", Lane: "cmd"},
	{Name: "turnavoid", Synopsis: "replay whole-model-turn avoidance traces with strict JSONL input and net-true attribution", Lane: "cmd"},
	{Name: "turntax", Synopsis: "turn-tax A/B: price the extra error-code model turns a SOTA loop fires vs fak's one-shot", Lane: "cmd"},
	{Name: "ui-quality-scorecard", Synopsis: "native terminal UI/UX control-pane payload (ui_quality_debt)", Lane: "cmd"},
	{Name: "unwired-debt-dispatch", Synopsis: "unwired-scorecard -> backlog: file one deduped issue per orphaned internal package", Lane: "cmd"},
	{Name: "unwired-scorecard", Synopsis: "score which code-complete internal packages are not wired into a runnable CLI surface", Lane: "cmd"},
	{Name: "usage", Synopsis: "read side of the CLI-invocation journal: how fak itself has been invoked (totals/errors/timing, per-verb)", Lane: "cmd"},
	{Name: "up", Synopsis: "start the local fak application surface and report readiness", Lane: "cmd"},
	{Name: "value-chain", Synopsis: "audit a value-chain manifest against observed artifacts and evidence", Lane: "cmd"},
	{Name: "vcache", Synopsis: "the virtual provider-cache status/proof surface (status/prove/prove-telemetry)", Lane: "cmd"},
	{Name: "version", Synopsis: "print the fak version", Aliases: []string{"-v", "--version"}, Lane: "cmd"},
	{Name: "waiting", Synopsis: "the waiting-on-human queue: fold blocked-on-operator loop events into tickets ranked by cost-of-delay", Lane: "cmd"},
	{Name: "watchdog", Synopsis: "operator surface over fak's default watchdog monitors: status (probe + heal-state) / heal (restart dead)", Lane: "cmd"},
	{Name: "watchdog-audit-health", Synopsis: "report whether the resume-watchdog audit loop is alive and productive", Lane: "cmd"},
	{Name: "watchdog-audit-run", Synopsis: "run one bounded resume-watchdog audit pass", Lane: "cmd"},
	{Name: "webbench", Synopsis: "frontier web/browser-agent benchmarking (describe/eval/compare)", Lane: "cmd"},
	{Name: "whats-changed", Synopsis: "peer code-diff readout: commits and files under target paths since a session/base ref", Lane: "cmd"},
	{Name: "wiki", Synopsis: "the witness-verified repo wiki: 'structure' emits the self-index page tree, 'verify' checks a page's cites", Lane: "cmd"},
	{Name: "windowgate", Synopsis: "the no-desktop-popup ratchet: scan for console-popup-prone automation, non-zero on violations", Lane: "cmd"},
	{Name: "wip", Synopsis: "checkpoint/restore the working-tree delta over refs/fak/wip/*: snapshot gc-safe, list, re-materialize", Lane: "cmd"},
	{Name: "work-delivery", Synopsis: "track, transition, and diagnose work-unit progress across delivery stages", Lane: "cmd"},
	{Name: "workflow", Synopsis: "keep ultracode Workflow scripts fak-native: lint for self-index/memory/shared-path use + seed the template", Lane: "cmd"},
	{Name: "workflow-audit", Synopsis: "classify .github/workflows branch/tag refs against the branch-role contract; gate unclassified dev-path refs", Lane: "cmd"},
	{Name: "workpattern", Synopsis: "mine recurring work patterns from source or recorded trajectories", Lane: "cmd"},
	{Name: "worktree", Synopsis: "guarded on-trunk-safe worktree verbs; 'witness' runs a check in a transient detached worktree at origin/main", Lane: "cmd"},
	{Name: "worktype", Synopsis: "attribute session token spend and outcomes to classified work types", Lane: "cmd"},
}

// Verbs returns the structured CLI-verb catalog, sorted by name. It is a live VIEW,
// not a frozen list: COVERAGE comes from the cmd/fak/main.go dispatch switch (every
// verb the binary actually routes), and the curated verbManifest supplies QUALITY
// (synopsis / owning lane / alias grouping / doc) for the verbs it names. A dispatched
// verb with no curated entry still appears, carrying a fallback synopsis that points at
// its own --help — so the catalog can never silently fall behind the binary the way a
// hand-maintained list does. When main.go cannot be read (an installed binary outside a
// repo), it falls back to the curated overlay alone. The cmd/ usage generator and
// `fak index verbs` (CLI + MCP) consume it.
func (c *Catalog) Verbs() []Verb {
	tokens := c.liveDispatchTokens()
	if len(tokens) == 0 {
		out := make([]Verb, len(verbManifest))
		copy(out, verbManifest)
		for i := range out {
			out[i].Tier = tierFor(out[i].Name)
		}
		sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
		return out
	}
	overlay := map[string]Verb{}
	for _, v := range verbManifest {
		for _, sp := range v.Spellings() {
			overlay[strings.ToLower(sp)] = v
		}
	}
	seen := map[string]bool{}
	var out []Verb
	for _, tok := range tokens {
		if v, ok := overlay[tok]; ok {
			if seen[v.Name] {
				continue // a curated verb reached via one of its alias spellings
			}
			seen[v.Name] = true
			v.Tier = tierFor(v.Name)
			out = append(out, v)
			continue
		}
		if seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, Verb{Name: tok, Synopsis: "not yet cataloged — `fak " + tok + " -h` for usage", Tier: tierFor(tok)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// liveDispatchTokens returns the lowercased verb tokens (canonical + alias spellings)
// of cmd/fak/main.go's top-level dispatch switch — the COVERAGE source Verbs() derives
// from. Nil when main.go cannot be read (an installed binary outside a repo), which
// sends Verbs() to its curated-overlay fallback. It reuses mainDispatchVerbs, the same
// brace-depth scan the freshness drift detector uses, so the catalog and the detector
// can never disagree on what a verb is.
func (c *Catalog) liveDispatchTokens() []string {
	b, err := os.ReadFile(filepath.Join(c.Root, "cmd", "fak", "main.go"))
	if err != nil {
		return nil
	}
	return mainDispatchVerbs(b)
}

// VerbByName returns the manifest entry matching the given (case-insensitive) token
// against the verb's canonical name OR any alias, and ok=false when nothing routes.
// The freshness gate uses this to ask "does main.go's case <tok> have a manifest
// entry?" without re-deriving the alias set at the call site.
func (c *Catalog) VerbByName(name string) (Verb, bool) {
	return manifestVerbByName(name)
}

// manifestVerbByName is the package-level manifest lookup behind VerbByName,
// factored out so TierOf (tiers.go) can canonicalize an alias spelling without a
// Catalog — the tier answer must not require a readable repo.
func manifestVerbByName(name string) (Verb, bool) {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return Verb{}, false
	}
	for _, v := range verbManifest {
		for _, sp := range v.Spellings() {
			if strings.ToLower(sp) == n {
				return v, true
			}
		}
	}
	return Verb{}, false
}

// SearchVerbs returns the catalog verbs matching the query, lexically scored (a name
// or alias hit weighs most, then the lane, then the synopsis) and ranked best-first.
// An empty query returns the full catalog in name order — `fak index verbs` with no
// term lists every verb, matching the leaf-search convention. It searches the live
// derived catalog (Verbs()), so a dispatched-but-uncurated verb is still found.
func (c *Catalog) SearchVerbs(query string) []Verb {
	all := c.Verbs()
	toks := tokens(query)
	if len(toks) == 0 {
		return all
	}
	type scored struct {
		v Verb
		s int
	}
	var hits []scored
	for _, v := range all {
		spellings := v.Spellings()
		names := strings.ToLower(strings.Join(spellings, " "))
		lane, syn := strings.ToLower(v.Lane), strings.ToLower(v.Synopsis)
		score := 0
		for _, tk := range toks {
			// An EXACT verb-name/alias match dominates: `fak index verbs guard` must
			// rank the `guard` verb above siblings like `guard-precompact` that merely
			// CONTAIN "guard" in their name or synopsis.
			for _, sp := range spellings {
				if strings.ToLower(sp) == tk {
					score += 10
					break
				}
			}
			if strings.Contains(names, tk) {
				score += 3
			}
			if strings.Contains(lane, tk) {
				score += 2
			}
			if strings.Contains(syn, tk) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{v, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].v.Name < hits[j].v.Name
	})
	out := make([]Verb, len(hits))
	for i, h := range hits {
		out[i] = h.v
	}
	return out
}
