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
	{Name: "ablate", Synopsis: "self-ablation: replay one frozen trace under N feature configs, one row per arm", Lane: "cmd"},
	{Name: "accounts", Synopsis: "config-home registry: every CLAUDE_CONFIG_DIR seat with its disk-true identity + tombstone rehome", Lane: "cmd"},
	{Name: "affected", Synopsis: "fast inner loop: run go test only for the packages your working-tree change can affect", Lane: "cmd"},
	{Name: "agent", Synopsis: "run a live agent task and A/B the turn count (offline or against a provider)", Lane: "cmd"},
	{Name: "ailuminate", Synopsis: "AILuminate safety-benchmark runner (describe/eval/compare)", Lane: "cmd"},
	{Name: "answer-shape", Synopsis: "degeneration/verbosity witness: grade how repetitive and how long a candidate answer is", Lane: "cmd"},
	{Name: "attest", Synopsis: "compliance attestation generator: prove the policy capability floor from preflight", Lane: "cmd"},
	{Name: "audit", Synopsis: "audit-trail consumer: verify/export a fak guard decision journal's hash chain", Lane: "cmd"},
	{Name: "bench", Synopsis: "transport A/B: in-process adjudication p50 vs spawned-hook p50", Lane: "cmd"},
	{Name: "bench-loop", Synopsis: "benchmark super-loop manager: status, next action, surface walk, and local collection entrypoint", Aliases: []string{"benchloop"}, Lane: "cmd"},
	{Name: "benchmarks", Synopsis: "the index of every benchmark fak ships (list/describe/run)", Lane: "cmd"},
	{Name: "bgloop", Synopsis: "background runner for the durable long-running-loop ledger", Lane: "cmd"},
	{Name: "blockers", Synopsis: "the blockers report/Slack surface: what is gating the fleet right now", Lane: "cmd"},
	{Name: "boundary", Synopsis: "the boundary-tell linter as a verb: pathlint/urllint/boundarylint witnesses (unexpanded paths, hardcoded URLs, no-timeout HTTP)", Lane: "cmd"},
	{Name: "c", Synopsis: "shorthand for 'fak console agent': launch a fak-guard-wrapped interactive Claude Code session", Lane: "cmd"},
	{Name: "cachevalue", Synopsis: "the cache-value rollup: realized agent-memory / KV-reuse savings", Lane: "cmd"},
	{Name: "cadence", Synopsis: "consolidated regular-cadence report: scores, maturity, work-done, releases in one envelope", Lane: "cmd"},
	{Name: "callavoid", Synopsis: "the call-avoidance report: identical-call dedup (vDSO) round-trips saved", Lane: "cmd"},
	{Name: "chat", Synopsis: "minimal chat client against a fak serve/guard gateway", Lane: "cmd"},
	{Name: "chatrelay", Synopsis: "the chat-relay Slack surface bridging a channel to a gateway", Lane: "cmd"},
	{Name: "check-tool-failure", Synopsis: "lookup the closed non-guard tool-failure vocabulary (summary/fix/retryable)", Lane: "cmd"},
	{Name: "claim-check", Synopsis: "grade an efficiency/perf claim against the six-question net-true-value rubric", Lane: "cmd"},
	{Name: "claude-mac-fak", Synopsis: "one-command Mac gateway dogfood through the node-macos-a fak serve gateway", Lane: "cmd"},
	{Name: "cluster", Synopsis: "multi-node compute: run a real cross-node collective over fak's process group", Lane: "cmd"},
	{Name: "codelint", Synopsis: "language-server-pack code linter: route each file to its pack and report parse/compile errors", Lane: "cmd"},
	{Name: "codex", Synopsis: "OpenAI Codex integration launcher (codex exec routed through the kernel)", Lane: "cmd"},
	{Name: "codex-mcp-health", Synopsis: "health check for the Codex MCP integration", Lane: "cmd"},
	{Name: "codex-memory", Synopsis: "the Codex agent-memory bridge", Lane: "cmd"},
	{Name: "commit", Synopsis: "commit staged paths with the lane ship-stamp trailer enforced (safe shared-trunk commit)", Lane: "cmd"},
	{Name: "complain", Synopsis: "file a dogfood complaint about an agent-experience friction", Lane: "cmd"},
	{Name: "concept-usage-score", Synopsis: "native concept-usage scorecard control-pane payload", Lane: "cmd"},
	{Name: "conflation-scorecard", Synopsis: "native provenance-honesty control-pane payload (WITNESSED vs OBSERVED), folded into conflation_debt", Lane: "cmd"},
	{Name: "console", Synopsis: "the native terminal control-pane spine: issue/loop/session lanes + garden/guard/overview", Lane: "cmd"},
	{Name: "coverage-matrix", Synopsis: "the generated model x backend coverage matrix", Lane: "cmd"},
	{Name: "cron", Synopsis: "project the in-kernel loop schedule down to a real OS scheduler unit (launchd/systemd/taskscheduler)", Lane: "cmd"},
	{Name: "debug", Synopsis: "the context debugger: attach to a finished session as a core dump and demand-page its working set", Lane: "cmd"},
	{Name: "dispatch", Synopsis: "the witness-gated issue-dispatch loop: spawn, ship #N, witness, close", Lane: "cmd"},
	{Name: "doctor", Synopsis: "operator diagnostic: run the answer-shape witness + the real kernel admit verdict and recommend", Lane: "cmd"},
	{Name: "dogfood-issues", Synopsis: "file dogfood issues from observed agent-experience defects", Lane: "cmd"},
	{Name: "dogfood-score", Synopsis: "scores the launched-session dogfooding loop: wired to run honestly + truthful self-report", Lane: "cmd"},
	{Name: "dojo", Synopsis: "the prediction-vs-reality gym: score each calibration lever's claimed vs realized behavior", Lane: "cmd"},
	{Name: "dojo-rsi", Synopsis: "the self-pacing dojo RSI loop (fold/propose/run/loop/trend)", Lane: "cmd"},
	{Name: "dream", Synopsis: "offline sleep pass over a core image: re-screen, pre-seal refuted witnesses, prune", Lane: "cmd"},
	{Name: "egress", Synopsis: "prove the network-egress floor on one destination (the cloud-metadata / SSRF class)", Lane: "cmd"},
	{Name: "experiments", Synopsis: "the experiments registry/runner over experiments/", Lane: "cmd"},
	{Name: "feature", Synopsis: "query the unified self-feature catalog (dev facts, live tools, memory drivers, capability cards)", Lane: "cmd"},
	{Name: "fleet-accounts", Synopsis: "fleet-wide account management across config-home seats", Lane: "cmd"},
	{Name: "garden", Synopsis: "the issue-garden: triage and groom GitHub issues (kind/priority/area)", Lane: "cmd"},
	{Name: "grafana", Synopsis: "export fak fleet metrics as a Grafana dashboard/datasource", Lane: "cmd"},
	{Name: "guard", Synopsis: "wrap an agent harness: deny/repair/quarantine proposed tool calls (the one-command front door)", Lane: "gateway"},
	{Name: "guard-precompact", Synopsis: "internal: Claude Code PreCompact hook actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-rsi-scorecard", Synopsis: "native control-pane payload for guard RSI loop maturity and realized value", Lane: "cmd"},
	{Name: "guard-stophook", Synopsis: "internal: Claude Code Stop hook actuator installed by fak guard", Lane: "cmd"},
	{Name: "guard-verdict-rsi", Synopsis: "the guard verdict RSI loop: fold the decision journal, score verdict-quality, keep on gain", Lane: "cmd"},
	{Name: "headroom", Synopsis: "the context-compression seam: shrink tool outputs/logs/files before they reach the model", Lane: "cmd"},
	{Name: "help", Synopsis: "print the top-level fak usage banner", Aliases: []string{"-h", "--help"}, Lane: "cmd"},
	{Name: "hook", Synopsis: "spawned-hook decide (the A/B baseline transport; reads call.json on stdin)", Lane: "cmd"},
	{Name: "hooks", Synopsis: "the commit-boundary git-hook gates in one process (pre-commit / commit-msg)", Lane: "cmd"},
	{Name: "horizon-recovery", Synopsis: "recover stranded / stalled long-horizon work", Lane: "cmd"},
	{Name: "hygiene", Synopsis: "the whole-tree hygiene gates in one process (the --audit-tree twin of fak hooks)", Lane: "cmd"},
	{Name: "index", Synopsis: "queryable self-index: lane/leaf/docs/claims/verbs (query, don't survey)", Lane: "devindex", Doc: "AGENTS.md"},
	{Name: "info", Synopsis: "the live fak-info overlay: poll a gateway's /debug/vars and print one plain-words line per tick", Lane: "cmd"},
	{Name: "issue", Synopsis: "the generated-issue contract: review machine-created GitHub issue candidates before sync", Lane: "cmd"},
	{Name: "lab", Synopsis: "the GPU-lab status surface: per-state / per-class node counts + readiness", Lane: "cmd"},
	{Name: "leaseref", Synopsis: "cross-machine lease visibility: read the refs/fak/locks/* lease ref namespace", Lane: "cmd"},
	{Name: "learning-debt-dispatch", Synopsis: "learning-scorecard -> backlog: file capped triage issues for HARD learning-debt defects", Lane: "cmd"},
	{Name: "lint", Synopsis: "the static tool linter: the definition-time dual of the kernel's call-time re-checks", Lane: "cmd"},
	{Name: "loop", Synopsis: "the durable long-running-loop ledger (append/run/status/admit)", Lane: "cmd"},
	{Name: "loop-index-scorecard", Synopsis: "fold the six agentic-loop stages (orient->plan->act->verify->ship->learn) into one loop-index + loopindex_debt", Lane: "cmd"},
	{Name: "loop-map", Synopsis: "the agentic-loop map: the stages, levers, and child issues of the dev-experience epic", Lane: "cmd"},
	{Name: "loop-score", Synopsis: "score a single loop run's outcome", Lane: "cmd"},
	{Name: "ls", Synopsis: "alias for 'fak model ls': list known model aliases + cache status", Lane: "cmd"},
	{Name: "marketing", Synopsis: "the marketing Slack surface", Lane: "cmd"},
	{Name: "maturity", Synopsis: "the feature-lifecycle maturity ladder report", Lane: "cmd"},
	{Name: "memory", Synopsis: "the memory-operation algebra: author a render/clean/compact/dream Op pipeline (drivers/explain/run)", Lane: "cmd"},
	{Name: "merge", Synopsis: "shared-trunk merge dry-run: predict empty-net-diff vs clean changed files vs conflicts", Lane: "cmd"},
	{Name: "milestone", Synopsis: "the roadmap/milestone report (discrete-deliverable epics, completion %)", Lane: "cmd"},
	{Name: "milestone-scorecard", Synopsis: "native milestone-roadmap scorecard control-pane payload", Lane: "cmd"},
	{Name: "model", Synopsis: "resolve an hf:// URI to a locally cached file path (Hub download + SHA256 verify)", Lane: "cmd"},
	{Name: "new-model", Synopsis: "scaffold a new model adapter/leaf", Lane: "cmd"},
	{Name: "news", Synopsis: "the news Slack surface for source-linked external industry/SOTA/OSS research updates", Lane: "cmd"},
	{Name: "nightrun", Synopsis: "run it all night: the local-capability-aware data-collection door (next/plan/run/ledger/caps)", Lane: "cmd"},
	{Name: "node", Synopsis: "the compute-node registry (register/list nodes)", Lane: "cmd"},
	{Name: "nodeusage", Synopsis: "the compute-node-usage Slack surface for #node-usage", Lane: "cmd"},
	{Name: "operator", Synopsis: "the human pacing brief: fold cadence/program/milestone into human/agent/watch/background buckets", Lane: "cmd"},
	{Name: "orient", Synopsis: "task-scoped convention orientation for path globs: lane, arch tier, owning tests, stamp, and live lease", Lane: "devindex"},
	{Name: "opt", Synopsis: "the optimization-fuser / RSI opt-target loop", Lane: "cmd"},
	{Name: "policy", Synopsis: "the deployable capability floor: --dump | --check a policy manifest", Lane: "cmd"},
	{Name: "preflight", Synopsis: "adjudicate one tool call against a policy (ALLOW/DENY by structure, no model in the loop)", Lane: "cmd"},
	{Name: "process-guard", Synopsis: "the host process-resource guard: detect / reap runaway or leaking processes", Lane: "cmd"},
	{Name: "product", Synopsis: "the product-direction Slack surface for #product", Lane: "cmd"},
	{Name: "profile", Synopsis: "host-aware profiler: capture CPU + allocation profiles of a package's benchmarks (Windows->WSL)", Lane: "cmd"},
	{Name: "program", Synopsis: "the ongoing-optimization program report (a frontier + a trend, never a completion %)", Lane: "cmd"},
	{Name: "ps", Synopsis: "the read-only process table: one aligned row per live served session", Lane: "cmd"},
	{Name: "pull", Synopsis: "alias for 'fak model pull': the Ollama-style run-by-name model download", Lane: "cmd"},
	{Name: "recall", Synopsis: "persist a finished session as a core dump and reload it in a fresh store (quarantine survives)", Lane: "cmd"},
	{Name: "recover", Synopsis: "closed-vocabulary refusal recovery: print or run the safe commands for a reason token", Lane: "cmd"},
	{Name: "release", Synopsis: "the release front door over the tools/release_*.py helpers (status/cut/tag/publish/...)", Lane: "cmd"},
	{Name: "release-staleness", Synopsis: "the publish-freshness signal: how far the latest @latest tag lags HEAD (commits + days)", Lane: "cmd"},
	{Name: "replay", Synopsis: "explicit spelling of the trace-replay path (fak run --trace)", Lane: "cmd"},
	{Name: "repo-hygiene-scorecard", Synopsis: "native repo-hygiene scorecard control-pane payload (hygiene_debt)", Lane: "cmd"},
	{Name: "resume", Synopsis: "the deterministic resume-cache decision: what happens to the prompt cache on resume, and what to do", Lane: "cmd"},
	{Name: "rollup", Synopsis: "the executive roll-up snapshot regenerated from the tree", Lane: "cmd"},
	{Name: "route", Synopsis: "the model-routing oracle: per-aspect + ensemble model routing for one classified subject", Lane: "cmd"},
	{Name: "routebench", Synopsis: "the offline routing benchmark: a per-aspect+ensemble policy vs a single-model baseline", Lane: "cmd"},
	{Name: "run", Synopsis: "run an agent turn (or a recorded trace via --trace) through the kernel", Lane: "cmd"},
	{Name: "rungstats", Synopsis: "stats over the verification-ladder rungs", Lane: "cmd"},
	{Name: "savings-vector", Synopsis: "the savings-vector report: realized token/cost savings by lever", Lane: "cmd"},
	{Name: "scoreboard", Synopsis: "the scoreboard Slack surface for #scoreboard", Lane: "cmd"},
	{Name: "scorecard", Synopsis: "the scorecard control pane: every metric's debt + grade + trend", Lane: "cmd"},
	{Name: "self-update", Synopsis: "converge a built-from-source fak binary on origin/main", Lane: "cmd"},
	{Name: "serve", Synopsis: "run the OpenAI-compatible gateway in front of a local or remote model", Lane: "gateway"},
	{Name: "serve-wiring", Synopsis: "audit fak serve flag -> gateway.Config -> runtime-read wiring", Lane: "gateway"},
	{Name: "session", Synopsis: "the operator control surface for a served session: read live DRIVE state, cancel/update in flight", Lane: "cmd"},
	{Name: "sessions", Synopsis: "ingest + score this host's agent transcripts (the session->outcome learn loop)", Lane: "cmd"},
	{Name: "signal", Synopsis: "job control for a running session (pause/resume/stop/steer) over the control plane", Lane: "cmd"},
	{Name: "skill-effectiveness-scorecard", Synopsis: "native skill-pack effectiveness control-pane payload", Lane: "cmd"},
	{Name: "slack", Synopsis: "debug + use the whole Slack surface from one place (check/health/beat/walk/refresh/send)", Lane: "cmd"},
	{Name: "snapshot", Synopsis: "dump/restore any primitive on the loops ladder to a portable sha256-integrity bundle", Lane: "cmd"},
	{Name: "steering", Synopsis: "the steerability Slack surface for #steering-guard (status/report/alert/pin)", Lane: "cmd"},
	{Name: "stopfailure", Synopsis: "operator surface for .dos/stop-failures breaker markers (plan/reset-stale/clear-reviewed)", Lane: "cmd"},
	{Name: "superloop", Synopsis: "operator-intent meta-loop: walk a set of member loops/scorecards/gardens worst-first (list/explain/walk)", Lane: "cmd"},
	{Name: "support", Synopsis: "per-cell support read-out: one line per model x backend cell (rung.regime.target.next-action)", Lane: "cmd"},
	{Name: "support-maturity-scorecard", Synopsis: "native support-maturity payload: fold the model x backend coverage matrix into a grade", Lane: "cmd"},
	{Name: "sweep", Synopsis: "drive a dirty multi-session tree toward zero: report by lane, then --apply one lane group", Lane: "cmd"},
	{Name: "swebench", Synopsis: "SWE-bench Verified benchmarking (describe/eval/compare)", Lane: "cmd"},
	{Name: "sync", Synopsis: "safe fast-forward sync for a dirty shared worktree (check/apply, never pull/stash/reset)", Lane: "cmd"},
	{Name: "task", Synopsis: "the process-local task manager snapshot (hardware/runtime + task/step/concept progress + ETA)", Lane: "cmd"},
	{Name: "test", Synopsis: "host-aware test runner: resolve the right go test invocation (Windows->WSL via test.ps1)", Lane: "cmd"},
	{Name: "token-defaults-scorecard", Synopsis: "native token-saving-defaults control-pane payload", Lane: "cmd"},
	{Name: "top", Synopsis: "= fak ps --watch (the live process-table top mode)", Lane: "cmd"},
	{Name: "traj", Synopsis: "the trajectory-corpus toolkit (similar/cluster/score/gc/export) over recorded turns", Lane: "cmd"},
	{Name: "tree-doctor", Synopsis: "the worktree doctor: detect and prune stray / dead git worktrees", Lane: "cmd"},
	{Name: "turntax", Synopsis: "turn-tax A/B: price the extra error-code model turns a SOTA loop fires vs fak's one-shot", Lane: "cmd"},
	{Name: "ui-quality-scorecard", Synopsis: "native terminal UI/UX control-pane payload (ui_quality_debt)", Lane: "cmd"},
	{Name: "vcache", Synopsis: "the virtual provider-cache status/proof surface (status/prove/prove-telemetry)", Lane: "cmd"},
	{Name: "version", Synopsis: "print the fak version", Aliases: []string{"-v", "--version"}, Lane: "cmd"},
	{Name: "webbench", Synopsis: "frontier web/browser-agent benchmarking (describe/eval/compare)", Lane: "cmd"},
	{Name: "workflow-audit", Synopsis: "classify every branch/tag ref in .github/workflows against the branch-role contract (#1697); gate on unclassified dev-path refs", Lane: "cmd"},
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
			out = append(out, v)
			continue
		}
		if seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, Verb{Name: tok, Synopsis: "not yet cataloged — `fak " + tok + " -h` for usage"})
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
