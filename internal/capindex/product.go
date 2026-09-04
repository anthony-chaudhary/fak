package capindex

import (
	"sort"
	"strings"
	"unicode"
)

// ProductOutcome is one operator-language answer to "what can fak do?". This
// catalog is intentionally stdlib-only and runtime-safe: both the shipped fak
// binary and repository self-query can consume it without importing devindex.
type ProductOutcome struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Summary string   `json:"summary"`
	Tags    []string `json:"tags"`
	Effect  string   `json:"effect"`
	Command []string `json:"command"`
	Detail  string   `json:"detail_ref"`
	Witness string   `json:"witness"`
}

// ProductOutcomes returns the small performance-first product catalog. Security
// remains indexed as a supporting floor and sorts behind performance outcomes
// unless the query asks for it.
func ProductOutcomes() []ProductOutcome {
	return []ProductOutcome{
		{ID: "turn-savings", Name: "Avoid unnecessary model turns", Summary: "measure turn tax and elide kernel-known work instead of paying for another model round trip", Tags: []string{"turn control", "turn tax", "turn savings", "fewer turns", "fused turn", "elision", "token efficiency", "latency"}, Effect: "read", Command: []string{"go", "run", "./cmd/turntaxdemo", "-selfcheck"}, Detail: "docs/CAPABILITIES.md#turn-savings", Witness: "internal/turntaxmeter + internal/fusedturn + cmd/turntaxdemo"},
		{ID: "wip-administration", Name: "Administer worktrees and unlanded WIP", Summary: "prioritize unlanded work, inspect bounded WIP inventory, and audit registered worker worktree lifecycle state", Tags: []string{"WIP", "worktree", "worktrees", "unlanded work", "worktree administration", "WIP administration", "WIP queue", "WIP inventory", "worker lifecycle", "stale worker residue", "checkpoint cleanup", "land", "reap"}, Effect: "read", Command: []string{"fak", "wip", "queue", "--json"}, Detail: "Use fak wip inventory --json (fak-wip-inventory/1) for bounded inventory and fak worktree worker list --json (fak-worker-worktree-lifecycle/1) for registered worker lifecycle state; the primary queue emits fak-wip-action-queue/1.", Witness: "docs/_witnesses/wip-action-queue-scale-2026-08-30.json + docs/_witnesses/wip-inventory-scale-2026-08-30.json + docs/_witnesses/worker-worktree-list-scale-2026-08-30.json"},
		{ID: "fleet-commit-health", Name: "Check whether an active fleet is landing commits", Summary: "measure real first-parent commits per ten-minute window and fail when active workers produce zero", Tags: []string{"fleet health", "commit throughput", "commits per 10 minutes", "commits per 10m", "zero commits", "self blocking", "stalled fleet", "blocked fleet", "progress check", "on demand check"}, Effect: "read", Command: []string{"fak", "fleet", "health", "--json"}, Detail: "docs/fleet.md#commit-throughput-health", Witness: "internal/fleetmetrics.CommitThroughput + superloop.GateCommitThroughput + fak_fleet_commits_per_10m"},
		{ID: "performance-rsi-health", Name: "Observe performance-loop health and debt", Summary: "inspect default loop-turn observability and score committed repository dogfood evidence for performance-RSI health, debt, and the dominant bottleneck", Tags: []string{"performance observability", "performance rsi", "performance loop", "loop health", "performance debt", "default loop turn", "automatic loop-turn", "dominant bottleneck", "scorecard"}, Effect: "read", Command: []string{"fak", "score", "performance-rsi", "--input", "docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json", "--json"}, Detail: "docs/notes/PERFORMANCE-RSI-DOGFOOD-2026-08-28.md", Witness: "internal/perfrsiscore.ScoreLoopTurn + docs/_witnesses/issue-9777-performance-rsi-loop-turn/loop-turn.txt"},
		{ID: "context-reuse", Name: "Reuse stable prompt and context work", Summary: "reuse stable prefixes, manage resident context, and price replay versus cut or reset after cache expiry", Tags: []string{"token savings", "save tokens", "prompt cache", "prefix reuse", "context compaction", "ctxmmu", "vdso", "resume", "cache efficiency"}, Effect: "read", Command: []string{"fak", "resume", "plan", "--resident-tokens", "250000", "--idle-seconds", "7200", "--json"}, Detail: "docs/CAPABILITIES.md#context-reuse", Witness: "internal/ctxmmu + internal/vdso + docs/managed-context-continuous-usage.md"},
		{ID: "session-control", Name: "Control a live session out of band", Summary: "budget, pause, resume, throttle, steer, or stop a served session without spending another prompt turn", Tags: []string{"turn control", "turn budget", "token budget", "context budget", "session control", "steer", "pause", "throttle", "cancel"}, Effect: "mutate", Command: []string{"fak", "session", "budget", "<id>", "--turns", "N", "--tokens", "N", "--context-tokens", "N"}, Detail: "docs/CAPABILITIES.md#session-control", Witness: "internal/sessionctl + internal/sessionsignals + docs/operator-control-plane.md"},
		{ID: "model-routing", Name: "Route each call to an appropriate model", Summary: "select cheaper or specialized inference per call instead of pinning a whole session to one expensive model", Tags: []string{"token savings", "cost efficiency", "model routing", "per call model", "model ladder", "cheap model", "inference efficiency"}, Effect: "read", Command: []string{"fak", "model", "--help"}, Detail: "docs/CAPABILITIES.md#model-routing", Witness: "internal/modelroute + internal/modelladder + docs/model-routing.md"},
		{ID: "savings-observability", Name: "Attribute cache and token savings", Summary: "show reused tokens, effective cost, and total savings; ablate one frozen trace to attribute the gain", Tags: []string{"token savings", "cache savings", "cost savings", "cache value", "observability", "ablate", "same trace", "efficiency"}, Effect: "read", Command: []string{"fak", "info", "--once"}, Detail: "docs/CAPABILITIES.md#savings-observability", Witness: "internal/cachevalue + internal/cachevaluereport + docs/cache-value-rollup.md"},
		{ID: "context-compression", Name: "Compress tool output before it reaches the model", Summary: "measure reversible context compression on built-in or captured tool output so fewer input tokens enter the model window", Tags: []string{"token savings", "save tokens", "context compression", "tool output compression", "headroom", "input tokens", "long context", "efficiency"}, Effect: "read", Command: []string{"fak", "headroom", "bench", "--json"}, Detail: "docs/CAPABILITIES.md#context-compression", Witness: "internal/headroom + docs/notes/CONTEXT-COMPRESSION-LIVE-2026-08-09.md"},
		{ID: "portable-session", Name: "Carry a session forward without transcript replay", Summary: "persist, inspect, query, and restore session state so a follow-up can page the needed working set instead of replaying the whole transcript", Tags: []string{"token savings", "save tokens", "session image", "portable session", "snapshot", "recall", "demand paging", "context replay", "long session"}, Effect: "read", Command: []string{"fak", "snapshot", "demo"}, Detail: "docs/CAPABILITIES.md#portable-session", Witness: "internal/sessionimage + internal/recall + cmd/fak/snapshot_cli.go"},
		{ID: "native-serve", Name: "Serve a model through fak-native inference", Summary: "start the in-kernel Metal model path and fail loudly instead of silently changing inference engines", Tags: []string{"serve native model", "native serving", "local model server", "fak native", "metal inference"}, Effect: "mutate", Command: []string{"fak", "serve", "--gguf", "<model.gguf>", "--metal"}, Detail: "docs/model-engine-env.md", Witness: "cmd/fak/serve.go + internal/modelengine + internal/metalgemm"},
		{ID: "model-benchmark", Name: "Discover the native model benchmark", Summary: "inspect the shipped in-kernel modelbench command, required weights, flags, and measurement contract before running it", Tags: []string{"benchmark native inference", "model benchmark", "modelbench", "throughput measurement", "latency measurement"}, Effect: "read", Command: []string{"fak", "benchmarks", "describe", "modelbench"}, Detail: "docs/model-engine-env.md", Witness: "internal/benchcatalog modelbench entry + cmd/modelbench"},
		{ID: "model-quality", Name: "Evaluate model output quality", Summary: "compare one versioned engine trace with its reference and emit a replayable pass or failure result", Tags: []string{"evaluate model quality", "model quality evaluation", "accuracy gate", "quality oracle", "failure replay"}, Effect: "read", Command: []string{"fak", "quality", "run", "--json"}, Detail: "docs/quality/output-quality-regression-runbook.md", Witness: "cmd/fak/quality.go + internal/quality"},
		{ID: "native-profile", Name: "Diagnose native inference bottlenecks", Summary: "classify phase and backend counters, then select the measured dependency-ready optimization lever", Tags: []string{"native bottleneck", "bottleneck diagnosis", "performance diagnosis", "phase counters", "next optimization"}, Effect: "read", Command: []string{"fak", "native-performance", "--profile-next", "profile.json"}, Detail: "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md", Witness: "cmd/fak/native_performance.go + internal/nativeperf profile classifier"},
		{ID: "performance-receipt", Name: "Validate a native performance receipt", Summary: "gate accepted and candidate receipts together under one pinned native engine, quality, memory, and throughput policy", Tags: []string{"performance receipt", "receipt validation", "performance regression", "benchmark gate", "native evidence"}, Effect: "read", Command: []string{"fak", "native-performance", "--gate", "gate-request.json"}, Detail: "docs/benchmarks/NATIVE-PERFORMANCE-REGRESSION-GATE.md", Witness: "cmd/fak/native_performance.go + internal/nativeperf regression gate"},
		{ID: "ple-disk-stream", Name: "Direct I/O PLE embedding disk streaming", Summary: "stream 16-row n-gram PLE embeddings directly from NVMe over Direct I/O with CUDA stream memop synchronization to keep 51GB table off host RAM", Tags: []string{"ple disk stream", "moe offload", "qwen4_exp", "direct io", "ngram embedding", "nvme stream"}, Effect: "read", Command: []string{"fak", "capabilities", "ple disk stream"}, Detail: "docs/notes/CONCEPT-STUDY-FREETOKEN-2026-09-03.md", Witness: "internal/qwen4exp/ple_stream.go + commit 4668ff578"},
		{ID: "capability-floor", Name: "Enforce the supporting capability floor", Summary: "check tool authority before execution so efficiency changes remain bounded and auditable", Tags: []string{"security", "policy", "capability floor", "preflight", "audit", "default deny"}, Effect: "read", Command: []string{"fak", "preflight", "--policy", "<file>", "--tool", "<name>", "--args", "{}"}, Detail: "docs/CAPABILITIES.md#capability-floor", Witness: "internal/policy + internal/adjudicator + docs/fak/security.md"},
	}
}

// QueryProductOutcomes ranks outcomes by operator intent. Exact phrases and
// complete token matches beat partial matches; stable declaration order breaks
// ties so the performance-first default remains deliberate and reproducible.
func QueryProductOutcomes(query string, limit int) []ProductOutcome {
	outcomes := ProductOutcomes()
	query = strings.TrimSpace(strings.ToLower(query))
	if query == "" {
		if limit > 0 && len(outcomes) > limit {
			return outcomes[:limit]
		}
		return outcomes
	}
	tokens := words(query)
	type ranked struct {
		outcome      ProductOutcome
		score, order int
	}
	var found []ranked
	for i, outcome := range outcomes {
		hay := strings.ToLower(outcome.Name + " " + outcome.Summary + " " + strings.Join(outcome.Tags, " "))
		score := 0
		if strings.Contains(hay, query) {
			score += 100
		}
		for _, token := range tokens {
			if containsWord(hay, token) {
				score += 12
			} else if strings.Contains(hay, token) {
				score += 3
			}
		}
		if score > 0 {
			found = append(found, ranked{outcome: outcome, score: score, order: i})
		}
	}
	sort.SliceStable(found, func(i, j int) bool {
		if found[i].score != found[j].score {
			return found[i].score > found[j].score
		}
		return found[i].order < found[j].order
	})
	result := make([]ProductOutcome, 0, len(found))
	for _, item := range found {
		result = append(result, item.outcome)
	}
	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}
	return result
}

func words(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
}
func containsWord(hay, needle string) bool {
	for _, word := range words(hay) {
		if word == needle {
			return true
		}
	}
	return false
}
