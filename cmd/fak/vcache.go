package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func cmdVCache(argv []string) {
	os.Exit(runVCache(os.Stdout, os.Stderr, argv))
}

func runVCache(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		vcacheUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "status":
		return runVCacheStatus(stdout, stderr, argv[1:])
	case "prove":
		return runVCacheProve(stdout, stderr, argv[1:])
	case "prove-telemetry":
		return runVCacheProveTelemetry(stdout, stderr, argv[1:])
	case "prove-recall":
		return runVCacheProveRecall(stdout, stderr, argv[1:])
	case "observe":
		return runVCacheObserve(stdout, stderr, argv[1:])
	case "calibrate":
		return runVCacheCalibrate(stdout, stderr, argv[1:])
	case "context-join":
		return runVCacheContextJoin(stdout, stderr, argv[1:])
	case "actions":
		return runVCacheActions(stdout, stderr, argv[1:])
	case "codex-session-extract":
		return runVCacheCodexSessionExtract(stdout, stderr, argv[1:])
	case "context-witness":
		return runVCacheContextWitness(stdout, stderr, argv[1:])
	case "score", "bench":
		return runVCacheScore(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		vcacheUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak vcache: unknown subcommand %q\n", argv[0])
		vcacheUsage(stderr)
		return 2
	}
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	seen := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			seen = true
		}
	})
	return seen
}

func vcacheUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak vcache status [--json] [--sessions] [--session-days N] [--session-max N]
                   [--session-ns-prefix PREFIX | --session-all]
  fak vcache prove [--json] [--anchor-tokens N] [--suffix-tokens N] [--requests N]
                   [--min-prefix-tokens N] [--read-mult F] [--write-mult F]
                   [--content public|secret|regulated]
  fak vcache prove-telemetry --file FILE [--json]
                   [--read-mult F] [--write-5m-mult F] [--write-1h-mult F]
  fak vcache prove-recall [--json] [--prefix-tokens N] [--unit-tokens N]
                   [--read-mult F] [--siblings N]
  fak vcache calibrate --samples FILE [--json] [--out FILE]
                   [--ttl-ms N] [--min-prefix-tokens N] [--read-mult F]
  fak vcache observe [--transcript FILE]... [--telemetry FILE] [--json]
                   [--calibration FILE] [--read-mult F]
                   [--write-5m-mult F] [--write-1h-mult F]
  fak vcache context-join [--transcript FILE]... [--telemetry FILE] --events FILE
                   [--json] [--before-millis N] [--after-millis N]
  fak vcache actions [--json] [--snapshot FILE|default|off] [--out FILE]
                   [--heartbeat-transport] [--explicit-cache-transport]
                   [--prefix-witness] [--deletion-capable] [--transport-source S]
  fak vcache codex-session-extract [--session FILE | --thread-id ID] --out FILE
                   [--snapshot-out FILE|default] [--score-out FILE] [--family NAME]
  fak vcache context-witness [--json] [--snapshot FILE] [--fixture FILE]
                   [--wire openai|anthropic]
  fak vcache score|bench [--json] [--out FILE] [--telemetry FILE] [--two-x F]
                   [--anchor-tokens N --suffix-tokens N --requests N]
                   [--read-mult F --write-mult F --write-5m-mult F --write-1h-mult F]
                   [--zipf-s F --anchors N --anchors-file FILE --target-coverage F]
                   [--kernel-ledger FILE|default|off] [--context-snapshot FILE|default|off]
                   [--kernel-kv-events N --context-events N]
                   [--kernel-kv-prompt-tokens N --kernel-kv-reused-tokens N]
                   [--context-shed-tokens N --context-resident-tokens N]
                   [--provider-vcache-decisions N --external-engine-events N]
                   [--external-engine-hit-rate F]
                   [--index-out FILE]
                   [--true-warm N --false-warm N --true-cold N --false-cold N]
                   [--recall-prefix-tokens N --recall-unit-tokens N --recall-siblings N --recall-read-mult F]

status reports what is actually up: the M5 governor is a local, off-path policy
engine; the M4 chains & recall engine is off-path and gated OFF by default;
provider calibration/warming remain tracked by #716-#718, and Codex/OpenAI cached-
token telemetry is proven by the replayable #727 artifacts.
prove runs the deterministic star-anchor token-savings proof. Exit 0 means PROVEN;
exit 1 means REFUTED; exit 2 means usage error.
prove-telemetry replays provider usage JSONL, such as Claude Code probe output,
OpenAI Responses/Chat usage objects, Codex CLI token_count rows, or codex exec
--json turn.completed usage rows, and proves realized savings from observed
cache counters.
prove-recall runs the deterministic M4 cost-gate proof (the §11.0 headline): a
single ~10-token unit recalled from a long warm prefix is almost always a net LOSS,
so the gate REFUSES it; rebuild wins only for amortized fan-out. Exit 0 = rebuild
allowed (PROVEN); exit 1 = refused (REFUTED); exit 2 = usage error.
calibrate fits provider TTL, minimum cacheable prefix, and cached-read multiplier
from replayed provider-cache probe samples (JSON array, JSONL, or {"samples":[...]})
and writes the calibration JSON consumed by observe --calibration.
score/bench composes planned or observed savings, workload concentration,
false-warm risk, recall risk, and a hot-anchor index into one 2x agent-dev gate.
observe is the 10x per-sub-concept observability lens: it ingests REAL Claude Code
transcripts (--transcript, repeatable) or a session-telemetry JSONL (--telemetry),
groups turns by prefix family (one session = one shared system prefix), and runs the
shipped M1-M5 decision leaves over that real data — one panel per sub-concept, each
labeled OBSERVED (relayed from the provider's counters) or DECISION (fak's verdict).
context-join (#1607) answers whether an observed cost change came from CONTEXT
PLANNING (a reset, compaction, page fault, or prefix mutation fak decided) or from
PROVIDER CACHE BEHAVIOR (a natural miss/TTL expiry unrelated to any context action).
It joins the same --transcript/--telemetry turn stream against a --events JSONL of
managed-context lifecycle events (see internal/vcacheobserve.LifecycleEvent).
actions renders the provider-cache action plan over a persisted observed-window
snapshot. It maps each observed prefix family from the M5 Governor verdict to a
concrete action row (ride_natural / heartbeat_pin / lazy_rebuild / evict_manifest /
no_cache / explicit_cache) and labels rows noop, ready, or gated. This is a
decision/API witness, not proof that a provider warm was spent. Transport flags are
witness inputs only: a heartbeat/explicit-cache row becomes ready only when its
required capability and byte-identical prefix evidence is supplied.

`)
}

type vcacheStatusReport struct {
	Status                 string                      `json:"status"`
	Governor               string                      `json:"governor"`
	Chains                 string                      `json:"chains"`
	LiveProvider           string                      `json:"live_provider"`
	Proof                  vcachegov.StarSavingsProof  `json:"proof"`
	RecallProof            vcachechain.RecallProof     `json:"recall_proof"`
	CodexOpenAI            vcacheCodexOpenAIStatus     `json:"codex_openai"`
	ContextAPI             vcacheContextAPIStatus      `json:"context_api"`
	ProviderCalibration    vcacheProviderCalStatus     `json:"provider_calibration"`
	ProviderActions        vcacheProviderActionStatus  `json:"provider_actions"`
	RecentObservation      *vcacheRecentObservation    `json:"recent_observation,omitempty"`
	RecentObservationError string                      `json:"recent_observation_error,omitempty"`
	RecentSessions         *sessionaudit.CompactReport `json:"recent_sessions,omitempty"`
	RecentSessionsError    string                      `json:"recent_sessions_error,omitempty"`
	M4Issue                string                      `json:"m4_issue"`
	M5Issue                string                      `json:"m5_issue"`
	Remaining              []vcacheRemainingIssue      `json:"remaining"`
	CorrectnessLaw         string                      `json:"correctness_law"`
}

type vcacheRemainingIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	URL    string `json:"url"`
}

type vcacheRecentObservation struct {
	Source                string  `json:"source"`
	Path                  string  `json:"path,omitempty"`
	Turns                 int     `json:"turns"`
	ProviderStatus        string  `json:"provider_status"`
	CacheReadTokens       float64 `json:"cache_read_tokens"`
	CacheCreationTokens   float64 `json:"cache_creation_tokens"`
	HitRate               float64 `json:"hit_rate"`
	Multiplier            float64 `json:"multiplier"`
	SavedTokenEquiv       float64 `json:"saved_token_equiv"`
	FalseWarmRate         float64 `json:"false_warm_rate"`
	FalseColdRate         float64 `json:"false_cold_rate"`
	GovernorDecision      string  `json:"governor_decision,omitempty"`
	ContextStatus         string  `json:"context_status"`
	ContextReason         string  `json:"context_reason,omitempty"`
	ContextPath           string  `json:"context_path,omitempty"`
	ContextEvents         int64   `json:"context_events,omitempty"`
	ContextShedTokens     int64   `json:"context_shed_tokens,omitempty"`
	ContextDroppedTurns   int64   `json:"context_dropped_turns,omitempty"`
	ContextBaselineTokens int64   `json:"context_baseline_tokens,omitempty"`
	ContextCostTokens     int64   `json:"context_cost_tokens,omitempty"`
}

type vcacheCodexOpenAIStatus struct {
	Verifier            string                          `json:"verifier"`
	LiveTelemetry       string                          `json:"live_telemetry"`
	Reason              string                          `json:"reason"`
	OpenAIAPIKeyPresent bool                            `json:"openai_api_key_present"`
	CachedTokenFields   []string                        `json:"cached_token_fields"`
	Issue               string                          `json:"issue"`
	CachedSampleProof   vcachegov.TelemetrySavingsProof `json:"cached_sample_proof"`
	NoCacheRefutation   vcachegov.TelemetrySavingsProof `json:"no_cache_refutation"`
}

type vcacheContextAPIStatus struct {
	Verifier            string   `json:"verifier"`
	HTTP                string   `json:"http"`
	MCPTool             string   `json:"mcp_tool"`
	AdviceOnly          bool     `json:"advice_only"`
	Provenance          []string `json:"provenance"`
	ScoreIntegration    string   `json:"score_integration"`
	NoKeyReplayFixture  string   `json:"no_key_replay_fixture"`
	NoKeyReplaySnapshot string   `json:"no_key_replay_snapshot"`
	NoKeyReplayCommand  string   `json:"no_key_replay_command"`
	NoKeyWitnessCommand string   `json:"no_key_witness_command"`
	DefaultSnapshot     string   `json:"default_snapshot"`
	NoKeyScoreCommand   string   `json:"no_key_score_command"`
	Reason              string   `json:"reason"`
}

type vcacheProviderCalStatus struct {
	Verifier  string `json:"verifier"`
	CLI       string `json:"cli"`
	Input     string `json:"input"`
	Output    string `json:"output"`
	Consumer  string `json:"consumer"`
	LiveProbe string `json:"live_probe"`
	Reason    string `json:"reason"`
}

type vcacheProviderActionStatus struct {
	Verifier  string `json:"verifier"`
	HTTP      string `json:"http"`
	CLI       string `json:"cli"`
	Schema    string `json:"schema"`
	ReadOnly  bool   `json:"read_only"`
	Transport string `json:"transport"`
	Reason    string `json:"reason"`
}

func runVCacheStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable status")
	includeSessions := fs.Bool("sessions", false, "include compact recent Claude session summary scoped to this workspace")
	sessionDays := fs.Float64("session-days", 7, "with --sessions, only include transcripts modified within N days")
	sessionMax := fs.Int("session-max", 40, "with --sessions, maximum recent transcripts to summarize")
	sessionNS := fs.String("session-ns-prefix", "", "with --sessions, namespace prefix to summarize (default: current workspace namespace)")
	sessionAll := fs.Bool("session-all", false, "with --sessions, include all non-excluded namespaces instead of the current workspace namespace")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	rep := defaultVCacheStatus()
	if *includeSessions {
		applyRecentSessionSummary(&rep, vcacheSessionSummaryOptions{
			SinceDays:       *sessionDays,
			Max:             *sessionMax,
			NamespacePrefix: *sessionNS,
			AllNamespaces:   *sessionAll,
		})
	}
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	fmt.Fprintf(stdout, "vCache status: %s\n", rep.Status)
	fmt.Fprintf(stdout, "vCache M5 governor: %s\n", rep.Governor)
	fmt.Fprintf(stdout, "vCache M4 chains & recall: %s\n", rep.Chains)
	fmt.Fprintf(stdout, "live provider loop: %s\n", rep.LiveProvider)
	if rep.RecentObservation != nil {
		recent := rep.RecentObservation
		if recent.ProviderStatus == "MISSING" {
			fmt.Fprintf(stdout, "recent snapshot: %d turns, provider MISSING, context %s (%d events)\n",
				recent.Turns, recent.ContextStatus, recent.ContextEvents)
		} else {
			fmt.Fprintf(stdout, "recent snapshot: %d turns, provider %s %.2fx, false-warm %.2f%%, governor %s, context %s (%d events)\n",
				recent.Turns, recent.ProviderStatus, recent.Multiplier, 100*recent.FalseWarmRate, recent.GovernorDecision, recent.ContextStatus, recent.ContextEvents)
		}
	} else if rep.RecentObservationError != "" {
		fmt.Fprintf(stdout, "recent snapshot: unreadable (%s)\n", rep.RecentObservationError)
	}
	if rep.RecentObservation != nil && rep.RecentObservation.ContextPath != "" && rep.RecentObservation.ContextPath != rep.RecentObservation.Path {
		fmt.Fprintf(stdout, "context witness snapshot: %s\n", rep.RecentObservation.ContextPath)
	}
	if rep.RecentSessions != nil {
		printVCacheSessionSummary(stdout, *rep.RecentSessions)
	} else if rep.RecentSessionsError != "" {
		fmt.Fprintf(stdout, "recent sessions: unreadable (%s)\n", rep.RecentSessionsError)
	}
	fmt.Fprintf(stdout, "context API: %s (%s; MCP %s; advice_only=%v)\n",
		rep.ContextAPI.Verifier, rep.ContextAPI.HTTP, rep.ContextAPI.MCPTool, rep.ContextAPI.AdviceOnly)
	fmt.Fprintf(stdout, "provider calibration: %s (CLI %s; output %s; consumer %s)\n",
		rep.ProviderCalibration.Verifier, rep.ProviderCalibration.CLI, rep.ProviderCalibration.Output, rep.ProviderCalibration.Consumer)
	fmt.Fprintf(stdout, "provider actions API: %s (%s; CLI %s; transport=%s)\n",
		rep.ProviderActions.Verifier, rep.ProviderActions.HTTP, rep.ProviderActions.CLI, rep.ProviderActions.Transport)
	fmt.Fprintf(stdout, "context witness replay: run `%s` (writes %s); score with `%s`\n",
		rep.ContextAPI.NoKeyWitnessCommand, rep.ContextAPI.DefaultSnapshot, rep.ContextAPI.NoKeyScoreCommand)
	fmt.Fprintf(stdout, "codex-like star proof: %s (%s)\n", rep.Proof.Status, rep.Proof.Reason)
	fmt.Fprintf(stdout, "token-equiv saved: %.1f / %.1f (%.1f%%)\n",
		rep.Proof.SavedTokenEquiv, rep.Proof.BaselineTokenEquiv, rep.Proof.SavedPct)
	fmt.Fprintf(stdout, "M4 recall cost-gate proof: %s — %s\n", rep.RecallProof.Status, rep.RecallProof.Decision)
	fmt.Fprintf(stdout, "M4 single-unit loss ratio: %.1fx (break-even %s siblings)\n",
		rep.RecallProof.LossRatio, formatBreakEven(rep.RecallProof.BreakEvenSiblings))
	fmt.Fprintf(stdout, "codex/openai verifier: %s\n", rep.CodexOpenAI.Verifier)
	fmt.Fprintf(stdout, "codex/openai live telemetry: %s (%s)\n",
		rep.CodexOpenAI.LiveTelemetry, rep.CodexOpenAI.Reason)
	fmt.Fprintf(stdout, "codex/openai cached-token sample: %s saved %.1f / %.1f (%.2f%%)\n",
		rep.CodexOpenAI.CachedSampleProof.Status,
		rep.CodexOpenAI.CachedSampleProof.SavedTokenEquiv,
		rep.CodexOpenAI.CachedSampleProof.BaselineTokenEquiv,
		rep.CodexOpenAI.CachedSampleProof.SavedPct)
	fmt.Fprintf(stdout, "codex/openai zero-cache sample: %s saved %.1f / %.1f (%.2f%%)\n",
		rep.CodexOpenAI.NoCacheRefutation.Status,
		rep.CodexOpenAI.NoCacheRefutation.SavedTokenEquiv,
		rep.CodexOpenAI.NoCacheRefutation.BaselineTokenEquiv,
		rep.CodexOpenAI.NoCacheRefutation.SavedPct)
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", rep.Proof.CorrectnessDependsOn)
	fmt.Fprintf(stdout, "m5 issue: %s\n", rep.M5Issue)
	fmt.Fprint(stdout, "remaining:")
	for _, issue := range rep.Remaining {
		fmt.Fprintf(stdout, " #%d", issue.Number)
	}
	fmt.Fprintln(stdout)
	return 0
}

func runVCacheActions(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache actions", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable provider action plan")
	snapshot := fs.String("snapshot", "default", "provider snapshot to read: FILE, default, or off")
	out := fs.String("out", "", "write the JSON action plan to this file")
	heartbeatTransport := fs.Bool("heartbeat-transport", false, "witness that the host can issue provider heartbeat refresh calls")
	explicitCacheTransport := fs.Bool("explicit-cache-transport", false, "witness that the provider exposes explicit cache create/delete controls")
	prefixWitness := fs.Bool("prefix-witness", false, "witness that action candidates use a byte-identical observed prefix")
	deletionCapable := fs.Bool("deletion-capable", false, "witness that explicit-cache entries can be deleted")
	transportSource := fs.String("transport-source", "", "short label for the transport witness source")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	var turns []vcacheobserve.Turn
	if path, readSnapshot := resolveVCacheProviderSnapshotPath(*snapshot); readSnapshot {
		readTurns, ok, err := vcachesnapshot.Read(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache actions: read snapshot %q: %v\n", path, err)
			return 1
		}
		if ok {
			turns = readTurns
		}
	}
	plan := vcacheobserve.PlanProviderActionsWithOptions(turns, false, vcacheobserve.ProviderActionOptions{
		Transport: vcacheobserve.ProviderTransportWitness{
			HeartbeatTransport:     *heartbeatTransport,
			ExplicitCacheTransport: *explicitCacheTransport,
			ByteIdenticalPrefix:    *prefixWitness,
			DeletionCapable:        *deletionCapable,
			Source:                 strings.TrimSpace(*transportSource),
		},
	})
	if strings.TrimSpace(*out) != "" {
		if err := writeJSONFile(*out, plan); err != nil {
			fmt.Fprintf(stderr, "fak vcache actions: write %q: %v\n", *out, err)
			return 1
		}
	}
	if *asJSON {
		return writeJSON(stdout, plan)
	}
	renderVCacheActions(stdout, plan)
	return 0
}

func renderVCacheActions(w io.Writer, plan vcacheobserve.ProviderActionPlan) {
	fmt.Fprintf(w, "vCache provider actions: %d turn(s), %d family(ies); noop=%d ready=%d gated=%d\n",
		plan.Turns,
		plan.FamilyCount,
		plan.Counts.Noop,
		plan.Counts.Ready,
		plan.Counts.Gated,
	)
	fmt.Fprintf(w, "transport: %s ready=%v — %s\n", plan.Transport.Mode, plan.Transport.Ready, plan.Transport.Reason)
	if len(plan.Actions) == 0 {
		fmt.Fprintln(w, "actions: none")
		return
	}
	fmt.Fprintln(w, "actions:")
	for _, row := range plan.Actions {
		fmt.Fprintf(w, "- %s: %s -> %s [%s], turns=%d, saved=%.1f, reason=%s\n",
			row.Family,
			row.Decision,
			row.Action,
			row.State,
			row.Turns,
			row.SavedTokenEquiv,
			row.Reason,
		)
	}
}

func runVCacheProve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	anchor := fs.Float64("anchor-tokens", 4096, "cacheable anchor size in input tokens")
	suffix := fs.Float64("suffix-tokens", 10, "fresh suffix tokens per sibling request")
	requests := fs.Int("requests", 7, "number of sibling requests sharing the anchor")
	minPrefix := fs.Float64("min-prefix-tokens", 1024, "provider minimum cacheable prefix")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read input-token multiplier")
	writeMult := fs.Float64("write-mult", vcachegov.WriteMult5Minutes, "provider cache-write input-token multiplier")
	content := fs.String("content", "public", "prefix content class: public, secret, regulated")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	proof := vcachegov.ProveStarSavings(vcachegov.StarSavingsInput{
		AnchorTokens:    *anchor,
		SuffixTokens:    *suffix,
		Requests:        *requests,
		MinPrefixTokens: *minPrefix,
		ReadMult:        *readMult,
		WriteMult:       *writeMult,
		Secret:          vcachegov.ClassifyPrefix(strings.ToLower(strings.TrimSpace(*content))),
	})
	if *asJSON {
		code := writeJSON(stdout, proof)
		if code != 0 {
			return code
		}
		return vcacheProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "requests: %d\n", proof.Requests)
	fmt.Fprintf(stdout, "anchor/suffix/min: %.0f / %.0f / %.0f tokens\n",
		proof.AnchorTokens, proof.SuffixTokens, proof.MinPrefixTokens)
	fmt.Fprintf(stdout, "read/write multipliers: %.3g / %.3g\n", proof.ReadMult, proof.WriteMult)
	fmt.Fprintf(stdout, "baseline token-equiv: %.1f\n", proof.BaselineTokenEquiv)
	fmt.Fprintf(stdout, "vcache token-equiv: %.1f\n", proof.VCacheTokenEquiv)
	fmt.Fprintf(stdout, "saved token-equiv: %.1f (%.1f%%)\n", proof.SavedTokenEquiv, proof.SavedPct)
	fmt.Fprintf(stdout, "break-even requests: %s\n", formatBreakEven(proof.BreakEvenRequests))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheProofExit(proof.Status)
}

func runVCacheProveRecall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove-recall", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	prefix := fs.Int64("prefix-tokens", 30000, "replayed warm prefix length in tokens (P)")
	unit := fs.Int64("unit-tokens", 10, "recalled unit fresh-prefill length in tokens (U)")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read token multiplier (r)")
	siblings := fs.Int("siblings", 1, "co-recalled sibling units sharing the prefix (S, the amortization)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	proof := vcachechain.ProveRecall(vcachechain.ProveRecallInput{
		PrefixTokens: *prefix,
		UnitTokens:   *unit,
		ReadMult:     *readMult,
		Siblings:     *siblings,
	})
	if *asJSON {
		if code := writeJSON(stdout, proof); code != 0 {
			return code
		}
		return vcacheRecallProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "decision: %s\n", proof.Decision)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "prefix/unit tokens: %d / %d\n", proof.PrefixTokens, proof.UnitTokens)
	fmt.Fprintf(stdout, "read multiplier: %.3g\n", proof.ReadMult)
	fmt.Fprintf(stdout, "siblings (amortization): %d\n", proof.Siblings)
	fmt.Fprintf(stdout, "replay cost (P·r): %.1f token-equiv\n", proof.ReplayCost)
	fmt.Fprintf(stdout, "fresh prefill (U): %.1f token-equiv\n", proof.FreshPrefillCost)
	fmt.Fprintf(stdout, "amortized savings (S·U): %.1f token-equiv\n", proof.AmortizedSavings)
	fmt.Fprintf(stdout, "single-unit loss ratio (P·r/U): %.1fx\n", proof.LossRatio)
	fmt.Fprintf(stdout, "break-even siblings: %s\n", formatBreakEven(proof.BreakEvenSiblings))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheRecallProofExit(proof.Status)
}

func vcacheRecallProofExit(s vcachechain.ProofStatus) int {
	if s == vcachechain.ProofProven {
		return 0
	}
	return 1
}

func runVCacheProveTelemetry(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove-telemetry", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	file := fs.String("file", "", "provider telemetry JSONL file ('-' for stdin)")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read input-token multiplier")
	write5mMult := fs.Float64("write-5m-mult", vcachegov.WriteMult5Minutes, "5m cache-write input-token multiplier")
	write1hMult := fs.Float64("write-1h-mult", vcachegov.WriteMult1Hour, "1h cache-write input-token multiplier")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if strings.TrimSpace(*file) == "" {
		fmt.Fprintln(stderr, "fak vcache prove-telemetry: --file is required")
		return 2
	}

	rows, err := readVCacheTelemetry(*file, os.Stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache prove-telemetry: %v\n", err)
		return 2
	}
	proof := vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
		Rows:        rows,
		ReadMult:    *readMult,
		Write5mMult: *write5mMult,
		Write1hMult: *write1hMult,
	})
	if *asJSON {
		code := writeJSON(stdout, proof)
		if code != 0 {
			return code
		}
		return vcacheProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "requests: %d\n", proof.Requests)
	fmt.Fprintf(stdout, "baseline token-equiv: %.1f\n", proof.BaselineTokenEquiv)
	fmt.Fprintf(stdout, "actual token-equiv: %.1f\n", proof.ActualTokenEquiv)
	fmt.Fprintf(stdout, "saved token-equiv: %.1f (%.2f%%)\n", proof.SavedTokenEquiv, proof.SavedPct)
	fmt.Fprintf(stdout, "cache read/write tokens: %.0f / %.0f\n", proof.CacheReadTokens, proof.CacheCreationTokens)
	fmt.Fprintf(stdout, "first positive request: %s\n", formatObservedPositive(proof.FirstPositiveRequest))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheProofExit(proof.Status)
}

type vcacheContextWitnessReport struct {
	Schema            string                       `json:"schema"`
	Fixture           string                       `json:"fixture"`
	Wire              string                       `json:"wire"`
	Snapshot          string                       `json:"snapshot"`
	ReplayExit        int                          `json:"replay_exit"`
	ScoreExit         int                          `json:"score_exit"`
	ScoreStatus       string                       `json:"score_status,omitempty"`
	ContextWitnessed  vcachescore.PlaneValueReport `json:"context_witnessed"`
	ContextEvents     int                          `json:"context_events"`
	ContextShedTokens float64                      `json:"context_shed_tokens"`
}

func runVCacheContextWitness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache context-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable witness report")
	snapshot := fs.String("snapshot", vcachesnapshot.DefaultContextPath(), "context witness snapshot path ('default' uses the per-user context path)")
	fixture := fs.String("fixture", defaultVCacheContextReplayFixturePath(), "guard replay trace fixture")
	wire := fs.String("wire", "openai", "replay wire: openai or anthropic")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	snapPath, ok := resolveVCacheContextSnapshotPath(*snapshot)
	if !ok {
		fmt.Fprintln(stderr, "fak vcache context-witness: --snapshot off disables the witness target")
		return 2
	}

	oldSnapshot, hadSnapshot := os.LookupEnv(vcachesnapshot.EnvPath)
	if err := os.Setenv(vcachesnapshot.EnvPath, snapPath); err != nil {
		fmt.Fprintf(stderr, "fak vcache context-witness: set %s: %v\n", vcachesnapshot.EnvPath, err)
		return 2
	}
	defer func() {
		if hadSnapshot {
			_ = os.Setenv(vcachesnapshot.EnvPath, oldSnapshot)
		} else {
			_ = os.Unsetenv(vcachesnapshot.EnvPath)
		}
	}()

	var replay bytes.Buffer
	replayExit := runGuardReplay(strings.TrimSpace(*fixture), strings.TrimSpace(*wire), "", &replay)

	var scoreOut, scoreErr bytes.Buffer
	scoreExit := runVCache(&scoreOut, &scoreErr, []string{"score", "--json", "--snapshot", "off", "--context-snapshot", snapPath, "--kernel-ledger", "off"})
	var score vcachescore.Report
	if err := json.Unmarshal(scoreOut.Bytes(), &score); err != nil && replayExit == 0 {
		fmt.Fprintf(stderr, "fak vcache context-witness: parse score: %v\n%s", err, scoreErr.String())
		return 2
	}

	report := vcacheContextWitnessReport{
		Schema:            "fak.vcache.context-witness.v1",
		Fixture:           strings.TrimSpace(*fixture),
		Wire:              normalizeReplayWire(*wire),
		Snapshot:          snapPath,
		ReplayExit:        replayExit,
		ScoreExit:         scoreExit,
		ScoreStatus:       score.Status,
		ContextWitnessed:  score.Planes.ContextWitnessed,
		ContextEvents:     score.AgenticActivation.ContextEvents,
		ContextShedTokens: score.Planes.ContextWitnessed.SavedTokenEquiv,
	}

	if *asJSON {
		if code := writeJSON(stdout, report); code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, replay.String())
		if strings.TrimSpace(scoreErr.String()) != "" {
			fmt.Fprintf(stdout, "\nfak vcache context-witness: score stderr:\n%s", scoreErr.String())
		}
		if report.ContextWitnessed.Available {
			fmt.Fprintf(stdout, "\nfak vcache context-witness: context %s - %d event(s), shed %.0f token-equiv; snapshot %s\n",
				report.ContextWitnessed.Provenance,
				report.ContextEvents,
				report.ContextShedTokens,
				report.Snapshot,
			)
			if report.Snapshot == vcachesnapshot.DefaultContextPath() {
				fmt.Fprintln(stdout, "fak vcache context-witness: default `fak vcache score --json` now composes this context snapshot unless FAK_VCACHE_SNAPSHOT is pinned.")
			} else {
				fmt.Fprintf(stdout, "fak vcache context-witness: score this custom snapshot with `fak vcache score --context-snapshot %s --json`.\n", report.Snapshot)
			}
		} else {
			fmt.Fprintf(stdout, "\nfak vcache context-witness: context MISSING after replay; snapshot %s\n", report.Snapshot)
		}
	}
	if replayExit != 0 {
		return replayExit
	}
	if !report.ContextWitnessed.Available || report.ContextEvents == 0 {
		return 1
	}
	return 0
}

func runVCacheScore(stdout, stderr io.Writer, argv []string) int {
	def := vcachescore.DefaultInput()
	fs := flag.NewFlagSet("vcache score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard")
	out := fs.String("out", "", "write machine-readable scorecard JSON to this file")
	telemetry := fs.String("telemetry", "", "optional provider telemetry JSONL file ('-' for stdin)")
	anchorsFile := fs.String("anchors-file", "", "optional ranked anchor workload JSONL/JSON/CSV file ('-' for stdin)")
	snapshotDefault := strings.TrimSpace(os.Getenv(vcachesnapshot.EnvPath))
	snapshot := fs.String("snapshot", snapshotDefault, "OBSERVED-by-default source: per-turn provider-cache window a finished `fak guard`/`fak serve` session persisted (default: $FAK_VCACHE_SNAPSHOT, then the well-known path under your config dir). When no --telemetry/--anchors-file is given and this snapshot has turns, the score reports the REALIZED cache multiplier from real traffic instead of the synthetic-Zipf FORECAST. Pass 'off' to force the planned forecast; an absent/empty snapshot falls open to the forecast (clearly labeled).")
	contextSnapshot := fs.String("context-snapshot", strings.TrimSpace(os.Getenv(vcachesnapshot.EnvContextPath)), "optional separate context-plane witness snapshot. With no explicit --snapshot this defaults to the well-known context path, so a no-key replay can prove context without overwriting provider telemetry. Pass 'off' to disable; 'default' uses the context snapshot path under your config dir.")
	indexOut := fs.String("index-out", "", "write selected hot-anchor index JSON to this file")
	anchor := fs.Float64("anchor-tokens", def.Star.AnchorTokens, "cacheable anchor size in input tokens")
	suffix := fs.Float64("suffix-tokens", def.Star.SuffixTokens, "fresh suffix tokens per sibling request")
	requests := fs.Int("requests", def.Star.Requests, "number of sibling requests sharing the anchor")
	minPrefix := fs.Float64("min-prefix-tokens", def.Star.MinPrefixTokens, "provider minimum cacheable prefix")
	readMult := fs.Float64("read-mult", def.Star.ReadMult, "provider cached-read input-token multiplier")
	writeMult := fs.Float64("write-mult", def.Star.WriteMult, "provider cache-write input-token multiplier")
	write5mMult := fs.Float64("write-5m-mult", vcachegov.WriteMult5Minutes, "5m cache-write input-token multiplier for telemetry")
	write1hMult := fs.Float64("write-1h-mult", vcachegov.WriteMult1Hour, "1h cache-write input-token multiplier for telemetry")
	content := fs.String("content", "public", "prefix content class: public, secret, regulated")
	zipfS := fs.Float64("zipf-s", 1.74, "synthetic workload Zipf exponent for hot-anchor concentration")
	anchors := fs.Int("anchors", 1000, "synthetic anchor universe size")
	targetCoverage := fs.Float64("target-coverage", def.TargetCoverage, "coverage target for the hot-anchor index")
	twoX := fs.Float64("two-x", def.TwoXThreshold, "multiplier gate required for success")
	maxFalseWarm := fs.Float64("max-false-warm-rate", def.MaxFalseWarmRate, "maximum tolerated false-warm rate")
	trueWarm := fs.Int("true-warm", 0, "prediction-error count: predicted warm and cache_read>0")
	falseWarm := fs.Int("false-warm", 0, "prediction-error count: predicted warm and cache_read=0")
	trueCold := fs.Int("true-cold", 0, "prediction-error count: predicted cold and cache_read=0")
	falseCold := fs.Int("false-cold", 0, "prediction-error count: predicted cold and cache_read>0")
	kernelKVEvents := fs.Int("kernel-kv-events", 0, "fak-authored pure-kernel KV cache events that fired")
	kernelKVPromptTokens := fs.Float64("kernel-kv-prompt-tokens", 0, "fak-owned KV witness: total prompt tokens prefetched by pure fak")
	kernelKVReusedTokens := fs.Float64("kernel-kv-reused-tokens", 0, "fak-owned KV witness: prompt tokens served from pure-fak KV prefix reuse")
	contextEvents := fs.Int("context-events", 0, "fak-authored O(1) context/query cache events that fired")
	contextShedTokens := fs.Float64("context-shed-tokens", 0, "O(1) context witness: prompt tokens removed from the live request body")
	contextResidentTokens := fs.Float64("context-resident-tokens", 0, "O(1) context witness: resident prompt tokens kept after compaction/planning")
	providerVCacheDecisions := fs.Int("provider-vcache-decisions", 0, "fak-authored provider-vcache action-plan decisions witnessed")
	externalEngineEvents := fs.Int("external-engine-events", 0, "fak-authored SGLang/vLLM/llama cache adapter events that fired")
	externalEngineHitRate := fs.Float64("external-engine-hit-rate", 0, "observed SGLang/vLLM/llama prefix-cache hit rate, 0..1")
	recallPrefix := fs.Int64("recall-prefix-tokens", def.Recall.PrefixTokens, "M4 recall proof prefix tokens (P)")
	recallUnit := fs.Int64("recall-unit-tokens", def.Recall.UnitTokens, "M4 recall proof unit tokens (U)")
	recallSiblings := fs.Int("recall-siblings", def.Recall.Siblings, "M4 recall proof sibling count (S)")
	recallReadMult := fs.Float64("recall-read-mult", def.Recall.ReadMult, "M4 recall cached-read token multiplier")
	kernelLedgerDefault := strings.TrimSpace(os.Getenv("FAK_VCACHE_KERNEL_LEDGER"))
	if kernelLedgerDefault == "" {
		kernelLedgerDefault = cachevalueledger.DefaultLedgerRel
	}
	kernelLedger := fs.String("kernel-ledger", kernelLedgerDefault, "durable cache-value ledger for fak-owned KV witness (default docs/nightrun/cache-value.jsonl; off disables)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	snapshotSet := flagWasSet(fs, "snapshot")
	contextSnapshotSet := flagWasSet(fs, "context-snapshot")
	if !contextSnapshotSet && !snapshotSet && snapshotDefault == "" && strings.TrimSpace(*contextSnapshot) == "" {
		*contextSnapshot = "default"
	}
	kernelLedgerSet := flagWasSet(fs, "kernel-ledger")
	if !kernelLedgerSet && (strings.TrimSpace(*telemetry) != "" || strings.TrimSpace(*anchorsFile) != "" || snapshotSet) {
		*kernelLedger = "off"
	}
	if strings.TrimSpace(*telemetry) == "-" && strings.TrimSpace(*anchorsFile) == "-" {
		fmt.Fprintln(stderr, "fak vcache score: --telemetry - and --anchors-file - cannot both read stdin")
		return 2
	}

	in := def
	in.Star = vcachegov.StarSavingsInput{
		AnchorTokens:    *anchor,
		SuffixTokens:    *suffix,
		Requests:        *requests,
		MinPrefixTokens: *minPrefix,
		ReadMult:        *readMult,
		WriteMult:       *writeMult,
		Secret:          vcachegov.ClassifyPrefix(strings.ToLower(strings.TrimSpace(*content))),
	}
	in.TelemetryReadMult = *readMult
	in.TelemetryWrite5m = *write5mMult
	in.TelemetryWrite1h = *write1hMult
	in.Ranked = vcachescore.SyntheticZipfWorkload(*zipfS, *anchors)
	in.TargetCoverage = *targetCoverage
	in.TwoXThreshold = *twoX
	in.MaxFalseWarmRate = *maxFalseWarm
	in.Prediction = vcachecal.PredictionError{
		Total:     *trueWarm + *falseWarm + *trueCold + *falseCold,
		TrueWarm:  *trueWarm,
		FalseWarm: *falseWarm,
		TrueCold:  *trueCold,
		FalseCold: *falseCold,
	}
	in.AgenticActivation = vcachescore.AgenticActivationInput{
		KernelKVEvents:          *kernelKVEvents,
		ContextEvents:           *contextEvents,
		ProviderVCacheDecisions: *providerVCacheDecisions,
		ExternalEngineEvents:    *externalEngineEvents,
	}
	if *kernelKVPromptTokens < 0 || *kernelKVReusedTokens < 0 {
		fmt.Fprintln(stderr, "fak vcache score: --kernel-kv-prompt-tokens and --kernel-kv-reused-tokens must be non-negative")
		return 2
	}
	if *kernelKVReusedTokens > 0 && *kernelKVPromptTokens <= 0 {
		fmt.Fprintln(stderr, "fak vcache score: --kernel-kv-reused-tokens requires --kernel-kv-prompt-tokens")
		return 2
	}
	if *kernelKVPromptTokens > 0 {
		reused := *kernelKVReusedTokens
		if reused > *kernelKVPromptTokens {
			reused = *kernelKVPromptTokens
		}
		if in.AgenticActivation.KernelKVEvents == 0 && reused > 0 {
			in.AgenticActivation.KernelKVEvents = 1
		}
		in.KernelKV = vcachescore.PlaneEvidenceInput{
			Available:          true,
			BaselineTokenEquiv: *kernelKVPromptTokens,
			SavedTokenEquiv:    reused,
			CostTokenEquiv:     *kernelKVPromptTokens - reused,
			Reason:             "fak-owned KV witness supplied by CLI",
		}
	}
	if !in.KernelKV.Available {
		applyVCacheKernelLedger(&in, *kernelLedger)
	}
	if *contextShedTokens < 0 || *contextResidentTokens < 0 {
		fmt.Fprintln(stderr, "fak vcache score: --context-shed-tokens and --context-resident-tokens must be non-negative")
		return 2
	}
	if *contextShedTokens > 0 {
		if in.AgenticActivation.ContextEvents == 0 {
			in.AgenticActivation.ContextEvents = 1
		}
		in.Context = vcachescore.PlaneEvidenceInput{
			Available:       true,
			SavedTokenEquiv: *contextShedTokens,
			Reason:          "O(1) context/query shed-token witness supplied by CLI",
		}
		if *contextResidentTokens > 0 {
			in.Context.BaselineTokenEquiv = *contextShedTokens + *contextResidentTokens
			in.Context.CostTokenEquiv = *contextResidentTokens
		}
	}
	if *externalEngineHitRate < 0 || *externalEngineHitRate > 1 {
		fmt.Fprintln(stderr, "fak vcache score: --external-engine-hit-rate must be between 0 and 1")
		return 2
	}
	if *externalEngineHitRate > 0 {
		in.ExternalEngine = vcachescore.PlaneEvidenceInput{
			Available:  true,
			Provenance: "OBSERVED",
			HitRate:    *externalEngineHitRate,
			Reason:     "external-engine prefix-cache hit rate supplied by CLI",
		}
	}
	in.Recall = vcachechain.ProveRecallInput{
		PrefixTokens: *recallPrefix,
		UnitTokens:   *recallUnit,
		ReadMult:     *recallReadMult,
		Siblings:     *recallSiblings,
	}
	if strings.TrimSpace(*anchorsFile) != "" {
		ranked, err := readVCacheAnchors(*anchorsFile, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache score: %v\n", err)
			return 2
		}
		in.Ranked = ranked
		in.AnchorSource = vcachescore.AnchorSourceMeasured
	}
	in.Ranked = vcachescore.NormalizeRanked(in.Ranked)
	if strings.TrimSpace(*telemetry) != "" {
		rows, err := readVCacheTelemetry(*telemetry, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache score: %v\n", err)
			return 2
		}
		in.TelemetryRows = rows
		in.TurnsObserved = len(rows)
	}
	// OBSERVED-by-default: with no explicit --telemetry and no --anchors-file, read the
	// persisted live cache window a finished guard/serve session left at the well-known path
	// and fold it through the SAME converter `fak vcache observe` uses. When it has turns the
	// score flips active_source to "telemetry" and reports the REALIZED multiplier; when it is
	// absent/empty/disabled we leave TelemetryRows nil so Score falls open to the planned
	// FORECAST (clearly labeled), never a phantom observed 0x.
	contextFromProviderSnapshot := false
	if len(in.TelemetryRows) == 0 && strings.TrimSpace(*anchorsFile) == "" {
		snapPath, readProviderSnapshot := resolveVCacheProviderSnapshotPath(*snapshot)
		if readProviderSnapshot {
			turns, ok, err := vcachesnapshot.Read(snapPath)
			if err != nil {
				fmt.Fprintf(stderr, "fak vcache score: snapshot %s: %v (falling open to the planned forecast)\n", snapPath, err)
			} else if ok {
				providerTurns := vcacheProviderTelemetryTurns(turns)
				if len(providerTurns) > 0 {
					observed := vcacheobserve.Observe(providerTurns, vcacheobserve.DefaultMultipliers())
					in.TelemetryRows = vcacheobserve.Rows(providerTurns)
					in.Ranked = vcacheobserve.RankedWorkload(providerTurns)
					in.Prediction = observed.Prediction
					in.AnchorSource = vcachescore.AnchorSourceMeasured
					in.TurnsObserved = len(providerTurns)
					applyVCacheProviderActionDecisions(&in, providerTurns)
				}
				contextFromProviderSnapshot = applyVCacheSnapshotContext(&in, turns, "persisted guard/serve context snapshot")
			}
		}
	}
	if !contextFromProviderSnapshot && strings.TrimSpace(*contextSnapshot) != "" {
		ctxPath, readContextSnapshot := resolveVCacheContextSnapshotPath(*contextSnapshot)
		if readContextSnapshot {
			turns, ok, err := vcachesnapshot.Read(ctxPath)
			if err != nil {
				fmt.Fprintf(stderr, "fak vcache score: context snapshot %s: %v (leaving context plane unchanged)\n", ctxPath, err)
			} else if ok {
				applyVCacheSnapshotContext(&in, turns, "persisted context snapshot "+ctxPath)
			}
		}
	}

	rep := vcachescore.Score(in)
	if strings.TrimSpace(*indexOut) != "" {
		artifact := vcachescore.BuildIndexArtifact(in.Ranked, rep.Index.TargetCoverage)
		if err := writeJSONFile(*indexOut, artifact); err != nil {
			fmt.Fprintf(stderr, "fak vcache score: %v\n", err)
			return 2
		}
	}
	if strings.TrimSpace(*out) != "" {
		if err := writeJSONFile(*out, rep); err != nil {
			fmt.Fprintf(stderr, "fak vcache score: %v\n", err)
			return 2
		}
	}
	if *asJSON {
		if code := writeJSON(stdout, rep); code != 0 {
			return code
		}
		if rep.TwoXBetter {
			return 0
		}
		return 1
	}

	fmt.Fprintf(stdout, "status: %s\n", rep.Status)
	fmt.Fprintf(stdout, "grade: %s (%d/100)\n", rep.Grade, rep.Score)
	fmt.Fprintf(stdout, "active source: %s\n", rep.ActiveSource)
	fmt.Fprintf(stdout, "anchor source: %s (turns observed %d)\n", rep.AnchorSource, rep.TurnsObserved)
	fmt.Fprintf(stdout, "active multiplier: %.2fx (target %.2fx)\n", rep.ActiveMultiplier, rep.TwoXThreshold)
	fmt.Fprintf(stdout, "2x gate: %s\n", passFail(rep.TwoXBetter))
	fmt.Fprintf(stdout, "planned proof: %s saved %.1f / %.1f (%.1f%%)\n",
		rep.Planned.Status, rep.Planned.SavedTokenEquiv, rep.Planned.BaselineTokenEquiv, rep.Planned.SavedPct)
	if rep.Observed != nil {
		fmt.Fprintf(stdout, "observed proof: %s saved %.1f / %.1f (%.2f%%), first positive request %s\n",
			rep.Observed.Status,
			rep.Observed.SavedTokenEquiv,
			rep.Observed.BaselineTokenEquiv,
			rep.Observed.SavedPct,
			formatObservedPositive(rep.Observed.FirstPositiveRequest))
	}
	if e := rep.Economics; e != nil {
		fmt.Fprintf(stdout, "economics (%s, %s): hit %.2f%% | read %.0f cached (write %.0f) | rebate %.1f (%.2f%%) | cost %.1f / %.1f baseline | %.2fx\n",
			e.Source, e.Witness, 100*e.HitRate, e.CacheReadTokens, e.CacheCreationTokens,
			e.RebateTokenEquiv, e.RebatePct, e.CostTokenEquiv, e.BaselineTokenEquiv, e.Multiplier)
	}
	fmt.Fprintf(stdout, "planes: provider=%s kernel=%s context=%s external=%s forecast=%s\n",
		planeLabel(rep.Planes.ProviderObserved),
		planeLabel(rep.Planes.KernelWitnessed),
		planeLabel(rep.Planes.ContextWitnessed),
		planeLabel(rep.Planes.ExternalEngineObserved),
		planeLabel(rep.Planes.Forecast))
	fmt.Fprintf(stdout, "agentic activation: %d events (kernel=%d context=%d provider-decisions=%d external=%d)\n",
		rep.AgenticActivation.Total,
		rep.AgenticActivation.KernelKVEvents,
		rep.AgenticActivation.ContextEvents,
		rep.AgenticActivation.ProviderVCacheDecisions,
		rep.AgenticActivation.ExternalEngineEvents)
	fmt.Fprintf(stdout, "default usefulness: %s (%s %d/100) - %s\n",
		rep.DefaultUsefulness.Verdict,
		rep.DefaultUsefulness.Grade,
		rep.DefaultUsefulness.Score,
		rep.DefaultUsefulness.Reason)
	fmt.Fprintf(stdout, "concentration: s=%.2f measured=%v defeated=%v\n",
		rep.Concentration.ZipfS, rep.Concentration.Measured, rep.Concentration.Defeated)
	fmt.Fprintf(stdout, "hot-anchor index: top %d covers %.1f%% (target %.1f%%)\n",
		rep.Index.AnchorCount, 100*rep.Index.Coverage, 100*rep.Index.TargetCoverage)
	if strings.TrimSpace(*indexOut) != "" {
		fmt.Fprintf(stdout, "hot-anchor index artifact: %s\n", *indexOut)
	}
	fmt.Fprintf(stdout, "prediction errors: false-warm %.2f%% false-cold %.2f%% (%d samples)\n",
		100*rep.Prediction.FalseWarmRate, 100*rep.Prediction.FalseColdRate, rep.Prediction.Total)
	fmt.Fprintf(stdout, "recall proof: %s decision=%s break-even siblings=%s\n",
		rep.Recall.Status, rep.Recall.Decision, formatBreakEven(rep.Recall.BreakEvenSiblings))
	if len(rep.Risks) > 0 {
		fmt.Fprintln(stdout, "risks:")
		for _, risk := range rep.Risks {
			fmt.Fprintf(stdout, "- %s\n", risk)
		}
	}
	fmt.Fprintln(stdout, "actions:")
	for _, action := range rep.Actions {
		fmt.Fprintf(stdout, "- %s\n", action)
	}
	fmt.Fprintln(stdout, "correctness depends on cache hit: false")
	if rep.TwoXBetter {
		return 0
	}
	return 1
}

func planeLabel(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "MISSING"
	}
	return p.Provenance
}

func applyVCacheKernelLedger(in *vcachescore.Input, path string) {
	if in == nil {
		return
	}
	path = strings.TrimSpace(path)
	if path == "" || strings.EqualFold(path, "off") {
		return
	}
	if strings.EqualFold(path, "default") {
		path = cachevalueledger.DefaultLedgerRel
	}
	res, err := cachevalueledger.ScoreLedger(path)
	if err != nil || !res.HasEnoughData() || res.GatePromptTokens == 0 || res.GateReusedTokens == 0 {
		return
	}
	cost := res.GatePromptTokens - res.GateReusedTokens
	if res.GateReusedTokens > res.GatePromptTokens {
		cost = 0
	}
	in.KernelKV = vcachescore.PlaneEvidenceInput{
		Available:          true,
		Provenance:         "WITNESSED",
		BaselineTokenEquiv: float64(res.GatePromptTokens),
		SavedTokenEquiv:    float64(res.GateReusedTokens),
		CostTokenEquiv:     float64(cost),
		Reason: fmt.Sprintf(
			"durable cache-value ledger %s witnessed %d multi-turn session(s), %d turn(s), and %.3f realized KV-prefix reuse",
			path,
			res.MultiTurnSessions,
			res.MultiTurnTurns,
			res.RealizedReuseRatio,
		),
	}
	if in.AgenticActivation.KernelKVEvents == 0 {
		in.AgenticActivation.KernelKVEvents = uint64ToIntVCache(res.MultiTurnTurns)
	}
}

func applyVCacheSnapshotContext(in *vcachescore.Input, turns []vcacheobserve.Turn, source string) bool {
	var events, shed, dropped, baseline, cost int64
	for _, t := range turns {
		events += t.ContextEvents
		shed += t.ContextShedTokens
		dropped += t.ContextDroppedTurns
		baseline += t.ContextBaselineTokens
		cost += t.ContextCostTokens
	}
	if events <= 0 && shed <= 0 && dropped <= 0 {
		return false
	}
	if events <= 0 && shed > 0 {
		events = 1
	}
	source = strings.TrimSpace(source)
	if source == "" {
		source = "persisted context snapshot"
	}
	ev := vcachescore.PlaneEvidenceInput{
		Available:       true,
		Provenance:      "WITNESSED",
		SavedTokenEquiv: float64(nonNegInt64(shed)),
		Reason: fmt.Sprintf(
			"%s witnessed %d context event(s), shed %d token(s), dropped %d turn(s)",
			source,
			nonNegInt64(events),
			nonNegInt64(shed),
			nonNegInt64(dropped),
		),
	}
	if baseline > 0 {
		ev.BaselineTokenEquiv = float64(baseline)
		if cost >= 0 {
			ev.CostTokenEquiv = float64(cost)
		}
	}
	in.Context = ev
	if in.AgenticActivation.ContextEvents == 0 {
		in.AgenticActivation.ContextEvents = int64ToInt(nonNegInt64(events))
	}
	return true
}

func applyVCacheProviderActionDecisions(in *vcachescore.Input, turns []vcacheobserve.Turn) bool {
	if in == nil {
		return false
	}
	plan := vcacheobserve.PlanProviderActions(turns, false)
	if len(plan.Actions) == 0 {
		return false
	}
	if in.AgenticActivation.ProviderVCacheDecisions == 0 {
		in.AgenticActivation.ProviderVCacheDecisions = len(plan.Actions)
	}
	return true
}

func nonNegInt64(n int64) int64 {
	if n < 0 {
		return 0
	}
	return n
}

func resolveVCacheProviderSnapshotPath(path string) (string, bool) {
	path = strings.TrimSpace(path)
	if strings.EqualFold(path, "off") {
		return "", false
	}
	if path == "" || strings.EqualFold(path, "default") {
		return vcachesnapshot.DefaultPath(), true
	}
	return path, true
}

func resolveVCacheContextSnapshotPath(path string) (string, bool) {
	path = strings.TrimSpace(path)
	if strings.EqualFold(path, "off") {
		return "", false
	}
	if path == "" || strings.EqualFold(path, "default") {
		return vcachesnapshot.DefaultContextPath(), true
	}
	return path, true
}

func int64ToInt(n int64) int {
	maxInt := int64(int(^uint(0) >> 1))
	if n > maxInt {
		return int(maxInt)
	}
	return int(n)
}

func uint64ToIntVCache(n uint64) int {
	maxInt := uint64(int(^uint(0) >> 1))
	if n > maxInt {
		return int(maxInt)
	}
	return int(n)
}

func defaultVCacheStatus() vcacheStatusReport {
	rep := vcacheStatusReport{
		Status:       "M5 governor decision witness and provider-action planner live; spendful heartbeat/explicit-cache transport still gated; full vCache provider loop not yet executing warms",
		Governor:     "decision witness live (/metrics + /debug/vars journal) and provider action plan live (GET /v1/fak/vcache/actions; fak vcache actions); heartbeat/explicit-cache transport gated",
		Chains:       "implemented (prefix DAG, topological replay, cost-gated rebuild); gated OFF by default; off-path",
		LiveProvider: "provider-cache window wired; provider action planner/API live; heartbeat/explicit-cache execution remains gated until prefix/capability evidence exists",
		Proof: vcachegov.ProveStarSavings(vcachegov.StarSavingsInput{
			AnchorTokens:    4096,
			SuffixTokens:    10,
			Requests:        7,
			MinPrefixTokens: 1024,
			ReadMult:        0.1,
			WriteMult:       vcachegov.WriteMult5Minutes,
			Secret:          vcachegov.Cacheable,
		}),
		RecallProof: vcachechain.ProveRecall(vcachechain.ProveRecallInput{
			PrefixTokens: 30000,
			UnitTokens:   10,
			ReadMult:     0.1,
			Siblings:     1,
		}),
		CodexOpenAI: defaultCodexOpenAIStatus(),
		ContextAPI:  defaultVCacheContextAPIStatus(),
		ProviderCalibration: vcacheProviderCalStatus{
			Verifier:  "ready",
			CLI:       "fak vcache calibrate",
			Input:     "provider-cache probe samples JSON/JSONL",
			Output:    "vcachecal.Calibration JSON",
			Consumer:  "fak vcache observe --calibration",
			LiveProbe: "operator-supplied samples; no spendful probe transport is auto-run",
			Reason:    "fits provider TTL, minimum prefix, and cached-read multiplier from replayed probe samples instead of hard-coded hypotheses",
		},
		ProviderActions: vcacheProviderActionStatus{
			Verifier:  "ready",
			HTTP:      "GET /v1/fak/vcache/actions",
			CLI:       "fak vcache actions",
			Schema:    vcacheobserve.ProviderActionSchema,
			ReadOnly:  true,
			Transport: "decision_only",
			Reason:    "maps observed provider-cache families to noop/ready/gated action rows; heartbeat/explicit-cache rows remain gated until prefix and provider-capability evidence exists",
		},
		M4Issue: "https://github.com/anthony-chaudhary/fak/issues/719",
		M5Issue: "https://github.com/anthony-chaudhary/fak/issues/720",
		Remaining: []vcacheRemainingIssue{
			{716, "M1 observe & calibrate", "https://github.com/anthony-chaudhary/fak/issues/716"},
			{717, "M2 star anchors", "https://github.com/anthony-chaudhary/fak/issues/717"},
			{718, "M3 dedicated warming", "https://github.com/anthony-chaudhary/fak/issues/718"},
		},
		CorrectnessLaw: "cost is budgeted at the uncached price; hits are realized rebates, never trust claims",
	}
	applyRecentVCacheObservation(&rep, statusVCacheSnapshotPath(), statusVCacheContextSnapshotPath())
	return rep
}

func statusVCacheSnapshotPath() string {
	path := strings.TrimSpace(os.Getenv(vcachesnapshot.EnvPath))
	if path == "" {
		return vcachesnapshot.DefaultPath()
	}
	return path
}

func statusVCacheContextSnapshotPath() string {
	path := strings.TrimSpace(os.Getenv(vcachesnapshot.EnvContextPath))
	if path != "" {
		return path
	}
	if strings.TrimSpace(os.Getenv(vcachesnapshot.EnvPath)) != "" {
		return ""
	}
	return vcachesnapshot.DefaultContextPath()
}

func applyRecentVCacheObservation(rep *vcacheStatusReport, path, contextPath string) {
	path = strings.TrimSpace(path)
	if path == "" || strings.EqualFold(path, "off") {
		applyRecentVCacheContextOnlyObservation(rep, contextPath)
		return
	}
	turns, ok, err := vcachesnapshot.Read(path)
	if err != nil {
		rep.RecentObservationError = fmt.Sprintf("%s: %v", path, err)
		return
	}
	if !ok {
		applyRecentVCacheContextOnlyObservation(rep, contextPath)
		return
	}
	providerTurns := vcacheProviderTelemetryTurns(turns)
	obs := vcacheobserve.Observe(providerTurns, vcacheobserve.DefaultMultipliers())
	recent := vcacheRecentObservation{
		Source:              "snapshot",
		Path:                path,
		Turns:               len(turns),
		ProviderStatus:      "MISSING",
		CacheReadTokens:     obs.Aggregate.CacheReadTokens,
		CacheCreationTokens: obs.Aggregate.CacheCreationTokens,
		HitRate:             obs.HitRate,
		Multiplier:          obs.Multiplier,
		SavedTokenEquiv:     obs.Aggregate.SavedTokenEquiv,
		FalseWarmRate:       obs.Prediction.FalseWarmRate(),
		FalseColdRate:       obs.Prediction.FalseColdRate(),
	}
	if len(providerTurns) > 0 {
		recent.ProviderStatus = string(obs.Aggregate.Status)
		recent.GovernorDecision = dominantVCacheGovernorDecision(obs.Families)
	}
	for _, turn := range turns {
		recent.ContextEvents += turn.ContextEvents
		recent.ContextShedTokens += turn.ContextShedTokens
		recent.ContextDroppedTurns += turn.ContextDroppedTurns
		recent.ContextBaselineTokens += turn.ContextBaselineTokens
		recent.ContextCostTokens += turn.ContextCostTokens
	}
	recent.ContextStatus, recent.ContextReason = recentVCacheContextStatus(recent)
	if recent.ContextStatus == "MISSING" {
		applyRecentVCacheContextSnapshot(&recent, contextPath)
	}
	rep.RecentObservation = &recent
	if len(providerTurns) > 0 {
		rep.LiveProvider = fmt.Sprintf("provider-cache window wired; recent snapshot observed %d provider turn(s) at %.2fx multiplier with %.2f%% false-warm; provider action planner live, heartbeat/explicit-cache execution gated",
			len(providerTurns), recent.Multiplier, 100*recent.FalseWarmRate)
	} else {
		rep.LiveProvider = fmt.Sprintf("provider-cache window wired; recent snapshot has no provider-cache telemetry; context status %s with %d event(s); provider action planner waiting on provider turns",
			recent.ContextStatus, recent.ContextEvents)
	}
}

func applyRecentVCacheContextOnlyObservation(rep *vcacheStatusReport, contextPath string) {
	contextPath = strings.TrimSpace(contextPath)
	if contextPath == "" || strings.EqualFold(contextPath, "off") {
		return
	}
	contextPath, readContextSnapshot := resolveVCacheContextSnapshotPath(contextPath)
	if !readContextSnapshot {
		return
	}
	turns, ok, err := vcachesnapshot.Read(contextPath)
	if err != nil {
		rep.RecentObservationError = fmt.Sprintf("%s: %v", contextPath, err)
		return
	}
	if !ok {
		return
	}
	recent := vcacheRecentObservation{
		Source:         "context_snapshot",
		Path:           contextPath,
		Turns:          len(turns),
		ProviderStatus: "MISSING",
		ContextPath:    contextPath,
	}
	for _, turn := range turns {
		recent.ContextEvents += turn.ContextEvents
		recent.ContextShedTokens += turn.ContextShedTokens
		recent.ContextDroppedTurns += turn.ContextDroppedTurns
		recent.ContextBaselineTokens += turn.ContextBaselineTokens
		recent.ContextCostTokens += turn.ContextCostTokens
	}
	recent.ContextStatus, recent.ContextReason = recentVCacheContextStatus(recent)
	if recent.ContextStatus != "WITNESSED" {
		return
	}
	recent.ContextReason = "separate context snapshot includes fak_context_* counters from a guard/serve context event"
	rep.RecentObservation = &recent
	rep.LiveProvider = fmt.Sprintf("provider-cache window wired; no provider-cache telemetry found; context status WITNESSED from %s with %d event(s); provider action planner waiting on provider turns",
		contextPath, recent.ContextEvents)
}

func applyRecentVCacheContextSnapshot(recent *vcacheRecentObservation, contextPath string) bool {
	if recent == nil {
		return false
	}
	contextPath = strings.TrimSpace(contextPath)
	if contextPath == "" || strings.EqualFold(contextPath, "off") {
		return false
	}
	resolved, readContextSnapshot := resolveVCacheContextSnapshotPath(contextPath)
	if !readContextSnapshot || resolved == recent.Path {
		return false
	}
	turns, ok, err := vcachesnapshot.Read(resolved)
	if err != nil || !ok {
		return false
	}
	var ctx vcacheRecentObservation
	for _, turn := range turns {
		ctx.ContextEvents += turn.ContextEvents
		ctx.ContextShedTokens += turn.ContextShedTokens
		ctx.ContextDroppedTurns += turn.ContextDroppedTurns
		ctx.ContextBaselineTokens += turn.ContextBaselineTokens
		ctx.ContextCostTokens += turn.ContextCostTokens
	}
	status, _ := recentVCacheContextStatus(ctx)
	if status != "WITNESSED" {
		return false
	}
	recent.ContextPath = resolved
	recent.ContextEvents = ctx.ContextEvents
	recent.ContextShedTokens = ctx.ContextShedTokens
	recent.ContextDroppedTurns = ctx.ContextDroppedTurns
	recent.ContextBaselineTokens = ctx.ContextBaselineTokens
	recent.ContextCostTokens = ctx.ContextCostTokens
	recent.ContextStatus = "WITNESSED"
	recent.ContextReason = "separate context snapshot includes fak_context_* counters from a guard/serve context event"
	return true
}

func recentVCacheContextStatus(recent vcacheRecentObservation) (string, string) {
	if recent.ContextEvents > 0 || recent.ContextShedTokens > 0 || recent.ContextDroppedTurns > 0 ||
		recent.ContextBaselineTokens > 0 || recent.ContextCostTokens > 0 {
		return "WITNESSED", "snapshot includes fak_context_* counters from a guard/serve context event"
	}
	return "MISSING", "snapshot has provider-cache turns but no fak_context_* counters; it predates context instrumentation or no managed-context event fired"
}

func vcacheProviderTelemetryTurns(turns []vcacheobserve.Turn) []vcacheobserve.Turn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]vcacheobserve.Turn, 0, len(turns))
	for _, turn := range turns {
		if vcacheTurnHasProviderTelemetry(turn) {
			out = append(out, turn)
		}
	}
	return out
}

func vcacheTurnHasProviderTelemetry(turn vcacheobserve.Turn) bool {
	return turn.InputTokens > 0 ||
		turn.CacheRead > 0 ||
		turn.CacheCreation > 0 ||
		turn.Ephemeral1h > 0 ||
		turn.Ephemeral5m > 0
}

type vcacheSessionSummaryOptions struct {
	SinceDays       float64
	Max             int
	NamespacePrefix string
	AllNamespaces   bool
}

func applyRecentSessionSummary(rep *vcacheStatusReport, opts vcacheSessionSummaryOptions) {
	var since *float64
	if opts.SinceDays >= 0 {
		v := opts.SinceDays
		since = &v
	}
	nsPrefix := strings.TrimSpace(opts.NamespacePrefix)
	if opts.AllNamespaces {
		nsPrefix = ""
	} else if nsPrefix == "" {
		cwd, err := os.Getwd()
		if err != nil {
			rep.RecentSessionsError = fmt.Sprintf("current workspace namespace: %v", err)
			return
		}
		nsPrefix = sessionaudit.ProjectNamespace(cwd)
	}
	discover := sessionaudit.DiscoverOptions{
		SinceDays:       since,
		NamespacePrefix: nsPrefix,
	}
	recs, err := sessionaudit.Discover(discover)
	if err != nil {
		rep.RecentSessionsError = err.Error()
		return
	}
	totalDiscovered := len(recs)
	if opts.Max > 0 && len(recs) > opts.Max {
		recs = recs[:opts.Max]
	}
	sessions := make([]sessionaudit.Session, 0, len(recs))
	for _, rec := range recs {
		if rec.Kind == "subagent" {
			continue
		}
		s := sessionaudit.Analyze(rec.Path)
		s.Kind = rec.Kind
		sessions = append(sessions, s)
	}
	agg := sessionaudit.AggregateSessions(sessions)
	summary := sessionaudit.BuildCompactReport(sessions, agg, nsPrefix, since, false, opts.Max, totalDiscovered, nil, time.Now())
	rep.RecentSessions = &summary
}

func printVCacheSessionSummary(w io.Writer, summary sessionaudit.CompactReport) {
	fmt.Fprintf(w, "recent sessions: %d/%d sessions, scope %s, context %d tok, cache-read %.1f%%, I:O %.1f, cost $%.2f",
		summary.Scope.Audited,
		summary.Scope.Discovered,
		summary.Scope.NamespaceFilter,
		summary.Totals.TotalContextTokens,
		100*summary.Totals.CacheReadShare,
		summary.Totals.IORatio,
		summary.Totals.EstimatedCostUSD)
	if summary.Scope.Clipped {
		fmt.Fprint(w, " (clipped)")
	}
	fmt.Fprintln(w)
	for _, tier := range summary.Tiers {
		if tier.Tier == "fable" || tier.Tier == "opus" {
			fmt.Fprintf(w, "  %s: output %d (%.1f%%), cost $%.2f (%.1f%%)\n",
				tier.Tier,
				tier.OutputTokens,
				100*tier.OutputShare,
				tier.EstimatedCostUSD,
				100*tier.CostShare)
		}
	}
	if len(summary.TopLongContext) > 0 {
		top := summary.TopLongContext[0]
		fmt.Fprintf(w, "  top long-context: %s context %d tok, cache-read %.1f%%, model %s\n",
			top.Session,
			top.TotalContextTokens,
			100*top.CacheReadShare,
			top.TopModel)
	}
	for _, rec := range summary.Recommendations {
		fmt.Fprintf(w, "  recommendation: %s [%s] %s (%s)\n",
			rec.Kind,
			rec.Severity,
			rec.Action,
			rec.Evidence,
		)
	}
}

func dominantVCacheGovernorDecision(families []vcacheobserve.Family) string {
	if len(families) == 0 {
		return ""
	}
	counts := make(map[string]int, len(families))
	best := ""
	for _, family := range families {
		decision := string(family.GovernorDecision)
		counts[decision]++
		if best == "" || counts[decision] > counts[best] || counts[decision] == counts[best] && decision < best {
			best = decision
		}
	}
	return best
}
