package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
	case "context-join":
		return runVCacheContextJoin(stdout, stderr, argv[1:])
	case "codex-session-extract":
		return runVCacheCodexSessionExtract(stdout, stderr, argv[1:])
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

func vcacheUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak vcache status [--json]
  fak vcache prove [--json] [--anchor-tokens N] [--suffix-tokens N] [--requests N]
                   [--min-prefix-tokens N] [--read-mult F] [--write-mult F]
                   [--content public|secret|regulated]
  fak vcache prove-telemetry --file FILE [--json]
                   [--read-mult F] [--write-5m-mult F] [--write-1h-mult F]
  fak vcache prove-recall [--json] [--prefix-tokens N] [--unit-tokens N]
                   [--read-mult F] [--siblings N]
  fak vcache observe [--transcript FILE]... [--telemetry FILE] [--json]
                   [--read-mult F] [--write-5m-mult F] [--write-1h-mult F]
  fak vcache context-join [--transcript FILE]... [--telemetry FILE] --events FILE
                   [--json] [--before-millis N] [--after-millis N]
  fak vcache codex-session-extract [--session FILE | --thread-id ID] --out FILE
                   [--snapshot-out FILE|default] [--score-out FILE] [--family NAME]
  fak vcache score|bench [--json] [--out FILE] [--telemetry FILE] [--two-x F]
                   [--anchor-tokens N --suffix-tokens N --requests N]
                   [--read-mult F --write-mult F --write-5m-mult F --write-1h-mult F]
                   [--zipf-s F --anchors N --anchors-file FILE --target-coverage F]
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

`)
}

type vcacheStatusReport struct {
	Status         string                     `json:"status"`
	Governor       string                     `json:"governor"`
	Chains         string                     `json:"chains"`
	LiveProvider   string                     `json:"live_provider"`
	Proof          vcachegov.StarSavingsProof `json:"proof"`
	RecallProof    vcachechain.RecallProof    `json:"recall_proof"`
	CodexOpenAI    vcacheCodexOpenAIStatus    `json:"codex_openai"`
	M4Issue        string                     `json:"m4_issue"`
	M5Issue        string                     `json:"m5_issue"`
	Remaining      []vcacheRemainingIssue     `json:"remaining"`
	CorrectnessLaw string                     `json:"correctness_law"`
}

type vcacheRemainingIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	URL    string `json:"url"`
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

func runVCacheStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable status")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	rep := defaultVCacheStatus()
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	fmt.Fprintf(stdout, "vCache status: %s\n", rep.Status)
	fmt.Fprintf(stdout, "vCache M5 governor: %s\n", rep.Governor)
	fmt.Fprintf(stdout, "vCache M4 chains & recall: %s\n", rep.Chains)
	fmt.Fprintf(stdout, "live provider loop: %s\n", rep.LiveProvider)
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

func runVCacheScore(stdout, stderr io.Writer, argv []string) int {
	def := vcachescore.DefaultInput()
	fs := flag.NewFlagSet("vcache score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard")
	out := fs.String("out", "", "write machine-readable scorecard JSON to this file")
	telemetry := fs.String("telemetry", "", "optional provider telemetry JSONL file ('-' for stdin)")
	anchorsFile := fs.String("anchors-file", "", "optional ranked anchor workload JSONL/JSON/CSV file ('-' for stdin)")
	snapshotDefault := strings.TrimSpace(os.Getenv("FAK_VCACHE_SNAPSHOT"))
	snapshot := fs.String("snapshot", snapshotDefault, "OBSERVED-by-default source: per-turn provider-cache window a finished `fak guard`/`fak serve` session persisted (default: $FAK_VCACHE_SNAPSHOT, then the well-known path under your config dir). When no --telemetry/--anchors-file is given and this snapshot has turns, the score reports the REALIZED cache multiplier from real traffic instead of the synthetic-Zipf FORECAST. Pass 'off' to force the planned forecast; an absent/empty snapshot falls open to the forecast (clearly labeled).")
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
	providerVCacheDecisions := fs.Int("provider-vcache-decisions", 0, "fak-authored provider-vcache warm/pin/evict decisions that fired")
	externalEngineEvents := fs.Int("external-engine-events", 0, "fak-authored SGLang/vLLM/llama cache adapter events that fired")
	externalEngineHitRate := fs.Float64("external-engine-hit-rate", 0, "observed SGLang/vLLM/llama prefix-cache hit rate, 0..1")
	recallPrefix := fs.Int64("recall-prefix-tokens", def.Recall.PrefixTokens, "M4 recall proof prefix tokens (P)")
	recallUnit := fs.Int64("recall-unit-tokens", def.Recall.UnitTokens, "M4 recall proof unit tokens (U)")
	recallSiblings := fs.Int("recall-siblings", def.Recall.Siblings, "M4 recall proof sibling count (S)")
	recallReadMult := fs.Float64("recall-read-mult", def.Recall.ReadMult, "M4 recall cached-read token multiplier")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
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
	if len(in.TelemetryRows) == 0 && strings.TrimSpace(*anchorsFile) == "" && !strings.EqualFold(strings.TrimSpace(*snapshot), "off") {
		snapPath := strings.TrimSpace(*snapshot)
		if snapPath == "" {
			snapPath = vcachesnapshot.DefaultPath()
		}
		turns, ok, err := vcachesnapshot.Read(snapPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache score: snapshot %s: %v (falling open to the planned forecast)\n", snapPath, err)
		} else if ok {
			in.TelemetryRows = vcacheobserve.Rows(turns)
			in.Ranked = vcacheobserve.RankedWorkload(turns)
			in.AnchorSource = vcachescore.AnchorSourceMeasured
			in.TurnsObserved = len(turns)
			applyVCacheSnapshotContext(&in, turns)
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

func applyVCacheSnapshotContext(in *vcachescore.Input, turns []vcacheobserve.Turn) {
	var events, shed, dropped, baseline, cost int64
	for _, t := range turns {
		events += t.ContextEvents
		shed += t.ContextShedTokens
		dropped += t.ContextDroppedTurns
		baseline += t.ContextBaselineTokens
		cost += t.ContextCostTokens
	}
	if events <= 0 && shed <= 0 && dropped <= 0 {
		return
	}
	if events <= 0 && shed > 0 {
		events = 1
	}
	ev := vcachescore.PlaneEvidenceInput{
		Available:       true,
		Provenance:      "WITNESSED",
		SavedTokenEquiv: float64(nonNegInt64(shed)),
		Reason: fmt.Sprintf(
			"persisted guard/serve context snapshot witnessed %d context event(s), shed %d token(s), dropped %d turn(s)",
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
}

func nonNegInt64(n int64) int64 {
	if n < 0 {
		return 0
	}
	return n
}

func int64ToInt(n int64) int {
	maxInt := int64(int(^uint(0) >> 1))
	if n > maxInt {
		return int(maxInt)
	}
	return int(n)
}

func defaultVCacheStatus() vcacheStatusReport {
	return vcacheStatusReport{
		Status:       "M5 governor decision witness live; warm/pin/evict actions still gated; full vCache provider loop not yet live",
		Governor:     "decision witness live (/metrics + /debug/vars journal); pin/lazy/evict actions not registered",
		Chains:       "implemented (prefix DAG, topological replay, cost-gated rebuild); gated OFF by default; off-path",
		LiveProvider: "passive provider-cache window wired; M1-M3 remain open; Codex/OpenAI telemetry #727 proven from replayable artifacts",
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
		M4Issue:     "https://github.com/anthony-chaudhary/fak/issues/719",
		M5Issue:     "https://github.com/anthony-chaudhary/fak/issues/720",
		Remaining: []vcacheRemainingIssue{
			{716, "M1 observe & calibrate", "https://github.com/anthony-chaudhary/fak/issues/716"},
			{717, "M2 star anchors", "https://github.com/anthony-chaudhary/fak/issues/717"},
			{718, "M3 dedicated warming", "https://github.com/anthony-chaudhary/fak/issues/718"},
		},
		CorrectnessLaw: "cost is budgeted at the uncached price; hits are realized rebates, never trust claims",
	}
}

func defaultCodexOpenAIStatus() vcacheCodexOpenAIStatus {
	hasKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != ""
	live := "proven (Codex CLI replay artifact)"
	reason := "replay experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl with prove-telemetry; raw OpenAI API probe not run because OPENAI_API_KEY is not present"
	if hasKey {
		reason = "Codex CLI replay artifact is tracked; OPENAI_API_KEY is present, so tools/vcache_openai_probe.py can refresh the optional raw API probe"
	}
	return vcacheCodexOpenAIStatus{
		Verifier:            "ready",
		LiveTelemetry:       live,
		Reason:              reason,
		OpenAIAPIKeyPresent: hasKey,
		CachedTokenFields: []string{
			"usage.input_tokens_details.cached_tokens",
			"usage.prompt_tokens_details.cached_tokens",
			"usage.cached_input_tokens",
			"payload.info.last_token_usage.cached_input_tokens",
		},
		Issue: "https://github.com/anthony-chaudhary/fak/issues/727",
		CachedSampleProof: vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows:     []vcachegov.TelemetryRow{openAITelemetryRow(2006, 1920)},
			ReadMult: 0.1,
		}),
		NoCacheRefutation: vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows:     []vcachegov.TelemetryRow{openAITelemetryRow(2006, 0)},
			ReadMult: 0.1,
		}),
	}
}

func writeJSON(w io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return 2
	}
	fmt.Fprintln(w, string(b))
	return 0
}

func writeJSONFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func vcacheProofExit(s vcachegov.ProofStatus) int {
	if s == vcachegov.ProofProven {
		return 0
	}
	return 1
}

func formatBreakEven(n int) string {
	if n == int(^uint(0)>>1) {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

func formatObservedPositive(n int) string {
	if n <= 0 {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

func passFail(ok bool) string {
	if ok {
		return "pass"
	}
	return "fail"
}

type vcacheAnchorInput struct {
	Key              string  `json:"key"`
	Anchor           string  `json:"anchor"`
	ID               string  `json:"id"`
	PrefixDigest     string  `json:"prefix_digest"`
	Frequency        float64 `json:"frequency"`
	Freq             float64 `json:"freq"`
	Count            float64 `json:"count"`
	AccessRatePerSec float64 `json:"access_rate_per_sec"`
	Size             float64 `json:"size"`
	Tokens           float64 `json:"tokens"`
	PrefixTokens     float64 `json:"prefix_tokens"`
	ReuseDensity     float64 `json:"reuse_density"`
	Reuse            float64 `json:"reuse"`
	Reuses           float64 `json:"reuses"`
	ExpectedReuse    float64 `json:"expected_reuse"`
	Weight           float64 `json:"weight"`
}

func readVCacheAnchors(path string, stdin io.Reader) ([]vcachecal.RankedVBlock, error) {
	var data []byte
	var err error
	if path == "-" {
		if stdin == nil {
			return nil, errors.New("stdin is not available")
		}
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, err
	}
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, errors.New("anchor workload is empty")
	}

	var rows []vcacheAnchorInput
	switch trimmed[0] {
	case '[':
		if err := json.Unmarshal(trimmed, &rows); err != nil {
			return nil, err
		}
	case '{':
		rows, err = readVCacheAnchorJSONL(trimmed)
	default:
		rows, err = readVCacheAnchorCSV(trimmed)
	}
	if err != nil {
		return nil, err
	}
	ranked := vcachescore.NormalizeRanked(anchorInputsToRanked(rows))
	if len(ranked) == 0 {
		return nil, errors.New("anchor workload has no positive-weight rows")
	}
	return ranked, nil
}

func readVCacheAnchorJSONL(raw []byte) ([]vcacheAnchorInput, error) {
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []vcacheAnchorInput
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row vcacheAnchorInput
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("anchor line %d: %w", lineNo, err)
		}
		rows = append(rows, row)
	}
	return rows, sc.Err()
}

func readVCacheAnchorCSV(raw []byte) ([]vcacheAnchorInput, error) {
	cr := csv.NewReader(bytes.NewReader(raw))
	cr.TrimLeadingSpace = true
	cr.FieldsPerRecord = -1
	records, err := cr.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, errors.New("anchor CSV is empty")
	}
	header := map[string]int{}
	for i, h := range records[0] {
		header[normalizeVCacheAnchorField(h)] = i
	}
	rows := make([]vcacheAnchorInput, 0, len(records)-1)
	for i, rec := range records[1:] {
		row, err := parseVCacheAnchorCSVRecord(header, rec)
		if err != nil {
			return nil, fmt.Errorf("anchor CSV row %d: %w", i+2, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func parseVCacheAnchorCSVRecord(header map[string]int, rec []string) (vcacheAnchorInput, error) {
	var row vcacheAnchorInput
	row.Key = csvString(header, rec, "key", "anchor", "id", "prefix_digest")
	var err error
	if row.Frequency, err = csvFloat(header, rec, "frequency", "freq", "count", "access_rate_per_sec"); err != nil {
		return row, err
	}
	if row.Size, err = csvFloat(header, rec, "size", "tokens", "prefix_tokens"); err != nil {
		return row, err
	}
	if row.ReuseDensity, err = csvFloat(header, rec, "reuse_density", "reuse", "reuses", "expected_reuse"); err != nil {
		return row, err
	}
	if row.Weight, err = csvFloat(header, rec, "weight"); err != nil {
		return row, err
	}
	return row, nil
}

func normalizeVCacheAnchorField(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.ReplaceAll(s, " ", "_")
	return s
}

func csvString(header map[string]int, rec []string, names ...string) string {
	for _, name := range names {
		if idx, ok := header[name]; ok && idx < len(rec) {
			return strings.TrimSpace(rec[idx])
		}
	}
	return ""
}

func csvFloat(header map[string]int, rec []string, names ...string) (float64, error) {
	for _, name := range names {
		idx, ok := header[name]
		if !ok || idx >= len(rec) {
			continue
		}
		s := strings.TrimSpace(rec[idx])
		if s == "" {
			return 0, nil
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, fmt.Errorf("%s=%q: %w", name, s, err)
		}
		return v, nil
	}
	return 0, nil
}

func anchorInputsToRanked(rows []vcacheAnchorInput) []vcachecal.RankedVBlock {
	out := make([]vcachecal.RankedVBlock, 0, len(rows))
	for _, row := range rows {
		v := vcachecal.RankedVBlock{
			Key:          firstAnchorString(row.Key, row.Anchor, row.ID, row.PrefixDigest),
			Frequency:    firstAnchorFloat(row.Frequency, row.Freq, row.Count, row.AccessRatePerSec),
			Size:         firstAnchorFloat(row.Size, row.Tokens, row.PrefixTokens),
			ReuseDensity: firstAnchorFloat(row.ReuseDensity, row.Reuse, row.Reuses, row.ExpectedReuse),
		}
		if row.Weight != 0 {
			v.Frequency = row.Weight
			v.Size = 1
			v.ReuseDensity = 1
		}
		out = append(out, v)
	}
	return out
}

func firstAnchorString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func firstAnchorFloat(values ...float64) float64 {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}

type vcacheTelemetryJSONLRow struct {
	InputTokens              float64             `json:"input_tokens"`
	PromptTokens             float64             `json:"prompt_tokens"`
	CachedTokens             float64             `json:"cached_tokens"`
	CacheCreationInputTokens float64             `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     float64             `json:"cache_read_input_tokens"`
	Ephemeral1hInputTokens   float64             `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens   float64             `json:"ephemeral_5m_input_tokens"`
	Usage                    *vcacheOpenAIUsage  `json:"usage"`
	Type                     string              `json:"type"`
	Payload                  *vcacheCodexPayload `json:"payload"`
}

type vcacheOpenAIUsage struct {
	InputTokens         float64                   `json:"input_tokens"`
	PromptTokens        float64                   `json:"prompt_tokens"`
	CachedInputTokens   float64                   `json:"cached_input_tokens"`
	InputTokensDetails  vcacheCachedTokensDetails `json:"input_tokens_details"`
	PromptTokensDetails vcacheCachedTokensDetails `json:"prompt_tokens_details"`
}

type vcacheCachedTokensDetails struct {
	CachedTokens float64 `json:"cached_tokens"`
}

type vcacheCodexPayload struct {
	Type string               `json:"type"`
	Info vcacheCodexTokenInfo `json:"info"`
}

type vcacheCodexTokenInfo struct {
	LastTokenUsage vcacheCodexTokenUsage `json:"last_token_usage"`
}

type vcacheCodexTokenUsage struct {
	InputTokens       float64 `json:"input_tokens"`
	CachedInputTokens float64 `json:"cached_input_tokens"`
}

// openInputOrStdin opens path for streaming, or returns stdin when path is "-". The
// returned closer MUST be deferred by the caller (it is a no-op on the stdin path); it
// keeps the file open for the lifetime of the caller's read, matching an inline
// `defer f.Close()`.
func openInputOrStdin(path string, stdin io.Reader) (io.Reader, func() error, error) {
	if path == "-" {
		return stdin, func() error { return nil }, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	return f, f.Close, nil
}

func readVCacheTelemetry(path string, stdin io.Reader) ([]vcachegov.TelemetryRow, error) {
	r, closeInput, err := openInputOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}
	defer closeInput()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []vcachegov.TelemetryRow
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var raw vcacheTelemetryJSONLRow
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		row, ok := raw.telemetryRow()
		if ok {
			rows = append(rows, row)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func (r vcacheTelemetryJSONLRow) telemetryRow() (vcachegov.TelemetryRow, bool) {
	if r.hasClaudeCounters() {
		return vcachegov.TelemetryRow{
			InputTokens:              r.InputTokens,
			CacheCreationInputTokens: r.CacheCreationInputTokens,
			CacheReadInputTokens:     r.CacheReadInputTokens,
			Ephemeral1hInputTokens:   r.Ephemeral1hInputTokens,
			Ephemeral5mInputTokens:   r.Ephemeral5mInputTokens,
		}, true
	}
	if r.Usage != nil {
		total, cached := r.Usage.openAITokens()
		return openAITelemetryRow(total, cached), true
	}
	if r.Payload != nil && r.Type == "event_msg" && r.Payload.Type == "token_count" {
		usage := r.Payload.Info.LastTokenUsage
		if usage.InputTokens != 0 || usage.CachedInputTokens != 0 {
			return openAITelemetryRow(usage.InputTokens, usage.CachedInputTokens), true
		}
	}
	if r.InputTokens != 0 || r.PromptTokens != 0 || r.CachedTokens != 0 {
		return openAITelemetryRow(firstNonZero(r.InputTokens, r.PromptTokens), r.CachedTokens), true
	}
	return vcachegov.TelemetryRow{}, false
}

func (r vcacheTelemetryJSONLRow) hasClaudeCounters() bool {
	return r.CacheCreationInputTokens != 0 ||
		r.CacheReadInputTokens != 0 ||
		r.Ephemeral1hInputTokens != 0 ||
		r.Ephemeral5mInputTokens != 0
}

func (u vcacheOpenAIUsage) openAITokens() (float64, float64) {
	total := firstNonZero(u.InputTokens, u.PromptTokens)
	cached := firstNonZero(u.InputTokensDetails.CachedTokens, u.PromptTokensDetails.CachedTokens, u.CachedInputTokens)
	return total, cached
}

func openAITelemetryRow(total, cached float64) vcachegov.TelemetryRow {
	if total < 0 {
		total = 0
	}
	if cached < 0 {
		cached = 0
	}
	if cached > total {
		cached = total
	}
	return vcachegov.TelemetryRow{
		InputTokens:          total - cached,
		CacheReadInputTokens: cached,
	}
}

func firstNonZero(values ...float64) float64 {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}
