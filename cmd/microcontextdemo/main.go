// Command microcontextdemo is the minimal runnable spine for the micro-context
// research program. It drives many logical agent contexts over a bounded set of
// physical workers while one immutable base context remains installed in the
// controlled model seam.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type config struct {
	Contexts       int
	Workers        int
	Delay          time.Duration
	Selfcheck      bool
	Endpoint       string
	APIKey         string
	Model          string
	Provider       string
	Hardware       string
	RequestTimeout time.Duration
	RunTimeout     time.Duration
	ControlledSoak bool
	PrefixMode     string
	APIShape       microagent.APIProviderShape
	LiveInput      string
	WorkUnits      []liveWorkUnit
}

type report struct {
	Schema                string         `json:"schema"`
	Verdict               string         `json:"verdict"`
	LogicalShards         int            `json:"logical_shards"`
	PhysicalWorkers       int            `json:"physical_workers"`
	Completed             int            `json:"completed"`
	Failed                int            `json:"failed"`
	SharedBaseInstalls    int64          `json:"shared_base_installs"`
	TurnCount             int64          `json:"turn_count"`
	PeakInFlight          int64          `json:"peak_in_flight"`
	ElapsedMS             int64          `json:"elapsed_ms"`
	ShardsPerSecond       float64        `json:"shards_per_second"`
	Scope                 string         `json:"scope"`
	FirstFailure          string         `json:"first_failure,omitempty"`
	Mode                  string         `json:"mode"`
	Endpoint              string         `json:"endpoint,omitempty"`
	Provider              string         `json:"provider,omitempty"`
	Model                 string         `json:"model,omitempty"`
	Hardware              string         `json:"hardware,omitempty"`
	BaseFingerprint       string         `json:"base_fingerprint"`
	PromptTokens          int64          `json:"prompt_tokens,omitempty"`
	CompletionTokens      int64          `json:"completion_tokens,omitempty"`
	CachedPromptTokens    int64          `json:"cached_prompt_tokens,omitempty"`
	UsageResponses        int            `json:"usage_responses,omitempty"`
	TTFTP50MS             float64        `json:"ttft_p50_ms,omitempty"`
	TTFTP95MS             float64        `json:"ttft_p95_ms,omitempty"`
	TTFTMaxMS             float64        `json:"ttft_max_ms,omitempty"`
	LatencyP50MS          float64        `json:"latency_p50_ms,omitempty"`
	LatencyP95MS          float64        `json:"latency_p95_ms,omitempty"`
	LatencyMaxMS          float64        `json:"latency_max_ms,omitempty"`
	PromptTokensPerSec    float64        `json:"prompt_tokens_per_wall_second,omitempty"`
	DecodeTokensPerSec    float64        `json:"decode_tokens_per_wall_second,omitempty"`
	ResourceSamples       int            `json:"resource_samples,omitempty"`
	ClientPeakRSSBytes    int64          `json:"client_peak_rss_bytes,omitempty"`
	ServerPeakRSSBytes    int64          `json:"server_peak_rss_bytes,omitempty"`
	ServerPeakHeapBytes   int64          `json:"server_peak_heap_alloc_bytes,omitempty"`
	EndpointPeakRequests  int            `json:"endpoint_peak_requests,omitempty"`
	KVCapacityEvidence    string         `json:"kv_capacity_evidence,omitempty"`
	QueueEvidence         string         `json:"queue_evidence,omitempty"`
	ResultCheck           string         `json:"result_check,omitempty"`
	VerifiedResultsPerSec float64        `json:"verified_nonempty_results_per_wall_second,omitempty"`
	SoakContract          string         `json:"soak_contract,omitempty"`
	CanaryContexts        int            `json:"canary_contexts,omitempty"`
	CanaryPassed          int            `json:"canary_passed,omitempty"`
	BaseRollbackCount     int            `json:"base_rollback_count,omitempty"`
	RetryInjected         int            `json:"retry_injected,omitempty"`
	RetryRecovered        int            `json:"retry_recovered,omitempty"`
	CancellationInjected  int            `json:"cancellation_injected,omitempty"`
	CancellationRecovered int            `json:"cancellation_recovered,omitempty"`
	ProviderFailures      int            `json:"provider_transient_failures,omitempty"`
	ProviderRecovered     int            `json:"provider_transient_recovered,omitempty"`
	MaxAttempts           int            `json:"max_attempts,omitempty"`
	QueuePeakContexts     int            `json:"queue_peak_contexts,omitempty"`
	HibernatedContexts    int            `json:"hibernated_contexts,omitempty"`
	RestoredContexts      int            `json:"restored_contexts,omitempty"`
	LiveInputRecords      []liveWorkUnit `json:"live_input_records,omitempty"`
}

type sharedBase struct {
	instructions string
	fingerprint  string
}

func canonicalBaseInstructions() string {
	return strings.Repeat("You are one worker in a bounded micro-context fabric. Preserve task isolation and return a short non-empty answer. Stable shared setup material follows. ", 24) + " Context identity: 00000000."
}

func canonicalBaseFingerprint() string { return "microcontext-base-v1" }

type fakeEndpoint struct {
	base     *sharedBase
	delay    time.Duration
	calls    atomic.Int64
	inFlight atomic.Int64
	peak     atomic.Int64
	seenMu   sync.Mutex
	seen     map[string]struct{}
}

func newFakeEndpoint(base *sharedBase, delay time.Duration) *fakeEndpoint {
	return &fakeEndpoint{base: base, delay: delay, seen: make(map[string]struct{})}
}

func (g *fakeEndpoint) Model() string { return "microcontext-synthetic" }

func (g *fakeEndpoint) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	current := g.inFlight.Add(1)
	defer g.inFlight.Add(-1)
	for {
		old := g.peak.Load()
		if current <= old || g.peak.CompareAndSwap(old, current) {
			break
		}
	}
	g.calls.Add(1)
	if len(messages) != 1 || messages[0].Role != agent.RoleUser {
		return nil, fmt.Errorf("delta contract: got %d messages", len(messages))
	}
	id := messages[0].Content
	g.seenMu.Lock()
	if _, duplicate := g.seen[id]; duplicate {
		g.seenMu.Unlock()
		return nil, fmt.Errorf("duplicate context %q", id)
	}
	g.seen[id] = struct{}{}
	g.seenMu.Unlock()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(g.delay):
	}
	// The shared base is intentionally read here, at the kernel seam, rather
	// than copied into every logical agent transcript.
	if g.base.instructions == "" || g.base.fingerprint == "" {
		return nil, fmt.Errorf("shared base is not installed")
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done:" + id}}, nil
}

type shardAgent struct {
	id         string
	done       bool
	exact      bool
	maxRetries int
	attempts   int
}

func (a *shardAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	var resp *agent.Completion
	var err error
	for {
		a.attempts++
		resp, err = gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: a.id}}, nil)
		if err == nil || a.attempts > a.maxRetries {
			break
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(time.Duration(a.attempts) * 100 * time.Millisecond):
		}
	}
	if err != nil {
		return false, err
	}
	if resp == nil || strings.TrimSpace(resp.Message.Content) == "" {
		return false, fmt.Errorf("empty completion for %s", a.id)
	}
	if a.exact && resp.Message.Content != "done:"+a.id {
		return false, fmt.Errorf("bad completion for %s", a.id)
	}
	a.done = true
	return true, nil
}

func percentileMS(values []time.Duration, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	idx := int(float64(len(values)-1) * q)
	return float64(values[idx].Microseconds()) / 1000
}

func run(ctx context.Context, cfg config) (report, error) {
	if cfg.LiveInput != "" && len(cfg.WorkUnits) == 0 {
		units, err := loadLiveWorkUnits(cfg.LiveInput)
		if err != nil {
			return report{}, err
		}
		cfg.WorkUnits = units
		cfg.Contexts = len(units)
	}
	if cfg.Contexts < 1 || cfg.Workers < 1 {
		return report{}, fmt.Errorf("contexts and workers must be positive")
	}
	base := &sharedBase{instructions: canonicalBaseInstructions(), fingerprint: canonicalBaseFingerprint()}
	var gw microagent.Gateway
	var live *openAIEndpoint
	var faults *controlledSoakGateway
	var soak controlledSoakEvidence
	mode := "synthetic"
	if cfg.Endpoint != "" {
		var err error
		live, err = newOpenAIEndpoint(cfg.Endpoint, cfg.APIKey, cfg.Model, base, cfg.RequestTimeout)
		if live != nil {
			live.prefixMode = cfg.PrefixMode
			if cfg.APIShape.RequestsPerMinute > 0 {
				cfg.APIShape.Name = cfg.Provider
				live.admission, err = microagent.NewAPIAdmission(cfg.APIShape)
			}
		}
		if err != nil {
			return report{}, err
		}
		gw = live
		if cfg.ControlledSoak {
			soak, err = runControlledSoakPreflight(ctx, cfg, live, base)
			if err != nil {
				return report{}, err
			}
			faults = newControlledSoakGateway(live)
			gw = faults
		}
		mode = "openai-compatible"
	} else {
		if cfg.ControlledSoak {
			return report{}, fmt.Errorf("controlled soak requires a live endpoint")
		}
		gw = newFakeEndpoint(base, cfg.Delay)
	}
	host, err := microagent.NewHost(gw, microagent.Config{Workers: cfg.Workers, Queue: cfg.Contexts})
	if err != nil {
		return report{}, err
	}
	defer host.Close()
	start := time.Now()
	for i := 0; i < cfg.Contexts; i++ {
		id := "ctx-" + strconv.Itoa(i)
		if len(cfg.WorkUnits) > 0 {
			id = cfg.WorkUnits[i].workID()
		}
		retries := 0
		if cfg.ControlledSoak {
			retries = 2
		}
		if err := host.Spawn(id, &shardAgent{id: id, exact: live == nil, maxRetries: retries}); err != nil {
			return report{}, fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	if err := host.Drain(ctx); err != nil {
		return report{}, err
	}
	elapsed := time.Since(start)
	results := host.Reap()
	sort.Slice(results, func(i, j int) bool { return results[i].ID < results[j].ID })
	failed := 0
	var firstFailure string
	for _, result := range results {
		if !result.Done || result.Err != nil {
			failed++
			if firstFailure == "" && result.Err != nil {
				firstFailure = result.ID + ": " + result.Err.Error()
			}
		}
	}
	r := report{
		Schema: "fak-microcontext-spine/1", Verdict: "PASS", LogicalShards: cfg.Contexts,
		PhysicalWorkers: cfg.Workers, Completed: len(results) - failed, Failed: failed,
		SharedBaseInstalls: 1, ElapsedMS: elapsed.Milliseconds(), ShardsPerSecond: float64(cfg.Contexts) / elapsed.Seconds(),
		Mode: mode, Provider: cfg.Provider, Model: cfg.Model, Hardware: cfg.Hardware, BaseFingerprint: base.fingerprint, FirstFailure: firstFailure,
		LiveInputRecords: cfg.WorkUnits,
	}
	if live == nil {
		fake := gw.(*fakeEndpoint)
		r.TurnCount = fake.calls.Load()
		r.PeakInFlight = fake.peak.Load()
		r.Scope = "synthetic endpoint; proves bounded harness fan-out and shared-base semantics, not model tokens/sec"
		if fake.calls.Load() != int64(cfg.Contexts) || len(fake.seen) != cfg.Contexts || fake.peak.Load() > int64(cfg.Workers) {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("synthetic spine invariant failed")
		}
		if cfg.Selfcheck && cfg.Contexts > 1 && cfg.Workers > 1 && fake.peak.Load() < 2 {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("parallelism was not observed")
		}
	} else {
		stats := live.snapshot()
		r.Endpoint = cfg.Endpoint
		r.TurnCount = int64(len(stats.latencies))
		r.PeakInFlight = int64(cfg.Workers) // admission ceiling; endpoint-internal batching is not inferred
		r.PromptTokens = stats.promptTokens
		r.CompletionTokens = stats.completionTokens
		r.CachedPromptTokens = stats.cachedTokens
		r.UsageResponses = stats.usageResponses
		r.TTFTP50MS = percentileMS(stats.ttfts, .50)
		r.TTFTP95MS = percentileMS(stats.ttfts, .95)
		r.TTFTMaxMS = percentileMS(stats.ttfts, 1)
		r.LatencyP50MS = percentileMS(stats.latencies, .50)
		r.LatencyP95MS = percentileMS(stats.latencies, .95)
		r.LatencyMaxMS = percentileMS(stats.latencies, 1)
		r.PromptTokensPerSec = float64(stats.promptTokens) / elapsed.Seconds()
		r.DecodeTokensPerSec = float64(stats.completionTokens) / elapsed.Seconds()
		r.Scope = "real streaming endpoint; token rates are aggregate observed usage divided by wall time, not server-internal kernel rates; critical-path latency is reported separately"
		if cfg.ControlledSoak {
			r.SoakContract = "controlled-10k-v1"
			r.CanaryContexts, r.CanaryPassed, r.BaseRollbackCount = soak.canaryContexts, soak.canaryPassed, soak.baseRollbackCount
			r.RetryInjected, r.RetryRecovered = int(faults.retryInjected.Load()), int(faults.retryRecovered.Load())
			r.CancellationInjected, r.CancellationRecovered = int(faults.cancelInjected.Load()), int(faults.cancelRecovered.Load())
			r.ProviderFailures, r.ProviderRecovered = int(faults.innerFailures.Load()), int(faults.innerRecovered.Load())
			r.MaxAttempts = 3
			r.QueuePeakContexts = soak.queuePeakContexts
			r.HibernatedContexts = soak.hibernatedContexts
			r.RestoredContexts = soak.restoredContexts
		}
		if stats.usageResponses != cfg.Contexts || len(stats.ttfts) != cfg.Contexts {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("live telemetry incomplete: usage=%d ttft=%d want=%d", stats.usageResponses, len(stats.ttfts), cfg.Contexts)
		}
	}
	if failed != 0 || len(results) != cfg.Contexts || r.TurnCount != int64(cfg.Contexts) {
		r.Verdict = "FAIL"
		return r, fmt.Errorf("spine invariant failed")
	}
	return r, nil
}
func main() {
	var cfg config
	var verifyPath string
	var abOutput string
	var verifyABPath string
	var s3Output string
	var verifyS3Path string
	var s3Resident, s3Low, s3Warm, s3Turns int
	var s3Memory uint64
	var descriptorOutput string
	var verifyDescriptorPath string
	var multiTurnDescriptorOutput string
	var verifyMultiTurnDescriptorPath string
	var verifyS2BPath string
	var compatOutput, verifyCompatPath string
	var batchModelPath, batchHardware, batchOutput, verifyBatchPath string
	var batchSize int
	var effectsOutput, verifyEffectsPath string
	var verifyAPIOnlyPath string
	var qualityInput, qualityOutput, verifyQualityPath string
	var qualitySamples int
	var largeInputOutput, verifyLargeInputPath string
	var largeInputSelfcheck bool
	var selectorOutput, verifySelectorPath string
	var selectorSelfcheck bool
	var toolEnrichmentOutput, verifyToolEnrichmentPath string
	var toolEnrichmentSelfcheck bool
	var provenanceFoldOutput, verifyProvenanceFoldPath string
	var provenanceFoldSelfcheck bool
	var falsificationOutput, verifyFalsificationPath string
	var falsificationSelfcheck bool
	var effectBatchOutput, verifyEffectBatchPath string
	var corpusInputPath, corpusPublicPath, corpusAnswersPath, corpusReportPath string
	var corpusSource, verifyCorpusPublic, verifyCorpusAnswers, verifyCorpusReport string
	var gradeCorpusAnswers, gradeCorpusSubmission, gradeCorpusOutput string
	var tunedBaselinesPublic, tunedBaselinesAnswers, tunedBaselinesOutput, verifyTunedBaselinesPath string
	var routingVOIOutput, verifyRoutingVOIPath string
	var routingVOISeed int64
	var semanticCorpus, semanticPacketOutput, semanticPacketInput, semanticJudgmentOutput string
	var semanticEndpoint, semanticAPIKey, semanticModel, semanticAdjudicator string
	var semanticPerSplit int
	var semanticFoldPacket, semanticFoldA, semanticFoldB, semanticGoldOutput string
	var semanticGradeGold, semanticGradeSubmission, semanticGradeOutput, semanticGradeSplit string
	var verifySemanticGoldPath, verifySemanticGradePath string
	var liveMatrixPacket, liveMatrixGold, liveMatrixOutput, verifyLiveMatrixPath string
	var liveMatrixEndpoint, liveMatrixAPIKey, liveMatrixModel, liveMatrixClass, liveMatrixHardware string
	var liveMatrixNativeBatch, liveMatrixPrefixCache, liveMatrixPricing string
	var liveMatrixTrials, liveMatrixWorkers int
	var liveMatrixInputPrice, liveMatrixOutputPrice float64
	var strongMatrixPacket, strongMatrixGold, strongMatrixOutput, verifyStrongMatrixPath string
	var strongMatrixEndpoint, strongMatrixAPIKey, strongMatrixModel, strongMatrixClass, strongMatrixHardware string
	var strongMatrixBatch, strongMatrixCache, strongMatrixPricing string
	var strongMatrixTrials, strongMatrixWorkers, strongMatrixK, strongMatrixChunk int
	var tailPacket, tailGold, tailOutput, verifyTailPath, tailEndpoint, tailAPIKey, tailModel, tailClass, tailHardware string
	var tailTrials, tailWorkers, tailSufficiency int
	var tailWindowMS, tailTaskMS, tailHedgeMS int64
	var routingVOITrials, routingVOIRecords int
	var effectBatchSelfcheck bool
	var fairnessOutput, verifyFairnessPath string
	var gradeInput, gradeOutput, verifyGradePath string
	flag.IntVar(&cfg.Contexts, "contexts", 10000, "logical micro-contexts")
	flag.IntVar(&cfg.Workers, "workers", 64, "bounded physical worker slots")
	flag.DurationVar(&cfg.Delay, "synthetic-latency", 100*time.Microsecond, "synthetic endpoint latency per context")
	flag.BoolVar(&cfg.Selfcheck, "selfcheck", false, "enforce spine invariants")
	flag.StringVar(&cfg.Endpoint, "endpoint", "", "OpenAI-compatible endpoint root; empty uses the synthetic S0 endpoint")
	flag.StringVar(&cfg.APIKey, "api-key", "", "endpoint API key (prefer environment expansion by the caller)")
	flag.StringVar(&cfg.Model, "model", "", "live endpoint model id")
	flag.StringVar(&cfg.Provider, "provider", "", "provider provenance label")
	flag.StringVar(&cfg.Hardware, "hardware", "", "hardware provenance label")
	flag.StringVar(&cfg.LiveInput, "live-issues", "", "bounded gh issue-list JSON snapshot to use as work units")
	flag.DurationVar(&cfg.RequestTimeout, "request-timeout", 2*time.Minute, "per-request live endpoint timeout")
	flag.DurationVar(&cfg.RunTimeout, "run-timeout", 15*time.Minute, "overall run timeout (0 disables the deadline)")
	flag.BoolVar(&cfg.ControlledSoak, "controlled-soak", false, "exercise the S5 canary/rollback, overload queue, cancellation, bounded retry, and hibernation contract")
	flag.IntVar(&cfg.APIShape.RequestsPerMinute, "api-rpm", 0, "API-only request-per-minute admission limit (0 disables adapter admission)")
	flag.IntVar(&cfg.APIShape.TokensPerMinute, "api-tpm", 0, "API-only estimated token-per-minute admission limit")
	flag.IntVar(&cfg.APIShape.Concurrency, "api-concurrency", 0, "API-only provider concurrency admission limit")
	flag.Int64Var(&cfg.APIShape.MaxSpendMicros, "api-spend-micros", 0, "API-only estimated spend envelope in provider micro-units")
	flag.Int64Var(&cfg.APIShape.PromptMicrosPerToken, "api-prompt-micros-per-token", 0, "API-only estimated prompt cost per token")
	flag.Int64Var(&cfg.APIShape.OutputMicrosPerToken, "api-output-micros-per-token", 0, "API-only estimated output cost per token")
	flag.StringVar(&cfg.APIShape.ReuseControl, "api-cache-control", "byte-identical-prefix", "API-only cache control shape")
	flag.StringVar(&cfg.APIShape.ReuseEvidence, "api-cache-telemetry", "opaque", "API-only cache telemetry shape")
	flag.StringVar(&verifyPath, "verify", "", "verify a captured S1 JSON artifact and exit")
	flag.StringVar(&abOutput, "prefix-ab", "", "run the S2 prefix A/B and write JSON to this path (or - for stdout)")
	flag.StringVar(&verifyABPath, "verify-prefix-ab", "", "verify a captured S2 prefix A/B artifact and exit")
	flag.StringVar(&s3Output, "hibernate-restart", "", "run the S3 hibernation/restart witness and write JSON to this path (or - for stdout)")
	flag.StringVar(&verifyS3Path, "verify-hibernate-restart", "", "verify a captured S3 artifact and exit")
	flag.IntVar(&s3Resident, "resident-high", 32, "S3 hard resident context cap")
	flag.IntVar(&s3Low, "resident-low", 16, "S3 warm-band low watermark")
	flag.IntVar(&s3Warm, "warm-cap", 8, "S3 warm reserve cap")
	flag.IntVar(&s3Turns, "turns", 2, "S3 synthetic turns per logical context")
	flag.Uint64Var(&s3Memory, "memory-envelope", 64<<20, "S3 peak Go allocation delta envelope in bytes")
	flag.StringVar(&descriptorOutput, "descriptor-bench", "", "run the 1,000-context descriptor/harness benchmark and write JSON")
	flag.StringVar(&verifyDescriptorPath, "verify-descriptor-bench", "", "verify a captured descriptor benchmark artifact and exit")
	flag.StringVar(&multiTurnDescriptorOutput, "multi-turn-descriptor", "", "run the 1,000-context multi-turn descriptor witness and write JSON")
	flag.StringVar(&verifyMultiTurnDescriptorPath, "verify-multi-turn-descriptor", "", "verify a captured multi-turn descriptor witness and exit")
	flag.StringVar(&verifyS2BPath, "verify-kernel-prefix-ab", "", "verify a controlled in-kernel prefix-cache A/B artifact and exit")
	flag.StringVar(&compatOutput, "compatibility-witness", "", "run compatibility scheduler witness and write JSON")
	flag.StringVar(&verifyCompatPath, "verify-compatibility", "", "verify compatibility artifact")
	registerBatchExecutionFlags(flag.CommandLine, &batchModelPath, &batchHardware, &batchOutput, &verifyBatchPath, &batchSize)
	flag.StringVar(&effectsOutput, "effects-witness", "", "run effect-safety witness and write JSON")
	flag.StringVar(&verifyEffectsPath, "verify-effects", "", "verify effect-safety artifact")
	flag.StringVar(&verifyAPIOnlyPath, "verify-api-only", "", "verify captured S6 API-only artifact")
	flag.BoolVar(&largeInputSelfcheck, "large-input-selfcheck", false, "run the fixture-backed 1,000-record large-input operator proof")
	flag.StringVar(&largeInputOutput, "large-input-output", "", "write the large-input operator proof artifact")
	flag.StringVar(&verifyLargeInputPath, "verify-large-input", "", "verify a captured large-input operator artifact")
	flag.BoolVar(&selectorSelfcheck, "filter-selector-selfcheck", false, "run the adaptive filter-selector proof")
	flag.StringVar(&selectorOutput, "filter-selector-output", "", "write the adaptive filter-selector proof artifact")
	flag.StringVar(&verifySelectorPath, "verify-filter-selector", "", "verify a captured filter-selector artifact")
	flag.BoolVar(&toolEnrichmentSelfcheck, "tool-enrichment-selfcheck", false, "run the read-only tool-enrichment fan-out proof")
	flag.StringVar(&toolEnrichmentOutput, "tool-enrichment-output", "", "write the read-only tool-enrichment proof artifact")
	flag.StringVar(&verifyToolEnrichmentPath, "verify-tool-enrichment", "", "verify a captured tool-enrichment artifact")
	flag.BoolVar(&provenanceFoldSelfcheck, "provenance-fold-selfcheck", false, "run the provenance-preserving hierarchical fold proof")
	flag.StringVar(&provenanceFoldOutput, "provenance-fold-output", "", "write the hierarchical fold proof artifact")
	flag.StringVar(&verifyProvenanceFoldPath, "verify-provenance-fold", "", "verify a captured hierarchical fold artifact")
	flag.BoolVar(&falsificationSelfcheck, "falsification-selfcheck", false, "run the tuned-baseline falsification benchmark")
	flag.StringVar(&falsificationOutput, "falsification-output", "", "write the falsification benchmark artifact")
	flag.StringVar(&verifyFalsificationPath, "verify-falsification", "", "verify a captured falsification artifact")
	flag.BoolVar(&effectBatchSelfcheck, "effect-batch-selfcheck", false, "run the witnessed effect-batch proof")
	flag.StringVar(&effectBatchOutput, "effect-batch-output", "", "write the effect-batch proof artifact")
	flag.StringVar(&verifyEffectBatchPath, "verify-effect-batch", "", "verify a captured effect-batch artifact")
	flag.StringVar(&corpusInputPath, "corpus-input", "", "freeze a public GitHub issue JSON export into leakage-controlled corpus artifacts")
	flag.StringVar(&corpusPublicPath, "corpus-public-output", "", "public candidate-input corpus output")
	flag.StringVar(&corpusAnswersPath, "corpus-answers-output", "", "separate answer bundle output")
	flag.StringVar(&corpusReportPath, "corpus-report-output", "", "corpus provenance/leak/grade report output")
	flag.StringVar(&corpusSource, "corpus-source", "github.com/anthony-chaudhary/fak/issues", "source provenance label")
	flag.StringVar(&verifyCorpusPublic, "verify-corpus-public", "", "verify public corpus with answer/report artifacts")
	flag.StringVar(&verifyCorpusAnswers, "verify-corpus-answers", "", "answer bundle paired with --verify-corpus-public")
	flag.StringVar(&verifyCorpusReport, "verify-corpus-report", "", "report paired with --verify-corpus-public")
	flag.StringVar(&gradeCorpusAnswers, "grade-corpus-answers", "", "hidden answer bundle for independent grading")
	flag.StringVar(&gradeCorpusSubmission, "grade-corpus-submission", "", "candidate submission to grade")
	flag.StringVar(&gradeCorpusOutput, "grade-corpus-output", "", "write independent grade report")
	flag.StringVar(&tunedBaselinesPublic, "tuned-baselines-public", "", "public corpus for tuned baseline dry-run")
	flag.StringVar(&tunedBaselinesAnswers, "tuned-baselines-answers", "", "answer bundle for tuning/grading")
	flag.StringVar(&tunedBaselinesOutput, "tuned-baselines-output", "", "write tuned baseline report")
	flag.StringVar(&verifyTunedBaselinesPath, "verify-tuned-baselines", "", "verify a tuned baseline report")
	flag.StringVar(&routingVOIOutput, "routing-voi-output", "", "run adaptive filter/tool routing experiment")
	flag.StringVar(&verifyRoutingVOIPath, "verify-routing-voi", "", "verify an adaptive routing experiment")
	flag.Int64Var(&routingVOISeed, "routing-voi-seed", 6105, "deterministic routing experiment seed")
	flag.IntVar(&routingVOITrials, "routing-voi-trials", 24, "routing experiment repetitions")
	flag.IntVar(&routingVOIRecords, "routing-voi-records", 200, "records per mixture and trial")
	flag.StringVar(&semanticCorpus, "semantic-packet-corpus", "", "public corpus used to build a blinded semantic packet")
	flag.StringVar(&semanticPacketOutput, "semantic-packet-output", "", "write a blinded semantic annotation packet")
	flag.IntVar(&semanticPerSplit, "semantic-per-split", 16, "semantic packet records per tune/test split")
	flag.StringVar(&semanticPacketInput, "semantic-adjudicate-packet", "", "semantic packet to adjudicate through a live endpoint")
	flag.StringVar(&semanticJudgmentOutput, "semantic-adjudicate-output", "", "write one adjudicator bundle")
	flag.StringVar(&semanticEndpoint, "semantic-endpoint", "", "OpenAI-compatible endpoint root for semantic adjudication")
	flag.StringVar(&semanticAPIKey, "semantic-api-key", "", "API key for semantic endpoint")
	flag.StringVar(&semanticModel, "semantic-model", "", "semantic adjudicator model id")
	flag.StringVar(&semanticAdjudicator, "semantic-adjudicator", "", "independent adjudicator identity")
	flag.StringVar(&semanticFoldPacket, "semantic-fold-packet", "", "packet bound to two independent adjudicators")
	flag.StringVar(&semanticFoldA, "semantic-fold-a", "", "first independent adjudicator bundle")
	flag.StringVar(&semanticFoldB, "semantic-fold-b", "", "second independent adjudicator bundle")
	flag.StringVar(&semanticGoldOutput, "semantic-gold-output", "", "write hidden semantic consensus/abstention answers")
	flag.StringVar(&semanticGradeGold, "semantic-grade-gold", "", "hidden semantic answers for blind grading")
	flag.StringVar(&semanticGradeSubmission, "semantic-grade-submission", "", "candidate semantic submission")
	flag.StringVar(&semanticGradeOutput, "semantic-grade-output", "", "write semantic blind grade")
	flag.StringVar(&semanticGradeSplit, "semantic-grade-split", "test", "split to grade")
	flag.StringVar(&verifySemanticGoldPath, "verify-semantic-gold", "", "verify semantic consensus artifact")
	flag.StringVar(&verifySemanticGradePath, "verify-semantic-grade", "", "verify semantic blind grade")
	flag.StringVar(&liveMatrixPacket, "live-matrix-packet", "", "S8i public semantic packet")
	flag.StringVar(&liveMatrixGold, "live-matrix-gold", "", "hidden S8i semantic gold")
	flag.StringVar(&liveMatrixOutput, "live-matrix-output", "", "write live comparative matrix")
	flag.StringVar(&verifyLiveMatrixPath, "verify-live-matrix", "", "verify live semantic matrix")
	flag.StringVar(&liveMatrixEndpoint, "live-matrix-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&liveMatrixAPIKey, "live-matrix-api-key", "", "live endpoint API key")
	flag.StringVar(&liveMatrixModel, "live-matrix-model", "", "live endpoint model")
	flag.StringVar(&liveMatrixClass, "live-matrix-endpoint-class", "", "public provenance class")
	flag.StringVar(&liveMatrixHardware, "live-matrix-hardware", "", "public hardware provenance")
	flag.StringVar(&liveMatrixNativeBatch, "live-matrix-native-batch", "unsupported", "endpoint native batching capability")
	flag.StringVar(&liveMatrixPrefixCache, "live-matrix-prefix-cache", "usage-observed-only", "endpoint prefix-cache capability")
	flag.StringVar(&liveMatrixPricing, "live-matrix-pricing", "unavailable", "pricing snapshot provenance")
	flag.IntVar(&liveMatrixTrials, "live-matrix-trials", 3, "repeated trials per pipeline")
	flag.IntVar(&liveMatrixWorkers, "live-matrix-workers", 8, "bounded micro-context concurrency")
	flag.Float64Var(&liveMatrixInputPrice, "live-matrix-input-per-mtok", -1, "input USD per million tokens; negative means unavailable")
	flag.Float64Var(&liveMatrixOutputPrice, "live-matrix-output-per-mtok", -1, "output USD per million tokens; negative means unavailable")
	flag.StringVar(&strongMatrixPacket, "strong-matrix-packet", "", "S8i packet for strengthened baselines")
	flag.StringVar(&strongMatrixGold, "strong-matrix-gold", "", "hidden S8i gold")
	flag.StringVar(&strongMatrixOutput, "strong-matrix-output", "", "write strengthened live matrix")
	flag.StringVar(&verifyStrongMatrixPath, "verify-strong-matrix", "", "verify strengthened live matrix")
	flag.StringVar(&strongMatrixEndpoint, "strong-matrix-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&strongMatrixAPIKey, "strong-matrix-api-key", "", "endpoint API key")
	flag.StringVar(&strongMatrixModel, "strong-matrix-model", "", "endpoint model")
	flag.StringVar(&strongMatrixClass, "strong-matrix-endpoint-class", "", "endpoint class")
	flag.StringVar(&strongMatrixHardware, "strong-matrix-hardware", "", "hardware provenance")
	flag.StringVar(&strongMatrixBatch, "strong-matrix-native-batch", "unsupported", "native batching capability")
	flag.StringVar(&strongMatrixCache, "strong-matrix-prefix-cache", "usage-observed-only", "prefix-cache capability")
	flag.StringVar(&strongMatrixPricing, "strong-matrix-pricing", "unavailable", "pricing provenance")
	flag.IntVar(&strongMatrixTrials, "strong-matrix-trials", 2, "held-out trials")
	flag.IntVar(&strongMatrixWorkers, "strong-matrix-workers", 8, "request admission limit")
	flag.IntVar(&strongMatrixK, "strong-matrix-retrieval-k", 3, "top-k tune examples")
	flag.IntVar(&strongMatrixChunk, "strong-matrix-chunk-size", 4, "parallel chunk size")
	flag.StringVar(&tailPacket, "tail-policy-packet", "", "S8i packet for tail policies")
	flag.StringVar(&tailGold, "tail-policy-gold", "", "hidden S8i gold")
	flag.StringVar(&tailOutput, "tail-policy-output", "", "write live tail-policy matrix")
	flag.StringVar(&verifyTailPath, "verify-tail-policy", "", "verify live tail-policy matrix")
	flag.StringVar(&tailEndpoint, "tail-policy-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&tailAPIKey, "tail-policy-api-key", "", "endpoint API key")
	flag.StringVar(&tailModel, "tail-policy-model", "", "endpoint model")
	flag.StringVar(&tailClass, "tail-policy-endpoint-class", "", "endpoint provenance class")
	flag.StringVar(&tailHardware, "tail-policy-hardware", "", "hardware provenance")
	flag.IntVar(&tailTrials, "tail-policy-trials", 2, "trials per policy")
	flag.IntVar(&tailWorkers, "tail-policy-workers", 8, "bounded workers")
	flag.IntVar(&tailSufficiency, "tail-policy-sufficiency", 12, "confirmed records before early stop")
	flag.Int64Var(&tailWindowMS, "tail-policy-window-ms", 15000, "per-window deadline milliseconds")
	flag.Int64Var(&tailTaskMS, "tail-policy-task-ms", 60000, "global task deadline milliseconds")
	flag.Int64Var(&tailHedgeMS, "tail-policy-hedge-ms", 6000, "bounded hedge delay milliseconds")
	flag.StringVar(&qualityInput, "quality-input", "", "ingest one run witness into a quality ledger")
	flag.StringVar(&qualityOutput, "quality-output", "", "write the quality ledger")
	flag.IntVar(&qualitySamples, "quality-samples", 16, "maximum sampled context IDs")
	flag.StringVar(&verifyQualityPath, "verify-quality", "", "verify a captured quality ledger")
	flag.StringVar(&fairnessOutput, "fairness-witness", "", "run the S7 mixed-tenant fairness fixture")
	flag.StringVar(&verifyFairnessPath, "verify-fairness", "", "verify a captured S7 fairness artifact")
	flag.StringVar(&gradeInput, "health-input", "", "quality ledger to grade for micro-context health")
	flag.StringVar(&gradeOutput, "health-scorecard", "", "write micro-context health scorecard")
	flag.StringVar(&verifyGradePath, "verify-health-scorecard", "", "verify micro-context health scorecard")
	flag.Parse()
	if verifyFairnessPath != "" {
		runVerify("verify-fairness", verifyFairnessPath, verifyFairnessArtifact)
		return
	}
	if fairnessOutput != "" {
		if err := writeFairness(fairnessOutput); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyGradePath != "" {
		runVerify("verify-health-scorecard", verifyGradePath, verifyHealthArtifact)
		return
	}
	if gradeOutput != "" {
		if err := writeHealthGrade(gradeInput, gradeOutput); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if corpusInputPath != "" {
		if err := freezeCorpus(corpusInputPath, corpusPublicPath, corpusAnswersPath, corpusReportPath, corpusSource); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: corpus freeze: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS corpus public=%s answers=%s report=%s\n", corpusPublicPath, corpusAnswersPath, corpusReportPath)
		return
	}
	if tailPacket != "" {
		if err := runTailPolicyMatrix(tailPacket, tailGold, tailOutput, tailEndpoint, tailAPIKey, tailModel, tailClass, tailHardware, tailTrials, tailWorkers, tailWindowMS, tailTaskMS, tailHedgeMS, tailSufficiency); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: tail policy: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS tail policy %s\n", tailOutput)
		return
	}
	if verifyTailPath != "" {
		runVerify("verify-tail-policy", verifyTailPath, verifyTailPolicyMatrix)
		return
	}
	if strongMatrixPacket != "" {
		if err := runStrongLiveMatrix(strongMatrixPacket, strongMatrixGold, strongMatrixOutput, strongMatrixEndpoint, strongMatrixAPIKey, strongMatrixModel, strongMatrixClass, strongMatrixHardware, strongMatrixBatch, strongMatrixCache, strongMatrixPricing, strongMatrixTrials, strongMatrixWorkers, strongMatrixK, strongMatrixChunk); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: strong matrix: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS strong matrix %s\n", strongMatrixOutput)
		return
	}
	if verifyStrongMatrixPath != "" {
		runVerify("verify-strong-matrix", verifyStrongMatrixPath, verifyStrongLiveMatrix)
		return
	}
	if liveMatrixPacket != "" {
		if err := runLiveSemanticMatrix(liveMatrixPacket, liveMatrixGold, liveMatrixOutput, liveMatrixEndpoint, liveMatrixAPIKey, liveMatrixModel, liveMatrixClass, liveMatrixHardware, liveMatrixNativeBatch, liveMatrixPrefixCache, liveMatrixPricing, liveMatrixInputPrice, liveMatrixOutputPrice, liveMatrixTrials, liveMatrixWorkers); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: live matrix: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS live matrix %s\n", liveMatrixOutput)
		return
	}
	if verifyLiveMatrixPath != "" {
		runVerify("verify-live-matrix", verifyLiveMatrixPath, verifyLiveSemanticMatrix)
		return
	}
	if semanticFoldPacket != "" {
		if err := foldSemanticAdjudicators(semanticFoldPacket, semanticFoldA, semanticFoldB, semanticGoldOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic fold: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic fold %s\n", semanticGoldOutput)
		return
	}
	if semanticGradeSubmission != "" {
		if err := gradeSemanticFiles(semanticGradeGold, semanticGradeSubmission, semanticGradeOutput, semanticGradeSplit); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic grade: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic grade %s\n", semanticGradeOutput)
		return
	}
	if verifySemanticGoldPath != "" {
		runVerify("verify-semantic-gold", verifySemanticGoldPath, verifySemanticGold)
		return
	}
	if verifySemanticGradePath != "" {
		runVerify("verify-semantic-grade", verifySemanticGradePath, verifySemanticGrade)
		return
	}
	if semanticCorpus != "" {
		if err := makeSemanticPacket(semanticCorpus, semanticPacketOutput, semanticPerSplit); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic packet: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic packet %s\n", semanticPacketOutput)
		return
	}
	if semanticPacketInput != "" {
		if err := runSemanticAdjudicator(semanticPacketInput, semanticJudgmentOutput, semanticEndpoint, semanticAPIKey, semanticModel, semanticAdjudicator); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic adjudicator: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic adjudicator %s\n", semanticJudgmentOutput)
		return
	}
	if routingVOIOutput != "" {
		if err := runRoutingVOI(routingVOIOutput, routingVOISeed, routingVOITrials, routingVOIRecords); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: routing VOI: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS routing VOI %s\n", routingVOIOutput)
		return
	}
	if verifyRoutingVOIPath != "" {
		runVerify("verify-routing-voi", verifyRoutingVOIPath, verifyRoutingVOI)
		return
	}
	if tunedBaselinesPublic != "" {
		if err := runTunedBaselines(tunedBaselinesPublic, tunedBaselinesAnswers, tunedBaselinesOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: tuned baselines: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS tuned baselines %s\n", tunedBaselinesOutput)
		return
	}
	if verifyTunedBaselinesPath != "" {
		runVerify("verify-tuned-baselines", verifyTunedBaselinesPath, verifyTunedBaselines)
		return
	}
	if gradeCorpusSubmission != "" {
		if err := gradeSubmissionFiles(gradeCorpusAnswers, gradeCorpusSubmission, gradeCorpusOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: corpus grade: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS corpus grade %s\n", gradeCorpusOutput)
		return
	}
	if verifyCorpusPublic != "" || verifyCorpusAnswers != "" || verifyCorpusReport != "" {
		verifyCorpusSet := func(string) error {
			return verifyCorpusArtifacts(verifyCorpusPublic, verifyCorpusAnswers, verifyCorpusReport)
		}
		if verifyCorpusPublic != "" {
			runVerify("verify-corpus-public", verifyCorpusPublic, verifyCorpusSet)
		}
		if verifyCorpusAnswers != "" {
			runVerify("verify-corpus-answers", verifyCorpusAnswers, verifyCorpusSet)
		}
		if verifyCorpusReport != "" {
			runVerify("verify-corpus-report", verifyCorpusReport, verifyCorpusSet)
		}
		return
	}
	if verifyEffectBatchPath != "" {
		runVerify("verify-effect-batch", verifyEffectBatchPath, verifyEffectBatchArtifact)
		return
	}
	if effectBatchSelfcheck {
		if err := runEffectBatchSelfcheck(effectBatchOutput); err != nil {
			fmt.Fprintf(os.Stderr, "effect-batch selfcheck: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if verifyFalsificationPath != "" {
		runVerify("verify-falsification", verifyFalsificationPath, verifyFalsificationArtifact)
		return
	}
	if falsificationSelfcheck {
		if err := runFalsificationBench(falsificationOutput); err != nil {
			fmt.Fprintf(os.Stderr, "falsification selfcheck: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if verifyProvenanceFoldPath != "" {
		runVerify("verify-provenance-fold", verifyProvenanceFoldPath, verifyProvenanceFoldArtifact)
		return
	}
	if provenanceFoldSelfcheck {
		if err := runProvenanceFoldSelfcheck(provenanceFoldOutput); err != nil {
			fmt.Fprintf(os.Stderr, "provenance-fold selfcheck: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if verifyToolEnrichmentPath != "" {
		runVerify("verify-tool-enrichment", verifyToolEnrichmentPath, verifyToolEnrichmentArtifact)
		return
	}
	if toolEnrichmentSelfcheck {
		if err := runToolEnrichmentSelfcheck(context.Background(), toolEnrichmentOutput, cfg.Workers); err != nil {
			fmt.Fprintf(os.Stderr, "tool-enrichment selfcheck: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if verifySelectorPath != "" {
		runVerify("verify-filter-selector", verifySelectorPath, verifySelectorArtifact)
		return
	}
	if selectorSelfcheck || selectorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		r, err := buildSelectorReport(ctx, cfg.Workers)
		r = compactSelectorReport(r)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil && selectorOutput != "" {
			merr = os.WriteFile(selectorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if selectorOutput == "" {
			fmt.Println(string(b))
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyLargeInputPath != "" {
		runVerify("verify-large-input", verifyLargeInputPath, verifyLargeInputArtifact)
		return
	}
	if largeInputSelfcheck || largeInputOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		r, err := buildLargeInputReport(ctx, cfg.Contexts, cfg.Workers)
		// The proof artifact keeps aggregate source accounting and citations; the
		// in-memory/test witness retains all 1,000 typed facts.
		r.Baseline.Facts = nil
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil && largeInputOutput != "" {
			merr = os.WriteFile(largeInputOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if largeInputOutput == "" {
			fmt.Println(string(b))
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyQualityPath != "" {
		runVerify("verify-quality", verifyQualityPath, verifyQualityLedgerArtifact)
		return
	}
	if qualityInput != "" {
		if qualityOutput == "" {
			fmt.Fprintln(os.Stderr, "quality-output is required")
			os.Exit(2)
		}
		if err := writeQualityLedger(qualityInput, qualityOutput, qualitySamples); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyAPIOnlyPath != "" {
		runVerify("verify-api-only", verifyAPIOnlyPath, verifyAPIOnlyArtifact)
		return
	}
	if verifyPath != "" {
		runVerify("verify", verifyPath, verifyArtifact)
		return
	}
	if verifyABPath != "" {
		runVerify("verify-prefix-ab", verifyABPath, verifyABArtifact)
		return
	}
	if verifyS2BPath != "" {
		runVerify("verify-kernel-prefix-ab", verifyS2BPath, verifyS2BArtifact)
		return
	}
	if verifyEffectsPath != "" {
		runVerify("verify-effects", verifyEffectsPath, verifyEffectsArtifact)
		return
	}
	if effectsOutput != "" {
		r, err := buildEffectsReport()
		b, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(effectsOutput, append(b, '\n'), 0o644)
		}
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyBatchPath != "" {
		runVerify("verify-compat-batch-execution", verifyBatchPath, verifyBatchExecutionArtifact)
		return
	}
	if batchOutput != "" {
		if err := runBatchExecution(batchModelPath, batchHardware, batchOutput, batchSize); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyCompatPath != "" {
		runVerify("verify-compatibility", verifyCompatPath, verifyCompatibilityArtifact)
		return
	}
	if compatOutput != "" {
		r, err := buildCompatReport()
		b, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(compatOutput, append(b, '\n'), 0o644)
		}
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyMultiTurnDescriptorPath != "" {
		runVerify("verify-multi-turn-descriptor", verifyMultiTurnDescriptorPath, verifyMultiTurnDescriptorArtifact)
		return
	}
	if multiTurnDescriptorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runMultiTurnDescriptor(ctx, cfg.Contexts, cfg.Workers, s3Turns)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(multiTurnDescriptorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyDescriptorPath != "" {
		runVerify("verify-descriptor-bench", verifyDescriptorPath, verifyDescriptorArtifact)
		return
	}
	if descriptorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runDescriptorBenchmark(ctx, cfg.Contexts)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(descriptorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if verifyS3Path != "" {
		runVerify("verify-hibernate-restart", verifyS3Path, verifyS3Artifact)
		return
	}
	if s3Output != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runS3(ctx, s3Config{Contexts: cfg.Contexts, Workers: cfg.Workers, ResidentHigh: s3Resident, ResidentLow: s3Low, WarmCap: s3Warm, Turns: s3Turns, MemoryBytes: s3Memory})
		var werr error
		if s3Output == "-" {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			werr = enc.Encode(r)
		} else {
			b, marshalErr := json.MarshalIndent(r, "", "  ")
			if marshalErr != nil {
				werr = marshalErr
			} else {
				werr = os.WriteFile(s3Output, append(b, '\n'), 0o644)
			}
		}
		if werr != nil {
			fmt.Fprintln(os.Stderr, werr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if abOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()
		r, err := runAB(ctx, cfg)
		if writeErr := writeAB(abOutput, r); writeErr != nil {
			fmt.Fprintln(os.Stderr, writeErr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	ctx, cancel := overallDeadline(context.Background(), cfg.RunTimeout)
	defer cancel()
	r, err := run(ctx, cfg)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func overallDeadline(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return parent, func() {}
	}
	return context.WithTimeout(parent, timeout)
}
func runVersionProbe(ctx context.Context, name string) (time.Duration, error) {
	args := []string{"--version"}
	if name == "fak" {
		args = []string{"version"}
	}
	start := time.Now()
	var resolved string
	var err error
	if name == "fak" {
		if home, homeErr := os.UserHomeDir(); homeErr == nil {
			candidate := filepath.Join(home, "bin", "fak.exe")
			if _, statErr := os.Stat(candidate); statErr == nil {
				resolved = candidate
			}
		}
	}
	if resolved == "" {
		resolved, err = exec.LookPath(name)
		if err != nil {
			return 0, err
		}
		if !filepath.IsAbs(resolved) {
			resolved, _ = filepath.Abs(resolved)
		}
	}
	cmd := exec.CommandContext(ctx, resolved, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	err = cmd.Run()
	return time.Since(start), err
}
