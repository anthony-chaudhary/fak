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
	Lineage        *microagent.Lineage
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
	Resources             resourceUsage  `json:"resources"`
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

func run(ctx context.Context, cfg config) (result report, err error) {
	monitor := startResourceMonitor()
	defer func() {
		usage := monitor.finish()
		if result.Schema != "" {
			result.Resources = usage
		}
	}()
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
		shard := microagent.WithLineage(id, &shardAgent{id: id, exact: live == nil, maxRetries: retries}, cfg.Lineage)
		if err := host.Spawn(id, shard); err != nil {
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

type demoOptions struct {
	cfg                           config
	verifyPath                    string
	abOutput                      string
	verifyABPath                  string
	s3Output                      string
	verifyS3Path                  string
	s3Resident                    int
	s3Low                         int
	s3Warm                        int
	s3Turns                       int
	s3Memory                      uint64
	descriptorOutput              string
	verifyDescriptorPath          string
	multiTurnDescriptorOutput     string
	verifyMultiTurnDescriptorPath string
	verifyS2BPath                 string
	compatOutput                  string
	verifyCompatPath              string
	batchModelPath                string
	batchHardware                 string
	batchOutput                   string
	verifyBatchPath               string
	batchSize                     int
	effectsOutput                 string
	verifyEffectsPath             string
	verifyAPIOnlyPath             string
	qualityInput                  string
	qualityOutput                 string
	verifyQualityPath             string
	qualitySamples                int
	largeInputOutput              string
	verifyLargeInputPath          string
	largeInputSelfcheck           bool
	selectorOutput                string
	verifySelectorPath            string
	selectorSelfcheck             bool
	toolEnrichmentOutput          string
	verifyToolEnrichmentPath      string
	toolEnrichmentSelfcheck       bool
	provenanceFoldOutput          string
	verifyProvenanceFoldPath      string
	provenanceFoldSelfcheck       bool
	falsificationOutput           string
	verifyFalsificationPath       string
	falsificationSelfcheck        bool
	effectBatchOutput             string
	verifyEffectBatchPath         string
	corpusInputPath               string
	corpusPublicPath              string
	corpusAnswersPath             string
	corpusReportPath              string
	corpusSource                  string
	verifyCorpusPublic            string
	verifyCorpusAnswers           string
	verifyCorpusReport            string
	gradeCorpusAnswers            string
	gradeCorpusSubmission         string
	gradeCorpusOutput             string
	tunedBaselinesPublic          string
	tunedBaselinesAnswers         string
	tunedBaselinesOutput          string
	verifyTunedBaselinesPath      string
	routingVOIOutput              string
	verifyRoutingVOIPath          string
	filterToolSchedulerFold       string
	filterToolSchedulerOutput     string
	verifyFilterToolSchedulerPath string
	liveFilterToolPacket          string
	liveFilterToolFold            string
	liveFilterToolOutput          string
	verifyLiveFilterToolPath      string
	disagreementPacket            string
	disagreementFold              string
	disagreementLive              string
	disagreementOutput            string
	verifyDisagreementPath        string
	counterfactualSource          string
	counterfactualCorpusOut       string
	counterfactualCorpusIn        string
	counterfactualJudgmentOut     string
	counterfactualAdjudicator     string
	counterfactualFoldA           string
	counterfactualFoldB           string
	counterfactualFoldOut         string
	verifyCounterfactualCorpus    string
	verifyCounterfactualFold      string
	trueAdmissionOut              string
	verifyTrueAdmissionPath       string
	naturalCorpusOut              string
	naturalCorpusIn               string
	naturalJudgeOut               string
	naturalAdjudicator            string
	naturalFoldA                  string
	naturalFoldB                  string
	naturalFoldOut                string
	verifyNaturalCorpus           string
	verifyNaturalFold             string
	naturalSurfaceOut             string
	verifyNaturalSurfacePath      string
	naturalTrafficCorpus          string
	naturalTrafficJudgeOut        string
	naturalTrafficAdjudicator     string
	naturalTrafficFoldA           string
	naturalTrafficFoldB           string
	naturalTrafficFoldOut         string
	naturalTrafficReportOut       string
	verifyNaturalTrafficPath      string
	filterToolSchedulerTrials     int
	routingVOISeed                int64
	semanticCorpus                string
	semanticPacketOutput          string
	semanticPacketInput           string
	semanticJudgmentOutput        string
	semanticEndpoint              string
	semanticAPIKey                string
	semanticModel                 string
	semanticAdjudicator           string
	semanticPromptVersion         string
	semanticPerSplit              int
	semanticFoldPacket            string
	semanticFoldA                 string
	semanticFoldB                 string
	semanticGoldOutput            string
	semanticTriplePacket          string
	semanticTripleOldA            string
	semanticTripleOldB            string
	semanticTripleV2A             string
	semanticTripleV2B             string
	semanticTripleOutput          string
	semanticGradeGold             string
	semanticGradeSubmission       string
	semanticGradeOutput           string
	semanticGradeSplit            string
	verifySemanticGoldPath        string
	verifySemanticGradePath       string
	liveMatrixPacket              string
	liveMatrixGold                string
	liveMatrixOutput              string
	verifyLiveMatrixPath          string
	liveMatrixEndpoint            string
	liveMatrixAPIKey              string
	liveMatrixModel               string
	liveMatrixClass               string
	liveMatrixHardware            string
	liveMatrixNativeBatch         string
	liveMatrixPrefixCache         string
	liveMatrixPricing             string
	liveMatrixTrials              int
	liveMatrixWorkers             int
	liveMatrixInputPrice          float64
	liveMatrixOutputPrice         float64
	strongMatrixPacket            string
	strongMatrixGold              string
	strongMatrixOutput            string
	verifyStrongMatrixPath        string
	strongMatrixEndpoint          string
	strongMatrixAPIKey            string
	strongMatrixModel             string
	strongMatrixClass             string
	strongMatrixHardware          string
	strongMatrixBatch             string
	strongMatrixCache             string
	strongMatrixPricing           string
	strongMatrixTrials            int
	strongMatrixWorkers           int
	strongMatrixK                 int
	strongMatrixChunk             int
	tailPacket                    string
	tailGold                      string
	tailOutput                    string
	verifyTailPath                string
	tailEndpoint                  string
	tailAPIKey                    string
	tailModel                     string
	tailClass                     string
	tailHardware                  string
	tailTrials                    int
	tailWorkers                   int
	tailSufficiency               int
	tailWindowMS                  int64
	tailTaskMS                    int64
	tailHedgeMS                   int64
	routingVOITrials              int
	routingVOIRecords             int
	effectBatchSelfcheck          bool
	fairnessOutput                string
	verifyFairnessPath            string
	gradeInput                    string
	gradeOutput                   string
	verifyGradePath               string
}

func main() {
	var o demoOptions
	registerDemoFlagsA(&o)
	registerDemoFlagsB(&o)
	flag.Parse()
	runDemoOptions(o)
}

func registerDemoFlagsA(o *demoOptions) {
	flag.IntVar(&o.cfg.Contexts, "contexts", 10000, "logical micro-contexts")
	flag.IntVar(&o.cfg.Workers, "workers", 64, "bounded physical worker slots")
	flag.DurationVar(&o.cfg.Delay, "synthetic-latency", 100*time.Microsecond, "synthetic endpoint latency per context")
	flag.BoolVar(&o.cfg.Selfcheck, "selfcheck", false, "enforce spine invariants")
	flag.StringVar(&o.cfg.Endpoint, "endpoint", "", "OpenAI-compatible endpoint root; empty uses the synthetic S0 endpoint")
	flag.StringVar(&o.cfg.APIKey, "api-key", "", "endpoint API key (prefer environment expansion by the caller)")
	flag.StringVar(&o.cfg.Model, "model", "", "live endpoint model id")
	flag.StringVar(&o.cfg.Provider, "provider", "", "provider provenance label")
	flag.StringVar(&o.cfg.Hardware, "hardware", "", "hardware provenance label")
	flag.StringVar(&o.cfg.LiveInput, "live-issues", "", "bounded gh issue-list JSON snapshot to use as work units")
	flag.DurationVar(&o.cfg.RequestTimeout, "request-timeout", 2*time.Minute, "per-request live endpoint timeout")
	flag.DurationVar(&o.cfg.RunTimeout, "run-timeout", 15*time.Minute, "overall run timeout (0 disables the deadline)")
	flag.BoolVar(&o.cfg.ControlledSoak, "controlled-soak", false, "exercise the S5 canary/rollback, overload queue, cancellation, bounded retry, and hibernation contract")
	flag.IntVar(&o.cfg.APIShape.RequestsPerMinute, "api-rpm", 0, "API-only request-per-minute admission limit (0 disables adapter admission)")
	flag.IntVar(&o.cfg.APIShape.TokensPerMinute, "api-tpm", 0, "API-only estimated token-per-minute admission limit")
	flag.IntVar(&o.cfg.APIShape.Concurrency, "api-concurrency", 0, "API-only provider concurrency admission limit")
	flag.Int64Var(&o.cfg.APIShape.MaxSpendMicros, "api-spend-micros", 0, "API-only estimated spend envelope in provider micro-units")
	flag.Int64Var(&o.cfg.APIShape.PromptMicrosPerToken, "api-prompt-micros-per-token", 0, "API-only estimated prompt cost per token")
	flag.Int64Var(&o.cfg.APIShape.OutputMicrosPerToken, "api-output-micros-per-token", 0, "API-only estimated output cost per token")
	flag.StringVar(&o.cfg.APIShape.ReuseControl, "api-cache-control", "byte-identical-prefix", "API-only cache control shape")
	flag.StringVar(&o.cfg.APIShape.ReuseEvidence, "api-cache-telemetry", "opaque", "API-only cache telemetry shape")
	flag.StringVar(&o.verifyPath, "verify", "", "verify a captured S1 JSON artifact and exit")
	flag.StringVar(&o.abOutput, "prefix-ab", "", "run the S2 prefix A/B and write JSON to this path (or - for stdout)")
	flag.StringVar(&o.verifyABPath, "verify-prefix-ab", "", "verify a captured S2 prefix A/B artifact and exit")
	flag.StringVar(&o.s3Output, "hibernate-restart", "", "run the S3 hibernation/restart witness and write JSON to this path (or - for stdout)")
	flag.StringVar(&o.verifyS3Path, "verify-hibernate-restart", "", "verify a captured S3 artifact and exit")
	flag.IntVar(&o.s3Resident, "resident-high", 32, "S3 hard resident context cap")
	flag.IntVar(&o.s3Low, "resident-low", 16, "S3 warm-band low watermark")
	flag.IntVar(&o.s3Warm, "warm-cap", 8, "S3 warm reserve cap")
	flag.IntVar(&o.s3Turns, "turns", 2, "S3 synthetic turns per logical context")
	flag.Uint64Var(&o.s3Memory, "memory-envelope", 64<<20, "S3 peak Go allocation delta envelope in bytes")
	flag.StringVar(&o.descriptorOutput, "descriptor-bench", "", "run the 1,000-context descriptor/harness benchmark and write JSON")
	flag.StringVar(&o.verifyDescriptorPath, "verify-descriptor-bench", "", "verify a captured descriptor benchmark artifact and exit")
	flag.StringVar(&o.multiTurnDescriptorOutput, "multi-turn-descriptor", "", "run the 1,000-context multi-turn descriptor witness and write JSON")
	flag.StringVar(&o.verifyMultiTurnDescriptorPath, "verify-multi-turn-descriptor", "", "verify a captured multi-turn descriptor witness and exit")
	flag.StringVar(&o.verifyS2BPath, "verify-kernel-prefix-ab", "", "verify a controlled in-kernel prefix-cache A/B artifact and exit")
	flag.StringVar(&o.compatOutput, "compatibility-witness", "", "run compatibility scheduler witness and write JSON")
	flag.StringVar(&o.verifyCompatPath, "verify-compatibility", "", "verify compatibility artifact")
	registerBatchExecutionFlags(flag.CommandLine, &o.batchModelPath, &o.batchHardware, &o.batchOutput, &o.verifyBatchPath, &o.batchSize)
	flag.StringVar(&o.effectsOutput, "effects-witness", "", "run effect-safety witness and write JSON")
	flag.StringVar(&o.verifyEffectsPath, "verify-effects", "", "verify effect-safety artifact")
	flag.StringVar(&o.verifyAPIOnlyPath, "verify-api-only", "", "verify captured S6 API-only artifact")
	flag.BoolVar(&o.largeInputSelfcheck, "large-input-selfcheck", false, "run the fixture-backed 1,000-record large-input operator proof")
	flag.StringVar(&o.largeInputOutput, "large-input-output", "", "write the large-input operator proof artifact")
	flag.StringVar(&o.verifyLargeInputPath, "verify-large-input", "", "verify a captured large-input operator artifact")
	flag.BoolVar(&o.selectorSelfcheck, "filter-selector-selfcheck", false, "run the adaptive filter-selector proof")
	flag.StringVar(&o.selectorOutput, "filter-selector-output", "", "write the adaptive filter-selector proof artifact")
	flag.StringVar(&o.verifySelectorPath, "verify-filter-selector", "", "verify a captured filter-selector artifact")
	flag.BoolVar(&o.toolEnrichmentSelfcheck, "tool-enrichment-selfcheck", false, "run the read-only tool-enrichment fan-out proof")
	flag.StringVar(&o.toolEnrichmentOutput, "tool-enrichment-output", "", "write the read-only tool-enrichment proof artifact")
	flag.StringVar(&o.verifyToolEnrichmentPath, "verify-tool-enrichment", "", "verify a captured tool-enrichment artifact")
	flag.BoolVar(&o.provenanceFoldSelfcheck, "provenance-fold-selfcheck", false, "run the provenance-preserving hierarchical fold proof")
	flag.StringVar(&o.provenanceFoldOutput, "provenance-fold-output", "", "write the hierarchical fold proof artifact")
	flag.StringVar(&o.verifyProvenanceFoldPath, "verify-provenance-fold", "", "verify a captured hierarchical fold artifact")
	flag.BoolVar(&o.falsificationSelfcheck, "falsification-selfcheck", false, "run the tuned-baseline falsification benchmark")
	flag.StringVar(&o.falsificationOutput, "falsification-output", "", "write the falsification benchmark artifact")
	flag.StringVar(&o.verifyFalsificationPath, "verify-falsification", "", "verify a captured falsification artifact")
	flag.BoolVar(&o.effectBatchSelfcheck, "effect-batch-selfcheck", false, "run the witnessed effect-batch proof")
	flag.StringVar(&o.effectBatchOutput, "effect-batch-output", "", "write the effect-batch proof artifact")
	flag.StringVar(&o.verifyEffectBatchPath, "verify-effect-batch", "", "verify a captured effect-batch artifact")
	flag.StringVar(&o.corpusInputPath, "corpus-input", "", "freeze a public GitHub issue JSON export into leakage-controlled corpus artifacts")
	flag.StringVar(&o.corpusPublicPath, "corpus-public-output", "", "public candidate-input corpus output")
	flag.StringVar(&o.corpusAnswersPath, "corpus-answers-output", "", "separate answer bundle output")
	flag.StringVar(&o.corpusReportPath, "corpus-report-output", "", "corpus provenance/leak/grade report output")
	flag.StringVar(&o.corpusSource, "corpus-source", "github.com/anthony-chaudhary/fak/issues", "source provenance label")
	flag.StringVar(&o.verifyCorpusPublic, "verify-corpus-public", "", "verify public corpus with answer/report artifacts")
	flag.StringVar(&o.verifyCorpusAnswers, "verify-corpus-answers", "", "answer bundle paired with --verify-corpus-public")
	flag.StringVar(&o.verifyCorpusReport, "verify-corpus-report", "", "report paired with --verify-corpus-public")
	flag.StringVar(&o.gradeCorpusAnswers, "grade-corpus-answers", "", "hidden answer bundle for independent grading")
	flag.StringVar(&o.gradeCorpusSubmission, "grade-corpus-submission", "", "candidate submission to grade")
	flag.StringVar(&o.gradeCorpusOutput, "grade-corpus-output", "", "write independent grade report")
	flag.StringVar(&o.tunedBaselinesPublic, "tuned-baselines-public", "", "public corpus for tuned baseline dry-run")
	flag.StringVar(&o.tunedBaselinesAnswers, "tuned-baselines-answers", "", "answer bundle for tuning/grading")
	flag.StringVar(&o.tunedBaselinesOutput, "tuned-baselines-output", "", "write tuned baseline report")
	flag.StringVar(&o.verifyTunedBaselinesPath, "verify-tuned-baselines", "", "verify a tuned baseline report")
	flag.StringVar(&o.routingVOIOutput, "routing-voi-output", "", "run adaptive filter/tool routing experiment")
	flag.StringVar(&o.verifyRoutingVOIPath, "verify-routing-voi", "", "verify an adaptive routing experiment")
	flag.StringVar(&o.filterToolSchedulerFold, "filter-tool-scheduler-fold", "", "stabilized tool fold input for scheduler matrix")
	flag.StringVar(&o.filterToolSchedulerOutput, "filter-tool-scheduler-output", "", "write filter/tool scheduler matrix")
	flag.IntVar(&o.filterToolSchedulerTrials, "filter-tool-scheduler-trials", 5, "controlled trials per scheduler policy")
	flag.StringVar(&o.verifyFilterToolSchedulerPath, "verify-filter-tool-scheduler", "", "verify filter/tool scheduler matrix")
	flag.StringVar(&o.liveFilterToolPacket, "live-filter-tool-packet", "", "frozen semantic packet for live filter/tool matrix")
	flag.StringVar(&o.liveFilterToolFold, "live-filter-tool-fold", "", "stabilized fold for live filter/tool matrix")
	flag.StringVar(&o.liveFilterToolOutput, "live-filter-tool-output", "", "write live filter/tool scheduler matrix")
	flag.StringVar(&o.disagreementPacket, "disagreement-audit-packet", "", "frozen semantic packet for live disagreement audit")
	flag.StringVar(&o.disagreementFold, "disagreement-audit-fold", "", "stabilized fold for live disagreement audit")
	flag.StringVar(&o.disagreementLive, "disagreement-audit-live", "", "S8o live artifact for disagreement audit")
	flag.StringVar(&o.disagreementOutput, "disagreement-audit-output", "", "write live disagreement audit")
	flag.StringVar(&o.verifyDisagreementPath, "verify-disagreement-audit", "", "verify a live disagreement audit artifact")
	flag.StringVar(&o.counterfactualSource, "counterfactual-source", "", "source semantic packet for paired counterfactual corpus")
	flag.StringVar(&o.counterfactualCorpusOut, "counterfactual-corpus-output", "", "write paired counterfactual corpus")
	flag.StringVar(&o.counterfactualCorpusIn, "counterfactual-corpus", "", "paired counterfactual corpus")
	flag.StringVar(&o.counterfactualJudgmentOut, "counterfactual-judgment-output", "", "write counterfactual judgments")
	flag.StringVar(&o.counterfactualAdjudicator, "counterfactual-adjudicator", "", "counterfactual adjudicator identity")
	flag.StringVar(&o.counterfactualFoldA, "counterfactual-fold-a", "", "first model-distinct judgment bundle")
	flag.StringVar(&o.counterfactualFoldB, "counterfactual-fold-b", "", "second model-distinct judgment bundle")
	flag.StringVar(&o.counterfactualFoldOut, "counterfactual-fold-output", "", "write counterfactual consensus fold")
	flag.StringVar(&o.verifyCounterfactualCorpus, "verify-counterfactual-corpus", "", "counterfactual corpus to verify")
	flag.StringVar(&o.verifyCounterfactualFold, "verify-counterfactual-fold", "", "counterfactual fold to verify")
}

func registerDemoFlagsB(o *demoOptions) {
	flag.StringVar(&o.trueAdmissionOut, "true-admission-output", "", "write true pre-answer admission matrix")
	flag.StringVar(&o.verifyTrueAdmissionPath, "verify-true-admission", "", "verify true pre-answer admission matrix")
	flag.StringVar(&o.naturalCorpusOut, "natural-multitool-corpus-output", "", "write natural multi-tool corpus")
	flag.StringVar(&o.naturalCorpusIn, "natural-multitool-corpus", "", "natural multi-tool corpus")
	flag.StringVar(&o.naturalJudgeOut, "natural-multitool-judgment-output", "", "write natural multi-tool judgments")
	flag.StringVar(&o.naturalAdjudicator, "natural-multitool-adjudicator", "", "natural multi-tool adjudicator identity")
	flag.StringVar(&o.naturalFoldA, "natural-multitool-fold-a", "", "first natural judgment bundle")
	flag.StringVar(&o.naturalFoldB, "natural-multitool-fold-b", "", "second natural judgment bundle")
	flag.StringVar(&o.naturalFoldOut, "natural-multitool-fold-output", "", "write natural consensus fold")
	flag.StringVar(&o.verifyNaturalCorpus, "verify-natural-multitool-corpus", "", "natural corpus to verify")
	flag.StringVar(&o.verifyNaturalFold, "verify-natural-multitool-fold", "", "natural fold to verify")
	flag.StringVar(&o.naturalSurfaceOut, "natural-multitool-surface-output", "", "write natural decision surface")
	flag.StringVar(&o.verifyNaturalSurfacePath, "verify-natural-multitool-surface", "", "verify natural decision surface")
	flag.StringVar(&o.naturalTrafficCorpus, "natural-traffic-corpus", "", "100+ record multi-label natural traffic corpus")
	flag.StringVar(&o.naturalTrafficJudgeOut, "natural-traffic-judgment-output", "", "write multi-label natural traffic judgments")
	flag.StringVar(&o.naturalTrafficAdjudicator, "natural-traffic-adjudicator", "", "natural traffic adjudicator identity")
	flag.StringVar(&o.naturalTrafficFoldA, "natural-traffic-fold-a", "", "first multi-label judgment bundle")
	flag.StringVar(&o.naturalTrafficFoldB, "natural-traffic-fold-b", "", "second multi-label judgment bundle")
	flag.StringVar(&o.naturalTrafficFoldOut, "natural-traffic-fold-output", "", "write frozen multi-label fold")
	flag.StringVar(&o.naturalTrafficReportOut, "natural-traffic-report-output", "", "run real seams and write held-out policy report")
	flag.StringVar(&o.verifyNaturalTrafficPath, "verify-natural-traffic", "", "verify multi-label fold or report")
	flag.StringVar(&o.verifyLiveFilterToolPath, "verify-live-filter-tool", "", "verify live filter/tool scheduler matrix")
	flag.Int64Var(&o.routingVOISeed, "routing-voi-seed", 6105, "deterministic routing experiment seed")
	flag.IntVar(&o.routingVOITrials, "routing-voi-trials", 24, "routing experiment repetitions")
	flag.IntVar(&o.routingVOIRecords, "routing-voi-records", 200, "records per mixture and trial")
	flag.StringVar(&o.semanticCorpus, "semantic-packet-corpus", "", "public corpus used to build a blinded semantic packet")
	flag.StringVar(&o.semanticPacketOutput, "semantic-packet-output", "", "write a blinded semantic annotation packet")
	flag.IntVar(&o.semanticPerSplit, "semantic-per-split", 16, "semantic packet records per tune/test split")
	flag.StringVar(&o.semanticPacketInput, "semantic-adjudicate-packet", "", "semantic packet to adjudicate through a live endpoint")
	flag.StringVar(&o.semanticJudgmentOutput, "semantic-adjudicate-output", "", "write one adjudicator bundle")
	flag.StringVar(&o.semanticEndpoint, "semantic-endpoint", "", "OpenAI-compatible endpoint root for semantic adjudication")
	flag.StringVar(&o.semanticAPIKey, "semantic-api-key", "", "API key for semantic endpoint")
	flag.StringVar(&o.semanticModel, "semantic-model", "", "semantic adjudicator model id")
	flag.StringVar(&o.semanticAdjudicator, "semantic-adjudicator", "", "independent adjudicator identity")
	flag.StringVar(&o.semanticPromptVersion, "semantic-prompt-version", semanticPromptV1, "semantic adjudication rubric version")
	flag.StringVar(&o.semanticFoldPacket, "semantic-fold-packet", "", "packet bound to two independent adjudicators")
	flag.StringVar(&o.semanticFoldA, "semantic-fold-a", "", "first independent adjudicator bundle")
	flag.StringVar(&o.semanticFoldB, "semantic-fold-b", "", "second independent adjudicator bundle")
	flag.StringVar(&o.semanticGoldOutput, "semantic-gold-output", "", "write hidden semantic consensus/abstention answers")
	flag.StringVar(&o.semanticTriplePacket, "semantic-triple-packet", "", "packet for three-adjudicator tool fold")
	flag.StringVar(&o.semanticTripleOldA, "semantic-triple-old-a", "", "legacy adjudicator A")
	flag.StringVar(&o.semanticTripleOldB, "semantic-triple-old-b", "", "legacy adjudicator B for old agreement")
	flag.StringVar(&o.semanticTripleV2A, "semantic-triple-v2-a", "", "v2 adjudicator A")
	flag.StringVar(&o.semanticTripleV2B, "semantic-triple-v2-b", "", "v2 adjudicator B")
	flag.StringVar(&o.semanticTripleOutput, "semantic-triple-output", "", "three-adjudicator tool fold output")
	flag.StringVar(&o.semanticGradeGold, "semantic-grade-gold", "", "hidden semantic answers for blind grading")
	flag.StringVar(&o.semanticGradeSubmission, "semantic-grade-submission", "", "candidate semantic submission")
	flag.StringVar(&o.semanticGradeOutput, "semantic-grade-output", "", "write semantic blind grade")
	flag.StringVar(&o.semanticGradeSplit, "semantic-grade-split", "test", "split to grade")
	flag.StringVar(&o.verifySemanticGoldPath, "verify-semantic-gold", "", "verify semantic consensus artifact")
	flag.StringVar(&o.verifySemanticGradePath, "verify-semantic-grade", "", "verify semantic blind grade")
	flag.StringVar(&o.liveMatrixPacket, "live-matrix-packet", "", "S8i public semantic packet")
	flag.StringVar(&o.liveMatrixGold, "live-matrix-gold", "", "hidden S8i semantic gold")
	flag.StringVar(&o.liveMatrixOutput, "live-matrix-output", "", "write live comparative matrix")
	flag.StringVar(&o.verifyLiveMatrixPath, "verify-live-matrix", "", "verify live semantic matrix")
	flag.StringVar(&o.liveMatrixEndpoint, "live-matrix-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&o.liveMatrixAPIKey, "live-matrix-api-key", "", "live endpoint API key")
	flag.StringVar(&o.liveMatrixModel, "live-matrix-model", "", "live endpoint model")
	flag.StringVar(&o.liveMatrixClass, "live-matrix-endpoint-class", "", "public provenance class")
	flag.StringVar(&o.liveMatrixHardware, "live-matrix-hardware", "", "public hardware provenance")
	flag.StringVar(&o.liveMatrixNativeBatch, "live-matrix-native-batch", "unsupported", "endpoint native batching capability")
	flag.StringVar(&o.liveMatrixPrefixCache, "live-matrix-prefix-cache", "usage-observed-only", "endpoint prefix-cache capability")
	flag.StringVar(&o.liveMatrixPricing, "live-matrix-pricing", "unavailable", "pricing snapshot provenance")
	flag.IntVar(&o.liveMatrixTrials, "live-matrix-trials", 3, "repeated trials per pipeline")
	flag.IntVar(&o.liveMatrixWorkers, "live-matrix-workers", 8, "bounded micro-context concurrency")
	flag.Float64Var(&o.liveMatrixInputPrice, "live-matrix-input-per-mtok", -1, "input USD per million tokens; negative means unavailable")
	flag.Float64Var(&o.liveMatrixOutputPrice, "live-matrix-output-per-mtok", -1, "output USD per million tokens; negative means unavailable")
	flag.StringVar(&o.strongMatrixPacket, "strong-matrix-packet", "", "S8i packet for strengthened baselines")
	flag.StringVar(&o.strongMatrixGold, "strong-matrix-gold", "", "hidden S8i gold")
	flag.StringVar(&o.strongMatrixOutput, "strong-matrix-output", "", "write strengthened live matrix")
	flag.StringVar(&o.verifyStrongMatrixPath, "verify-strong-matrix", "", "verify strengthened live matrix")
	flag.StringVar(&o.strongMatrixEndpoint, "strong-matrix-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&o.strongMatrixAPIKey, "strong-matrix-api-key", "", "endpoint API key")
	flag.StringVar(&o.strongMatrixModel, "strong-matrix-model", "", "endpoint model")
	flag.StringVar(&o.strongMatrixClass, "strong-matrix-endpoint-class", "", "endpoint class")
	flag.StringVar(&o.strongMatrixHardware, "strong-matrix-hardware", "", "hardware provenance")
	flag.StringVar(&o.strongMatrixBatch, "strong-matrix-native-batch", "unsupported", "native batching capability")
	flag.StringVar(&o.strongMatrixCache, "strong-matrix-prefix-cache", "usage-observed-only", "prefix-cache capability")
	flag.StringVar(&o.strongMatrixPricing, "strong-matrix-pricing", "unavailable", "pricing provenance")
	flag.IntVar(&o.strongMatrixTrials, "strong-matrix-trials", 2, "held-out trials")
	flag.IntVar(&o.strongMatrixWorkers, "strong-matrix-workers", 8, "request admission limit")
	flag.IntVar(&o.strongMatrixK, "strong-matrix-retrieval-k", 3, "top-k tune examples")
	flag.IntVar(&o.strongMatrixChunk, "strong-matrix-chunk-size", 4, "parallel chunk size")
	flag.StringVar(&o.tailPacket, "tail-policy-packet", "", "S8i packet for tail policies")
	flag.StringVar(&o.tailGold, "tail-policy-gold", "", "hidden S8i gold")
	flag.StringVar(&o.tailOutput, "tail-policy-output", "", "write live tail-policy matrix")
	flag.StringVar(&o.verifyTailPath, "verify-tail-policy", "", "verify live tail-policy matrix")
	flag.StringVar(&o.tailEndpoint, "tail-policy-endpoint", "", "OpenAI-compatible endpoint")
	flag.StringVar(&o.tailAPIKey, "tail-policy-api-key", "", "endpoint API key")
	flag.StringVar(&o.tailModel, "tail-policy-model", "", "endpoint model")
	flag.StringVar(&o.tailClass, "tail-policy-endpoint-class", "", "endpoint provenance class")
	flag.StringVar(&o.tailHardware, "tail-policy-hardware", "", "hardware provenance")
	flag.IntVar(&o.tailTrials, "tail-policy-trials", 2, "trials per policy")
	flag.IntVar(&o.tailWorkers, "tail-policy-workers", 8, "bounded workers")
	flag.IntVar(&o.tailSufficiency, "tail-policy-sufficiency", 12, "confirmed records before early stop")
	flag.Int64Var(&o.tailWindowMS, "tail-policy-window-ms", 15000, "per-window deadline milliseconds")
	flag.Int64Var(&o.tailTaskMS, "tail-policy-task-ms", 60000, "global task deadline milliseconds")
	flag.Int64Var(&o.tailHedgeMS, "tail-policy-hedge-ms", 6000, "bounded hedge delay milliseconds")
	flag.StringVar(&o.qualityInput, "quality-input", "", "ingest one run witness into a quality ledger")
	flag.StringVar(&o.qualityOutput, "quality-output", "", "write the quality ledger")
	flag.IntVar(&o.qualitySamples, "quality-samples", 16, "maximum sampled context IDs")
	flag.StringVar(&o.verifyQualityPath, "verify-quality", "", "verify a captured quality ledger")
	flag.StringVar(&o.fairnessOutput, "fairness-witness", "", "run the S7 mixed-tenant fairness fixture")
	flag.StringVar(&o.verifyFairnessPath, "verify-fairness", "", "verify a captured S7 fairness artifact")
	flag.StringVar(&o.gradeInput, "health-input", "", "quality ledger to grade for micro-context health")
	flag.StringVar(&o.gradeOutput, "health-scorecard", "", "write micro-context health scorecard")
	flag.StringVar(&o.verifyGradePath, "verify-health-scorecard", "", "verify micro-context health scorecard")
}

func runDemoCommands1(o demoOptions) bool {
	if o.verifyFairnessPath != "" {
		runVerify("verify-fairness", o.verifyFairnessPath, verifyFairnessArtifact)
		return true
	}
	if o.fairnessOutput != "" {
		if err := writeFairness(o.fairnessOutput); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyGradePath != "" {
		runVerify("verify-health-scorecard", o.verifyGradePath, verifyHealthArtifact)
		return true
	}
	if o.gradeOutput != "" {
		if err := writeHealthGrade(o.gradeInput, o.gradeOutput); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.corpusInputPath != "" {
		if err := freezeCorpus(o.corpusInputPath, o.corpusPublicPath, o.corpusAnswersPath, o.corpusReportPath, o.corpusSource); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: corpus freeze: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS corpus public=%s answers=%s report=%s\n", o.corpusPublicPath, o.corpusAnswersPath, o.corpusReportPath)
		return true
	}
	if o.tailPacket != "" {
		if err := runTailPolicyMatrix(o.tailPacket, o.tailGold, o.tailOutput, o.tailEndpoint, o.tailAPIKey, o.tailModel, o.tailClass, o.tailHardware, o.tailTrials, o.tailWorkers, o.tailWindowMS, o.tailTaskMS, o.tailHedgeMS, o.tailSufficiency); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: tail policy: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS tail policy %s\n", o.tailOutput)
		return true
	}
	if o.verifyTailPath != "" {
		runVerify("verify-tail-policy", o.verifyTailPath, verifyTailPolicyMatrix)
		return true
	}
	if o.strongMatrixPacket != "" {
		if err := runStrongLiveMatrix(o.strongMatrixPacket, o.strongMatrixGold, o.strongMatrixOutput, o.strongMatrixEndpoint, o.strongMatrixAPIKey, o.strongMatrixModel, o.strongMatrixClass, o.strongMatrixHardware, o.strongMatrixBatch, o.strongMatrixCache, o.strongMatrixPricing, o.strongMatrixTrials, o.strongMatrixWorkers, o.strongMatrixK, o.strongMatrixChunk); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: strong matrix: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS strong matrix %s\n", o.strongMatrixOutput)
		return true
	}
	if o.verifyStrongMatrixPath != "" {
		runVerify("verify-strong-matrix", o.verifyStrongMatrixPath, verifyStrongLiveMatrix)
		return true
	}
	if o.liveMatrixPacket != "" {
		if err := runLiveSemanticMatrix(o.liveMatrixPacket, o.liveMatrixGold, o.liveMatrixOutput, o.liveMatrixEndpoint, o.liveMatrixAPIKey, o.liveMatrixModel, o.liveMatrixClass, o.liveMatrixHardware, o.liveMatrixNativeBatch, o.liveMatrixPrefixCache, o.liveMatrixPricing, o.liveMatrixInputPrice, o.liveMatrixOutputPrice, o.liveMatrixTrials, o.liveMatrixWorkers); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: live matrix: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS live matrix %s\n", o.liveMatrixOutput)
		return true
	}
	if o.verifyLiveMatrixPath != "" {
		runVerify("verify-live-matrix", o.verifyLiveMatrixPath, verifyLiveSemanticMatrix)
		return true
	}
	if o.semanticTriplePacket != "" {
		if o.semanticTripleOldA == "" || o.semanticTripleOldB == "" || o.semanticTripleV2A == "" || o.semanticTripleV2B == "" || o.semanticTripleOutput == "" {
			fmt.Fprintln(os.Stderr, "microcontextdemo: semantic triple fold requires all inputs and output")
			os.Exit(2)
		}
		if err := foldSemanticToolTriple(o.semanticTriplePacket, o.semanticTripleOldA, o.semanticTripleOldB, o.semanticTripleV2A, o.semanticTripleV2B, o.semanticTripleOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic triple fold: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic triple fold %s\n", o.semanticTripleOutput)
		return true
	}
	if o.semanticFoldPacket != "" {
		if err := foldSemanticAdjudicators(o.semanticFoldPacket, o.semanticFoldA, o.semanticFoldB, o.semanticGoldOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic fold: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic fold %s\n", o.semanticGoldOutput)
		return true
	}
	if o.semanticGradeSubmission != "" {
		if err := gradeSemanticFiles(o.semanticGradeGold, o.semanticGradeSubmission, o.semanticGradeOutput, o.semanticGradeSplit); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic grade: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic grade %s\n", o.semanticGradeOutput)
		return true
	}
	if o.verifySemanticGoldPath != "" {
		runVerify("verify-semantic-gold", o.verifySemanticGoldPath, verifySemanticGold)
		return true
	}
	if o.verifySemanticGradePath != "" {
		runVerify("verify-semantic-grade", o.verifySemanticGradePath, verifySemanticGrade)
		return true
	}
	if o.semanticCorpus != "" {
		if err := makeSemanticPacket(o.semanticCorpus, o.semanticPacketOutput, o.semanticPerSplit); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic packet: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic packet %s\n", o.semanticPacketOutput)
		return true
	}
	if o.semanticPacketInput != "" {
		if err := runSemanticAdjudicator(o.semanticPacketInput, o.semanticJudgmentOutput, o.semanticEndpoint, o.semanticAPIKey, o.semanticModel, o.semanticAdjudicator, o.semanticPromptVersion); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: semantic adjudicator: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS semantic adjudicator %s\n", o.semanticJudgmentOutput)
		return true
	}
	if o.verifyLiveFilterToolPath != "" {
		runVerify("verify-live-filter-tool", o.verifyLiveFilterToolPath, verifyLiveFilterToolMatrix)
		return true
	}
	if o.naturalCorpusOut != "" {
		if err := buildNaturalCorpus(o.naturalCorpusOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural corpus %s\n", o.naturalCorpusOut)
		return true
	}
	return false
}

func runDemoCommands2A(o demoOptions) bool {
	if o.naturalTrafficJudgeOut != "" {
		if err := runNaturalTrafficJudge(context.Background(), o.naturalTrafficCorpus, o.naturalTrafficJudgeOut, o.semanticEndpoint, o.semanticAPIKey, o.semanticModel, o.naturalTrafficAdjudicator); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural traffic judgments %s\n", o.naturalTrafficJudgeOut)
		return true
	}
	if o.naturalTrafficFoldOut != "" {
		if err := foldNaturalTraffic(o.naturalTrafficCorpus, o.naturalTrafficFoldA, o.naturalTrafficFoldB, o.naturalTrafficFoldOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural traffic fold %s\n", o.naturalTrafficFoldOut)
		return true
	}
	if o.naturalTrafficReportOut != "" {
		if err := runNaturalTrafficReport(context.Background(), o.naturalTrafficCorpus, o.naturalTrafficFoldA, o.naturalTrafficReportOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural traffic report %s\n", o.naturalTrafficReportOut)
		return true
	}
	if o.verifyNaturalTrafficPath != "" {
		if err := verifyNaturalTraffic(o.naturalTrafficCorpus, o.verifyNaturalTrafficPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural traffic verified %s\n", o.verifyNaturalTrafficPath)
		return true
	}
	if o.naturalJudgeOut != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if err := runNaturalJudge(context.Background(), o.naturalCorpusIn, o.naturalJudgeOut, endpoint, apiKey, o.semanticModel, o.naturalAdjudicator); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural judgments %s\n", o.naturalJudgeOut)
		return true
	}
	if o.naturalFoldOut != "" {
		if err := foldNatural(o.naturalCorpusIn, o.naturalFoldA, o.naturalFoldB, o.naturalFoldOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural fold %s\n", o.naturalFoldOut)
		return true
	}
	if o.naturalSurfaceOut != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if err := runNaturalSurface(context.Background(), o.naturalCorpusIn, o.verifyNaturalFold, o.naturalSurfaceOut, endpoint, apiKey, o.liveMatrixModel); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural surface %s\n", o.naturalSurfaceOut)
		return true
	}
	if o.verifyNaturalFold != "" {
		if err := verifyNatural(o.verifyNaturalCorpus, o.verifyNaturalFold); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural verified %s\n", o.verifyNaturalFold)
		return true
	}
	if o.verifyNaturalSurfacePath != "" {
		if err := verifyNaturalSurface(o.verifyNaturalSurfacePath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS natural surface verified %s\n", o.verifyNaturalSurfacePath)
		return true
	}
	if o.counterfactualCorpusOut != "" {
		if err := buildCounterfactualCorpus(o.counterfactualSource, o.counterfactualCorpusOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS counterfactual corpus %s\n", o.counterfactualCorpusOut)
		return true
	}
	if o.counterfactualJudgmentOut != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if err := runCounterfactualAdjudicator(context.Background(), o.counterfactualCorpusIn, o.counterfactualJudgmentOut, endpoint, apiKey, o.semanticModel, o.counterfactualAdjudicator); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS counterfactual judgments %s\n", o.counterfactualJudgmentOut)
		return true
	}
	if o.counterfactualFoldOut != "" {
		if err := foldCounterfactual(o.counterfactualCorpusIn, o.counterfactualFoldA, o.counterfactualFoldB, o.counterfactualFoldOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS counterfactual fold %s\n", o.counterfactualFoldOut)
		return true
	}
	if o.trueAdmissionOut != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if err := runTrueAdmission(context.Background(), o.counterfactualCorpusIn, o.verifyCounterfactualFold, o.trueAdmissionOut, endpoint, apiKey, o.liveMatrixModel); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS true admission %s\n", o.trueAdmissionOut)
		return true
	}
	if o.verifyCounterfactualFold != "" {
		if err := verifyCounterfactual(o.verifyCounterfactualCorpus, o.verifyCounterfactualFold); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS counterfactual verified %s\n", o.verifyCounterfactualFold)
		return true
	}
	return false
}

func runDemoCommands2B(o demoOptions) bool {
	if o.verifyTrueAdmissionPath != "" {
		if err := verifyTrueAdmission(o.verifyTrueAdmissionPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS true admission verified %s\n", o.verifyTrueAdmissionPath)
		return true
	}
	if o.verifyDisagreementPath != "" {
		if err := verifyDisagreementAudit(o.verifyDisagreementPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS: verified %s\n", o.verifyDisagreementPath)
		return true
	}
	if o.disagreementOutput != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if o.disagreementPacket == "" || o.disagreementFold == "" || o.disagreementLive == "" {
			fmt.Fprintln(os.Stderr, "disagreement audit requires packet, fold, and live artifact")
			os.Exit(2)
		}
		if err := runDisagreementAudit(context.Background(), o.disagreementPacket, o.disagreementFold, o.disagreementLive, o.disagreementOutput, endpoint, apiKey, o.liveMatrixModel); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("PASS disagreement audit %s\n", o.disagreementOutput)
		return true
	}
	if o.liveFilterToolOutput != "" {
		endpoint, apiKey := o.semanticEndpoint, o.semanticAPIKey
		if endpoint == "" {
			endpoint = os.Getenv("OPENAI_BASE_URL")
		}
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if o.liveFilterToolPacket == "" || o.liveFilterToolFold == "" {
			fmt.Fprintln(os.Stderr, "microcontextdemo: live filter/tool packet and fold required")
			os.Exit(2)
		}
		if err := runLiveFilterToolMatrix(context.Background(), o.liveFilterToolPacket, o.liveFilterToolFold, o.liveFilterToolOutput, endpoint, apiKey, o.liveMatrixModel); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: live filter/tool: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS live filter/tool %s\n", o.liveFilterToolOutput)
		return true
	}
	if o.verifyFilterToolSchedulerPath != "" {
		runVerify("verify-filter-tool-scheduler", o.verifyFilterToolSchedulerPath, verifyFilterToolScheduler)
		return true
	}
	if o.filterToolSchedulerOutput != "" {
		if o.filterToolSchedulerFold == "" {
			fmt.Fprintln(os.Stderr, "microcontextdemo: -filter-tool-scheduler-fold required")
			os.Exit(2)
		}
		if err := runFilterToolScheduler(o.filterToolSchedulerFold, o.filterToolSchedulerOutput, o.filterToolSchedulerTrials); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: filter/tool scheduler: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS filter/tool scheduler %s\n", o.filterToolSchedulerOutput)
		return true
	}
	if o.routingVOIOutput != "" {
		if err := runRoutingVOI(o.routingVOIOutput, o.routingVOISeed, o.routingVOITrials, o.routingVOIRecords); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: routing VOI: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS routing VOI %s\n", o.routingVOIOutput)
		return true
	}
	if o.verifyRoutingVOIPath != "" {
		runVerify("verify-routing-voi", o.verifyRoutingVOIPath, verifyRoutingVOI)
		return true
	}
	if o.tunedBaselinesPublic != "" {
		if err := runTunedBaselines(o.tunedBaselinesPublic, o.tunedBaselinesAnswers, o.tunedBaselinesOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: tuned baselines: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS tuned baselines %s\n", o.tunedBaselinesOutput)
		return true
	}
	if o.verifyTunedBaselinesPath != "" {
		runVerify("verify-tuned-baselines", o.verifyTunedBaselinesPath, verifyTunedBaselines)
		return true
	}
	if o.gradeCorpusSubmission != "" {
		if err := gradeSubmissionFiles(o.gradeCorpusAnswers, o.gradeCorpusSubmission, o.gradeCorpusOutput); err != nil {
			fmt.Fprintf(os.Stderr, "microcontextdemo: corpus grade: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("PASS corpus grade %s\n", o.gradeCorpusOutput)
		return true
	}
	return false
}

func runDemoCommands3(o demoOptions) bool {
	if o.verifyCorpusPublic != "" || o.verifyCorpusAnswers != "" || o.verifyCorpusReport != "" {
		verifyCorpusSet := func(string) error {
			return verifyCorpusArtifacts(o.verifyCorpusPublic, o.verifyCorpusAnswers, o.verifyCorpusReport)
		}
		if o.verifyCorpusPublic != "" {
			runVerify("verify-corpus-public", o.verifyCorpusPublic, verifyCorpusSet)
		}
		if o.verifyCorpusAnswers != "" {
			runVerify("verify-corpus-answers", o.verifyCorpusAnswers, verifyCorpusSet)
		}
		if o.verifyCorpusReport != "" {
			runVerify("verify-corpus-report", o.verifyCorpusReport, verifyCorpusSet)
		}
		return true
	}
	if o.verifyEffectBatchPath != "" {
		runVerify("verify-effect-batch", o.verifyEffectBatchPath, verifyEffectBatchArtifact)
		return true
	}
	if o.effectBatchSelfcheck {
		if err := runEffectBatchSelfcheck(o.effectBatchOutput); err != nil {
			fmt.Fprintf(os.Stderr, "effect-batch selfcheck: %v\n", err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyFalsificationPath != "" {
		runVerify("verify-falsification", o.verifyFalsificationPath, verifyFalsificationArtifact)
		return true
	}
	if o.falsificationSelfcheck {
		if err := runFalsificationBench(o.falsificationOutput); err != nil {
			fmt.Fprintf(os.Stderr, "falsification selfcheck: %v\n", err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyProvenanceFoldPath != "" {
		runVerify("verify-provenance-fold", o.verifyProvenanceFoldPath, verifyProvenanceFoldArtifact)
		return true
	}
	if o.provenanceFoldSelfcheck {
		if err := runProvenanceFoldSelfcheck(o.provenanceFoldOutput); err != nil {
			fmt.Fprintf(os.Stderr, "provenance-fold selfcheck: %v\n", err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyToolEnrichmentPath != "" {
		runVerify("verify-tool-enrichment", o.verifyToolEnrichmentPath, verifyToolEnrichmentArtifact)
		return true
	}
	if o.toolEnrichmentSelfcheck {
		if err := runToolEnrichmentSelfcheck(context.Background(), o.toolEnrichmentOutput, o.cfg.Workers); err != nil {
			fmt.Fprintf(os.Stderr, "tool-enrichment selfcheck: %v\n", err)
			os.Exit(1)
		}
		return true
	}
	if o.verifySelectorPath != "" {
		runVerify("verify-filter-selector", o.verifySelectorPath, verifySelectorArtifact)
		return true
	}
	if o.selectorSelfcheck || o.selectorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		r, err := buildSelectorReport(ctx, o.cfg.Workers)
		r = compactSelectorReport(r)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil && o.selectorOutput != "" {
			merr = os.WriteFile(o.selectorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if o.selectorOutput == "" {
			fmt.Println(string(b))
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyLargeInputPath != "" {
		runVerify("verify-large-input", o.verifyLargeInputPath, verifyLargeInputArtifact)
		return true
	}
	if o.largeInputSelfcheck || o.largeInputOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		r, err := buildLargeInputReport(ctx, o.cfg.Contexts, o.cfg.Workers)
		// The proof artifact keeps aggregate source accounting and citations; the
		// in-memory/test witness retains all 1,000 typed facts.
		r.Baseline.Facts = nil
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil && o.largeInputOutput != "" {
			merr = os.WriteFile(o.largeInputOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if o.largeInputOutput == "" {
			fmt.Println(string(b))
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyQualityPath != "" {
		runVerify("verify-quality", o.verifyQualityPath, verifyQualityLedgerArtifact)
		return true
	}
	if o.qualityInput != "" {
		if o.qualityOutput == "" {
			fmt.Fprintln(os.Stderr, "quality-output is required")
			os.Exit(2)
		}
		if err := writeQualityLedger(o.qualityInput, o.qualityOutput, o.qualitySamples); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyAPIOnlyPath != "" {
		runVerify("verify-api-only", o.verifyAPIOnlyPath, verifyAPIOnlyArtifact)
		return true
	}
	if o.verifyPath != "" {
		runVerify("verify", o.verifyPath, verifyArtifact)
		return true
	}
	if o.verifyABPath != "" {
		runVerify("verify-prefix-ab", o.verifyABPath, verifyABArtifact)
		return true
	}
	if o.verifyS2BPath != "" {
		runVerify("verify-kernel-prefix-ab", o.verifyS2BPath, verifyS2BArtifact)
		return true
	}
	if o.verifyEffectsPath != "" {
		runVerify("verify-effects", o.verifyEffectsPath, verifyEffectsArtifact)
		return true
	}
	if o.effectsOutput != "" {
		r, err := buildEffectsReport()
		b, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(o.effectsOutput, append(b, '\n'), 0o644)
		}
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	return false
}

func runDemoCommands4(o demoOptions) bool {
	if o.verifyBatchPath != "" {
		runVerify("verify-compat-batch-execution", o.verifyBatchPath, verifyBatchExecutionArtifact)
		return true
	}
	if o.batchOutput != "" {
		if err := runBatchExecution(o.batchModelPath, o.batchHardware, o.batchOutput, o.batchSize); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyCompatPath != "" {
		runVerify("verify-compatibility", o.verifyCompatPath, verifyCompatibilityArtifact)
		return true
	}
	if o.compatOutput != "" {
		r, err := buildCompatReport()
		b, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(o.compatOutput, append(b, '\n'), 0o644)
		}
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyMultiTurnDescriptorPath != "" {
		runVerify("verify-multi-turn-descriptor", o.verifyMultiTurnDescriptorPath, verifyMultiTurnDescriptorArtifact)
		return true
	}
	if o.multiTurnDescriptorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runMultiTurnDescriptor(ctx, o.cfg.Contexts, o.cfg.Workers, o.s3Turns)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(o.multiTurnDescriptorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyDescriptorPath != "" {
		runVerify("verify-descriptor-bench", o.verifyDescriptorPath, verifyDescriptorArtifact)
		return true
	}
	if o.descriptorOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runDescriptorBenchmark(ctx, o.cfg.Contexts)
		b, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(o.descriptorOutput, append(b, '\n'), 0o644)
		}
		if merr != nil {
			fmt.Fprintln(os.Stderr, merr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	if o.verifyS3Path != "" {
		runVerify("verify-hibernate-restart", o.verifyS3Path, verifyS3Artifact)
		return true
	}
	if o.s3Output != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		r, err := runS3(ctx, s3Config{Contexts: o.cfg.Contexts, Workers: o.cfg.Workers, ResidentHigh: o.s3Resident, ResidentLow: o.s3Low, WarmCap: o.s3Warm, Turns: o.s3Turns, MemoryBytes: o.s3Memory})
		var werr error
		if o.s3Output == "-" {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			werr = enc.Encode(r)
		} else {
			b, marshalErr := json.MarshalIndent(r, "", "  ")
			if marshalErr != nil {
				werr = marshalErr
			} else {
				werr = os.WriteFile(o.s3Output, append(b, '\n'), 0o644)
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
		return true
	}
	if o.abOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()
		r, err := runAB(ctx, o.cfg)
		if writeErr := writeAB(o.abOutput, r); writeErr != nil {
			fmt.Fprintln(os.Stderr, writeErr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return true
	}
	lineage, lineageErr := microagent.LineageFromEnv()
	if lineageErr != nil {
		fmt.Fprintln(os.Stderr, lineageErr)
		os.Exit(1)
	}
	o.cfg.Lineage = lineage
	ctx, cancel := overallDeadline(context.Background(), o.cfg.RunTimeout)
	defer cancel()
	r, err := run(ctx, o.cfg)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return false
}

func runDemoOptions(o demoOptions) {
	if runDemoCommands1(o) || runDemoCommands2A(o) || runDemoCommands2B(o) || runDemoCommands3(o) || runDemoCommands4(o) {
		return
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
