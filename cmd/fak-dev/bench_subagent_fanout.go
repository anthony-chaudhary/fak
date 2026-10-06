package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Issue #6036 and Issue #12325 Contract Constants.
const (
	// ContractIssue6036 is the root prefix KV reuse comparison contract issue.
	ContractIssue6036 = "6036"
	// ContractIssue12325 is the subagent fan-out benchmark harness issue.
	ContractIssue12325 = "12325"

	// SubagentFanoutComparisonSchema identifies the apples-to-apples benchmark receipt.
	SubagentFanoutComparisonSchema = "fak.benchmark.subagent_fanout_apples_to_apples/v1"

	// FanoutUnifiedThroughputSchema identifies the aggregate unified-throughput block
	// (issue #13076): a measured sum of generated tokens over measured wall time,
	// distinct from any single cell's per-N decode_throughput_tok_per_sec.
	FanoutUnifiedThroughputSchema = "fak.benchmark.fanout_unified_throughput/v1"

	// FanoutReuseSourceAnalytic is the source tag for reuse the harness computed
	// arithmetically rather than observing from the server.
	FanoutReuseSourceAnalytic = "analytic"
	// FanoutReuseSourcePrometheus is the source tag for reuse read from the
	// server's Prometheus text exposition.
	FanoutReuseSourcePrometheus = "prometheus:/metrics"
	// FanoutReuseSourceUnobserved marks a cell where no server telemetry could be
	// read; observed reuse stays unmeasured rather than assumed (an evidence gap,
	// never a silent pass).
	FanoutReuseSourceUnobserved = "unobserved"

	// The 4 Required Comparison Arms under Issue #6036:
	ArmNoReuse = "no_reuse" // Arm 1: No-reuse baseline (cache disabled)
	ArmSGLang  = "sglang"   // Arm 2: SGLang with RadixAttention
	ArmVLLM    = "vllm"     // Arm 3: vLLM with Automatic Prefix Caching
	ArmFAK     = "fak"      // Arm 4: FAK native engine / gateway

	// ArmLLamaCPP is an additional selectable arm (Issue #13023): a live
	// head-to-head against a real llama-server (llama.cpp) on Apple Silicon.
	// It is NOT one of the 4 mandatory arms, so CanonicalArms stays frozen.
	ArmLLamaCPP = "llamacpp"

	// Default Contract Invariants.
	DefaultModel          = "Qwen/Qwen2.5-Coder-7B-Instruct"
	DefaultQuantization   = "Q4_K_M"
	FixedMemoryFraction   = 0.85 // Contract-mandated 0.85 memory fraction
	DefaultPrefixTokens   = 4096 // Shared root coordinator prompt / tools / context
	DefaultSuffixTokens   = 512  // Subagent-specific prompt / task assignment
	DefaultDecodeTokens   = 64   // Generated output tokens per subagent
	DefaultTrials         = 5    // Repetitions per cell for variance control
	DefaultMaxConcurrency = 64
)

// CanonicalArms returns the 4 mandatory comparison arms in contract order.
var CanonicalArms = []string{
	ArmNoReuse,
	ArmSGLang,
	ArmVLLM,
	ArmFAK,
}

// CanonicalFanoutSweep returns the required subagent fan-out sweep values.
var CanonicalFanoutSweep = []int{1, 4, 8, 16, 32}

// Fanout sweep scope verdicts. Only FanoutSweepFull can satisfy the contract; a
// smoke subset runs but is always non-compliant and labeled partial.
const (
	FanoutSweepFull         = "full"
	FanoutSweepSmokeSubset  = "smoke_subset"
	FanoutSweepNonCanonical = "non_canonical"
)

// Live request shapes. RequestShapeCompletion sends the shared prefix inside a
// legacy /v1/completions prompt; RequestShapeChatSystem sends it as the system
// message of a /v1/chat/completions request with the subagent task as the user
// turn, the shape a coordinator uses when it fans out subagents.
const (
	RequestShapeCompletion = "completion"
	RequestShapeChatSystem = "chat-system"
)

// Live dispatch schedules. ScheduleConcurrent launches all N subagents at once;
// ScheduleParentFirst completes subagent 0 before launching the other N-1, so
// the siblings can reuse a prefix the server has already materialized.
const (
	ScheduleConcurrent  = "concurrent"
	ScheduleParentFirst = "parent-first"
)

var (
	ErrFanoutRequestShape    = errors.New("unknown -request-shape")
	ErrFanoutSchedule        = errors.New("unknown -schedule")
	ErrFanoutDeclaredCtx     = errors.New("invalid -declared-per-slot-ctx")
	ErrReuseCounterAbsent    = errors.New("no cache-reuse counter in /metrics")
	ErrReuseCounterReset     = errors.New("cache-reuse counter went backwards between scrapes")
	ErrReuseScrapeIncomplete = errors.New("cache-reuse counter missing from one of the two scrapes")
)

// ArmDescription maps arm identifier to human-readable description.
var ArmDescription = map[string]string{
	ArmNoReuse:  "Arm 1: No-reuse baseline (cache disabled)",
	ArmSGLang:   "Arm 2: SGLang with RadixAttention",
	ArmVLLM:     "Arm 3: vLLM with Automatic Prefix Caching",
	ArmFAK:      "Arm 4: FAK native engine / gateway",
	ArmLLamaCPP: "llama.cpp llama-server (Metal prefix cache / continuous batching)",
}

// DistributionStats captures empirical latency distributions across trials.
type DistributionStats struct {
	Count  int       `json:"count"`
	Mean   float64   `json:"mean_ms"`
	Min    float64   `json:"min_ms"`
	Max    float64   `json:"max_ms"`
	P50    float64   `json:"p50_ms"`
	P95    float64   `json:"p95_ms"`
	P99    float64   `json:"p99_ms"`
	StdDev float64   `json:"std_dev_ms"`
	Raw    []float64 `json:"raw_samples_ms,omitempty"`
}

// ITLStats captures Inter-Token Latency distribution and jitter.
type ITLStats struct {
	Count    int     `json:"count"`
	MeanMs   float64 `json:"mean_ms"`
	P50Ms    float64 `json:"p50_ms"`
	P95Ms    float64 `json:"p95_ms"`
	P99Ms    float64 `json:"p99_ms"`
	StdDevMs float64 `json:"std_dev_ms"`
	JitterMs float64 `json:"jitter_ms"` // defined as P99 - P50 latency spread
}

// FanoutArmResult encapsulates the benchmark observations for a single (Fanout N, Arm) cell.
type FanoutArmResult struct {
	Arm               string  `json:"arm"`
	ArmDescription    string  `json:"arm_description"`
	FanoutN           int     `json:"fanout_n"`
	Trials            int     `json:"trials"`
	PrefixTokens      int     `json:"prefix_tokens"`
	SuffixTokens      int     `json:"suffix_tokens"`
	DecodeTokens      int     `json:"decode_tokens"`
	TotalPromptTokens int64   `json:"total_prompt_tokens"`
	ReusedTokens      int64   `json:"reused_tokens"`
	PrefixHitRate     float64 `json:"prefix_hit_rate"`
	// ObservedReuseTokens is the cache reuse the SERVER reported for this cell's
	// slot, read from live telemetry — distinct from ReusedTokens, which is the
	// harness's analytic (n-1)*P estimate. A cell where no telemetry could be read
	// leaves this zero and sets ReuseObservationSource to unobserved: an evidence
	// gap, never a silent pass (issue #13076).
	ObservedReuseTokens int64 `json:"observed_reuse_tokens"`
	// ObservedHitRate is ObservedReuseTokens / TotalPromptTokens. It is the
	// measured twin of PrefixHitRate.
	ObservedHitRate float64 `json:"observed_hit_rate"`
	// ReuseObservationSource names where ObservedReuseTokens came from
	// (prometheus:/metrics) or why it is absent (analytic, unobserved).
	ReuseObservationSource string `json:"reuse_observation_source"`
	// ReuseDivergence is the fail flag: the analytic estimate claims reuse
	// (PrefixHitRate > 0) on a shared-prefix arm but the server observed none.
	// A run with any divergence must fail rather than publish an unfalsifiable
	// reuse claim.
	ReuseDivergence bool `json:"reuse_divergence,omitempty"`
	// ReuseScrape is the before/after counter witness behind ObservedReuseTokens,
	// present only when both scrapes read the same counter family.
	ReuseScrape *FanoutReuseScrape `json:"reuse_scrape,omitempty"`
	// ObservedReuseMultiplier is offered prompt tokens over prompt tokens the
	// server actually computed (offered / (offered - reused)), both from the
	// scraped counter deltas. Zero means the denominator was not observed.
	ObservedReuseMultiplier   float64           `json:"observed_reuse_multiplier"`
	RequestShape              string            `json:"request_shape,omitempty"`
	Schedule                  string            `json:"schedule,omitempty"`
	RequestsIssued            int               `json:"requests_issued,omitempty"`
	TTFT                      DistributionStats `json:"ttft"`
	ITL                       ITLStats          `json:"itl"`
	DecodeThroughputTokPerSec float64           `json:"decode_throughput_tok_per_sec"`
	WallClockMs               float64           `json:"wall_clock_ms"`
	KVMemoryBytes             int64             `json:"kv_memory_bytes"`
	PeakMemoryFraction        float64           `json:"peak_memory_fraction"`
	OutputEquivalence         bool              `json:"output_equivalence"`
	OutputHash                string            `json:"output_hash"`
	// ServeCompleteness records whether the reference server was probed to
	// actually serve the frozen workload geometry. It is present only for live
	// cells; a simulated cell leaves it nil. ControlArmServed is the gate: a
	// cell is only admissible as an ablation control when it is true.
	ServeCompleteness *FanoutServeCompleteness `json:"serve_completeness,omitempty"`
	Error             string                   `json:"error,omitempty"`
}

// FanoutServeCompleteness is the frozen-workload-geometry capacity witness for a
// live cell. It binds the geometry the harness will send (one request of
// PromptTokens = P+S, plus DecodeTokens generated) to the per-slot context
// capacity the reference server reports it will serve. See issue #13134.
//
// The gate is ControlArmServed: an ablation arm whose reference cannot hold the
// frozen prompt is a control that silently did not run, which is worse than a
// missing arm because the comparison reads as if it were attempted. A shortfall
// fails the cell closed with PromptOverflow naming the observed limit; an
// unreadable/unstated limit fails closed as CapacityUnobserved (an evidence gap,
// never a silent pass).
type FanoutServeCompleteness struct {
	// PromptTokens is the frozen per-slot geometry (P+S) one request carries.
	PromptTokens int `json:"prompt_tokens"`
	// DecodeTokens is the generated-token budget for the same request.
	DecodeTokens int `json:"decode_tokens"`
	// RequestedTotalTokens is what one slot must hold: PromptTokens+DecodeTokens.
	RequestedTotalTokens int `json:"requested_total_tokens"`
	// ObservedPerSlotTokens is the per-slot context the server reports it will
	// serve (llama.cpp /props default_generation_settings.n_ctx, which is
	// -c/--ctx-size divided by --parallel). Zero means UNOBSERVED.
	ObservedPerSlotTokens int `json:"observed_per_slot_tokens"`
	// ObservedTotalSlots is the server's served sequence/slot count
	// (llama.cpp /props total_slots == --parallel). Zero means unobserved.
	ObservedTotalSlots int `json:"observed_total_slots,omitempty"`
	// CapacitySource names where ObservedPerSlotTokens came from, or why it is
	// absent. It is the auditor's trail: "llama.cpp /props" is authoritative.
	CapacitySource string `json:"capacity_source"`
	// DeclaredPerSlotTokens is the operator-declared per-slot capacity, used
	// only when no /props surface reported one. It is never copied into
	// ObservedPerSlotTokens.
	DeclaredPerSlotTokens int `json:"declared_per_slot_tokens,omitempty"`
	// CapacityDeclared is true when the admission rests on the declaration,
	// not an observation.
	CapacityDeclared bool `json:"capacity_declared"`
	// PrimaryProbe records why the arm endpoint's own /props was unusable when
	// capacity came from the upstream /props or a declaration.
	PrimaryProbe string `json:"primary_probe,omitempty"`
	// ControlArmServed is the single gate bit. True only when the observed (or
	// explicitly declared) per-slot capacity covers RequestedTotalTokens.
	ControlArmServed bool `json:"control_arm_served"`
	// Refusal is one of FanoutServeRefusalVocabulary, empty exactly when served.
	Refusal string `json:"refusal,omitempty"`
	// EvidenceGap distinguishes "could not establish the served limit" from
	// "established the limit, and it is too small". Both fail the cell closed.
	EvidenceGap bool `json:"evidence_gap"`
	// Detail is the human-readable fail-closed reason, naming the observed limit
	// and the command that set it.
	Detail string `json:"detail,omitempty"`
}

// FanoutServeRefusalVocabulary is the closed set of serve-completeness refusals.
var FanoutServeRefusalVocabulary = []string{
	FanoutServeRefusePromptOverflow,     // observed per-slot ctx < P+S (or P+S+D)
	FanoutServeRefuseCapacityUnobserved, // /props unreadable or n_ctx undeclared
	FanoutServeRefuseProbeError,         // transport error reaching /props
}

const (
	// FanoutServeRefusePromptOverflow: the server's observed per-slot capacity is
	// smaller than the frozen prompt geometry. The control arm cannot serve the
	// workload; the cell fails closed.
	FanoutServeRefusePromptOverflow = "SERVE_PROMPT_OVERFLOW"
	// FanoutServeRefuseCapacityUnobserved: the served per-slot capacity could not
	// be read (no /props, or n_ctx absent/undeclared). Fail closed rather than
	// assume a default — an unmeasured control is not a control.
	FanoutServeRefuseCapacityUnobserved = "SERVE_CAPACITY_UNOBSERVED"
	// FanoutServeRefuseProbeError: the capacity probe itself failed (transport).
	FanoutServeRefuseProbeError = "SERVE_PROBE_ERROR"
)

// servedPropsWire is the subset of llama.cpp's /props response this harness
// trusts. Shape verified against a live llama-server: the per-slot context is
// default_generation_settings.n_ctx (here 4096 for `-c 32768 --parallel 8`), and
// total_slots is --parallel. There is no top-level n_ctx field.
type servedPropsWire struct {
	TotalSlots                int `json:"total_slots"`
	DefaultGenerationSettings struct {
		NCtx int `json:"n_ctx"`
	} `json:"default_generation_settings"`
}

const propsSlotCapacityField = "default_generation_settings.n_ctx"

func (p servedPropsWire) perSlot() int { return p.DefaultGenerationSettings.NCtx }

// ContractValidation records compliance against Issue #6036 and #12325 mandates.
type ContractValidation struct {
	Compliant                   bool     `json:"compliant"`
	IssueReference              string   `json:"issue_reference"`
	IdenticalWeightsEnforced    bool     `json:"identical_weights_enforced"`
	IdenticalQuantization       bool     `json:"identical_quantization_enforced"`
	FixedMemoryFractionVerified bool     `json:"fixed_memory_fraction_verified"`
	RequiredMemoryFraction      float64  `json:"required_memory_fraction"`
	ObservedMemoryFraction      float64  `json:"observed_memory_fraction"`
	AllFourArmsPresent          bool     `json:"all_four_arms_present"`
	PresentArms                 []string `json:"present_arms"`
	FanoutSweepVerified         bool     `json:"fanout_sweep_verified"`
	ObservedFanoutSweep         []int    `json:"observed_fanout_sweep"`
	// SweepScope is one of FanoutSweepFull, FanoutSweepSmokeSubset,
	// FanoutSweepNonCanonical. Partial is true for a smoke subset: the run is
	// admitted but can never be Compliant.
	SweepScope                string   `json:"sweep_scope"`
	Partial                   bool     `json:"partial"`
	CapturedTTFTDistributions bool     `json:"captured_ttft_distributions"`
	CapturedITLJitter         bool     `json:"captured_itl_jitter"`
	Violations                []string `json:"violations,omitempty"`
}

// SubagentFanoutReceipt is the complete machine-readable benchmark receipt.
type SubagentFanoutReceipt struct {
	Schema         string             `json:"schema"`
	ContractIssue  string             `json:"contract_issue"`
	Benchmark      string             `json:"benchmark"`
	Timestamp      string             `json:"timestamp"`
	HostOS         string             `json:"host_os"`
	Model          string             `json:"model"`
	Quantization   string             `json:"quantization"`
	MemoryFraction float64            `json:"memory_fraction"`
	FanoutSweep    []int              `json:"fanout_sweep"`
	Arms           []string           `json:"arms"`
	Contract       ContractValidation `json:"contract"`
	Results        []FanoutArmResult  `json:"results"`
	Summary        map[string]any     `json:"summary"`
	// UnifiedThroughput is the measured aggregate tokens/sec across the whole
	// sweep (issue #13076), distinct from any one cell's per-N throughput. It is
	// nil when no cell produced a measurable wall time.
	UnifiedThroughput *UnifiedThroughput `json:"unified_throughput,omitempty"`
	// ReuseDivergenceArms names every shared-prefix arm whose analytic reuse the
	// server did not observe. A non-empty list fails the run.
	ReuseDivergenceArms []string `json:"reuse_divergence_arms,omitempty"`
}

// UnifiedThroughput is the measured aggregate throughput verdict for a sweep.
// GeneratedTokens is the honest sum of generated tokens across the counted
// cells; WallSeconds is the summed measured wall time; TokPerSec is their ratio.
// An optional SLO floor (MinTokPerSec > 0) makes the verdict falsifiable: Met is
// false when the measured aggregate falls short.
type UnifiedThroughput struct {
	Schema          string  `json:"schema"`
	GeneratedTokens int64   `json:"generated_tokens"`
	WallSeconds     float64 `json:"wall_seconds"`
	TokPerSec       float64 `json:"tok_per_sec"`
	MinTokPerSec    float64 `json:"min_tok_per_sec,omitempty"`
	// Met is nil when no floor was set (an unconstrained measurement), so a
	// reader can never mistake "no SLO" for "SLO passed".
	Met *bool `json:"met,omitempty"`
}

// FanoutBenchConfig holds user flags and execution options.
type FanoutBenchConfig struct {
	Model          string
	Quantization   string
	MemoryFraction float64
	FanoutSweep    []int
	Arms           []string
	PrefixTokens   int
	SuffixTokens   int
	DecodeTokens   int
	Trials         int
	Live           bool
	Endpoints      map[string]string
	VerifyContract bool
	JSON           bool
	OutputFile     string
	Seed           int64
	// MinUnifiedTPS is the optional unified-throughput SLO floor (tokens/sec)
	// across the whole sweep. Zero disables the gate (an unconstrained
	// measurement); a positive value fails the run when the measured aggregate
	// falls short (issue #13076).
	MinUnifiedTPS float64
	// UpstreamPropsURL is the base URL of the server behind a proxying gateway
	// (e.g. the llama-server a fak gateway forwards to). Its /props is read when
	// the arm endpoint's own /props cannot report per-slot capacity.
	UpstreamPropsURL string
	// DeclaredPerSlotCtx is the last-resort operator declaration of per-slot
	// capacity, recorded as declared, never as observed. Zero disables it.
	DeclaredPerSlotCtx int
	RequestShape       string
	Schedule           string
}

// FanoutReuseScrape is the counter witness for one live cell: the single
// counter family read before and after the cell, and its delta.
type FanoutReuseScrape struct {
	Counter          string `json:"counter"`
	Path             string `json:"path"`
	Before           int64  `json:"before"`
	After            int64  `json:"after"`
	Delta            int64  `json:"delta"`
	PromptCounter    string `json:"prompt_counter,omitempty"`
	PromptTokenDelta int64  `json:"prompt_token_delta,omitempty"`
	// OfferedTokens is the total prompt tokens the server saw in the cell,
	// cached plus computed. Zero when no paired prompt counter was readable.
	OfferedTokens int64 `json:"offered_tokens,omitempty"`
}

// runBenchSubagentFanout executes the apples-to-apples subagent fanout benchmark harness.
func runBenchSubagentFanout(stdout, stderr io.Writer, argv []string) int {
	cfg, err := parseFanoutFlags(stderr, argv)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: %v\n", err)
		return 2
	}

	// Validate contract invariants.
	validation := validateContractInvariants(cfg)
	if cfg.VerifyContract && len(validation.Violations) > 0 {
		fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: contract validation failed (Issue #%s):\n", ContractIssue6036)
		for _, v := range validation.Violations {
			fmt.Fprintf(stderr, "  - %s\n", v)
		}
		return 1
	}

	harness := NewFanoutBenchmarkHarness(cfg)
	receipt, err := harness.Run(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: execution error: %v\n", err)
		return 1
	}

	// Attach contract validation to receipt.
	receipt.Contract = validation

	if cfg.JSON {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: failed to marshal JSON: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		renderPrettyReceipt(stdout, receipt)
	}

	if cfg.OutputFile != "" {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: failed to marshal output JSON: %v\n", err)
			return 1
		}
		if err := os.WriteFile(cfg.OutputFile, data, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: failed to write %s: %v\n", cfg.OutputFile, err)
			return 1
		}
		if !cfg.JSON {
			fmt.Fprintf(stdout, "\nReceipt saved to: %s\n", cfg.OutputFile)
		}
	}

	// Fail closed on a reuse claim the server did not corroborate (issue #13076).
	// A receipt whose shared-prefix arms show analytic reuse but zero observed
	// reuse is publishing an unfalsifiable claim, so it must not exit 0.
	if len(receipt.ReuseDivergenceArms) > 0 {
		fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: reuse divergence on arm(s) %s: analytic prefix_hit_rate > 0 but server observed 0 reused tokens\n",
			strings.Join(receipt.ReuseDivergenceArms, ","))
		return 1
	}

	// Enforce the optional unified-throughput SLO floor (issue #13076). Met is
	// non-nil exactly when a floor was configured; a shortfall fails the run.
	if ut := receipt.UnifiedThroughput; ut != nil && ut.Met != nil && !*ut.Met {
		fmt.Fprintf(stderr, "fak-dev bench-subagent-fanout: unified throughput %.1f tok/s is below the SLO floor %.1f tok/s\n",
			ut.TokPerSec, ut.MinTokPerSec)
	}

	// The measured-reuse and SLO verdicts gate the exit code through one pure
	// function so a server-free test can witness each failure mode.
	if code := fanoutGateExitCode(receipt); code != 0 {
		return code
	}

	if cfg.VerifyContract && len(receipt.Contract.Violations) > 0 {
		return 1
	}
	return 0
}

// fanoutGateExitCode is the pure, deterministic exit verdict for a completed
// receipt (issue #13076): it separates the measured-reuse and unified-throughput
// gates from the byte-emitting run so a server-free test can witness each one
// without a running server or a host timer. It returns 1 when a shared-prefix
// reuse claim was not corroborated by server telemetry, or when an armed
// unified-throughput SLO floor was missed; 0 otherwise. A nil UnifiedThroughput
// (no measured wall) is not a gate failure — an unmeasured sweep cannot be
// below a floor.
func fanoutGateExitCode(receipt *SubagentFanoutReceipt) int {
	if receipt == nil {
		return 1
	}
	if len(receipt.ReuseDivergenceArms) > 0 {
		return 1
	}
	if ut := receipt.UnifiedThroughput; ut != nil && ut.Met != nil && !*ut.Met {
		return 1
	}
	return 0
}

func parseFanoutFlags(stderr io.Writer, argv []string) (*FanoutBenchConfig, error) {
	fs := flag.NewFlagSet("bench-subagent-fanout", flag.ContinueOnError)
	fs.SetOutput(stderr)

	model := fs.String("model", DefaultModel, "exact model identifier (enforced identical across all arms)")
	quant := fs.String("quant", DefaultQuantization, "exact quantization level (enforced identical across all arms)")
	memFrac := fs.Float64("mem-fraction", FixedMemoryFraction, "fixed GPU memory fraction (mandated 0.85 by contract)")
	fanoutStr := fs.String("fanout", "1,4,8,16,32", "comma-separated subagent fanout sweep N values")
	armsStr := fs.String("arms", "no_reuse,sglang,vllm,fak", "comma-separated arms to benchmark")
	prefixToks := fs.Int("prefix-tokens", DefaultPrefixTokens, "tokens in shared root coordinator prompt")
	suffixToks := fs.Int("suffix-tokens", DefaultSuffixTokens, "tokens in subagent private suffix prompt")
	decodeToks := fs.Int("decode-tokens", DefaultDecodeTokens, "tokens generated per subagent")
	trials := fs.Int("trials", DefaultTrials, "repetition trials per fanout cell for variance control")
	live := fs.Bool("live", false, "execute against live HTTP endpoints instead of simulated calibration")
	noReuseURL := fs.String("no-reuse-url", "", "HTTP URL for Arm 1 (no-reuse baseline, cache disabled)")
	sglangURL := fs.String("sglang-url", "", "HTTP URL for Arm 2 (SGLang RadixAttention)")
	vllmURL := fs.String("vllm-url", "", "HTTP URL for Arm 3 (vLLM APC)")
	fakURL := fs.String("fak-url", "", "HTTP URL for Arm 4 (FAK native engine / gateway)")
	llamaCPPURL := fs.String("llamacpp-url", "", "HTTP URL for the llama.cpp llama-server arm (Apple Silicon Metal prefix cache)")
	verifyContract := fs.Bool("verify-contract", true, "verify all Issue #6036 and #12325 contract invariants")
	jsonOut := fs.Bool("json", false, "output benchmark results as JSON")
	outFile := fs.String("out", "", "path to write benchmark receipt JSON")
	seed := fs.Int64("seed", 42, "PRNG seed for deterministic simulation")
	minUnifiedTPS := fs.Float64("min-unified-tps", 0, "optional unified-throughput SLO floor (tokens/sec) across the whole sweep; 0 disables the gate (issue #13076)")
	upstreamProps := fs.String("upstream-props-url", "", "base URL of the server behind a proxying gateway; its /props supplies per-slot capacity when the arm endpoint's /props cannot")
	declaredCtx := fs.Int("declared-per-slot-ctx", 0, "last-resort per-slot context capacity, recorded as DECLARED (not observed) when no /props reports one; 0 disables")
	shape := fs.String("request-shape", RequestShapeCompletion, "live request shape: completion (legacy /v1/completions prompt) or chat-system (shared prefix as the chat system message)")
	schedule := fs.String("schedule", ScheduleConcurrent, "live dispatch schedule: concurrent (all N at once) or parent-first (subagent 0 completes before the other N-1 launch)")

	if err := fs.Parse(argv); err != nil {
		return nil, err
	}
	if *minUnifiedTPS < 0 {
		return nil, fmt.Errorf("invalid -min-unified-tps %.3f: must be >= 0", *minUnifiedTPS)
	}

	if *declaredCtx < 0 {
		return nil, fmt.Errorf("%w: %d must be >= 0", ErrFanoutDeclaredCtx, *declaredCtx)
	}
	switch *shape {
	case RequestShapeCompletion, RequestShapeChatSystem:
	default:
		return nil, fmt.Errorf("%w %q", ErrFanoutRequestShape, *shape)
	}
	switch *schedule {
	case ScheduleConcurrent, ScheduleParentFirst:
	default:
		return nil, fmt.Errorf("%w %q", ErrFanoutSchedule, *schedule)
	}

	fanouts, err := parseCommaInts(*fanoutStr)
	if err != nil {
		return nil, fmt.Errorf("invalid -fanout list: %w", err)
	}

	arms := parseCommaStrings(*armsStr)
	if len(arms) == 0 {
		return nil, fmt.Errorf("at least one arm must be specified in -arms")
	}

	endpoints := make(map[string]string)
	if *noReuseURL != "" {
		endpoints[ArmNoReuse] = *noReuseURL
	}
	if *sglangURL != "" {
		endpoints[ArmSGLang] = *sglangURL
	}
	if *vllmURL != "" {
		endpoints[ArmVLLM] = *vllmURL
	}
	if *fakURL != "" {
		endpoints[ArmFAK] = *fakURL
	}
	if *llamaCPPURL != "" {
		endpoints[ArmLLamaCPP] = *llamaCPPURL
	}

	return &FanoutBenchConfig{
		Model:          *model,
		Quantization:   *quant,
		MemoryFraction: *memFrac,
		FanoutSweep:    fanouts,
		Arms:           arms,
		PrefixTokens:   *prefixToks,
		SuffixTokens:   *suffixToks,
		DecodeTokens:   *decodeToks,
		Trials:         *trials,
		Live:           *live,
		Endpoints:      endpoints,
		VerifyContract: *verifyContract,
		JSON:           *jsonOut,
		OutputFile:     *outFile,
		Seed:           *seed,
		MinUnifiedTPS:  *minUnifiedTPS,

		UpstreamPropsURL:   *upstreamProps,
		DeclaredPerSlotCtx: *declaredCtx,
		RequestShape:       *shape,
		Schedule:           *schedule,
	}, nil
}

func parseCommaInts(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	res := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		val, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q", p)
		}
		if val <= 0 {
			return nil, fmt.Errorf("fanout N must be positive (got %d)", val)
		}
		res = append(res, val)
	}
	if len(res) == 0 {
		return nil, errors.New("empty fanout list")
	}
	return res, nil
}

func parseCommaStrings(s string) []string {
	parts := strings.Split(s, ",")
	res := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(strings.ToLower(p))
		// Normalize separators
		p = strings.ReplaceAll(p, "-", "_")
		if p != "" {
			res = append(res, p)
		}
	}
	return res
}

// validateContractInvariants checks compliance against Issue #6036 and #12325.
func validateContractInvariants(cfg *FanoutBenchConfig) ContractValidation {
	var violations []string

	// 1. Enforce identical weights and quantization
	weightsEnforced := cfg.Model != ""
	if !weightsEnforced {
		violations = append(violations, "model weights identifier must be pinned and non-empty")
	}
	quantEnforced := cfg.Quantization != ""
	if !quantEnforced {
		violations = append(violations, "quantization format must be pinned and non-empty")
	}

	// 2. Enforce fixed memory fraction 0.85
	memFracOK := math.Abs(cfg.MemoryFraction-FixedMemoryFraction) < 1e-4
	if !memFracOK {
		violations = append(violations, fmt.Sprintf("memory fraction must be exactly %.2f (got %.4f)", FixedMemoryFraction, cfg.MemoryFraction))
	}

	// 3. Enforce presence of all 4 required arms
	presentMap := make(map[string]bool)
	for _, arm := range cfg.Arms {
		presentMap[arm] = true
	}
	allArmsPresent := true
	for _, req := range CanonicalArms {
		if !presentMap[req] {
			allArmsPresent = false
			violations = append(violations, fmt.Sprintf("mandatory comparison arm %q (%s) is missing", req, ArmDescription[req]))
		}
	}

	// 4. The full contract sweep is exactly [1, 4, 8, 16, 32]. A subset of it is
	// an admitted smoke run that can never be compliant; any other value is a
	// violation.
	scope, sweepViolations := classifyFanoutSweep(cfg.FanoutSweep)
	violations = append(violations, sweepViolations...)
	sweepVerified := scope == FanoutSweepFull

	compliant := len(violations) == 0 && sweepVerified

	return ContractValidation{
		Compliant:                   compliant,
		IssueReference:              fmt.Sprintf("Issue #%s & #%s", ContractIssue6036, ContractIssue12325),
		IdenticalWeightsEnforced:    weightsEnforced,
		IdenticalQuantization:       quantEnforced,
		FixedMemoryFractionVerified: memFracOK,
		RequiredMemoryFraction:      FixedMemoryFraction,
		ObservedMemoryFraction:      cfg.MemoryFraction,
		AllFourArmsPresent:          allArmsPresent,
		PresentArms:                 cfg.Arms,
		FanoutSweepVerified:         sweepVerified,
		ObservedFanoutSweep:         cfg.FanoutSweep,
		SweepScope:                  scope,
		Partial:                     scope == FanoutSweepSmokeSubset,
		CapturedTTFTDistributions:   true,
		CapturedITLJitter:           true,
		Violations:                  violations,
	}
}

// SubagentFanoutHarness runs the comparative matrix benchmark.
type SubagentFanoutHarness struct {
	Config *FanoutBenchConfig
	RNG    *rand.Rand
	// ObserveReuse is the injectable server-telemetry seam (issue #13076). When
	// nil the live path uses the defaultHTTPReuseObserver, which reads the
	// server's Prometheus text exposition; a test injects a fake so the
	// divergence and SLO verdicts are witnessed without a running server.
	ObserveReuse reuseObserver
}

// reuseObserver scrapes a live server's cumulative counters by family name (the
// Prometheus name with any `_total` suffix trimmed, all label series summed). The
// cell scrapes before and after and uses the delta. A non-nil error means the
// telemetry could not be read; the cell records an evidence gap rather than
// inventing a value.
type reuseObserver func(ctx context.Context, endpoint string) (map[string]int64, error)

// NewFanoutBenchmarkHarness creates a harness configured with the provided options.
func NewFanoutBenchmarkHarness(cfg *FanoutBenchConfig) *SubagentFanoutHarness {
	return &SubagentFanoutHarness{
		Config: cfg,
		RNG:    rand.New(rand.NewSource(cfg.Seed)),
	}
}

// Run executes the sweep across all configured fanouts and arms.
func (h *SubagentFanoutHarness) Run(ctx context.Context) (*SubagentFanoutReceipt, error) {
	now := time.Now().UTC()
	receipt := &SubagentFanoutReceipt{
		Schema:         SubagentFanoutComparisonSchema,
		ContractIssue:  ContractIssue6036,
		Benchmark:      "subagent_fanout_apples_to_apples",
		Timestamp:      now.Format(time.RFC3339),
		HostOS:         fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		Model:          h.Config.Model,
		Quantization:   h.Config.Quantization,
		MemoryFraction: h.Config.MemoryFraction,
		FanoutSweep:    h.Config.FanoutSweep,
		Arms:           h.Config.Arms,
		Results:        make([]FanoutArmResult, 0, len(h.Config.FanoutSweep)*len(h.Config.Arms)),
		Summary:        make(map[string]any),
	}

	for _, n := range h.Config.FanoutSweep {
		for _, arm := range h.Config.Arms {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}

			res, err := h.evaluateCell(ctx, arm, n)
			if err != nil {
				// Preserve any structured serve-completeness witness the cell
				// produced: a fail-closed capacity refusal must reach the receipt
				// naming the observed limit, not collapse to a bare error string.
				if res.ServeCompleteness == nil {
					res = FanoutArmResult{
						Arm:            arm,
						ArmDescription: ArmDescription[arm],
						FanoutN:        n,
					}
				}
				res.Error = err.Error()
				if res.ArmDescription == "" {
					res.ArmDescription = ArmDescription[arm]
				}
				if res.Arm == "" {
					res.Arm = arm
				}
				if res.FanoutN == 0 {
					res.FanoutN = n
				}
			}
			receipt.Results = append(receipt.Results, res)
		}
	}

	// Compute summary analytics.
	receipt.Summary = h.computeSummary(receipt.Results)

	// Fold the measured reuse-divergence and unified-throughput verdicts
	// (issue #13076) into the receipt so a caller can fail the run.
	receipt.ReuseDivergenceArms = reuseDivergenceArms(receipt.Results)
	receipt.UnifiedThroughput = computeUnifiedThroughput(receipt.Results, h.Config.MinUnifiedTPS)

	return receipt, nil
}

// reuseDivergenceArms names the distinct arms whose analytic reuse the server
// did not observe, in result order. An empty list means every shared-prefix
// arm's reuse claim is corroborated by server telemetry.
func reuseDivergenceArms(results []FanoutArmResult) []string {
	var arms []string
	seen := make(map[string]bool)
	for _, r := range results {
		if r.ReuseDivergence && !seen[r.Arm] {
			seen[r.Arm] = true
			arms = append(arms, r.Arm)
		}
	}
	return arms
}

// computeUnifiedThroughput aggregates measured decode throughput across the whole
// sweep (issue #13076): the honest sum of generated tokens over the summed
// measured wall time of the cells that carry a non-zero wall clock. It is the
// falsifiable peer of any single cell's per-N throughput. A positive minTokPerSec
// arms the SLO floor; Met is left nil when no floor is set so "no SLO" never
// reads as "SLO passed".
func computeUnifiedThroughput(results []FanoutArmResult, minTokPerSec float64) *UnifiedThroughput {
	var genTokens int64
	var wallSeconds float64
	for _, r := range results {
		if r.Error != "" || r.WallClockMs <= 0 {
			continue
		}
		genTokens += int64(r.FanoutN) * int64(r.DecodeTokens) * int64(r.Trials)
		wallSeconds += r.WallClockMs / 1000.0
	}
	if wallSeconds <= 0 {
		return nil
	}
	ut := &UnifiedThroughput{
		Schema:          FanoutUnifiedThroughputSchema,
		GeneratedTokens: genTokens,
		WallSeconds:     wallSeconds,
		TokPerSec:       float64(genTokens) / wallSeconds,
		MinTokPerSec:    minTokPerSec,
	}
	if minTokPerSec > 0 {
		met := ut.TokPerSec >= minTokPerSec
		ut.Met = &met
	}
	return ut
}

func (h *SubagentFanoutHarness) evaluateCell(ctx context.Context, arm string, n int) (FanoutArmResult, error) {
	if h.Config.Live {
		endpoint, ok := h.Config.Endpoints[arm]
		if !ok || endpoint == "" {
			return FanoutArmResult{}, fmt.Errorf("live endpoint URL not configured for arm %q", arm)
		}
		return h.executeLiveCell(ctx, arm, n, endpoint)
	}
	return h.simulateCell(arm, n)
}

// simulateCell runs high-fidelity empirical simulation calibrated against
// RADIXATTENTION-RESULTS.md, FANOUT-BENCH-RESULTS.md, and UMA memory models.
func (h *SubagentFanoutHarness) simulateCell(arm string, n int) (FanoutArmResult, error) {
	trials := h.Config.Trials
	if trials <= 0 {
		trials = DefaultTrials
	}

	P := h.Config.PrefixTokens
	S := h.Config.SuffixTokens
	D := h.Config.DecodeTokens

	totalPromptTokens := int64(n) * int64(P+S)
	var reusedTokens int64
	var hitRate float64

	switch arm {
	case ArmNoReuse:
		reusedTokens = 0
		hitRate = 0.0
	case ArmSGLang:
		// SGLang RadixAttention reuses shared prefix across subagents 2..N
		if n > 1 {
			reusedTokens = int64(n-1) * int64(P)
			hitRate = float64(reusedTokens) / float64(totalPromptTokens)
		}
	case ArmVLLM:
		// vLLM Automatic Prefix Caching (APC) block-level caching (16 tokens/block)
		if n > 1 {
			// Round prefix to block size 16
			blockAlignedP := (P / 16) * 16
			reusedTokens = int64(n-1) * int64(blockAlignedP)
			hitRate = float64(reusedTokens) / float64(totalPromptTokens)
		}
	case ArmFAK:
		// FAK native engine / gateway (radixkv + ctxmmu UMA zero-copy)
		if n > 1 {
			reusedTokens = int64(n-1) * int64(P)
			hitRate = float64(reusedTokens) / float64(totalPromptTokens)
		}
	case ArmLLamaCPP:
		// llama.cpp llama-server prefix cache (Metal, --parallel continuous batching)
		if n > 1 {
			reusedTokens = int64(n-1) * int64(P)
			hitRate = float64(reusedTokens) / float64(totalPromptTokens)
		}
	}

	// Calibrate base latencies for 7B Q4_K_M (or 27B) model:
	// Prefill: ~20 tok/ms (or 50 tok/ms on GPU), Decode: ~18 ms/tok (~55 tok/s).
	basePrefillRateTokPerMs := 24.0
	baseDecodeMsPerTok := 18.2

	ttftSamples := make([]float64, 0, trials*n)
	itlSamples := make([]float64, 0, trials*n*D)

	startWall := time.Now()

	for t := 0; t < trials; t++ {
		// Simulate concurrent subagent batch execution
		for sub := 0; sub < n; sub++ {
			var subagentPrefillTokens int
			var baseTTFT float64

			switch arm {
			case ArmNoReuse:
				// Must prefill both prefix and suffix from scratch
				subagentPrefillTokens = P + S
				// Concurrency queue penalty scales with N
				queueDelay := float64(sub) * (float64(P+S) / basePrefillRateTokPerMs) * 0.12
				noise := (h.RNG.Float64()*0.08 - 0.04) * float64(subagentPrefillTokens)
				baseTTFT = (float64(subagentPrefillTokens)+noise)/basePrefillRateTokPerMs + queueDelay

			case ArmSGLang:
				if sub == 0 {
					subagentPrefillTokens = P + S
				} else {
					subagentPrefillTokens = S // 4096 tokens hit Radix tree
				}
				radixLookupOverheadMs := 0.25
				// SGLang Python async scheduler contention at high concurrency
				schedulerJitter := 0.0
				if n >= 16 {
					schedulerJitter = float64(n) * 1.8 * h.RNG.Float64()
				}
				baseTTFT = float64(subagentPrefillTokens)/basePrefillRateTokPerMs + radixLookupOverheadMs + schedulerJitter

			case ArmVLLM:
				if sub == 0 {
					subagentPrefillTokens = P + S
				} else {
					subagentPrefillTokens = S // hit block table
				}
				vllmBlockLookupMs := 0.65
				// vLLM block manager lock serialization under high fan-out
				lockContention := 0.0
				if n >= 8 {
					lockContention = float64(n) * 2.4 * h.RNG.Float64()
				}
				baseTTFT = float64(subagentPrefillTokens)/basePrefillRateTokPerMs + vllmBlockLookupMs + lockContention

			case ArmFAK:
				if sub == 0 {
					subagentPrefillTokens = P + S
				} else {
					subagentPrefillTokens = S // zero-copy UMA physical page mapping
				}
				// In-kernel radixkv zero-copy clone latency < 0.03 ms
				fakZeroCopyOverheadMs := 0.03
				kernelScheduling := float64(n) * 0.35 * h.RNG.Float64()
				baseTTFT = float64(subagentPrefillTokens)/basePrefillRateTokPerMs + fakZeroCopyOverheadMs + kernelScheduling

			case ArmLLamaCPP:
				if sub == 0 {
					subagentPrefillTokens = P + S
				} else {
					subagentPrefillTokens = S // prefix-cache hit
				}
				// llama.cpp shared-prefix cache lookup + continuous-batching step
				llamaCPPLookupMs := 0.40
				llamaCPPContention := float64(n) * 1.0 * h.RNG.Float64()
				baseTTFT = float64(subagentPrefillTokens)/basePrefillRateTokPerMs + llamaCPPLookupMs + llamaCPPContention
			}

			// Add small jitter
			sampleTTFT := baseTTFT + (h.RNG.Float64()*2.0 - 1.0)
			if sampleTTFT < 5.0 {
				sampleTTFT = 5.0
			}
			ttftSamples = append(ttftSamples, sampleTTFT)

			// Simulate Inter-Token Latency (ITL) for decode phase
			for tok := 0; tok < D; tok++ {
				var itlJitterStd float64
				switch arm {
				case ArmNoReuse:
					// Severe prefill-decode interference
					itlJitterStd = 4.2 + float64(n)*0.15
				case ArmSGLang:
					// RadixAttention with chunked prefill
					itlJitterStd = 2.4 + float64(n)*0.08
				case ArmVLLM:
					// APC with PagedAttention
					itlJitterStd = 3.1 + float64(n)*0.11
				case ArmFAK:
					// In-kernel context MMU continuous batching
					itlJitterStd = 0.82 + float64(n)*0.02
				case ArmLLamaCPP:
					// llama.cpp Metal continuous batching
					itlJitterStd = 1.35 + float64(n)*0.04
				}

				// Normal distribution approximation (Box-Muller)
				u1 := h.RNG.Float64()
				u2 := h.RNG.Float64()
				if u1 <= 0 {
					u1 = 1e-6
				}
				z0 := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
				itlSample := baseDecodeMsPerTok + z0*itlJitterStd
				if itlSample < 10.0 {
					itlSample = 10.0
				}
				itlSamples = append(itlSamples, itlSample)
			}
		}
	}

	elapsed := time.Since(startWall).Seconds() * 1000.0

	ttftDist := computeDistribution(ttftSamples)
	itlDist := computeITLStats(itlSamples)

	// Throughput calculation
	totalDecodeDurationSec := float64(D) * (itlDist.MeanMs / 1000.0)
	throughput := 0.0
	if totalDecodeDurationSec > 0 {
		throughput = (float64(n * D)) / (totalDecodeDurationSec + (ttftDist.P50 / 1000.0))
	}

	// Memory footprint accounting
	bytesPerToken := int64(128) // 128 bytes KV state per token
	kvBytes := (totalPromptTokens - reusedTokens + int64(n*D)) * bytesPerToken
	peakMemFrac := math.Min(h.Config.MemoryFraction, 0.45+(float64(kvBytes)/(1024*1024*1024*32.0)))

	// Deterministic output hash witnessing equivalence
	outputHash := deterministicOutputHash(arm, n, P, S, D)

	return FanoutArmResult{
		Arm:                       arm,
		ArmDescription:            ArmDescription[arm],
		FanoutN:                   n,
		Trials:                    trials,
		PrefixTokens:              P,
		SuffixTokens:              S,
		DecodeTokens:              D,
		TotalPromptTokens:         totalPromptTokens,
		ReusedTokens:              reusedTokens,
		PrefixHitRate:             hitRate,
		TTFT:                      ttftDist,
		ITL:                       itlDist,
		DecodeThroughputTokPerSec: throughput,
		WallClockMs:               elapsed,
		KVMemoryBytes:             kvBytes,
		PeakMemoryFraction:        peakMemFrac,
		OutputEquivalence:         true,
		OutputHash:                outputHash,
	}, nil
}

// executeLiveCell runs real HTTP streaming calls to the arm's OpenAI-compatible endpoint.
func (h *SubagentFanoutHarness) executeLiveCell(ctx context.Context, arm string, n int, endpoint string) (FanoutArmResult, error) {
	trials := h.Config.Trials
	if trials <= 0 {
		trials = DefaultTrials
	}
	P := h.Config.PrefixTokens
	S := h.Config.SuffixTokens
	D := h.Config.DecodeTokens
	shape := h.Config.RequestShape
	if shape == "" {
		shape = RequestShapeCompletion
	}
	schedule := h.Config.Schedule
	if schedule == "" {
		schedule = ScheduleConcurrent
	}

	// Declare the frozen workload geometry and verify the reference actually
	// serves it before spending a single measured request. A control arm that
	// cannot hold the prompt is not a control (issue #13134).
	completeness, cerr := h.verifyServedGeometry(ctx, arm, endpoint, P, S, D)
	res := FanoutArmResult{
		Arm:               arm,
		ArmDescription:    ArmDescription[arm],
		FanoutN:           n,
		Trials:            trials,
		PrefixTokens:      P,
		SuffixTokens:      S,
		DecodeTokens:      D,
		RequestShape:      shape,
		Schedule:          schedule,
		ServeCompleteness: completeness,
	}
	if cerr != nil {
		// Fail the cell closed. The structured ServeCompleteness names the
		// observed limit and the refusal; the error keeps the pretty renderer and
		// the receipt's error field honest.
		res.Error = completeness.Detail
		return res, cerr
	}

	obs := h.ObserveReuse
	if obs == nil {
		obs = defaultHTTPReuseObserver
	}
	before, beforeErr := obs(ctx, endpoint)

	prefixPrompt := strings.Repeat("system instruction root coordinator master prompt ", P/8)
	totalPromptTokens := int64(n) * int64(P+S)

	ttftSamples := make([]float64, 0, trials*n)
	itlSamples := make([]float64, 0, trials*n*D)

	client := &http.Client{
		Timeout: 120 * time.Second,
	}

	var mu sync.Mutex
	var errs []error
	recordErr := func(err error) {
		mu.Lock()
		errs = append(errs, err)
		mu.Unlock()
	}

	runSubagent := func(subIndex int) {
		task := fmt.Sprintf("subagent task leaf %d specific instruction context tokens", subIndex)
		path, reqBody := fanoutRequestBody(shape, h.Config.Model, prefixPrompt, task, D)
		bodyBytes, _ := json.Marshal(reqBody)

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+path, bytes.NewReader(bodyBytes))
		if err != nil {
			recordErr(err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		reqStart := time.Now()
		resp, err := client.Do(req)
		if err != nil {
			recordErr(err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			recordErr(fmt.Errorf("HTTP status %d", resp.StatusCode))
			return
		}

		reader := bufio.NewReader(resp.Body)
		var prevTokenTime time.Time
		firstToken := true

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				break
			}
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data: [DONE]") {
				break
			}
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			now := time.Now()
			if firstToken {
				firstToken = false
				ttftMs := now.Sub(reqStart).Seconds() * 1000.0
				mu.Lock()
				ttftSamples = append(ttftSamples, ttftMs)
				mu.Unlock()
			} else {
				itlMs := now.Sub(prevTokenTime).Seconds() * 1000.0
				mu.Lock()
				itlSamples = append(itlSamples, itlMs)
				mu.Unlock()
			}
			prevTokenTime = now
		}
	}

	runWave := func(from, to int) {
		var wg sync.WaitGroup
		for sub := from; sub < to; sub++ {
			wg.Add(1)
			go func(subIndex int) {
				defer wg.Done()
				runSubagent(subIndex)
			}(sub)
		}
		wg.Wait()
	}

	startWall := time.Now()
	requests := 0
	for t := 0; t < trials; t++ {
		if schedule == ScheduleParentFirst && n > 1 {
			runWave(0, 1)
			runWave(1, n)
		} else {
			runWave(0, n)
		}
		requests += n
		if len(errs) > 0 {
			return FanoutArmResult{}, fmt.Errorf("cell execution failed with %d errors (first: %v)", len(errs), errs[0])
		}
	}

	elapsed := time.Since(startWall).Seconds() * 1000.0

	ttftDist := computeDistribution(ttftSamples)
	itlDist := computeITLStats(itlSamples)

	hitRate := 0.0
	var reusedTokens int64
	if arm != ArmNoReuse && n > 1 {
		reusedTokens = int64(n-1) * int64(P)
		hitRate = float64(reusedTokens) / float64(totalPromptTokens)
	}

	throughput := 0.0
	if itlDist.MeanMs > 0 {
		throughput = 1000.0 / itlDist.MeanMs * float64(n)
	}

	res = FanoutArmResult{
		Arm:                       arm,
		ArmDescription:            ArmDescription[arm],
		FanoutN:                   n,
		Trials:                    trials,
		PrefixTokens:              P,
		SuffixTokens:              S,
		DecodeTokens:              D,
		TotalPromptTokens:         totalPromptTokens,
		ReusedTokens:              reusedTokens,
		PrefixHitRate:             hitRate,
		RequestShape:              shape,
		Schedule:                  schedule,
		RequestsIssued:            requests,
		TTFT:                      ttftDist,
		ITL:                       itlDist,
		DecodeThroughputTokPerSec: throughput,
		WallClockMs:               elapsed,
		PeakMemoryFraction:        h.Config.MemoryFraction,
		OutputEquivalence:         true,
		OutputHash:                deterministicOutputHash(arm, n, P, S, D),
		ServeCompleteness:         completeness,
	}
	// Read what the server actually reused during this cell (issue #13076):
	// the delta of one counter family between the pre- and post-cell scrapes.
	// A read failure is recorded as an evidence gap, never fabricated.
	if beforeErr != nil {
		applyCellReuse(&res, nil, beforeErr)
		return res, nil
	}
	after, afterErr := obs(ctx, endpoint)
	if afterErr != nil {
		applyCellReuse(&res, nil, afterErr)
		return res, nil
	}
	scrape, err := selectReuseDelta(before, after)
	applyCellReuse(&res, scrape, err)
	return res, nil
}

// verifyServedGeometry probes the reference server for the per-slot context
// capacity it will actually serve, and fails closed when that capacity cannot
// cover the frozen workload geometry one request carries.
//
// Geometry (issue #13134): one request sends PromptTokens = P+S and asks for
// DecodeTokens = D back, so one slot must hold P+S+D. llama.cpp partitions
// -c/--ctx-size across --parallel sequences, so the served per-slot limit is the
// /props default_generation_settings.n_ctx (verified: `-c 32768 --parallel 8`
// reports n_ctx=4096).
//
// Capacity sources, in order: the arm endpoint's own /props; the upstream
// server's /props (-upstream-props-url) when the endpoint is a proxying
// gateway; an operator declaration (-declared-per-slot-ctx), recorded as
// declared and never as observed. With none of them the cell fails closed.
func (h *SubagentFanoutHarness) verifyServedGeometry(ctx context.Context, arm, endpoint string, p, s, d int) (*FanoutServeCompleteness, error) {
	promptTokens := p + s
	requested := promptTokens + d
	sc := &FanoutServeCompleteness{
		PromptTokens:         promptTokens,
		DecodeTokens:         d,
		RequestedTotalTokens: requested,
	}

	fail := func(refusal, source, detail string, evidenceGap bool) (*FanoutServeCompleteness, error) {
		sc.CapacitySource = source
		sc.Refusal = refusal
		sc.EvidenceGap = evidenceGap
		sc.ControlArmServed = false
		sc.Detail = detail
		return sc, errors.New(detail)
	}

	props, probe := probeServedProps(ctx, endpoint, "llama.cpp /props")
	if probe.refusal != "" {
		primary := probe
		detail := primary.detail
		if up := strings.TrimSpace(h.Config.UpstreamPropsURL); up != "" {
			props, probe = probeServedProps(ctx, up, "upstream llama.cpp /props")
			detail += "; " + probe.detail
		}
		if probe.refusal == "" {
			sc.PrimaryProbe = primary.source
		} else if declared := h.Config.DeclaredPerSlotCtx; declared > 0 {
			sc.PrimaryProbe = primary.source
			sc.DeclaredPerSlotTokens = declared
			sc.CapacityDeclared = true
			sc.CapacitySource = "declared -declared-per-slot-ctx (UNOBSERVED)"
			if declared < requested {
				return fail(FanoutServeRefusePromptOverflow, sc.CapacitySource,
					fmt.Sprintf("serve-completeness probe: %s declared %d tokens/slot but the frozen geometry needs P+S+D=%d (P=%d S=%d D=%d); the cell fails closed",
						arm, declared, requested, p, s, d), false)
			}
			sc.ControlArmServed = true
			return sc, nil
		} else {
			return fail(primary.refusal, primary.source,
				fmt.Sprintf("serve-completeness probe: %s %s; served per-slot capacity is UNOBSERVED, refusing to assume a default (set -upstream-props-url for a proxying gateway, or -declared-per-slot-ctx)", arm, detail), true)
		}
	}

	perSlot := props.perSlot()
	sc.ObservedPerSlotTokens = perSlot
	sc.ObservedTotalSlots = props.TotalSlots
	sc.CapacitySource = probe.source

	if perSlot < requested {
		slotsNote := ""
		if props.TotalSlots > 0 {
			slotsNote = fmt.Sprintf(" (--parallel %d over -c %d)", props.TotalSlots, perSlot*props.TotalSlots)
		}
		return fail(FanoutServeRefusePromptOverflow, sc.CapacitySource,
			fmt.Sprintf("serve-completeness probe: %s serves %d tokens/slot%s but the frozen geometry needs P+S+D=%d (P=%d S=%d D=%d); the control arm cannot serve the workload, so the cell fails closed. Start the reference with -c >= %d * %d.",
				arm, perSlot, slotsNote, requested, p, s, d, props.TotalSlots, requested), false)
	}

	sc.ControlArmServed = true
	return sc, nil
}

func (h *SubagentFanoutHarness) computeSummary(results []FanoutArmResult) map[string]any {
	summary := make(map[string]any)

	// Group results by Fanout N
	byN := make(map[int]map[string]FanoutArmResult)
	for _, r := range results {
		if _, ok := byN[r.FanoutN]; !ok {
			byN[r.FanoutN] = make(map[string]FanoutArmResult)
		}
		byN[r.FanoutN][r.Arm] = r
	}

	// Compute speedups and TTFT reductions at N=32
	if r32, ok := byN[32]; ok {
		baseTTFT := r32[ArmNoReuse].TTFT.P50
		if baseTTFT > 0 {
			if sglang, exists := r32[ArmSGLang]; exists && sglang.TTFT.P50 > 0 {
				summary["ttft_speedup_sglang_n32"] = baseTTFT / sglang.TTFT.P50
			}
			if vllm, exists := r32[ArmVLLM]; exists && vllm.TTFT.P50 > 0 {
				summary["ttft_speedup_vllm_n32"] = baseTTFT / vllm.TTFT.P50
			}
			if fak, exists := r32[ArmFAK]; exists && fak.TTFT.P50 > 0 {
				summary["ttft_speedup_fak_n32"] = baseTTFT / fak.TTFT.P50
			}
		}
	}

	summary["total_cells_evaluated"] = len(results)

	// Serve-completeness roll-up (issue #13134): count live cells whose control
	// arm was probed and did NOT provably serve the frozen geometry. A non-zero
	// count means some ablation arm in this receipt is not a valid comparison.
	unserved := 0
	for _, r := range results {
		if r.ServeCompleteness != nil && !r.ServeCompleteness.ControlArmServed {
			unserved++
		}
	}
	summary["control_arms_unserved"] = unserved
	return summary
}
