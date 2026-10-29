// Package nativeperfslo turns matched fak-native benchmark observations into
// stable time-series state. It refuses cross-envelope comparisons and preserves
// unavailable evidence instead of manufacturing zeroes.
package nativeperfslo

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// State represents the tri-state debounced condition of a native performance SLO.
// Unobserved or invalid metrics force StateMissing immediately rather than
// coercing zeros, so consumers never read fabricated health.
type State string

const (
	// StateGood indicates all monitored objectives satisfy their relative and absolute thresholds.
	// Reached only after consecutive healthy evaluations, so one sample cannot flip state.
	StateGood State = "good"

	// StateRegression indicates one or more objectives have persistently violated thresholds.
	// Triggered only after consecutive violating evaluations, debouncing transient spikes.
	StateRegression State = "regression"

	// StateMissing indicates insufficient, malformed, or missing evidence.
	// Fail-closed fallback that immediately resets streaks to zero.
	StateMissing State = "missing_evidence"
)

// Objective identifies an individual measured performance dimension in benchmark evaluation.
// Names are lowercase alphanumeric tokens matched across baseline and candidate.
type Objective string

const (
	// TTFT measures time to first token in seconds (lower is better).
	TTFT Objective = "ttft"

	// TPOT measures time per output token in seconds (lower is better).
	TPOT Objective = "tpot"

	// Throughput measures generated tokens per second (higher is better).
	Throughput Objective = "throughput"

	// QueueDelay measures scheduler queue dwell time in seconds (lower is better).
	QueueDelay Objective = "queue_delay"

	// CacheEfficiency measures prompt and prefix cache hit efficiency ratio in [0, 1] (higher is better).
	CacheEfficiency Objective = "cache_efficiency"

	// TransferShare measures memory bus transfer time fraction in [0, 1] (lower is better).
	TransferShare Objective = "transfer_share"

	// KernelShare measures compute execution time fraction in [0, 1] (lower is better).
	KernelShare Objective = "kernel_share"

	// MemoryPressure measures accelerator memory utilization ratio in [0, 1].
	MemoryPressure Objective = "memory_pressure"

	// EvidenceFresh measures elapsed seconds since candidate benchmark execution.
	EvidenceFresh Objective = "evidence_freshness"

	// ReceiptCoverage measures fraction of expected execution receipts collected.
	ReceiptCoverage Objective = "receipt_coverage"
)

var objectives = [...]Objective{
	TTFT, TPOT, Throughput, QueueDelay, CacheEfficiency, TransferShare,
	KernelShare, MemoryPressure, EvidenceFresh, ReceiptCoverage,
}

// LifecycleEvent represents a release or deployment milestone attached to performance series.
type LifecycleEvent string

const (
	// EventPromotion marks promotion of an evaluated revision to active serving.
	EventPromotion LifecycleEvent = "promotion"

	// EventRevert marks reversion of an evaluated revision following a regression.
	EventRevert LifecycleEvent = "revert"

	// EventRelease marks an official software release tag.
	EventRelease LifecycleEvent = "release"
)

// LifecycleAnnotation records deployment or release metadata associated with an observation.
// Both fields must be non-empty for lifecycle info metrics to render; absent tags are
// omitted rather than fabricated.
type LifecycleAnnotation struct {
	Event   LifecycleEvent `json:"event"`
	Release string         `json:"release"`
}

// SupervisionStatus describes the operational state of the supervising benchmark harness.
// Only healthy statuses permit series emission; failure fails closed as unavailable.
type SupervisionStatus string

const (
	// SupervisionActive indicates the benchmark harness operated normally.
	SupervisionActive SupervisionStatus = "active"

	// SupervisionDegraded indicates the harness experienced non-fatal operational anomalies.
	SupervisionDegraded SupervisionStatus = "degraded"

	// SupervisionFailed indicates harness breakdown, invalidating benchmark output.
	SupervisionFailed SupervisionStatus = "failed"
)

// LifecycleStatus indicates the lifecycle classification of an evaluated candidate.
// Invalid inputs fail closed to LifecycleMissing or LifecycleMismatched.
type LifecycleStatus string

const (
	// LifecycleGood indicates performance meets or exceeds baseline requirements.
	LifecycleGood LifecycleStatus = "good"

	// LifecycleRegressed indicates verified performance regression.
	LifecycleRegressed LifecycleStatus = "regression"

	// LifecycleStale indicates observation timestamp exceeds maximum allowed age.
	LifecycleStale LifecycleStatus = "stale"

	// LifecycleMissing indicates absent, incomplete, or unvalidated benchmark metrics.
	LifecycleMissing LifecycleStatus = "missing_evidence"

	// LifecycleMismatched indicates baseline and candidate envelopes do not match.
	LifecycleMismatched LifecycleStatus = "mismatched_envelope"
)

// SeriesStatus indicates whether time-series output could be generated.
// Unavailable runs publish no series rather than misleading or zero-filled ones.
type SeriesStatus string

const (
	// SeriesProduced indicates valid metrics were evaluated and rendered.
	SeriesProduced SeriesStatus = "produced"

	// SeriesUnavailable indicates evaluation was refused due to invalid or stale inputs.
	SeriesUnavailable SeriesStatus = "unavailable"
)

// UnavailableReason details why time-series generation was refused.
// ReasonNone only accompanies SeriesProduced; every refusal names its cause.
type UnavailableReason string

const (
	// ReasonNone indicates no refusal occurred.
	ReasonNone UnavailableReason = "none"

	// ReasonIncomplete indicates missing data, invalid envelope, or harness failure.
	ReasonIncomplete UnavailableReason = "incomplete"

	// ReasonStale indicates observation timestamp is older than maxAge threshold.
	ReasonStale UnavailableReason = "stale"

	// ReasonMismatched indicates baseline and candidate comparison envelopes differ.
	ReasonMismatched UnavailableReason = "mismatched_envelope"

	// ReasonPrivate indicates data cannot be published due to privacy constraints.
	ReasonPrivate UnavailableReason = "private"

	// ReasonUntrustedScheme indicates an unrecognized or unverified scheme.
	ReasonUntrustedScheme UnavailableReason = "untrusted_scheme"
)

// LiveRunInput bundles input parameters required to produce live SLO time-series.
// Baseline and Candidate envelopes must validate and match identically; ObservedAt
// must be non-zero and not in the future.
type LiveRunInput struct {
	ObservedAt        time.Time
	Baseline          Observation
	Candidate         Observation
	SupervisionStatus SupervisionStatus
	Lifecycle         LifecycleAnnotation
}

// LiveSeriesResult contains the outcome of live time-series generation.
// Unavailable runs emit zero Prometheus metrics to avoid polluting dashboards.
type LiveSeriesResult struct {
	Status            SeriesStatus      `json:"status"`
	UnavailableReason UnavailableReason `json:"unavailable_reason"`
	LifecycleStatus   LifecycleStatus   `json:"lifecycle_status"`
	Result            Result            `json:"result,omitempty"`
	Prometheus        string            `json:"prometheus,omitempty"`
}

// ProduceLiveSeries evaluates a candidate observation against a matched baseline,
// verifies envelope validity and freshness, and emits bounded SLO and lifecycle Prometheus series.
// Validation failures return SeriesUnavailable with a specific reason; it never
// manufactures zeroes.
func ProduceLiveSeries(now time.Time, maxAge time.Duration, evaluator *Evaluator, input LiveRunInput) LiveSeriesResult {
	unavailable := func(reason UnavailableReason, lstatus LifecycleStatus) LiveSeriesResult {
		return LiveSeriesResult{
			Status:            SeriesUnavailable,
			UnavailableReason: reason,
			LifecycleStatus:   lstatus,
		}
	}

	if now.IsZero() || maxAge <= 0 {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}
	if input.ObservedAt.IsZero() || now.Before(input.ObservedAt) {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}
	if now.Sub(input.ObservedAt) > maxAge {
		return unavailable(ReasonStale, LifecycleStale)
	}

	if err := input.Baseline.Envelope.Validate(); err != nil {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}
	if err := input.Candidate.Envelope.Validate(); err != nil {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}

	if input.Baseline.Envelope != input.Candidate.Envelope {
		return unavailable(ReasonMismatched, LifecycleMismatched)
	}

	if input.SupervisionStatus == SupervisionFailed {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}

	if evaluator == nil {
		evaluator = New(DefaultThresholds())
	}

	res, err := evaluator.Observe(now, input.Baseline, input.Candidate)
	if err != nil {
		return unavailable(ReasonIncomplete, LifecycleMissing)
	}

	var lstatus LifecycleStatus
	switch res.State {
	case StateGood:
		lstatus = LifecycleGood
	case StateRegression:
		lstatus = LifecycleRegressed
	case StateMissing:
		lstatus = LifecycleMissing
	default:
		lstatus = LifecycleMissing
	}

	if res.State == StateMissing && len(res.Violations) == 0 {
		return LiveSeriesResult{
			Status:            SeriesUnavailable,
			UnavailableReason: ReasonIncomplete,
			LifecycleStatus:   LifecycleMissing,
			Result:            res,
		}
	}

	var b strings.Builder
	b.WriteString(RenderPrometheus(res))

	if input.Lifecycle.Event != "" && input.Lifecycle.Release != "" {
		b.WriteString("# HELP fak_native_lifecycle_event_info Native performance lifecycle event annotation.\n")
		b.WriteString("# TYPE fak_native_lifecycle_event_info gauge\n")
		fmt.Fprintf(&b, "fak_native_lifecycle_event_info{engine=\"fak-native\",module_rev=%q,benchmark_envelope=%q,model=%q,backend=%q,event=%q,release=%q} 1\n",
			res.Envelope.ModuleRev, res.Envelope.Benchmark, res.Envelope.Model, res.Envelope.Backend, input.Lifecycle.Event, input.Lifecycle.Release)
	}

	return LiveSeriesResult{
		Status:            SeriesProduced,
		UnavailableReason: ReasonNone,
		LifecycleStatus:   lstatus,
		Result:            res,
		Prometheus:        b.String(),
	}
}

// Envelope defines the complete benchmark comparison identity. ModuleRev is
// intentionally a module@rev value, not a bare commit SHA. Comparisons are valid
// only between identical envelopes.
type Envelope struct {
	ModuleRev string
	Benchmark string
	Model     string
	Backend   string
}

// Validate checks whether the Envelope contains structurally sound and supported identity fields.
// It fails closed on any empty, malformed, or unsupported configuration.
func (e Envelope) Validate() error {
	if !strings.Contains(e.ModuleRev, "@r") {
		return errors.New("module_rev must be module@rev")
	}
	if strings.TrimSpace(e.Benchmark) == "" {
		return errors.New("benchmark envelope is required")
	}
	if !strings.HasPrefix(e.Model, "Qwen3.8") {
		return errors.New("native performance evidence must use Qwen3.8")
	}
	switch e.Backend {
	case "metal", "cuda", "cpu":
	default:
		return fmt.Errorf("unsupported fak-native backend %q", e.Backend)
	}
	return nil
}

// Value represents a single measured objective value with explicit availability tracking.
// An unavailable value is distinct from a measured zero and is never rendered.
type Value struct {
	Available bool
	Value     float64
}

// Observation records one quality-constrained benchmark receipt projection.
// Receipts must not exceed Expected when receipts are tracked; At is the run time.
type Observation struct {
	At       time.Time
	Envelope Envelope
	Values   map[Objective]Value
	Receipts uint64
	Expected uint64
}

// Thresholds configure relative candidate/baseline limits and absolute bounds.
// Evaluator uses them to classify objective violations and manage debouncing streaks.
type Thresholds struct {
	MaxRegression      float64
	MinRetention       float64
	MaxMemoryPressure  float64
	MaxEvidenceAge     time.Duration
	MinReceiptCoverage float64
	Consecutive        int
}

// DefaultThresholds returns recommended baseline SLO thresholds for native inference.
// The defaults are deliberately strict: no permissive unconfigured evaluations.
func DefaultThresholds() Thresholds {
	return Thresholds{
		MaxRegression:      1.10,
		MinRetention:       0.90,
		MaxMemoryPressure:  0.90,
		MaxEvidenceAge:     5 * time.Minute,
		MinReceiptCoverage: 0.95,
		Consecutive:        2,
	}
}

// Result captures the debounced evaluation state and detailed objective ratios.
// GoodStreak and BadStreak are mutually exclusive counters; Ratios and Violations are
// populated only when both sides' values are available and finite.
type Result struct {
	At         time.Time
	Envelope   Envelope
	State      State
	Values     map[Objective]Value
	Ratios     map[Objective]Value
	Violations map[Objective]bool
	GoodStreak int
	BadStreak  int
}

// Evaluator debounces regression and recovery transitions across benchmark observations.
// Missing evidence immediately resets both streaks to zero, and Consecutive complete
// samples are required to transition state, preventing single-sample flapping.
type Evaluator struct {
	thresholds Thresholds
	state      State
	goodStreak int
	badStreak  int
}

// New constructs an Evaluator configured with sanitized thresholds and initial StateMissing.
// Out-of-bound threshold values are clamped to safe defaults.
func New(thresholds Thresholds) *Evaluator {
	if thresholds.MaxRegression <= 1 {
		thresholds.MaxRegression = 1.10
	}
	if thresholds.MinRetention <= 0 || thresholds.MinRetention >= 1 {
		thresholds.MinRetention = 0.90
	}
	if thresholds.MaxMemoryPressure <= 0 || thresholds.MaxMemoryPressure > 1 {
		thresholds.MaxMemoryPressure = 0.90
	}
	if thresholds.MaxEvidenceAge <= 0 {
		thresholds.MaxEvidenceAge = 5 * time.Minute
	}
	if thresholds.MinReceiptCoverage <= 0 || thresholds.MinReceiptCoverage > 1 {
		thresholds.MinReceiptCoverage = 0.95
	}
	if thresholds.Consecutive < 1 {
		thresholds.Consecutive = 2
	}
	return &Evaluator{thresholds: thresholds, state: StateMissing}
}

// Observe evaluates candidate performance against baseline and updates internal debouncing state.
// Both envelopes must validate and be identical; baseline values must be positive and
// finite. Any unavailable or non-finite required objective fails closed to StateMissing.
func (e *Evaluator) Observe(now time.Time, baseline, candidate Observation) (Result, error) {
	if err := baseline.Envelope.Validate(); err != nil {
		return Result{}, fmt.Errorf("baseline: %w", err)
	}
	if err := candidate.Envelope.Validate(); err != nil {
		return Result{}, fmt.Errorf("candidate: %w", err)
	}
	if baseline.Envelope != candidate.Envelope {
		return Result{}, errors.New("unmatched benchmark envelopes cannot be compared")
	}
	if now.IsZero() {
		now = time.Now()
	}
	result := Result{
		At:         now,
		Envelope:   candidate.Envelope,
		State:      e.state,
		Values:     cloneValues(candidate.Values),
		Ratios:     make(map[Objective]Value, len(objectives)),
		Violations: make(map[Objective]bool, len(objectives)),
	}

	coverage := Value{}
	if candidate.Expected > 0 {
		coverage = Value{Available: true, Value: float64(candidate.Receipts) / float64(candidate.Expected)}
	}
	result.Values[ReceiptCoverage] = coverage
	age := Value{}
	if !candidate.At.IsZero() && !now.Before(candidate.At) {
		age = Value{Available: true, Value: now.Sub(candidate.At).Seconds()}
	}
	result.Values[EvidenceFresh] = age

	missing := false
	for _, objective := range objectives {
		if !result.Values[objective].Available || !finite(result.Values[objective].Value) {
			missing = true
		}
	}
	if missing {
		e.state, e.goodStreak, e.badStreak = StateMissing, 0, 0
		result.State = e.state
		return result, nil
	}

	for _, objective := range objectives {
		candidateValue := result.Values[objective]
		switch objective {
		case EvidenceFresh:
			result.Violations[objective] = time.Duration(candidateValue.Value*float64(time.Second)) > e.thresholds.MaxEvidenceAge
		case ReceiptCoverage:
			result.Violations[objective] = candidateValue.Value < e.thresholds.MinReceiptCoverage
		case MemoryPressure:
			result.Violations[objective] = candidateValue.Value > e.thresholds.MaxMemoryPressure
		default:
			base := baseline.Values[objective]
			if !base.Available || !finite(base.Value) || base.Value <= 0 {
				return Result{}, fmt.Errorf("baseline %s is unavailable or non-positive", objective)
			}
			ratio := candidateValue.Value / base.Value
			result.Ratios[objective] = Value{Available: true, Value: ratio}
			if objective == Throughput || objective == CacheEfficiency {
				result.Violations[objective] = ratio < e.thresholds.MinRetention
			} else {
				result.Violations[objective] = ratio > e.thresholds.MaxRegression
			}
		}
	}

	bad := false
	for _, violation := range result.Violations {
		bad = bad || violation
	}
	if bad {
		e.badStreak++
		e.goodStreak = 0
		if e.badStreak >= e.thresholds.Consecutive {
			e.state = StateRegression
		}
	} else {
		e.goodStreak++
		e.badStreak = 0
		if e.goodStreak >= e.thresholds.Consecutive {
			e.state = StateGood
		}
	}
	result.State, result.GoodStreak, result.BadStreak = e.state, e.goodStreak, e.badStreak
	return result, nil
}

func cloneValues(in map[Objective]Value) map[Objective]Value {
	out := make(map[Objective]Value, len(in)+2)
	for k, v := range in {
		out[k] = v
	}
	return out
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 }

// RenderPrometheus formats evaluation results into standard Prometheus text exposition format.
// Unavailable objectives are omitted entirely rather than coerced to zero.
func RenderPrometheus(result Result) string {
	labels := fmt.Sprintf("engine=\"fak-native\",module_rev=%q,benchmark_envelope=%q,model=%q,backend=%q",
		result.Envelope.ModuleRev, result.Envelope.Benchmark, result.Envelope.Model, result.Envelope.Backend)
	var b strings.Builder
	b.WriteString("# HELP fak_native_slo_state Current debounced native performance SLO state.\n")
	b.WriteString("# TYPE fak_native_slo_state gauge\n")
	for _, state := range []State{StateGood, StateRegression, StateMissing} {
		value := 0
		if result.State == state {
			value = 1
		}
		fmt.Fprintf(&b, "fak_native_slo_state{%s,state=%q} %d\n", labels, state, value)
	}
	b.WriteString("# HELP fak_native_slo_value Latest available native performance objective value.\n")
	b.WriteString("# TYPE fak_native_slo_value gauge\n")
	b.WriteString("# HELP fak_native_slo_ratio Candidate divided by matched-envelope baseline.\n")
	b.WriteString("# TYPE fak_native_slo_ratio gauge\n")
	b.WriteString("# HELP fak_native_slo_violation Whether an available objective currently violates its SLO.\n")
	b.WriteString("# TYPE fak_native_slo_violation gauge\n")
	ordered := append([]Objective(nil), objectives[:]...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	for _, objective := range ordered {
		if value := result.Values[objective]; value.Available && finite(value.Value) {
			fmt.Fprintf(&b, "fak_native_slo_value{%s,objective=%q} %.9g\n", labels, objective, value.Value)
			violation := 0
			if result.Violations[objective] {
				violation = 1
			}
			fmt.Fprintf(&b, "fak_native_slo_violation{%s,objective=%q} %d\n", labels, objective, violation)
		}
		if ratio := result.Ratios[objective]; ratio.Available && finite(ratio.Value) {
			fmt.Fprintf(&b, "fak_native_slo_ratio{%s,objective=%q} %.9g\n", labels, objective, ratio.Value)
		}
	}
	return b.String()
}
