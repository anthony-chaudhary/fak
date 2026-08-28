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

type State string

const (
	StateGood       State = "good"
	StateRegression State = "regression"
	StateMissing    State = "missing_evidence"
)

type Objective string

const (
	TTFT            Objective = "ttft"
	TPOT            Objective = "tpot"
	Throughput      Objective = "throughput"
	QueueDelay      Objective = "queue_delay"
	CacheEfficiency Objective = "cache_efficiency"
	TransferShare   Objective = "transfer_share"
	KernelShare     Objective = "kernel_share"
	MemoryPressure  Objective = "memory_pressure"
	EvidenceFresh   Objective = "evidence_freshness"
	ReceiptCoverage Objective = "receipt_coverage"
)

var objectives = [...]Objective{
	TTFT, TPOT, Throughput, QueueDelay, CacheEfficiency, TransferShare,
	KernelShare, MemoryPressure, EvidenceFresh, ReceiptCoverage,
}

// Envelope is the complete comparison identity. ModuleRev is intentionally a
// module@rev value, not a bare commit SHA.
type Envelope struct {
	ModuleRev string
	Benchmark string
	Model     string
	Backend   string
}

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

// Value is one measured objective. Available=false is distinct from a measured
// zero and is rendered as no series.
type Value struct {
	Available bool
	Value     float64
}

// Observation is one quality-constrained benchmark receipt projection.
type Observation struct {
	At       time.Time
	Envelope Envelope
	Values   map[Objective]Value
	Receipts uint64
	Expected uint64
}

// Thresholds define relative candidate/baseline limits. MaxRegression is used
// for lower-is-better objectives; MinRetention is used for higher-is-better.
type Thresholds struct {
	MaxRegression      float64
	MinRetention       float64
	MaxMemoryPressure  float64
	MaxEvidenceAge     time.Duration
	MinReceiptCoverage float64
	Consecutive        int
}

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

// Evaluator debounces regression and recovery transitions. Missing evidence is
// immediate and resets both streaks; two fresh complete samples are required to
// recover, so one scrape cannot flap the state.
type Evaluator struct {
	thresholds Thresholds
	state      State
	goodStreak int
	badStreak  int
}

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

// RenderPrometheus renders only available measurements. No absent value is
// zero-coerced. The stable labels make dashboard comparisons envelope-matched.
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
