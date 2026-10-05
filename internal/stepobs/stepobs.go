// Package stepobs renders the two sub-seams of the inference cycle that record real
// per-call data but had no Prometheus surface at all: the SUB-KERNEL seam (every GEMM
// the compute backends time) and the SUB-PLANNER step seam (every leg of a serving
// step). Before this package both were write-only — computetrace.Event and
// enginestep/microtrace spans existed, were populated with real timings, and stopped
// there. An operator could not see kernel or planner-step latency on /metrics.
//
// It is instrumentation only. No kernel math, scheduling policy, quantization, or ABI
// is touched. The shape deliberately imitates internal/enginestep so the two live in
// one family group:
//
//   - fixed-bucket histograms over the shared seconds bucket ladder,
//   - a closed label vocabulary per family,
//   - always emit every family so a dashboard distinguishes "idle" from "missing",
//   - a process-wide Default recorder the gateway renders.
//
// Presence is reported HONESTLY. Both registries carry an observed gauge that is 1
// only once a real event has actually flowed through the recorder. A 0 with the
// family present means the producer is ABSENT — never "zero kernels ran", which is the
// flat-zero lie this repo's metrics doctrine forbids (internal/gateway/metrics_render.go
// uses the same _known / _enabled bit for exactly this reason). The families are
// emitted unconditionally so the board shape is stable while the bit tells the truth.
//
// Label cardinality is bounded. Kernel and backend are free-form strings minted per
// GEMM by the compute layer, so the (kernel, backend, timer_domain) key set is capped
// and overflow folds into one explicit bucket counted by its own counter, following
// internal/engine/cacheevents.go (fak_engine_cache_keys_capped). An unbounded label
// from a per-GEMM string is a production incident, not a detail.
//
// Memory is fixed at construction: every histogram slot and every bounded key slot is
// allocated up front, so an Observe is one short mutex, one map lookup, and O(buckets)
// integer work — negligible against a millisecond-scale decode forward, and it never
// allocates once the key set is full.
package stepobs

import (
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// StepKind classifies one planner-step leg. The vocabulary is CLOSED and identical to
// metrics.MicroSpanKind — the span vocabulary already in the tree — so the planner-step
// family labels read the same whether the leg came from a microagent span or from the
// native serving loop. TestStepKindVocabularyTracksMicroSpanKind in internal/gateway is
// the pin that keeps the two mirrors from drifting, in both directions; gateway holds the
// type-to-type bridge because metrics must not import this package (metrics sits below
// agent, and agent reaches this package).
//
// ONE PRODUCER PER KIND is the invariant, so a leg is never paid for twice and two
// conventions never compete over what "one planner step" means:
//
//   - step, seat, admission are produced by internal/enginestep. Attach installs the phase
//     projection below and internal/gateway installs the decode-step observer, so a real
//     InKernelPlanner.Complete reaches this registry with no per-request plumbing.
//   - tool, verdict are produced by a completed MicroSpanScope leg, whose store is
//     per-instance. The only producer on the tree today is the opt-in `fak micro` host
//     (cmd/fak/micro.go), so on a plain `fak serve` these two render an honest 0 beside a
//     real step value. That is ABSENCE, not idle — every kind is always rendered, and
//     fak_engine_planner_step_observed reports "some leg flowed", not "this leg flowed".
type StepKind string

const (
	// StepKindStep is one agent step / model turn (prefill, decode, sampling).
	StepKindStep StepKind = "step"
	// StepKindTool is a tool call a step made.
	StepKindTool StepKind = "tool"
	// StepKindAdmission is an admission (token/concurrency) decision.
	StepKindAdmission StepKind = "admission"
	// StepKindSeat is a slot/seat acquire or release.
	StepKindSeat StepKind = "seat"
	// StepKindVerdict is an adjudication verdict.
	StepKindVerdict StepKind = "verdict"
)

// StepKinds is the closed, ordered planner-step vocabulary (render and dashboard order).
var StepKinds = []StepKind{
	StepKindStep, StepKindTool, StepKindAdmission, StepKindSeat, StepKindVerdict,
}

// UnknownStepKind labels a leg whose kind is outside the closed vocabulary. It is a
// fixed constant rather than the offending string, so an unexpected producer cannot
// mint unbounded labels.
const UnknownStepKind StepKind = "unknown"

// Overflow labels a kernel key that arrived after the bounded key set was full.
const (
	OverflowKernelLabel      = "overflow"
	OverflowBackendLabel     = "overflow"
	OverflowTimerDomainLabel = "overflow"
)

// Metric family names emitted by WritePrometheus. Dashboards and readers pin against
// these; renaming one is a contract change.
const (
	// MetricKernelCalls counts kernel invocations observed, by (kernel, backend).
	MetricKernelCalls = "fak_engine_kernel_calls_total"
	// MetricKernelSeconds is the per-call duration, by (kernel, backend, timer_domain).
	MetricKernelSeconds = "fak_engine_kernel_seconds"
	// MetricKernelObserved is 1 only when a kernel observer is attached AND fed.
	MetricKernelObserved = "fak_engine_kernel_observed"
	// MetricKernelKeysCapped is 1 once the bounded key set is full.
	MetricKernelKeysCapped = "fak_engine_kernel_keys_capped"
	// MetricKernelEventOverflowTotal counts events folded into the overflow key.
	MetricKernelEventOverflowTotal = "fak_engine_kernel_event_overflow_total"

	// MetricPlannerStepSeconds is per-leg duration over the closed StepKind vocabulary.
	MetricPlannerStepSeconds = "fak_engine_planner_step_seconds"
	// MetricPlannerStepEvents counts legs observed, by kind.
	MetricPlannerStepEvents = "fak_engine_planner_step_events_total"
	// MetricPlannerStepObserved is 1 only when a planner-step observer is attached AND fed.
	MetricPlannerStepObserved = "fak_engine_planner_step_observed"
	// MetricPlannerStepKindOverflowTotal counts legs dropped because their kind fell
	// outside the closed vocabulary.
	MetricPlannerStepKindOverflowTotal = "fak_engine_planner_step_kind_overflow_total"
)

// MetricFamilies lists every family WritePrometheus emits, in render order.
var MetricFamilies = []string{
	MetricKernelCalls, MetricKernelSeconds, MetricKernelObserved,
	MetricKernelKeysCapped, MetricKernelEventOverflowTotal,
	MetricPlannerStepSeconds, MetricPlannerStepEvents, MetricPlannerStepObserved,
	MetricPlannerStepKindOverflowTotal,
}

// secondsBuckets is the same ladder internal/enginestep uses, so a kernel or
// planner-step quantile is directly comparable to a fak_engine_phase_seconds quantile
// on the same board. A decode GEMV and a full decode step differ by ~3 decades, which
// is exactly the span this ladder covers.
var secondsBuckets = []float64{0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}

// MaxKernelKeys bounds the (kernel, backend, timer_domain) key set. Kernel is minted
// per GEMM by the compute layer as "<dtype>_matmul" and backend by the HAL, so the set
// is small in practice but bounded in principle: 256 keys x 19 buckets is ~39 KB of
// counters, flat for the process lifetime, which is the trade internal/engine/cacheevents.go
// already made for the same hazard.
const MaxKernelKeys = 256

// Recorder accumulates sub-kernel and sub-planner step telemetry. The zero value is
// not usable; use New. A nil *Recorder is a safe no-op receiver so an unattached
// producer path costs a nil check.
type Recorder struct {
	mu      sync.Mutex
	kernel  kernelRegistry
	planner plannerRegistry

	// attached records that this recorder was wired to a real producer seam. It is
	// reported separately from "fed" so a wired-but-starved observer is
	// distinguishable in a snapshot without inventing a second gauge in the contract.
	attached bool
}

// New returns a recorder with every bounded slot pre-allocated.
func New() *Recorder {
	r := &Recorder{}
	r.kernel.init()
	r.planner.init()
	return r
}

// Default is the process-wide recorder the compute/planar seams feed and the gateway
// renders on /metrics.
var Default = New()

// Attach wires this recorder to the production producer seams. It is idempotent and
// safe to call from any server construction path: the setters below replace the
// process seam with this recorder's sink rather than only installing once, so a
// re-construction also REPAIRS a seam something else detached. (An install-once guard
// here would make a detached seam permanently dark — the failure this package exists to
// prevent.)
//
// This is the mandatory production attachment: without it the kernel family would have
// writers and no reader, which is the exact defect this package exists to fix. Attach
// is deliberately NOT an init() — the seams stay inert until a server asks for them,
// so a library embedder pays nothing until it renders /metrics.
func (r *Recorder) Attach() {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.attached = true
	r.mu.Unlock()
	computetrace.SetObserver(r.ObserveKernel)
	enginestep.SetPhaseObserver(r.ObserveEnginePhase)
}

// Attach installs the process recorder. See Recorder.Attach.
func Attach() { Default.Attach() }

// Attached reports whether the recorder was wired to a producer seam.
func (r *Recorder) Attached() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.attached
}

// Snapshot is the bounded, agent-facing summary of both registries.
type Snapshot struct {
	Schema string `json:"schema"`

	KernelObserved      bool              `json:"kernel_observed"`
	KernelEvents        uint64            `json:"kernel_events"`
	KernelKeys          []string          `json:"kernel_keys"`
	KernelKeysCapped    bool              `json:"kernel_keys_capped"`
	KernelOverflow      uint64            `json:"kernel_event_overflow"`
	PlannerObserved     bool              `json:"planner_step_observed"`
	PlannerEvents       map[string]uint64 `json:"planner_step_events"`
	PlannerKindOverflow uint64            `json:"planner_step_kind_overflow"`
}

// SnapshotSchema versions the Snapshot JSON shape.
const SnapshotSchema = "fak-step-observations/1"

// Snapshot returns the summary without exposing live histogram internals.
func (r *Recorder) Snapshot() Snapshot {
	s := Snapshot{Schema: SnapshotSchema, PlannerEvents: map[string]uint64{}}
	if r == nil {
		return s
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	s.KernelObserved = r.kernel.events > 0
	s.KernelEvents = r.kernel.events
	s.KernelKeys = make([]string, 0, len(r.kernel.series()))
	for _, k := range r.kernel.series() {
		s.KernelKeys = append(s.KernelKeys, k.render())
	}
	s.KernelKeysCapped = r.kernel.capped()
	s.KernelOverflow = r.kernel.overflowEvents
	s.PlannerObserved = r.planner.events > 0
	for _, k := range StepKinds {
		s.PlannerEvents[string(k)] = r.planner.eventsByKind[k]
	}
	s.PlannerKindOverflow = r.planner.kindOverflow
	return s
}

// Compact renders the snapshot as one bounded operator line.
func (s Snapshot) Compact() string {
	var b strings.Builder
	if s.KernelObserved {
		b.WriteString("KERNEL observed calls=" + strconv.FormatUint(s.KernelEvents, 10) + " keys=" + strconv.Itoa(len(s.KernelKeys)))
	} else {
		b.WriteString("KERNEL absent")
	}
	if s.KernelOverflow > 0 {
		b.WriteString(" overflow=" + strconv.FormatUint(s.KernelOverflow, 10))
	}
	if s.KernelKeysCapped {
		b.WriteString(" capped=1")
	}
	if s.PlannerObserved {
		var kinds []string
		for _, k := range StepKinds {
			if n := s.PlannerEvents[string(k)]; n > 0 {
				kinds = append(kinds, string(k)+"="+strconv.FormatUint(n, 10))
			}
		}
		b.WriteString(" | STEP observed " + strings.Join(kinds, " "))
	} else {
		b.WriteString(" | STEP absent")
	}
	return b.String()
}

// WritePrometheus renders the fak_engine_kernel_* and fak_engine_planner_step_* families
// in Prometheus text format. Every family is always emitted, with the observed bit at 0
// while no producer is attached, so a board band is present-but-dark instead of absent.
func (r *Recorder) WritePrometheus(w io.Writer) {
	if r == nil || w == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kernel.write(w)
	r.planner.write(w)
}

func helpType(w io.Writer, name, help, typ string) {
	writeString(w, "# HELP "+name+" "+help+"\n")
	writeString(w, "# TYPE "+name+" "+typ+"\n")
}

func writeString(w io.Writer, s string) {
	_, _ = io.WriteString(w, s)
}

func formatFloat(v float64) string {
	if math.IsInf(v, 1) {
		return "+Inf"
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// quoteLabel escapes a label value per the Prometheus text exposition format.
func quoteLabel(v string) string {
	r := strings.NewReplacer(`\`, `\\`, "\n", `\n`, `"`, `\"`)
	return r.Replace(v)
}

// seconds clamps a duration to a non-negative float the histogram can accept. A negative
// or NaN duration is recorded as 0 rather than dropped, so the counter and the
// histogram can never disagree about how many calls happened.
func seconds(d time.Duration) float64 {
	v := d.Seconds()
	if v < 0 || math.IsNaN(v) {
		return 0
	}
	return v
}

// normalizeLabel collapses a blank label value to a fixed token. An empty label value is
// legal Prometheus but reads as a board gap, and the compute layer genuinely can report
// an unnamed kernel or backend; the token is a constant, so it cannot smuggle
// cardinality either.
func normalizeLabel(v, fallback string) string {
	if s := strings.TrimSpace(v); s != "" {
		return s
	}
	return fallback
}

// histogram is a fixed-bucket histogram: bounds and counts are allocated once, so an
// observe never grows the structure. counts is non-cumulative with +Inf last; the
// renderer accumulates for exposition.
type histogram struct {
	bounds []float64
	counts []uint64
	count  uint64
	sum    float64
	max    float64
}

func newHistogram(bounds []float64) *histogram {
	return &histogram{bounds: bounds, counts: make([]uint64, len(bounds)+1)}
}

func (h *histogram) observe(v float64) {
	if math.IsNaN(v) || v < 0 {
		v = 0
	}
	i := sort.SearchFloat64s(h.bounds, v)
	h.counts[i]++
	h.count++
	h.sum += v
	if v > h.max {
		h.max = v
	}
}

// quantile returns the upper bound of the bucket holding quantile q (an over-estimate by
// at most one bucket width, clamped to the observed max).
func (h *histogram) quantile(q float64) float64 {
	if h.count == 0 {
		return 0
	}
	rank := uint64(math.Ceil(q * float64(h.count)))
	if rank == 0 {
		rank = 1
	}
	var cum uint64
	for i, c := range h.counts {
		cum += c
		if cum >= rank {
			if i < len(h.bounds) {
				return math.Min(h.bounds[i], h.max)
			}
			return h.max
		}
	}
	return h.max
}

// write renders the cumulative Prometheus histogram body under `labels`.
func (h *histogram) write(w io.Writer, name, labels string) {
	sep := ""
	if labels != "" {
		sep = ","
	}
	var cum uint64
	for i, b := range h.bounds {
		cum += h.counts[i]
		writeString(w, name+"_bucket{"+labels+sep+"le=\""+formatFloat(b)+"\"} "+strconv.FormatUint(cum, 10)+"\n")
	}
	cum += h.counts[len(h.bounds)]
	writeString(w, name+"_bucket{"+labels+sep+"le=\"+Inf\"} "+strconv.FormatUint(cum, 10)+"\n")
	if labels == "" {
		writeString(w, name+"_sum "+formatFloat(h.sum)+"\n")
		writeString(w, name+"_count "+strconv.FormatUint(h.count, 10)+"\n")
		return
	}
	writeString(w, name+"_sum{"+labels+"} "+formatFloat(h.sum)+"\n")
	writeString(w, name+"_count{"+labels+"} "+strconv.FormatUint(h.count, 10)+"\n")
}

// StepKindForEnginePhase projects one native serving-loop phase onto the closed planner
// -step vocabulary. ok is false for a phase that names no span leg, and the caller
// records nothing — a phase with no honest kind is dropped rather than folded into a
// neighbouring label.
//
// The mapping is deliberately partial. Phases that are sub-legs of a step (prefix lookup,
// prefix admit) and the whole-request envelope have no MicroSpanKind counterpart, so
// they read on fak_engine_phase_seconds only.
//
// KNOWN, MEASURED, NOT YET RESOLVED: the three phases mapped here are three different
// granularities under one kind label. On a four-token CPU-reference turn a single
// InKernelPlanner.Complete folds prefill (one chunked prompt forward) + decode (one
// request's WHOLE decode loop) + sample (one token, four times) into kind="step", while
// internal/gateway's decode-step observer folds the four individual forwards into the
// same label. A kind="step" quantile therefore mixes seconds with milliseconds. Making
// kind="step" mean exactly one model turn is a metric-contract change to an already
// rendered family and board, so it is deliberately NOT taken here; read kind="step" as
// "a model-turn-shaped leg" and use fak_engine_phase_seconds{phase=...} plus
// fak_engine_decode_step_seconds{path=...} for per-granularity latency.
func StepKindForEnginePhase(p enginestep.Phase) (StepKind, bool) {
	switch p {
	case enginestep.PhasePrefill, enginestep.PhaseDecode, enginestep.PhaseSample:
		return StepKindStep, true
	case enginestep.PhaseAdmissionWait:
		return StepKindAdmission, true
	case enginestep.PhaseDeviceWait, enginestep.PhaseCohortWait:
		return StepKindSeat, true
	default:
		return "", false
	}
}
