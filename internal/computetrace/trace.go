// Package computetrace records bounded, opt-in compute events in a stable local artifact.
//
// Invariant: compute trace recording is fail-closed and bounded. When disabled
// or uninitialized, tracing calls are zero-allocation no-ops and emit no events.
// When enabled, recorders strictly enforce an upper bound on retained events;
// excess events are dropped with dropped counters incremented, protecting host
// memory from unbounded growth.
//
// A separate, optional observer (SetObserver) is fanned out on every Record so a
// metrics sink can read per-kernel timings without the compute backends importing a
// metrics package, and without the trace artifact or its cost ever becoming
// unconditional. With no observer attached Record is unchanged.
package computetrace

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"time"
)

const Schema = "fak.compute_trace.v1"

// Bounds on optional activation-sample capture. Sampling is a diagnostic aid,
// never a tensor dump: each event may carry at most MaxActivationSamplesPerEvent
// values and a recorder retains at most MaxActivationSamplesTotal across all of
// its events. Excess values are dropped and counted, keeping a trace bounded on
// a memory-constrained host and keeping the comparator fail-closed.
const (
	MaxActivationSamplesPerEvent = 32
	MaxActivationSamplesTotal    = 4096
)

type Event struct {
	Sequence         int64     `json:"sequence"`
	RunID            string    `json:"run_id"`
	RequestID        string    `json:"request_id"`
	Operation        string    `json:"operation"`
	Phase            string    `json:"phase"`
	Backend          string    `json:"backend"`
	Device           string    `json:"device"`
	Kernel           string    `json:"kernel_id,omitempty"`
	Candidate        string    `json:"candidate_id,omitempty"`
	StartedAt        time.Time `json:"started_at"`
	DurationNS       int64     `json:"duration_ns"`
	DeviceDurationNS int64     `json:"device_duration_ns,omitempty"`
	TimerDomain      string    `json:"timer_domain"`
	Route            string    `json:"route,omitempty"`
	InputDType       string    `json:"input_dtype,omitempty"`
	WeightDType      string    `json:"weight_dtype,omitempty"`
	OutputDType      string    `json:"output_dtype,omitempty"`
	Bytes            int64     `json:"bytes,omitempty"`
	BytesRead        int64     `json:"bytes_read,omitempty"`
	BytesWritten     int64     `json:"bytes_written,omitempty"`
	EstimatedFLOPs   int64     `json:"estimated_flops,omitempty"`
	Shapes           [][]int   `json:"shapes,omitempty"`
	Status           string    `json:"status"`
	ProvenanceDigest string    `json:"provenance_digest"`

	// Optional bounded activation-sample record (additive; historical v1
	// artifacts without these fields still decode). Layer and Token give the
	// sample a stable identity within a forward; InputDigest and WeightDigest
	// carry the provenance a comparator uses to refuse an apples-to-oranges
	// comparison. SampleIndices are logical indices into the stage's output;
	// SampleValues are the matching values. Layer and Token are omitempty, so a
	// zero layer or token is ABSENT from the JSON: a reader outside Go must
	// default a missing layer/token to 0 rather than treat it as null.
	Layer         int       `json:"layer,omitempty"`
	Token         int       `json:"token,omitempty"`
	InputDigest   string    `json:"input_digest,omitempty"`
	WeightDigest  string    `json:"weight_digest,omitempty"`
	SampleIndices []int     `json:"sample_indices,omitempty"`
	SampleValues  []float32 `json:"sample_values,omitempty"`

	// ExcludedSampleIndices names sample indices a producer declares not
	// comparable (for example a reference captured after a transform the other
	// side samples before). ActivationErrorProfile skips them and counts them;
	// CompareActivationTraces ignores the field and stays strict.
	ExcludedSampleIndices []int `json:"excluded_sample_indices,omitempty"`
}

type Artifact struct {
	Schema             string  `json:"schema"`
	Events             []Event `json:"events"`
	Dropped            uint64  `json:"dropped_events"`
	ObserverOverheadNS int64   `json:"observer_overhead_ns"`

	// Activation-sample accounting, distinct from Dropped (dropped EVENTS). A
	// nonzero sample-drop counter means the trace is incomplete and the
	// comparator must report INCOMPARABLE rather than agreement.
	RetainedSampleValues uint64 `json:"retained_sample_values,omitempty"`
	DroppedSampleValues  uint64 `json:"dropped_sample_values,omitempty"`

	// InputTokens optionally records the token ids the traced forward consumed,
	// so AdoptActivationDigests can prove two traces saw the same input before a
	// reference adopts the other trace's provenance digests.
	InputTokens []int `json:"input_tokens,omitempty"`
}

type Recorder struct {
	mu          sync.Mutex
	limit       int
	events      []Event
	dropped     uint64
	overhead    int64
	retained    uint64
	droppedSamp uint64
	sampleTotal int
	operations  map[string]bool
}

// New returns a disabled recorder when limit is zero. Enabled recorders retain at most limit events.
func New(limit int) *Recorder { return &Recorder{limit: limit} }

// NewWithSampleBudget is New with a recorder-total activation-sample budget in
// place of MaxActivationSamplesTotal. A budget <= 0 keeps the default; the
// per-event cap MaxActivationSamplesPerEvent is unchanged.
func NewWithSampleBudget(limit, maxSamples int) *Recorder {
	return &Recorder{limit: limit, sampleTotal: maxSamples}
}

// RecorderOptions configures a recorder beyond its event limit.
type RecorderOptions struct {
	// MaxSamples replaces MaxActivationSamplesTotal when > 0.
	MaxSamples int
	// Operations, when non-empty, is an allowlist: an event whose Operation is
	// not listed is ignored (neither retained nor counted as dropped), so other
	// producers sharing the process recorder cannot spend its event budget.
	Operations []string
}

// NewWithOptions is New configured by opts.
func NewWithOptions(limit int, opts RecorderOptions) *Recorder {
	r := &Recorder{limit: limit, sampleTotal: opts.MaxSamples}
	if len(opts.Operations) > 0 {
		r.operations = make(map[string]bool, len(opts.Operations))
		for _, op := range opts.Operations {
			r.operations[op] = true
		}
	}
	return r
}

func (r *Recorder) sampleBudget() int {
	if r.sampleTotal > 0 {
		return r.sampleTotal
	}
	return MaxActivationSamplesTotal
}

func (r *Recorder) Enabled() bool { return r != nil && r.limit > 0 }
func (r *Recorder) Record(e Event) {
	if !r.Enabled() {
		return
	}
	if r.operations != nil && !r.operations[e.Operation] {
		return
	}
	began := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.events) >= r.limit {
		r.dropped++
		r.overhead += time.Since(began).Nanoseconds()
		return
	}
	e.Sequence = int64(len(r.events) + 1)
	if e.Status == "" {
		e.Status = "ok"
	}
	e.Shapes = cloneShapes(e.Shapes)
	e.SampleIndices, e.SampleValues = r.boundSamples(e.SampleIndices, e.SampleValues)
	e.ExcludedSampleIndices = append([]int(nil), e.ExcludedSampleIndices...)
	r.events = append(r.events, e)
	r.overhead += time.Since(began).Nanoseconds()
}

// boundSamples enforces the per-event and recorder-total sample caps, returning
// deep copies so a caller mutating its slice after Record cannot alias the
// retained record. Values dropped past either cap increment droppedSamp; the
// indices are clipped to the retained values so the two slices stay parallel.
func (r *Recorder) boundSamples(idx []int, vals []float32) ([]int, []float32) {
	if len(vals) == 0 {
		return nil, nil
	}
	keep := len(vals)
	if keep > MaxActivationSamplesPerEvent {
		keep = MaxActivationSamplesPerEvent
	}
	if room := r.sampleBudget() - int(r.retained); keep > room {
		keep = room
	}
	if dropped := len(vals) - keep; dropped > 0 {
		r.droppedSamp += uint64(dropped)
	}
	if keep <= 0 {
		return nil, nil
	}
	outIdx := make([]int, keep)
	for i := 0; i < keep; i++ {
		if i < len(idx) {
			outIdx[i] = idx[i]
		}
	}
	outVals := make([]float32, keep)
	copy(outVals, vals[:keep])
	r.retained += uint64(keep)
	return outIdx, outVals
}
func (r *Recorder) Artifact() Artifact {
	if r == nil {
		return Artifact{Schema: Schema}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	events := make([]Event, len(r.events))
	for i := range r.events {
		events[i] = r.events[i]
		events[i].SampleIndices = append([]int(nil), r.events[i].SampleIndices...)
		events[i].SampleValues = append([]float32(nil), r.events[i].SampleValues...)
		events[i].ExcludedSampleIndices = append([]int(nil), r.events[i].ExcludedSampleIndices...)
	}
	return Artifact{
		Schema:               Schema,
		Events:               events,
		Dropped:              r.dropped,
		ObserverOverheadNS:   r.overhead,
		RetainedSampleValues: r.retained,
		DroppedSampleValues:  r.droppedSamp,
	}
}
func (r *Recorder) Write(w io.Writer) error { return json.NewEncoder(w).Encode(r.Artifact()) }
func Read(rd io.Reader) (Artifact, error) {
	var a Artifact
	if err := json.NewDecoder(rd).Decode(&a); err != nil {
		return Artifact{}, err
	}
	if a.Schema != Schema {
		return Artifact{}, errors.New("unsupported compute trace schema")
	}
	return a, nil
}
func Digest(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		_, _ = io.WriteString(h, p)
		_, _ = h.Write([]byte{0})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
func cloneShapes(in [][]int) [][]int {
	out := make([][]int, len(in))
	for i := range in {
		out[i] = append([]int(nil), in[i]...)
	}
	return out
}

var active struct {
	sync.RWMutex
	recorder     *Recorder
	run, request string
	observer     func(Event)
}

// Enable installs the process recorder. Calling the returned function disables it.
func Enable(limit int, run, request string) (*Recorder, func()) {
	return install(New(limit), run, request)
}

// EnableWithSampleBudget is Enable with a recorder-total activation-sample
// budget (see NewWithSampleBudget), for a producer whose full-depth forward
// needs more than the MaxActivationSamplesTotal default.
func EnableWithSampleBudget(limit, maxSamples int, run, request string) (*Recorder, func()) {
	return install(NewWithSampleBudget(limit, maxSamples), run, request)
}

// EnableWithOptions is Enable configured by opts (see RecorderOptions).
func EnableWithOptions(limit int, opts RecorderOptions, run, request string) (*Recorder, func()) {
	return install(NewWithOptions(limit, opts), run, request)
}

// TryEnableWithSampleBudget installs a recorder only when no recorder is
// installed. A non-nil recorder with a zero event limit still owns the slot.
// It returns nil, nil if the slot is occupied. Check and install are atomic;
// a later explicit Enable call retains its existing replacement semantics.
func TryEnableWithSampleBudget(limit, maxSamples int, run, request string) (*Recorder, func()) {
	return TryEnableWithOptions(limit, RecorderOptions{MaxSamples: maxSamples}, run, request)
}

// TryEnableWithOptions atomically installs an options-configured recorder only
// when the global slot has no owner, including no zero-limit owner. Like the
// other install paths, its disable function cannot remove a later replacement.
func TryEnableWithOptions(limit int, opts RecorderOptions, run, request string) (*Recorder, func()) {
	active.Lock()
	defer active.Unlock()
	if active.recorder != nil {
		return nil, nil
	}
	return installLocked(NewWithOptions(limit, opts), run, request)
}

func install(r *Recorder, run, request string) (*Recorder, func()) {
	active.Lock()
	defer active.Unlock()
	return installLocked(r, run, request)
}

// installLocked requires active's write lock.
func installLocked(r *Recorder, run, request string) (*Recorder, func()) {
	active.recorder, active.run, active.request = r, run, request
	return r, func() {
		active.Lock()
		if active.recorder == r {
			active.recorder = nil
		}
		active.Unlock()
	}
}

// SetObserver installs the process observer: a sink fed EVERY Record call, whether or
// not a recorder is enabled. This is the seam that lets per-kernel timings reach a
// metric surface without any compute backend knowing a metrics package exists — the
// fan-out lives here, in the trace package the backends already call.
//
// The observer is a metrics sink, not a second trace store: it must not retain the
// event, any slice the event references, or the caller's tensors. Passing nil detaches
// it and restores the pre-observer no-op. Enabled() is deliberately unaffected, so the
// device backends keep gating their (more expensive) per-GEMM measurement on explicit
// tracing rather than on a metrics observer happening to be attached.
func SetObserver(fn func(Event)) {
	active.Lock()
	active.observer = fn
	active.Unlock()
}

// Observed reports whether a process observer is attached.
func Observed() bool {
	active.RLock()
	defer active.RUnlock()
	return active.observer != nil
}

// Record fans one event out to the process observer, then adds it to the recorder when
// tracing was explicitly enabled. The observer call is a nil check plus a struct copy
// when unattached nothing at all, so the per-GEMM path stays allocation-free.
func Record(e Event) {
	active.RLock()
	r, run, request, observe := active.recorder, active.run, active.request, active.observer
	active.RUnlock()
	if observe != nil {
		observe(e)
	}
	if r == nil {
		return
	}
	if e.RunID == "" {
		e.RunID = run
	}
	if e.RequestID == "" {
		e.RequestID = request
	}
	r.Record(e)
}
func Enabled() bool {
	active.RLock()
	defer active.RUnlock()
	return active.recorder != nil && active.recorder.Enabled()
}

// Emitting reports whether an Event would reach ANY sink: the artifact recorder
// is enabled OR a process observer is attached. Enabled() is deliberately NOT
// widened to cover the observer case — it also gates the bounded
// activation-sample capture on the decode hot path
// (internal/model/v41_activation_trace.go), which must stay opt-in behind the
// `fak computetrace` recorder.
//
// Use Emitting() only at an emission point whose per-call work is a nil check
// plus a struct copy (an event the producer already builds, or one whose
// measurement it takes anyway). Use Enabled() for anything that captures
// samples, tensors, or per-GEMM device timing.
func Emitting() bool {
	active.RLock()
	defer active.RUnlock()
	return (active.recorder != nil && active.recorder.Enabled()) || active.observer != nil
}
