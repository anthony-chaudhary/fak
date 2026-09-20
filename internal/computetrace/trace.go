// Package computetrace records bounded, opt-in compute events in a stable local artifact.
//
// Invariant: compute trace recording is fail-closed and bounded. When disabled
// or uninitialized, tracing calls are zero-allocation no-ops and emit no events.
// When enabled, recorders strictly enforce an upper bound on retained events;
// excess events are dropped with dropped counters incremented, protecting host
// memory from unbounded growth.
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
	// SampleValues are the matching values.
	Layer         int       `json:"layer,omitempty"`
	Token         int       `json:"token,omitempty"`
	InputDigest   string    `json:"input_digest,omitempty"`
	WeightDigest  string    `json:"weight_digest,omitempty"`
	SampleIndices []int     `json:"sample_indices,omitempty"`
	SampleValues  []float32 `json:"sample_values,omitempty"`
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
}

type Recorder struct {
	mu          sync.Mutex
	limit       int
	events      []Event
	dropped     uint64
	overhead    int64
	retained    uint64
	droppedSamp uint64
}

// New returns a disabled recorder when limit is zero. Enabled recorders retain at most limit events.
func New(limit int) *Recorder     { return &Recorder{limit: limit} }
func (r *Recorder) Enabled() bool { return r != nil && r.limit > 0 }
func (r *Recorder) Record(e Event) {
	if !r.Enabled() {
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
	if room := MaxActivationSamplesTotal - int(r.retained); keep > room {
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
}

// Enable installs the process recorder. Calling the returned function disables it.
func Enable(limit int, run, request string) (*Recorder, func()) {
	r := New(limit)
	active.Lock()
	active.recorder, active.run, active.request = r, run, request
	active.Unlock()
	return r, func() {
		active.Lock()
		if active.recorder == r {
			active.recorder = nil
		}
		active.Unlock()
	}
}

// Record adds an event only when tracing was explicitly enabled.
func Record(e Event) {
	active.RLock()
	r, run, request := active.recorder, active.run, active.request
	active.RUnlock()
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
