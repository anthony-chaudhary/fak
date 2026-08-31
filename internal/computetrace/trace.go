// Package computetrace records bounded, opt-in compute events in a stable local artifact.
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
}

type Artifact struct {
	Schema             string  `json:"schema"`
	Events             []Event `json:"events"`
	Dropped            uint64  `json:"dropped_events"`
	ObserverOverheadNS int64   `json:"observer_overhead_ns"`
}

type Recorder struct {
	mu       sync.Mutex
	limit    int
	events   []Event
	dropped  uint64
	overhead int64
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
	r.events = append(r.events, e)
	r.overhead += time.Since(began).Nanoseconds()
}
func (r *Recorder) Artifact() Artifact {
	if r == nil {
		return Artifact{Schema: Schema}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return Artifact{Schema: Schema, Events: append([]Event(nil), r.events...), Dropped: r.dropped, ObserverOverheadNS: r.overhead}
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
