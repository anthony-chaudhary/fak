package metalgemm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

// Matches MG_GRAPH_TIMING_MAX_ENCODERS: two 8-byte timestamp results per
// encoder fit the documented 32 KiB Apple counter sample buffer ceiling.
const executionEncoderSampleLimit = 2048

// ExecutionOperation identifies the native operation observed by one call.
type ExecutionOperation string

const (
	ExecutionQ4KGEMV                ExecutionOperation = "q4_k-gemv"
	ExecutionQ4KGEMVBatch           ExecutionOperation = "q4_k-gemv-batch"
	ExecutionQ4KGEMVGroup           ExecutionOperation = "q4_k-gemv-group"
	ExecutionQ4KGEMM                ExecutionOperation = "q4_k-gemm"
	ExecutionQ4KGEMMGroup           ExecutionOperation = "q4_k-gemm-group"
	ExecutionQ8GEMV                 ExecutionOperation = "q8-gemv"
	ExecutionQ8GEMVGroup            ExecutionOperation = "q8-gemv-group"
	ExecutionQ8GEMM                 ExecutionOperation = "q8-gemm"
	ExecutionQ8GEMMGroup            ExecutionOperation = "q8-gemm-group"
	ExecutionQ6KGEMV                ExecutionOperation = "q6_k-gemv"
	ExecutionQ6KGEMM                ExecutionOperation = "q6_k-gemm"
	ExecutionQ4KFusedMLP            ExecutionOperation = "q4_k-fused-mlp"
	ExecutionQ4KFusedMLPQ6Down      ExecutionOperation = "q4_k-fused-mlp-q6_k-down"
	ExecutionQ4KFusedMLPQ6DownBatch ExecutionOperation = "q4_k-fused-mlp-q6_k-down-batch"
	ExecutionPrefillAttention       ExecutionOperation = "prefill-attention"
)

// ExecutionEventsUnavailableError reports that this build cannot observe Metal execution.
type ExecutionEventsUnavailableError struct{}

func (ExecutionEventsUnavailableError) Error() string { return "Metal execution events unavailable" }

// IsExecutionEventsUnavailable reports whether err is the typed unsupported-build result.
func IsExecutionEventsUnavailable(err error) bool {
	var unavailable ExecutionEventsUnavailableError
	return errors.As(err, &unavailable)
}

// ExecutionEvent contains lifecycle facts witnessed inside one native Metal call. CommandBufferID
// is opaque and local to this observation; it never exposes a process-global native identity.
type ExecutionEvent struct {
	Operation        ExecutionOperation `json:"operation"`
	CommandBufferID  uint64             `json:"command_buffer_id"`
	Committed        bool               `json:"committed"`
	CompletedWait    bool               `json:"completed_wait"`
	HostReadback     bool               `json:"host_readback"`
	Encoders         int                `json:"encoders"`
	GPUMilliseconds  float64            `json:"gpu_milliseconds"`
	WaitMilliseconds float64            `json:"wait_milliseconds"`
	TimingAvailable  bool               `json:"timing_available"`

	EncoderTiming *ExecutionEncoderTiming `json:"encoder_timing,omitempty"`
}

// ExecutionEncoderSpan is one measured encoder interval, in submission order.
// Ordinal is one-based within the declared encoder census for its segment.
// Times are milliseconds in the enclosing timing window's GPU clock domain.
type ExecutionEncoderSpan struct {
	Ordinal           int     `json:"ordinal"`
	Operation         string  `json:"operation"`
	StartMilliseconds float64 `json:"start_milliseconds"`
	EndMilliseconds   float64 `json:"end_milliseconds"`
}

// ExecutionEncoderSegment retains a separate window for one submitted buffer.
// CapturedEncoders counts instrumented sites, not allocation-derived graph counters.
type ExecutionEncoderSegment struct {
	CommandBufferID  uint64                 `json:"command_buffer_id"`
	CapturedEncoders int                    `json:"captured_encoders"`
	Timing           ExecutionEncoderTiming `json:"timing"`
}

// ExecutionEncoderTiming describes one command buffer or separate segment windows.
// A nil value means capture was not requested. Complete means all encoders in the declared census were sampled, not that
// every instant of the command buffer is attributable to work.
// Unavailable/disjoint captures carry a reason and no comparable intervals.
type ExecutionEncoderTiming struct {
	Segments              []ExecutionEncoderSegment `json:"segments,omitempty"`
	WindowScope           string                    `json:"window_scope,omitempty"`
	EncoderCensus         string                    `json:"encoder_census,omitempty"`
	UnmeasuredPrefix      bool                      `json:"unmeasured_prefix,omitempty"`
	UnmeasuredSuffix      bool                      `json:"unmeasured_suffix,omitempty"`
	UncoveredEncoderScope string                    `json:"uncovered_encoder_scope,omitempty"`

	Status                  string                 `json:"status"`
	Reason                  string                 `json:"reason,omitempty"`
	ClockDomain             string                 `json:"clock_domain,omitempty"`
	WindowStartMilliseconds float64                `json:"window_start_milliseconds,omitempty"`
	WindowEndMilliseconds   float64                `json:"window_end_milliseconds,omitempty"`
	Spans                   []ExecutionEncoderSpan `json:"spans,omitempty"`
}

// ExecutionInterval is a half-open interval in the capture's GPU clock domain.
type ExecutionInterval struct {
	StartMilliseconds float64 `json:"start_milliseconds"`
	EndMilliseconds   float64 `json:"end_milliseconds"`
}

// ExecutionEncoderCoverage partitions a comparable timing window. Uncovered is
// unattributed time, NOT idle time: omitted work and instrumentation may occupy it.
// Host submission and waits are deliberately outside this accounting.
type ExecutionEncoderCoverage struct {
	Available                bool                `json:"available"`
	Covered                  []ExecutionInterval `json:"covered,omitempty"`
	Uncovered                []ExecutionInterval `json:"uncovered,omitempty"`
	AttributedMilliseconds   float64             `json:"attributed_milliseconds"`
	UnattributedMilliseconds float64             `json:"unattributed_milliseconds"`
}

// EncoderCoverage validates capture and unions overlapping intervals without
// changing their original submission order. It never compares separate buffers.
func (t *ExecutionEncoderTiming) EncoderCoverage(encoders int) (ExecutionEncoderCoverage, error) {
	var out ExecutionEncoderCoverage
	if t == nil {
		return out, nil
	}
	invalid := func(detail string) (ExecutionEncoderCoverage, error) {
		return ExecutionEncoderCoverage{}, fmt.Errorf("invalid Metal encoder timing: %s", detail)
	}
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if t.Status == "segmented" {
		if len(t.Segments) == 0 || len(t.Segments) > executionEncoderSampleLimit || t.Reason == "" || t.ClockDomain != "" || len(t.Spans) != 0 || t.WindowStartMilliseconds != 0 || t.WindowEndMilliseconds != 0 || t.WindowScope != "" || t.EncoderCensus != "" || t.UnmeasuredPrefix || t.UnmeasuredSuffix || t.UncoveredEncoderScope != "" {
			return invalid("segmented capture must preserve separate windows")
		}
		totalCaptured := 0
		for i, segment := range t.Segments {
			if segment.CommandBufferID != uint64(i+1) || segment.CapturedEncoders < 0 || segment.CapturedEncoders > executionEncoderSampleLimit || len(segment.Timing.Segments) != 0 {
				return invalid("invalid segment identity or encoder census")
			}
			totalCaptured += segment.CapturedEncoders
			if totalCaptured > executionEncoderSampleLimit {
				return invalid("encoder census exceeds capture capacity")
			}
			if _, err := segment.Timing.EncoderCoverage(segment.CapturedEncoders); err != nil {
				return invalid(err.Error())
			}
		}
		// A cross-buffer union is deliberately unavailable. Inspect each segment.
		return out, nil
	}
	if len(t.Segments) != 0 {
		return invalid("unexpected segment windows")
	}
	if t.Status == "unavailable" || t.Status == "disjoint" {
		if t.Reason == "" || t.ClockDomain != "" || len(t.Spans) != 0 || t.WindowStartMilliseconds != 0 || t.WindowEndMilliseconds != 0 || t.WindowScope != "" || t.EncoderCensus != "" || t.UnmeasuredPrefix || t.UnmeasuredSuffix || t.UncoveredEncoderScope != "" {
			return invalid("noncomparable capture requires a reason and no comparable intervals")
		}
		return out, nil
	}
	if t.Status != "complete" && t.Status != "partial" {
		return invalid("unknown capture status")
	}
	if encoders <= 0 || len(t.Spans) == 0 || len(t.Spans) > encoders || (t.Status == "complete" && len(t.Spans) != encoders) || (t.Status == "partial" && len(t.Spans) == encoders && t.WindowScope != "sampled-encoder-envelope") {
		return invalid("capture status does not match encoder coverage")
	}
	if t.WindowScope == "sampled-encoder-envelope" {
		if t.Status != "partial" || t.Reason == "" || !t.UnmeasuredPrefix || !t.UnmeasuredSuffix || t.EncoderCensus == "" || t.UncoveredEncoderScope == "" {
			return invalid("sampled envelope must disclose unknown tails and uncovered encoder scope")
		}
	} else if t.WindowScope != "" || t.UnmeasuredPrefix || t.UnmeasuredSuffix || t.UncoveredEncoderScope != "" {
		return invalid("unknown window scope")
	}
	if t.ClockDomain != "metal-gpu-counter" || !finite(t.WindowStartMilliseconds) || !finite(t.WindowEndMilliseconds) || t.WindowStartMilliseconds < 0 || t.WindowEndMilliseconds <= t.WindowStartMilliseconds {
		return invalid("missing or invalid GPU counter window")
	}
	intervals := make([]ExecutionInterval, len(t.Spans))
	ordinal := 0
	for i, span := range t.Spans {
		if span.Ordinal <= ordinal || span.Ordinal > encoders || span.Operation == "" || !finite(span.StartMilliseconds) || !finite(span.EndMilliseconds) || span.StartMilliseconds < t.WindowStartMilliseconds || span.EndMilliseconds > t.WindowEndMilliseconds || span.EndMilliseconds <= span.StartMilliseconds {
			return invalid("unordered encoder identity or out-of-range interval")
		}
		ordinal = span.Ordinal
		intervals[i] = ExecutionInterval{span.StartMilliseconds, span.EndMilliseconds}
	}
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].StartMilliseconds < intervals[j].StartMilliseconds })
	for _, interval := range intervals {
		n := len(out.Covered)
		if n == 0 || interval.StartMilliseconds > out.Covered[n-1].EndMilliseconds {
			out.Covered = append(out.Covered, interval)
		} else if interval.EndMilliseconds > out.Covered[n-1].EndMilliseconds {
			out.Covered[n-1].EndMilliseconds = interval.EndMilliseconds
		}
	}
	cursor := t.WindowStartMilliseconds
	for _, interval := range out.Covered {
		if cursor < interval.StartMilliseconds {
			out.Uncovered = append(out.Uncovered, ExecutionInterval{cursor, interval.StartMilliseconds})
		}
		out.AttributedMilliseconds += interval.EndMilliseconds - interval.StartMilliseconds
		cursor = interval.EndMilliseconds
	}
	if cursor < t.WindowEndMilliseconds {
		out.Uncovered = append(out.Uncovered, ExecutionInterval{cursor, t.WindowEndMilliseconds})
	}
	for _, interval := range out.Uncovered {
		out.UnattributedMilliseconds += interval.EndMilliseconds - interval.StartMilliseconds
	}
	out.Available = true
	return out, nil
}

// Clone returns a detached copy suitable for immutable receipt snapshots.
func (t *ExecutionEncoderTiming) Clone() *ExecutionEncoderTiming {
	if t == nil {
		return nil
	}
	out := *t
	out.Spans = append([]ExecutionEncoderSpan(nil), t.Spans...)
	out.Segments = append([]ExecutionEncoderSegment(nil), t.Segments...)
	for i := range out.Segments {
		out.Segments[i].Timing = *out.Segments[i].Timing.Clone()
	}
	return &out
}

func cloneExecutionEvents(events []ExecutionEvent) []ExecutionEvent {
	out := append([]ExecutionEvent(nil), events...)
	for i := range out {
		out[i].EncoderTiming = out[i].EncoderTiming.Clone()
	}
	return out
}

// ExecutionSnapshot is an immutable copy of one call's observations.
type ExecutionSnapshot struct {
	Events []ExecutionEvent `json:"events"`
}

// ExecutionObservation owns events for exactly one native call. It is safe to snapshot while
// unrelated calls execute; no package-global attribution state or whole-token lock is used.
type ExecutionObservation struct {
	operation ExecutionOperation
	available bool
	mu        sync.Mutex
	events    []ExecutionEvent
}

// NewExecutionObservation starts a call-local observation.
func NewExecutionObservation(operation ExecutionOperation) *ExecutionObservation {
	return newExecutionObservation(operation, executionEventsAvailable())
}

func newExecutionObservation(operation ExecutionOperation, available bool) *ExecutionObservation {
	return &ExecutionObservation{operation: operation, available: available}
}

func (o *ExecutionObservation) record(commandBuffer uintptr, committed, completedWait, hostReadback bool, encoders int, gpuMS, waitMS float64, timingAvailable bool) {
	if o == nil || !o.available || commandBuffer == 0 {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, ExecutionEvent{
		Operation:        o.operation,
		CommandBufferID:  uint64(len(o.events) + 1),
		Committed:        committed,
		CompletedWait:    completedWait,
		HostReadback:     hostReadback,
		Encoders:         encoders,
		GPUMilliseconds:  gpuMS,
		WaitMilliseconds: waitMS,
		TimingAvailable:  timingAvailable,
	})
}

// ExecutionCounters is the lossless Metal subset required by the native-performance v1 profile.
type ExecutionCounters struct {
	CommandBuffers       int     `json:"command_buffers"`
	Encoders             int     `json:"encoders"`
	DispatchMilliseconds float64 `json:"dispatch_milliseconds"`
	WaitMilliseconds     float64 `json:"wait_milliseconds"`
}

// ExecutionReceipt preserves the raw ordered event stream behind an aggregate.
type ExecutionReceipt struct {
	Schema       string            `json:"schema"`
	Events       []ExecutionEvent  `json:"events"`
	EventsSHA256 string            `json:"events_sha256"`
	Counters     ExecutionCounters `json:"counters"`
}

// ExecutionCountersIncompleteError reports the first lifecycle fact that could not be captured.
type ExecutionCountersIncompleteError struct{ Detail string }

func (e ExecutionCountersIncompleteError) Error() string {
	return "Metal execution counters incomplete: " + e.Detail
}

// IsExecutionCountersIncomplete reports whether err is a typed incomplete-capture result.
func IsExecutionCountersIncomplete(err error) bool {
	var incomplete ExecutionCountersIncompleteError
	return errors.As(err, &incomplete)
}

// ExecutionSession owns every observation for one model session. It never aggregates process-wide.
type ExecutionSession struct {
	mu     sync.Mutex
	events []ExecutionEvent
	err    error
}

// NewExecutionSession creates an empty session-local aggregate.
func NewExecutionSession() *ExecutionSession { return &ExecutionSession{} }

// Record appends one native call, poisoning the aggregate when capture is unavailable or empty.
func (s *ExecutionSession) Record(snapshot ExecutionSnapshot, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return
	}
	if err != nil {
		s.err = err
		return
	}
	if len(snapshot.Events) == 0 {
		s.err = ExecutionCountersIncompleteError{Detail: "native call emitted no command-buffer event"}
		return
	}
	s.events = append(s.events, cloneExecutionEvents(snapshot.Events)...)
}

// Counters validates every captured lifecycle before returning the session aggregate.
func (s *ExecutionSession) Counters() (ExecutionCounters, error) {
	if s == nil {
		return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: "session capture is nil"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return ExecutionCounters{}, s.err
	}
	return countersForEvents(s.events)
}

// Receipt returns the raw stream, its deterministic digest, and independently recomputed totals.
func (s *ExecutionSession) Receipt() (ExecutionReceipt, error) {
	if s == nil {
		return ExecutionReceipt{}, ExecutionCountersIncompleteError{Detail: "session capture is nil"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return ExecutionReceipt{}, s.err
	}
	events := cloneExecutionEvents(s.events)
	counters, err := countersForEvents(events)
	if err != nil {
		return ExecutionReceipt{}, err
	}
	raw, err := json.Marshal(events)
	if err != nil {
		return ExecutionReceipt{}, err
	}
	digest := sha256.Sum256(raw)
	return ExecutionReceipt{Schema: "fak-metal-execution-receipt/v1", Events: events, EventsSHA256: hex.EncodeToString(digest[:]), Counters: counters}, nil
}

// ValidateExecutionReceipt recomputes both digest and totals from the raw event stream.
func ValidateExecutionReceipt(receipt ExecutionReceipt) error {
	if receipt.Schema != "fak-metal-execution-receipt/v1" {
		return fmt.Errorf("unexpected execution receipt schema %q", receipt.Schema)
	}
	raw, err := json.Marshal(receipt.Events)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(raw)
	if receipt.EventsSHA256 != hex.EncodeToString(digest[:]) {
		return fmt.Errorf("execution event digest mismatch")
	}
	counters, err := countersForEvents(receipt.Events)
	if err != nil {
		return err
	}
	if counters != receipt.Counters {
		return fmt.Errorf("execution counter aggregate mismatch")
	}
	return nil
}

func countersForEvents(events []ExecutionEvent) (ExecutionCounters, error) {
	if len(events) == 0 {
		return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: "session captured no command buffers"}
	}
	var out ExecutionCounters
	for i, event := range events {
		if !event.Committed || !event.CompletedWait || !event.HostReadback || event.Encoders <= 0 || !event.TimingAvailable || math.IsNaN(event.GPUMilliseconds) || math.IsInf(event.GPUMilliseconds, 0) || event.GPUMilliseconds <= 0 || math.IsNaN(event.WaitMilliseconds) || math.IsInf(event.WaitMilliseconds, 0) || event.WaitMilliseconds < 0 {
			return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: fmt.Sprintf("event %d (%s) lacks a complete committed/waited/readback/timed lifecycle", i+1, event.Operation)}
		}
		if _, err := event.EncoderTiming.EncoderCoverage(event.Encoders); err != nil {
			return ExecutionCounters{}, ExecutionCountersIncompleteError{Detail: fmt.Sprintf("event %d: %v", i+1, err)}
		}
		out.CommandBuffers++
		out.Encoders += event.Encoders
		out.DispatchMilliseconds += event.GPUMilliseconds
		out.WaitMilliseconds += event.WaitMilliseconds
	}
	return out, nil
}

// Snapshot returns only events owned by this observation.
func (o *ExecutionObservation) Snapshot() (ExecutionSnapshot, error) {
	if o == nil || !o.available {
		return ExecutionSnapshot{}, ExecutionEventsUnavailableError{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	return ExecutionSnapshot{Events: cloneExecutionEvents(o.events)}, nil
}

// rawEncoderSample contains actual stage-boundary timestamp data on supported
// Apple GPUs. Keep integer nanoseconds until subtracting each segment's origin.
type rawEncoderSample struct {
	segment, ordinal int
	start, end       uint64
	operation        string
}

func encoderTimingFromSamples(samples []rawEncoderSample, segments int) *ExecutionEncoderTiming {
	unavailable := func(reason string) *ExecutionEncoderTiming {
		return &ExecutionEncoderTiming{Status: "unavailable", Reason: reason}
	}
	if len(samples) == 0 || len(samples) > executionEncoderSampleLimit || segments <= 0 || segments > executionEncoderSampleLimit {
		return unavailable("no bounded native encoder samples")
	}
	out := &ExecutionEncoderTiming{Status: "segmented", Reason: "partial instrumented-site coverage; segment windows are not a whole-token timeline"}
	lastSegment, ordinal := 0, 0
	for _, sample := range samples {
		if sample.segment < lastSegment || sample.segment < 1 || sample.segment > segments || sample.operation == "" || sample.start == 0 || sample.end <= sample.start || sample.end == ^uint64(0) {
			return unavailable("invalid native encoder sample")
		}
		if sample.segment != lastSegment {
			ordinal = 0
		}
		if sample.ordinal != ordinal+1 {
			return unavailable("invalid native encoder ordering")
		}
		lastSegment, ordinal = sample.segment, sample.ordinal
	}
	cursor := 0
	for segment := 1; segment <= segments; segment++ {
		begin := cursor
		var origin, end uint64
		for cursor < len(samples) && samples[cursor].segment == segment {
			sample := samples[cursor]
			if cursor == begin || sample.start < origin {
				origin = sample.start
			}
			if sample.end > end {
				end = sample.end
			}
			cursor++
		}
		timing := ExecutionEncoderTiming{Status: "unavailable", Reason: "segment has no instrumented encoder samples"}
		if cursor > begin {
			timing = ExecutionEncoderTiming{
				Status: "partial", Reason: "only q4k and qwen35 graph encoder sites sampled; GDN and other producers are not counted",
				ClockDomain: "metal-gpu-counter", WindowScope: "sampled-encoder-envelope",
				EncoderCensus: "q4k-and-qwen35-graph-sites", UnmeasuredPrefix: true, UnmeasuredSuffix: true,
				UncoveredEncoderScope: "gdn-and-other-graph-producers", WindowEndMilliseconds: float64(end-origin) / 1e6,
			}
			for _, sample := range samples[begin:cursor] {
				timing.Spans = append(timing.Spans, ExecutionEncoderSpan{sample.ordinal, sample.operation, float64(sample.start-origin) / 1e6, float64(sample.end-origin) / 1e6})
			}
		}
		out.Segments = append(out.Segments, ExecutionEncoderSegment{uint64(segment), cursor - begin, timing})
	}
	if _, err := out.EncoderCoverage(0); err != nil {
		return unavailable(err.Error())
	}
	return out
}
