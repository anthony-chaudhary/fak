package servingsim

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
)

// TraceEvent represents an event in the Chrome / Perfetto Trace Event Format.
//
// Invariant: TS is non-negative timestamp in microseconds; Dur is non-negative when present.
// Guard: Ph must match Chrome trace specification phases ("X", "B", "E", "i", "C").
type TraceEvent struct {
	Cat  string         `json:"cat"`
	PID  int            `json:"pid"`
	TID  int            `json:"tid"`
	TS   float64        `json:"ts"` // timestamp in microseconds
	Ph   string         `json:"ph"` // phase: "X" (complete), "B", "E", "i", "C" (counter)
	Name string         `json:"name"`
	Dur  float64        `json:"dur,omitempty"` // duration in microseconds (for "X" phase)
	Args map[string]any `json:"args,omitempty"`
}

// ChromeTraceContainer formats trace events as a top-level Perfetto / Chrome Tracing document.
//
// Invariant: TraceEvents contains valid Chrome trace events serializable to standard Perfetto JSON.
type ChromeTraceContainer struct {
	TraceEvents     []TraceEvent `json:"traceEvents"`
	DisplayTimeUnit string       `json:"displayTimeUnit,omitempty"`
}

// workloadLine represents raw JSONL trace entry fields.
type workloadLine struct {
	ID            string   `json:"id"`
	RequestID     string   `json:"request_id"`
	ArrivalTimeMS *float64 `json:"arrival_time_ms"`
	ArrivalTime   *float64 `json:"arrival_time"`
	PromptTokens  int      `json:"prompt_tokens"`
	OutputTokens  int      `json:"output_tokens"`
	OutputTarget  int      `json:"output_target"`
}

// ReadTraceJSONL reads workload traces from a JSONL stream, returning initialized RequestState objects.
//
// Invariant: Preserves parsed request order while populating IDs and positive token counts.
// Guard: Empty lines and comments are skipped; negative timestamps or non-positive token counts return an error fail-closed.
func ReadTraceJSONL(r io.Reader) ([]RequestState, error) {
	scanner := bufio.NewScanner(r)
	var requests []RequestState
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}

		var raw workloadLine
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("servingsim: invalid JSON at line %d: %w", lineNum, err)
		}

		id := raw.ID
		if id == "" {
			id = raw.RequestID
		}
		if id == "" {
			id = fmt.Sprintf("req-%d", lineNum)
		}

		var arrivalMS float64
		if raw.ArrivalTimeMS != nil {
			arrivalMS = *raw.ArrivalTimeMS
		} else if raw.ArrivalTime != nil {
			arrivalMS = *raw.ArrivalTime
		}
		if arrivalMS < 0 {
			return nil, fmt.Errorf("servingsim: negative arrival_time_ms at line %d: %f", lineNum, arrivalMS)
		}

		promptTokens := raw.PromptTokens
		if promptTokens <= 0 {
			return nil, fmt.Errorf("servingsim: prompt_tokens must be positive at line %d, got %d", lineNum, promptTokens)
		}

		outputTokens := raw.OutputTokens
		if outputTokens <= 0 {
			outputTokens = raw.OutputTarget
		}
		if outputTokens <= 0 {
			return nil, fmt.Errorf("servingsim: output_tokens must be positive at line %d, got %d", lineNum, outputTokens)
		}

		requests = append(requests, RequestState{
			ID:            id,
			ArrivalTimeMS: arrivalMS,
			PromptTokens:  promptTokens,
			OutputTarget:  outputTokens,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("servingsim: scanning error: %w", err)
	}

	return requests, nil
}

// WriteTraceJSONL serializes a slice of requests into JSONL format.
//
// Invariant: Each request is encoded as a single-line JSON record terminated by a newline.
// Guard: I/O write or marshaling errors are returned immediately fail-closed.
func WriteTraceJSONL(w io.Writer, requests []RequestState) error {
	for _, req := range requests {
		data, err := json.Marshal(struct {
			ID            string  `json:"id"`
			ArrivalTimeMS float64 `json:"arrival_time_ms"`
			PromptTokens  int     `json:"prompt_tokens"`
			OutputTokens  int     `json:"output_tokens"`
		}{
			ID:            req.ID,
			ArrivalTimeMS: req.ArrivalTimeMS,
			PromptTokens:  req.PromptTokens,
			OutputTokens:  req.OutputTarget,
		})
		if err != nil {
			return fmt.Errorf("servingsim: marshal error: %w", err)
		}
		if _, err := w.Write(append(data, '\n')); err != nil {
			return fmt.Errorf("servingsim: write error: %w", err)
		}
	}
	return nil
}

// TraceCollector gathers TraceEvents in memory during simulation.
//
// Invariant: All mutations and reads are concurrency-safe under internal mutex locking.
// Guard: Unbounded event streams consume memory proportional to total simulated steps.
type TraceCollector struct {
	mu     sync.Mutex
	events []TraceEvent
}

// NewTraceCollector creates an empty trace collector with pre-allocated buffer capacity.
//
// Invariant: Returns an initialized, non-nil collector ready for concurrent recording.
func NewTraceCollector() *TraceCollector {
	return &TraceCollector{
		events: make([]TraceEvent, 0, 128),
	}
}

// Add appends a trace event to the collector.
//
// Invariant: Concurrency-safe under internal mutex; preserves FIFO append order.
func (c *TraceCollector) Add(ev TraceEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
}

// Events returns a snapshot slice of all recorded events.
//
// Invariant: Returns an isolated copy preventing data races on subsequent collector mutations.
func (c *TraceCollector) Events() []TraceEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([]TraceEvent, len(c.events))
	copy(res, c.events)
	return res
}

// Reset clears collected events while retaining underlying slice capacity.
//
// Invariant: Concurrency-safe under internal mutex; leaves zero events in the buffer.
func (c *TraceCollector) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = c.events[:0]
}

// RecordStep records a hardware execution step duration in microsecond units.
//
// Invariant: Event phase is set to "X" (complete event) with TS and Dur converted from milliseconds to microseconds.
func (c *TraceCollector) RecordStep(name, cat string, startMS, durationMS float64, pid, tid int, args map[string]any) {
	c.Add(TraceEvent{
		Cat:  cat,
		PID:  pid,
		TID:  tid,
		TS:   startMS * 1000.0,    // convert ms to microseconds
		Dur:  durationMS * 1000.0, // convert ms to microseconds
		Ph:   "X",
		Name: name,
		Args: args,
	})
}

// RecordCounter records a counter metric value at a point in time.
//
// Invariant: Event phase is set to "C" with timestamp converted from milliseconds to microseconds.
func (c *TraceCollector) RecordCounter(name, cat string, timeMS float64, pid, tid int, args map[string]any) {
	c.Add(TraceEvent{
		Cat:  cat,
		PID:  pid,
		TID:  tid,
		TS:   timeMS * 1000.0,
		Ph:   "C",
		Name: name,
		Args: args,
	})
}

// RecordInstant records a point-in-time instant event.
//
// Invariant: Event phase is set to "i" with timestamp converted from milliseconds to microseconds.
func (c *TraceCollector) RecordInstant(name, cat string, timeMS float64, pid, tid int, args map[string]any) {
	c.Add(TraceEvent{
		Cat:  cat,
		PID:  pid,
		TID:  tid,
		TS:   timeMS * 1000.0,
		Ph:   "i",
		Name: name,
		Args: args,
	})
}

// ExportChromeTrace exports trace events wrapped in a standard Chrome / Perfetto container JSON.
//
// Invariant: Output JSON adheres to Perfetto/Chrome trace schema with displayTimeUnit set to "ms".
// Guard: Encoding failures on invalid JSON writers return an error fail-closed.
func ExportChromeTrace(w io.Writer, events []TraceEvent) error {
	container := ChromeTraceContainer{
		TraceEvents:     events,
		DisplayTimeUnit: "ms",
	}
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(container); err != nil {
		return fmt.Errorf("servingsim: failed to encode Chrome trace JSON: %w", err)
	}
	return nil
}

// ExportTraceEventsJSON exports the raw array of trace events as JSON.
//
// Invariant: Output is an indented JSON array of TraceEvent objects.
// Guard: Encoding failures return an error fail-closed.
func ExportTraceEventsJSON(w io.Writer, events []TraceEvent) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(events); err != nil {
		return fmt.Errorf("servingsim: failed to encode trace events array JSON: %w", err)
	}
	return nil
}

// ExportTrace writes Chrome trace format to the given writer.
//
// Invariant: Forwards directly to ExportChromeTrace for standard Perfetto viewing.
// Guard: Encoding errors are returned fail-closed.
func ExportTrace(w io.Writer, events []TraceEvent) error {
	return ExportChromeTrace(w, events)
}
