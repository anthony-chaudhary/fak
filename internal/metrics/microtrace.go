package metrics

// microtrace.go — structured per-microagent tracing (#2031, epic #2000 M32).
//
// The native in-process microagent host (internal/microagent) runs 1000s of
// goroutine agents in ONE process, so the detached path's one-log-file-per-process
// isolation (resolve-<issue>-<stamp>.log) does not apply — a single interleaved
// stream is unreadable. This primitive is the "spans for the rest" half of the
// issue: the security-relevant rows (spawn/admission/verdict) already land in the
// shared hash-chained audit ring (M11, internal/journal via microagent.JournalSink);
// the full per-agent timeline (step legs, tool calls, seat/admission decisions,
// adjudication verdicts, tokens) is carried here as structured spans multiplexed by
// trace id. Separability is by construction — ONE trace per trace id — so a readout
// (`fak micro trace <id>`) pulls exactly one agent's timeline out of the interleaved
// fleet.
//
// Generation intent: gen/second-next architectural OPTION (#2031, part of #2002).
// This is an observability primitive behind the explicit `fak micro` gate — nothing
// in the default serve/guard/dispatch path constructs a MicroTracer.
//   - Promotion evidence: the tracer multiplexes N concurrent agents' spans keyed by
//     trace id and renders one agent's timeline in isolation (TestMicroTracer*), and
//     round-trips through JSONL so a separate `fak micro trace` process can read a
//     persisted fleet run. Promote past the Mock planner once the dispatch path can
//     target the host (#2030) and real gateway verdicts/token counts flow in.
//   - Demotion / retirement: retire this primitive if the host is retired (#2033
//     shows per-agent cost is provider-seat-bound, not local process weight, so the
//     goroutine host buys no density), or if OpenTelemetry-style external tracing
//     supersedes an in-process span store.
//   - Invalidating assumption: spans live in a bounded in-memory map for one host's
//     lifetime (persisted only on explicit --trace-out). If a fleet must retain full
//     traces for 1000s of long-lived agents past a host restart, this must grow a
//     ring-bounded or on-disk span store rather than an unbounded map.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"
)

// MicroSpanKind classifies one leg of a per-microagent trace timeline. The
// security-relevant kinds (admission/seat/verdict) mirror rows the shared audit
// ring also carries; the timeline carries every kind so the readout is complete.
type MicroSpanKind string

const (
	SpanStep      MicroSpanKind = "step"      // one agent step / model turn
	SpanTool      MicroSpanKind = "tool"      // a tool call the step made
	SpanAdmission MicroSpanKind = "admission" // an admission (token/concurrency) decision
	SpanSeat      MicroSpanKind = "seat"      // a slot/seat acquire or release
	SpanVerdict   MicroSpanKind = "verdict"   // an adjudication verdict
)

// MicroSpan is one leg of a per-agent trace timeline. Seq is assigned by the
// tracer in record order within a trace; the remaining fields are populated by
// kind (a step carries Tokens, a seat carries Seat, a verdict carries Verdict).
type MicroSpan struct {
	Seq     int           `json:"seq"`
	Kind    MicroSpanKind `json:"kind"`
	Label   string        `json:"label,omitempty"`
	Tokens  int           `json:"tokens,omitempty"`
	Seat    string        `json:"seat,omitempty"`
	Verdict string        `json:"verdict,omitempty"`
	Dur     time.Duration `json:"dur_ns,omitempty"`
}

// MicroTrace is one microagent's ordered span timeline, keyed by its trace id
// (the agent id, which is also its session.Table TraceID in the host).
type MicroTrace struct {
	TraceID string      `json:"trace_id"`
	Spans   []MicroSpan `json:"spans"`
}

// Tokens sums the token count across every span in the trace.
func (t MicroTrace) Tokens() int {
	n := 0
	for _, s := range t.Spans {
		n += s.Tokens
	}
	return n
}

// Verdicts returns the distinct verdicts recorded on the trace, in first-seen
// order — the short answer to "what did the kernel decide for this agent".
func (t MicroTrace) Verdicts() []string {
	var out []string
	seen := map[string]bool{}
	for _, s := range t.Spans {
		if s.Verdict == "" || seen[s.Verdict] {
			continue
		}
		seen[s.Verdict] = true
		out = append(out, s.Verdict)
	}
	return out
}

// MicroTracer multiplexes many per-agent traces in ONE single-process host, keyed
// by trace id. It is safe for concurrent use: the host drives K agent goroutines
// that each Record spans against their own trace id, so the mutex protects the
// shared map. Separability is structural — one entry per trace id — so Trace/Render
// pull exactly one agent's timeline out of the interleaved fleet.
type MicroTracer struct {
	mu     sync.Mutex
	traces map[string]*MicroTrace
	order  []string // first-seen trace ids, for a stable listing
}

// NewMicroTracer returns an empty tracer.
func NewMicroTracer() *MicroTracer {
	return &MicroTracer{traces: map[string]*MicroTrace{}}
}

// Record appends one span to the trace named by id, assigning it the next Seq in
// that trace. A new id is registered on first use. Safe for concurrent callers.
func (t *MicroTracer) Record(id string, span MicroSpan) {
	t.mu.Lock()
	defer t.mu.Unlock()
	tr := t.traces[id]
	if tr == nil {
		tr = &MicroTrace{TraceID: id}
		t.traces[id] = tr
		t.order = append(t.order, id)
	}
	span.Seq = len(tr.Spans)
	tr.Spans = append(tr.Spans, span)
}

// Trace returns a copy of one agent's timeline and whether it exists.
func (t *MicroTracer) Trace(id string) (MicroTrace, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	tr := t.traces[id]
	if tr == nil {
		return MicroTrace{}, false
	}
	return MicroTrace{TraceID: tr.TraceID, Spans: append([]MicroSpan(nil), tr.Spans...)}, true
}

// IDs returns the known trace ids in first-seen order.
func (t *MicroTracer) IDs() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.order...)
}

// Traces returns every trace, sorted by trace id (stable for a report).
func (t *MicroTracer) Traces() []MicroTrace {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]MicroTrace, 0, len(t.traces))
	for _, id := range t.order {
		tr := t.traces[id]
		out = append(out, MicroTrace{TraceID: tr.TraceID, Spans: append([]MicroSpan(nil), tr.Spans...)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].TraceID < out[j].TraceID })
	return out
}

// Render formats one agent's timeline as a human-readable readout — the body of
// `fak micro trace <id>` (legs, tokens, seat, verdicts). It reports false when the
// id names no trace.
func (t *MicroTracer) Render(id string) (string, bool) {
	tr, ok := t.Trace(id)
	if !ok {
		return "", false
	}
	return tr.Render(), true
}

// Render formats this single trace's timeline. Standalone so a trace loaded from
// JSONL renders identically to one read live from a tracer.
func (t MicroTrace) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "trace %s — %d span(s), %d token(s)\n", t.TraceID, len(t.Spans), t.Tokens())
	if v := t.Verdicts(); len(v) > 0 {
		fmt.Fprintf(&b, "  verdicts: %s\n", strings.Join(v, ", "))
	}
	for _, s := range t.Spans {
		line := fmt.Sprintf("  #%-3d %-9s", s.Seq, s.Kind)
		if s.Label != "" {
			line += " " + s.Label
		}
		var tags []string
		if s.Tokens != 0 {
			tags = append(tags, fmt.Sprintf("tokens=%d", s.Tokens))
		}
		if s.Seat != "" {
			tags = append(tags, "seat="+s.Seat)
		}
		if s.Verdict != "" {
			tags = append(tags, "verdict="+s.Verdict)
		}
		if s.Dur != 0 {
			tags = append(tags, "dur="+s.Dur.String())
		}
		if len(tags) > 0 {
			line += "  (" + strings.Join(tags, " ") + ")"
		}
		b.WriteString(line + "\n")
	}
	return b.String()
}

// WriteJSONL writes every trace as one JSON object per line (sorted by trace id),
// so a fleet run can persist its traces (`--trace-out`) for a later, separate-process
// readout (`fak micro trace <id> --trace-in`).
func (t *MicroTracer) WriteJSONL(w io.Writer) error {
	bw := bufio.NewWriter(w)
	for _, tr := range t.Traces() {
		raw, err := json.Marshal(tr)
		if err != nil {
			return err
		}
		if _, err := bw.Write(raw); err != nil {
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// ReadTracesJSONL reconstructs a tracer from JSONL written by WriteJSONL. Blank
// lines are skipped; a malformed line is a hard error (a truncated trace file is
// not silently half-read).
func ReadTracesJSONL(r io.Reader) (*MicroTracer, error) {
	t := NewMicroTracer()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var tr MicroTrace
		if err := json.Unmarshal([]byte(raw), &tr); err != nil {
			return nil, fmt.Errorf("microtrace: parse line %d: %w", line, err)
		}
		if tr.TraceID == "" {
			return nil, fmt.Errorf("microtrace: line %d has empty trace_id", line)
		}
		cp := MicroTrace{TraceID: tr.TraceID, Spans: append([]MicroSpan(nil), tr.Spans...)}
		t.traces[tr.TraceID] = &cp
		t.order = append(t.order, tr.TraceID)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return t, nil
}
