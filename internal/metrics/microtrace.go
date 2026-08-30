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
	"sync/atomic"
	"time"
)

// Outcome counters (#5838, follow-on to the micro-context fabric spine 28846558d0).
//
// The spans above answer "what did ONE agent do"; they did not answer "how did the
// fabric do". A fleet run's success/refusal/error split lived only in the live host's
// in-process audit sink, so once a run persisted its traces (--trace-out) the outcome
// distribution was unrecoverable without hand-parsing the JSONL — a regression in the
// refusal rate surfaced only as a bug report. The fold below derives the outcome from
// the verdict legs already recorded, so it costs no new span kind, no new file format,
// and no new dashboard: `fak micro` and `fak micro trace <id>` render it in place.
//   - Promotion evidence: the counts fold identically from a live tracer and from a
//     tracer reconstructed out of persisted JSONL (TestMicroOutcomeCountsSurviveJSONL),
//     which is what makes them queryable after the run that produced them exited.
//     Promote past the Mock planner once #2030 wires real gateway verdicts in — the
//     vocabulary below is already keyed on verdict strings, not on the Mock's ALLOW.
//   - Demotion / retirement: retire with the tracer itself (see the MicroTracer note
//     above), or fold into an OpenTelemetry counter export if in-process spans are
//     superseded — the classification would move, the vocabulary would not.
//   - Invalidating assumption: a verdict string outside the closed refusal/error
//     vocabularies is read as an ADMIT leg. That is the safe default while the Mock
//     planner only ever emits ALLOW, but if a real gateway starts emitting a refusal
//     spelled some third way, it would be silently counted as a success. The unknown
//     bucket covers only the no-verdict-at-all case, not a misspelled refusal.

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

// MicroSpanEvent marks the lifecycle record emitted by a scoped duration.
// Empty remains valid for existing one-record spans.
type MicroSpanEvent string

const (
	MicroSpanStart MicroSpanEvent = "span_start"
	MicroSpanEnd   MicroSpanEvent = "span_end"
)

// MicroSpan is one leg of a per-agent trace timeline. Seq is assigned by the
// tracer in record order within a trace; the remaining fields are populated by
// kind (a step carries Tokens, a seat carries Seat, a verdict carries Verdict).
type MicroSpan struct {
	Seq     int            `json:"seq"`
	Kind    MicroSpanKind  `json:"kind"`
	Event   MicroSpanEvent `json:"event,omitempty"`
	SpanID  uint64         `json:"span_id,omitempty"`
	Label   string         `json:"label,omitempty"`
	Tokens  int            `json:"tokens,omitempty"`
	Seat    string         `json:"seat,omitempty"`
	Verdict string         `json:"verdict,omitempty"`
	Dur     time.Duration  `json:"dur_ns,omitempty"`
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

// MicroOutcome is the terminal classification of ONE micro-context invocation.
// It is derived from the trace's verdict legs rather than stored alongside them, so
// a tracer reconstructed from persisted JSONL folds to exactly the counts the live
// host would have reported — no extra field to write, migrate, or keep in sync.
type MicroOutcome string

const (
	// MicroOutcomeSuccess: the invocation ran its legs and the kernel admitted them.
	MicroOutcomeSuccess MicroOutcome = "success"
	// MicroOutcomeRefusal: the kernel refused at least one of this agent's calls.
	// A refusal is a WORKING kernel, not a fault — it is counted apart from error
	// precisely so a rising refusal rate is legible without looking like breakage.
	MicroOutcomeRefusal MicroOutcome = "refusal"
	// MicroOutcomeError: the invocation hit a hard failure (gateway/transport/step).
	MicroOutcomeError MicroOutcome = "error"
	// MicroOutcomeUnknown: the trace carries no verdict leg at all, so its outcome
	// is not evidence either way. Kept as its own bucket rather than folded into
	// success — an untraced agent must not inflate the success count.
	MicroOutcomeUnknown MicroOutcome = "unknown"
)

// microRefusalVerdicts / microErrorVerdicts are the closed vocabularies the fold
// classifies against, matched case-insensitively. Any other non-empty verdict is an
// ADMIT leg (the Mock planner's ALLOW, and whatever a real gateway spells success).
var (
	microRefusalVerdicts = map[string]bool{"DENY": true, "DENIED": true, "REFUSE": true, "REFUSED": true, "BLOCK": true, "BLOCKED": true}
	microErrorVerdicts   = map[string]bool{"ERROR": true, "ERRORED": true, "FAIL": true, "FAILED": true}
)

// Outcome classifies this whole trace. Precedence is error > refusal > success: an
// agent that was admitted for two turns and then failed on the third is an error, and
// one that was admitted twice and refused once is a refusal — the terminal fact about
// the invocation, not the majority of its legs.
func (t MicroTrace) Outcome() MicroOutcome {
	sawVerdict, refused := false, false
	for _, s := range t.Spans {
		if s.Verdict == "" {
			continue
		}
		sawVerdict = true
		switch v := strings.ToUpper(strings.TrimSpace(s.Verdict)); {
		case microErrorVerdicts[v]:
			return MicroOutcomeError
		case microRefusalVerdicts[v]:
			refused = true
		}
	}
	switch {
	case !sawVerdict:
		return MicroOutcomeUnknown
	case refused:
		return MicroOutcomeRefusal
	}
	return MicroOutcomeSuccess
}

// MicroOutcomeCounts is the fabric-level fold: how many invocations succeeded, were
// refused, errored, or recorded no verdict. This is the counter surface #5838 names.
type MicroOutcomeCounts struct {
	Success int `json:"success"`
	Refusal int `json:"refusal"`
	Error   int `json:"error"`
	Unknown int `json:"unknown"`
}

// Total is the number of invocations folded — every trace lands in exactly one
// bucket, so the buckets always sum to the trace count.
func (c MicroOutcomeCounts) Total() int {
	return c.Success + c.Refusal + c.Error + c.Unknown
}

// Add counts one more invocation under the given outcome.
func (c *MicroOutcomeCounts) Add(o MicroOutcome) {
	switch o {
	case MicroOutcomeSuccess:
		c.Success++
	case MicroOutcomeRefusal:
		c.Refusal++
	case MicroOutcomeError:
		c.Error++
	default:
		c.Unknown++
	}
}

// Render formats the counts as one operator-readable line. The unknown bucket is
// printed only when it is non-zero, so the ordinary all-traced run reads clean while
// a partially-traced one cannot hide its gap.
func (c MicroOutcomeCounts) Render() string {
	line := fmt.Sprintf("success=%d refusal=%d error=%d", c.Success, c.Refusal, c.Error)
	if c.Unknown != 0 {
		line += fmt.Sprintf(" unknown=%d", c.Unknown)
	}
	return line + fmt.Sprintf("  (%d invocation(s))", c.Total())
}

// Outcomes folds every known trace into the fabric-level counters. It is the whole
// query surface: one call answers "how did this fleet run go", live or replayed.
func (t *MicroTracer) Outcomes() MicroOutcomeCounts {
	var c MicroOutcomeCounts
	for _, tr := range t.Traces() {
		c.Add(tr.Outcome())
	}
	return c
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
	now    func() time.Time
}

// NewMicroTracer returns an empty tracer.
func NewMicroTracer() *MicroTracer {
	return newMicroTracerWithClock(time.Now)
}

func newMicroTracerWithClock(now func() time.Time) *MicroTracer {
	return &MicroTracer{traces: map[string]*MicroTrace{}, now: now}
}

func (t *MicroTracer) nowTime() time.Time {
	if t.now != nil {
		return t.now()
	}
	return time.Now()
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

var nextMicroSpanID atomic.Uint64

// MicroSpanScope emits one start record immediately and one terminal record from
// End. It adapts the paired-lifecycle contract from Modular SpanGuard
// (Support/include/Support/SpanGuard.h@1c9fd2e0) to MicroTracer records.
type MicroSpanScope struct {
	tracer  *MicroTracer
	traceID string
	span    MicroSpan
	started time.Time
	once    sync.Once
}

// Scope starts a paired duration span. The returned scope is intended for:
//
//	scope := tracer.Scope(traceID, span)
//	defer scope.End()
//
// End is idempotent, so explicit cleanup plus a deferred fallback still emits
// exactly one terminal record. IDs are process-wide and safe under concurrency.
func (t *MicroTracer) Scope(traceID string, span MicroSpan) *MicroSpanScope {
	started := t.nowTime()
	span.Event = MicroSpanStart
	span.SpanID = nextMicroSpanID.Add(1)
	if span.SpanID == 0 {
		span.SpanID = nextMicroSpanID.Add(1)
	}
	span.Dur = 0
	t.Record(traceID, span)
	return &MicroSpanScope{
		tracer:  t,
		traceID: traceID,
		span:    span,
		started: started,
	}
}

// ID returns the identifier shared by this scope's start and terminal records.
func (s *MicroSpanScope) ID() uint64 {
	if s == nil {
		return 0
	}
	return s.span.SpanID
}

// End records the terminal half of the pair. time.Now carries a monotonic clock
// reading in normal operation; the defensive clamp also keeps injected or
// platform clocks from producing a negative duration.
func (s *MicroSpanScope) End() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		dur := s.tracer.nowTime().Sub(s.started)
		if dur < 0 {
			dur = 0
		}
		span := s.span
		span.Event = MicroSpanEnd
		span.Dur = dur
		s.tracer.Record(s.traceID, span)
	})
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
	fmt.Fprintf(&b, "  outcome: %s\n", t.Outcome())
	if v := t.Verdicts(); len(v) > 0 {
		fmt.Fprintf(&b, "  verdicts: %s\n", strings.Join(v, ", "))
	}
	for _, s := range t.Spans {
		line := fmt.Sprintf("  #%-3d %-9s", s.Seq, s.Kind)
		if s.Label != "" {
			line += " " + s.Label
		}
		var tags []string
		if s.Event != "" {
			tags = append(tags, "event="+string(s.Event))
		}
		if s.SpanID != 0 {
			tags = append(tags, fmt.Sprintf("span_id=%d", s.SpanID))
		}
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
