package metrics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMicroTracerSeparability is the acceptance witness: many agents record spans
// into ONE tracer concurrently (as the single-process host drives K goroutines),
// and each agent's timeline pulls out cleanly by trace id — no cross-contamination.
func TestMicroTracerSeparability(t *testing.T) {
	tr := NewMicroTracer()
	const agents, steps = 50, 4
	var wg sync.WaitGroup
	for a := 0; a < agents; a++ {
		wg.Add(1)
		go func(a int) {
			defer wg.Done()
			id := micoID(a)
			for s := 0; s < steps; s++ {
				tr.Record(id, MicroSpan{Kind: SpanSeat, Seat: "slot", Label: "acquire"})
				tr.Record(id, MicroSpan{Kind: SpanStep, Label: "turn", Tokens: 10 + a})
				tr.Record(id, MicroSpan{Kind: SpanVerdict, Verdict: "ALLOW"})
			}
		}(a)
	}
	wg.Wait()

	if got := len(tr.IDs()); got != agents {
		t.Fatalf("IDs: got %d trace ids, want %d", got, agents)
	}
	for a := 0; a < agents; a++ {
		id := micoID(a)
		got, ok := tr.Trace(id)
		if !ok {
			t.Fatalf("Trace(%s): missing", id)
		}
		if want := steps * 3; len(got.Spans) != want {
			t.Fatalf("Trace(%s): got %d spans, want %d", id, len(got.Spans), want)
		}
		// Separability: every span in THIS trace carries THIS agent's token
		// signature (10+a); a leaked span from another agent would break it.
		for _, s := range got.Spans {
			if s.Kind == SpanStep && s.Tokens != 10+a {
				t.Fatalf("Trace(%s): step token %d, want %d (cross-agent leak?)", id, s.Tokens, 10+a)
			}
		}
		// Seq is dense and in record order within the trace.
		for i, s := range got.Spans {
			if s.Seq != i {
				t.Fatalf("Trace(%s): span %d has Seq %d", id, i, s.Seq)
			}
		}
		if want := steps * (10 + a); got.Tokens() != want {
			t.Fatalf("Trace(%s): Tokens()=%d, want %d", id, got.Tokens(), want)
		}
		if v := got.Verdicts(); len(v) != 1 || v[0] != "ALLOW" {
			t.Fatalf("Trace(%s): Verdicts()=%v, want [ALLOW]", id, v)
		}
	}
	if _, ok := tr.Trace("nope"); ok {
		t.Fatal("Trace of unknown id reported present")
	}
}

func TestMicroTracerRender(t *testing.T) {
	tr := NewMicroTracer()
	tr.Record("micro-000", MicroSpan{Kind: SpanSeat, Seat: "slot", Label: "acquire"})
	tr.Record("micro-000", MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 42})
	tr.Record("micro-000", MicroSpan{Kind: SpanVerdict, Verdict: "ALLOW"})

	out, ok := tr.Render("micro-000")
	if !ok {
		t.Fatal("Render: trace not found")
	}
	for _, want := range []string{"trace micro-000", "3 span(s)", "42 token(s)", "verdicts: ALLOW", "seat=slot", "verdict=ALLOW", "tokens=42"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Render missing %q:\n%s", want, out)
		}
	}
	if _, ok := tr.Render("ghost"); ok {
		t.Fatal("Render of unknown id reported present")
	}
}

// TestMicroTracerJSONLRoundTrip proves a fleet run can persist its traces and a
// separate-process readout reconstructs them identically (the --trace-out /
// --trace-in path behind `fak micro trace`).
func TestMicroTracerJSONLRoundTrip(t *testing.T) {
	src := NewMicroTracer()
	src.Record("micro-001", MicroSpan{Kind: SpanStep, Label: "turn", Tokens: 7})
	src.Record("micro-000", MicroSpan{Kind: SpanSeat, Seat: "slot"})
	src.Record("micro-000", MicroSpan{Kind: SpanVerdict, Verdict: "DENY"})

	var buf bytes.Buffer
	if err := src.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	// Two traces, one line each, sorted by id (micro-000 before micro-001).
	if n := strings.Count(strings.TrimSpace(buf.String()), "\n"); n != 1 {
		t.Fatalf("WriteJSONL: got %d newlines, want 1 (two trace lines)", n)
	}

	got, err := ReadTracesJSONL(&buf)
	if err != nil {
		t.Fatalf("ReadTracesJSONL: %v", err)
	}
	want, _ := src.Render("micro-000")
	round, ok := got.Render("micro-000")
	if !ok {
		t.Fatal("round-trip lost micro-000")
	}
	if round != want {
		t.Fatalf("round-trip render mismatch:\n got %q\nwant %q", round, want)
	}

	if _, err := ReadTracesJSONL(strings.NewReader("{bad json")); err == nil {
		t.Fatal("ReadTracesJSONL accepted malformed line")
	}
	if _, err := ReadTracesJSONL(strings.NewReader(`{"spans":[]}`)); err == nil {
		t.Fatal("ReadTracesJSONL accepted empty trace_id")
	}
}

func micoID(a int) string {
	return fmt.Sprintf("micro-%03d", a)
}

func TestMicroTraceWritePerfetto(t *testing.T) {
	tr := &MicroTrace{
		TraceID: "micro-042",
		Spans: []MicroSpan{
			{
				Seq:   0,
				Kind:  SpanSeat,
				Label: "acquire",
				Seat:  "slot-pool/4",
				Dur:   10 * time.Millisecond,
			},
			{
				Seq:    1,
				Kind:   SpanStep,
				Label:  "turn 1",
				Tokens: 42,
				Dur:    25 * time.Millisecond,
			},
			{
				Seq:     2,
				Kind:    SpanVerdict,
				Label:   "mock-planner",
				Verdict: "ALLOW",
				Dur:     5 * time.Millisecond,
			},
		},
	}

	var buf bytes.Buffer
	if err := tr.WritePerfetto(&buf); err != nil {
		t.Fatalf("WritePerfetto: %v", err)
	}

	// 1. Verify parseable into PerfettoTraceContainer struct
	var container PerfettoTraceContainer
	if err := json.Unmarshal(buf.Bytes(), &container); err != nil {
		t.Fatalf("json.Unmarshal into PerfettoTraceContainer: %v", err)
	}

	if container.DisplayTimeUnit != "ms" {
		t.Fatalf("displayTimeUnit = %q, want %q", container.DisplayTimeUnit, "ms")
	}

	if len(container.TraceEvents) != len(tr.Spans) {
		t.Fatalf("traceEvents count = %d, want %d", len(container.TraceEvents), len(tr.Spans))
	}

	expected := []struct {
		name string
		cat  string
		ph   string
		ts   float64
		dur  float64
		pid  string
		tid  int
		args map[string]any
	}{
		{
			name: "acquire",
			cat:  "microtrace",
			ph:   "X",
			ts:   0.0,
			dur:  10000.0, // 10ms in microseconds
			pid:  "micro-042",
			tid:  0,
			args: map[string]any{
				"seat":  "slot-pool/4",
				"kind":  "seat",
				"label": "acquire",
			},
		},
		{
			name: "turn 1",
			cat:  "microtrace",
			ph:   "X",
			ts:   10000.0,
			dur:  25000.0, // 25ms in microseconds
			pid:  "micro-042",
			tid:  1,
			args: map[string]any{
				"tokens": float64(42),
				"kind":   "step",
				"label":  "turn 1",
				"seq":    float64(1),
			},
		},
		{
			name: "mock-planner",
			cat:  "microtrace",
			ph:   "X",
			ts:   35000.0,
			dur:  5000.0, // 5ms in microseconds
			pid:  "micro-042",
			tid:  2,
			args: map[string]any{
				"verdict": "ALLOW",
				"kind":    "verdict",
				"label":   "mock-planner",
				"seq":     float64(2),
			},
		},
	}

	for i, ev := range container.TraceEvents {
		exp := expected[i]
		if ev.Name != exp.name {
			t.Errorf("event[%d].Name = %q, want %q", i, ev.Name, exp.name)
		}
		if ev.Cat != exp.cat {
			t.Errorf("event[%d].Cat = %q, want %q", i, ev.Cat, exp.cat)
		}
		if ev.Ph != exp.ph {
			t.Errorf("event[%d].Ph = %q, want %q", i, ev.Ph, exp.ph)
		}
		if ev.TS != exp.ts {
			t.Errorf("event[%d].TS = %f, want %f", i, ev.TS, exp.ts)
		}
		if ev.Dur != exp.dur {
			t.Errorf("event[%d].Dur = %f, want %f", i, ev.Dur, exp.dur)
		}
		if ev.PID != exp.pid {
			t.Errorf("event[%d].PID = %q, want %q", i, ev.PID, exp.pid)
		}
		if ev.TID != exp.tid {
			t.Errorf("event[%d].TID = %d, want %d", i, ev.TID, exp.tid)
		}
		for k, v := range exp.args {
			got, ok := ev.Args[k]
			if !ok {
				t.Errorf("event[%d].Args[%q] missing", i, k)
				continue
			}
			if gotNum, ok := got.(float64); ok {
				if expNum, ok := v.(float64); ok && gotNum != expNum {
					t.Errorf("event[%d].Args[%q] = %v, want %v", i, k, got, v)
				}
			} else if fmt.Sprintf("%v", got) != fmt.Sprintf("%v", v) {
				t.Errorf("event[%d].Args[%q] = %v, want %v", i, k, got, v)
			}
		}
	}

	// 2. Also verify parseable into a generic map[string]any
	var generic map[string]any
	if err := json.Unmarshal(buf.Bytes(), &generic); err != nil {
		t.Fatalf("json.Unmarshal into map[string]any: %v", err)
	}
	if generic["displayTimeUnit"] != "ms" {
		t.Errorf("generic displayTimeUnit = %v, want ms", generic["displayTimeUnit"])
	}
	rawEvents, ok := generic["traceEvents"].([]any)
	if !ok || len(rawEvents) != 3 {
		t.Fatalf("generic traceEvents = %v, want 3 items", generic["traceEvents"])
	}
}

func TestMicroTraceWritePerfettoEmptyTrace(t *testing.T) {
	for _, tc := range []struct {
		name  string
		trace *MicroTrace
	}{
		{"nil trace", nil},
		{"empty spans nil", &MicroTrace{TraceID: "micro-empty"}},
		{"empty spans slice", &MicroTrace{TraceID: "micro-empty", Spans: []MicroSpan{}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := tc.trace.WritePerfetto(&buf); err != nil {
				t.Fatalf("WritePerfetto: %v", err)
			}

			// Must produce valid JSON
			var container PerfettoTraceContainer
			if err := json.Unmarshal(buf.Bytes(), &container); err != nil {
				t.Fatalf("Unmarshal into struct failed: %v", err)
			}
			if container.DisplayTimeUnit != "ms" {
				t.Fatalf("displayTimeUnit = %q, want ms", container.DisplayTimeUnit)
			}
			if len(container.TraceEvents) != 0 {
				t.Fatalf("traceEvents count = %d, want 0", len(container.TraceEvents))
			}

			// Verify literal JSON contains "traceEvents": [] (not null)
			var raw map[string]any
			if err := json.Unmarshal(buf.Bytes(), &raw); err != nil {
				t.Fatalf("Unmarshal into map failed: %v", err)
			}
			evList, ok := raw["traceEvents"].([]any)
			if !ok {
				t.Fatalf("raw traceEvents is not an array: %v", raw["traceEvents"])
			}
			if len(evList) != 0 {
				t.Fatalf("raw traceEvents length = %d, want 0", len(evList))
			}
			if strings.Contains(buf.String(), `"traceEvents": null`) {
				t.Fatal("traceEvents serialized as null instead of empty array []")
			}
		})
	}
}

func TestMicroTraceWritePerfettoFallbacks(t *testing.T) {
	// 1. Default agent name when TraceID is empty
	tr := &MicroTrace{
		TraceID: "",
		Spans: []MicroSpan{
			// Span with empty label falls back to Kind
			{Kind: SpanStep, Dur: 0},
			// Span with empty label and empty kind falls back to "span"
			{Label: "", Dur: 0},
		},
	}
	var buf bytes.Buffer
	if err := tr.WritePerfetto(&buf); err != nil {
		t.Fatalf("WritePerfetto: %v", err)
	}
	var container PerfettoTraceContainer
	if err := json.Unmarshal(buf.Bytes(), &container); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(container.TraceEvents) != 2 {
		t.Fatalf("traceEvents count = %d, want 2", len(container.TraceEvents))
	}
	if container.TraceEvents[0].PID != "agent" {
		t.Errorf("PID = %q, want default 'agent'", container.TraceEvents[0].PID)
	}
	if container.TraceEvents[0].Name != "step" {
		t.Errorf("Name = %q, want 'step'", container.TraceEvents[0].Name)
	}
	if container.TraceEvents[1].Name != "span" {
		t.Errorf("Name = %q, want 'span'", container.TraceEvents[1].Name)
	}
}

func TestMicroTracerWritePerfetto(t *testing.T) {
	src := NewMicroTracer()
	src.Record("micro-000", MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 10, Dur: 5 * time.Millisecond})
	src.Record("micro-001", MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 20, Dur: 8 * time.Millisecond})

	var buf bytes.Buffer
	if err := src.WritePerfetto(&buf); err != nil {
		t.Fatalf("MicroTracer.WritePerfetto: %v", err)
	}
	var container PerfettoTraceContainer
	if err := json.Unmarshal(buf.Bytes(), &container); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(container.TraceEvents) != 2 {
		t.Fatalf("traceEvents count = %d, want 2", len(container.TraceEvents))
	}
	if container.TraceEvents[0].PID != "micro-000" || container.TraceEvents[1].PID != "micro-001" {
		t.Errorf("PIDs = (%q, %q), want (micro-000, micro-001)", container.TraceEvents[0].PID, container.TraceEvents[1].PID)
	}
}

func TestMicroTraceWritePerfettoNilWriter(t *testing.T) {
	tr := &MicroTrace{TraceID: "micro-000"}
	if err := tr.WritePerfetto(nil); err == nil {
		t.Fatal("WritePerfetto(nil) expected error, got nil")
	}
	tracer := NewMicroTracer()
	if err := tracer.WritePerfetto(nil); err == nil {
		t.Fatal("tracer.WritePerfetto(nil) expected error, got nil")
	}
}
