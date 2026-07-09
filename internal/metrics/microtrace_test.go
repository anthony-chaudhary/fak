package metrics

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"testing"
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
