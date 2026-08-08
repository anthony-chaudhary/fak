package metrics

// microtrace_outcome_test.go — the outcome-counter contract for the micro-context
// fabric (#5838). The spine's per-agent spans already existed; what did not exist was
// a fabric-level success/refusal/error fold that survives the run that produced it.

import (
	"bytes"
	"strings"
	"testing"
)

// TestMicroTraceOutcomeClassification pins the per-invocation vocabulary, including
// the precedence rule (error outranks refusal outranks success) and the no-verdict
// case that must NOT be counted as a success.
func TestMicroTraceOutcomeClassification(t *testing.T) {
	for _, tc := range []struct {
		name     string
		verdicts []string
		want     MicroOutcome
	}{
		{"admitted", []string{"ALLOW", "ALLOW"}, MicroOutcomeSuccess},
		{"refused", []string{"ALLOW", "DENY"}, MicroOutcomeRefusal},
		{"errored", []string{"ALLOW", "ERROR"}, MicroOutcomeError},
		{"error outranks refusal", []string{"DENY", "ERROR"}, MicroOutcomeError},
		{"refusal outranks later allow", []string{"REFUSED", "ALLOW"}, MicroOutcomeRefusal},
		{"case and space insensitive", []string{" denied "}, MicroOutcomeRefusal},
		{"no verdict leg at all", nil, MicroOutcomeUnknown},
		// An unrecognised verdict reads as an admit leg — the documented invalidating
		// assumption on the fold, pinned so a future vocabulary change trips this test
		// instead of silently moving refusals into the success bucket.
		{"unknown verdict reads as admit", []string{"PROCEED"}, MicroOutcomeSuccess},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := MicroTrace{TraceID: "micro-000"}
			// A step leg with no verdict must never decide the outcome on its own.
			tr.Spans = append(tr.Spans, MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 3})
			for _, v := range tc.verdicts {
				tr.Spans = append(tr.Spans, MicroSpan{Kind: SpanVerdict, Verdict: v})
			}
			if got := tr.Outcome(); got != tc.want {
				t.Fatalf("Outcome()=%q, want %q (verdicts=%v)", got, tc.want, tc.verdicts)
			}
		})
	}
}

// fixtureTracer builds a mixed fleet: two clean agents, one refused, one errored,
// one that recorded legs but no verdict.
func fixtureTracer() *MicroTracer {
	tr := NewMicroTracer()
	for _, id := range []string{"micro-000", "micro-001"} {
		tr.Record(id, MicroSpan{Kind: SpanSeat, Seat: "slot-pool/4", Label: "acquire"})
		tr.Record(id, MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 12})
		tr.Record(id, MicroSpan{Kind: SpanVerdict, Label: "mock-planner", Verdict: "ALLOW"})
	}
	tr.Record("micro-002", MicroSpan{Kind: SpanAdmission, Label: "token budget"})
	tr.Record("micro-002", MicroSpan{Kind: SpanVerdict, Label: "budget", Verdict: "DENY"})
	tr.Record("micro-003", MicroSpan{Kind: SpanStep, Label: "turn 1", Verdict: "ERROR"})
	tr.Record("micro-004", MicroSpan{Kind: SpanStep, Label: "turn 1", Tokens: 5})
	return tr
}

// TestMicroTracerOutcomes folds a mixed fleet and pins the counts and the readout —
// the captured evidence that the surface reports REAL counts, not a shape.
func TestMicroTracerOutcomes(t *testing.T) {
	got := fixtureTracer().Outcomes()
	want := MicroOutcomeCounts{Success: 2, Refusal: 1, Error: 1, Unknown: 1}
	if got != want {
		t.Fatalf("Outcomes()=%+v, want %+v", got, want)
	}
	if got.Total() != 5 {
		t.Fatalf("Total()=%d, want 5 (buckets must sum to the trace count)", got.Total())
	}
	const wantLine = "success=2 refusal=1 error=1 unknown=1  (5 invocation(s))"
	if line := got.Render(); line != wantLine {
		t.Fatalf("Render()=%q, want %q", line, wantLine)
	}
	// A fully-traced run hides the unknown bucket rather than printing a zero.
	clean := MicroOutcomeCounts{Success: 3}
	if line := clean.Render(); line != "success=3 refusal=0 error=0  (3 invocation(s))" {
		t.Fatalf("clean Render()=%q", line)
	}
}

// TestMicroOutcomeCountsSurviveJSONL is the promotion evidence the fold rests on:
// the counts are queryable AFTER the run that produced them exited, because they
// re-derive identically from a tracer reconstructed out of persisted --trace-out
// JSONL. Without this the counters would be a live-only readout.
func TestMicroOutcomeCountsSurviveJSONL(t *testing.T) {
	live := fixtureTracer()
	var buf bytes.Buffer
	if err := live.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	replayed, err := ReadTracesJSONL(&buf)
	if err != nil {
		t.Fatalf("ReadTracesJSONL: %v", err)
	}
	if got, want := replayed.Outcomes(), live.Outcomes(); got != want {
		t.Fatalf("replayed Outcomes()=%+v, want %+v (live)", got, want)
	}
}

// TestMicroTraceRenderCarriesOutcome proves the counter joined an EXISTING surface:
// `fak micro trace <id>` prints MicroTrace.Render() verbatim, so the per-invocation
// outcome is visible there with no new command and no new dashboard.
func TestMicroTraceRenderCarriesOutcome(t *testing.T) {
	tr := fixtureTracer()
	for id, want := range map[string]MicroOutcome{
		"micro-000": MicroOutcomeSuccess,
		"micro-002": MicroOutcomeRefusal,
		"micro-003": MicroOutcomeError,
		"micro-004": MicroOutcomeUnknown,
	} {
		out, ok := tr.Render(id)
		if !ok {
			t.Fatalf("Render(%s): trace not found", id)
		}
		if !strings.Contains(out, "outcome: "+string(want)) {
			t.Fatalf("Render(%s) missing %q:\n%s", id, "outcome: "+string(want), out)
		}
	}
}
