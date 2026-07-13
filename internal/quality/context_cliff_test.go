package quality

import (
	"fmt"
	"strings"
	"testing"
)

// ctxCliffVariedTokens returns n pairwise-distinct tokens — the bounded,
// non-repeating decode shape at any position.
func ctxCliffVariedTokens(n int, prefix string) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("%s%d", prefix, i)
	}
	return out
}

// ctxCliffCollapsedTokens injects the position-scaling defect: the engine
// decodes varied tokens up to cliffAt, then collapses into repeating one token
// for the rest of the trace — the RoPE-beyond-trained-length loop failure.
func ctxCliffCollapsedTokens(n, cliffAt int) []string {
	out := ctxCliffVariedTokens(n, "tok")
	for i := cliffAt; i < n; i++ {
		out[i] = "loop"
	}
	return out
}

// ctxCliffCase is a hermetic long-decode case judged only by the context-cliff
// oracle. The reference is 32 all-distinct tokens (worst windowed repetition 0,
// so the derived bound is exactly ctxCliffMargin); the engine decodes 96 tokens
// — three reference lengths, modeling positions well past the trained window.
func ctxCliffCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "context-cliff-long-decode",
		Version: 1,
		Prompt:  "Continue the document well beyond the trained context length.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 96},
		Reference: Trace{
			Tokens: ctxCliffVariedTokens(32, "ref"),
			Text:   "bounded reference decode",
		},
		Oracles: []string{"context-cliff"},
	}
}

// ctxCliffVerdict pulls the context-cliff verdict out of a result or fails the
// test.
func ctxCliffVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "context-cliff" {
			return v
		}
	}
	t.Fatalf("no context-cliff verdict in %s", Explain(res))
	return Verdict{}
}

// TestCtxCliffBoundedLongDecodePasses is the faithful path: an engine that keeps
// emitting varied tokens for three reference lengths — deep past the reference's
// own positions — stays within the reference-derived bound at every position and
// passes with a full score and no failure bundle.
func TestCtxCliffBoundedLongDecodePasses(t *testing.T) {
	c := ctxCliffCase()
	eng := ScriptedRunner{Label: "engine-bounded", Trace: Trace{
		Tokens: ctxCliffVariedTokens(96, "tok"),
		Text:   "bounded engine decode",
	}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("bounded long decode should pass; got %s", Explain(res))
	}
	v := ctxCliffVerdict(t, res)
	if v.Score != 1 {
		t.Errorf("bounded decode score = %v, want 1", v.Score)
	}
	if v.FirstDivergence != nil {
		t.Errorf("passing verdict must not carry a divergence: %+v", v.FirstDivergence)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestCtxCliffRepetitionCollapseFailsAtPosition is the defect Witness for #4546:
// past position 40 the engine collapses into a single-token loop. With an
// all-distinct reference the bound is ctxCliffMargin (0.15); a 16-token window
// holding k copies of the loop token has repetition (k-1)/16, which first
// exceeds 0.15 at k=4, i.e. window end position 40+3 = 43. The oracle must fail,
// localize the cliff at exactly that position, and score the bounded fraction:
// windows end at positions 15..95 (81 total) and stay bounded through 42 (28).
func TestCtxCliffRepetitionCollapseFailsAtPosition(t *testing.T) {
	c := ctxCliffCase()
	eng := ScriptedRunner{Label: "engine-cliffs", Trace: Trace{
		Tokens: ctxCliffCollapsedTokens(96, 40),
		Text:   "engine decode that loops past the trained length",
	}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("repetition collapse past the trained length must not pass; got %s", Explain(res))
	}
	v := ctxCliffVerdict(t, res)
	if v.Pass {
		t.Fatal("context-cliff verdict should have failed")
	}
	if v.FirstDivergence == nil {
		t.Fatalf("failing verdict must localize the cliff position; got %s", Explain(res))
	}
	if v.FirstDivergence.Index != 43 {
		t.Errorf("cliff position = %d, want 43 (first window with 4 loop tokens)", v.FirstDivergence.Index)
	}
	if want := 28.0 / 81.0; v.Score != want {
		t.Errorf("score = %v, want %v (28 of 81 positions bounded)", v.Score, want)
	}
	if !strings.Contains(v.Detail, "position 43") {
		t.Errorf("Detail %q does not name the cliff position 43", v.Detail)
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "context-cliff" {
		t.Errorf("first failing oracle = %q, want context-cliff", fb.FailingOracle)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 43 {
		t.Errorf("failure bundle divergence = %+v, want index 43", fb.FirstDivergence)
	}
}

// TestCtxCliffReferenceRepetitionRaisesBound proves the bound is
// reference-derived, not absolute: a reference that legitimately cycles with
// period 8 (windowed repetition 0.5) raises the bound to 0.65, so an engine
// replaying the same cycling shape — including past the reference length —
// passes instead of false-failing, while a MinScore below 1 tolerates a bounded
// late crossing and says so.
func TestCtxCliffReferenceRepetitionRaisesBound(t *testing.T) {
	c := ctxCliffCase()
	cycle := make([]string, 32)
	for i := range cycle {
		cycle[i] = fmt.Sprintf("cyc%d", i%8)
	}
	c.Reference = Trace{Tokens: cycle, Text: "cycling reference"}
	long := make([]string, 96)
	for i := range long {
		long[i] = fmt.Sprintf("cyc%d", i%8)
	}
	eng := ScriptedRunner{Label: "engine-cycling", Trace: Trace{Tokens: long, Text: "cycling engine"}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("engine matching the reference's own repetition shape must pass; got %s", Explain(res))
	}

	tol := ctxCliffCase() // back to the all-distinct reference: bound is 0.15
	tol.Rubric = RubricSpec{MinScore: 0.3}
	engCliff := ScriptedRunner{Label: "engine-late-cliff", Trace: Trace{
		Tokens: ctxCliffCollapsedTokens(96, 88),
		Text:   "mostly varied, loops only at the tail",
	}}
	res, err = RunCase(tol, ReferenceRunner{}, engCliff, oraclesFor(t, tol))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("MinScore 0.3 should tolerate a tail-only crossing; got %s", Explain(res))
	}
	if v := ctxCliffVerdict(t, res); !strings.Contains(v.Detail, "tolerated crossing at position") {
		t.Errorf("Detail %q should report the tolerated crossing position", v.Detail)
	}
}
