package quality

import (
	"strings"
	"testing"
)

// TestLogprobParityFaithfulPasses is the happy path: an engine that reports
// aligned, log-softmax-normalized logprob rows for every prompt and generated
// token passes the oracle with no failure bundle — parity within tolerance is
// a green gate, not a flake.
func TestLogprobParityFaithfulPasses(t *testing.T) {
	c := LogprobParityCase()
	res, err := RunCase(c, ReferenceRunner{}, LogprobParityEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful logprob engine should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	// The case's own reference must cover prompt AND generation: one row per
	// token, prompt tokens included.
	promptLen := len(strings.Fields(c.Prompt))
	wantRows := promptLen + c.Params.MaxTokens
	if got := len(c.Reference.Logits); got != wantRows {
		t.Fatalf("reference carries %d logprob rows, want %d (%d prompt + %d generation)",
			got, wantRows, promptLen, c.Params.MaxTokens)
	}
}

// TestLogprobParityShiftFailsAtFirstShiftedToken is the off-by-one alignment
// witness: an engine whose logprob rows are shifted by one position keeps
// byte-identical token text, yet fails at exactly index 1 — the first token
// whose row belongs to its neighbor — with a Detail that names the defect
// class. The intact index 0 proves the localization is doing work.
func TestLogprobParityShiftFailsAtFirstShiftedToken(t *testing.T) {
	c := LogprobParityCase()
	res, err := RunCase(c, ReferenceRunner{}, LogprobParityEngine("logprob-shift"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("shifted-logprob engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "logprob-parity" {
		t.Errorf("first failing oracle = %q, want logprob-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != 1 {
		t.Fatalf("expected first divergence at token 1 (first shifted row), got %+v", d)
	}
	if !strings.Contains(fb.Detail, "off-by-one") {
		t.Errorf("detail should classify the defect as off-by-one alignment; got %q", fb.Detail)
	}
	// The token text is byte-identical — this defect class is invisible to any
	// token-differential oracle, which is why this gate reads Logits.
	if fb.Engine.Text != c.Reference.Text {
		t.Fatalf("shift defect must not change token text: engine %q, reference %q", fb.Engine.Text, c.Reference.Text)
	}
	// And the misalignment is literal: engine row 1 IS reference row 0.
	if len(fb.Engine.Logits) < 2 || !lpApproxEqual(fb.Engine.Logits[1], c.Reference.Logits[0]) {
		t.Errorf("engine row 1 should equal reference row 0 under the shift defect")
	}
}

// TestLogprobParityRawLogitsFailAtIndexZero is the wrong-normalization
// witness: an engine that reports raw pre-softmax logits where logprobs were
// promised fails at the first row (index 0), and the Detail names the raw
// logit defect class rather than leaving a bare numeric mismatch.
func TestLogprobParityRawLogitsFailAtIndexZero(t *testing.T) {
	c := LogprobParityCase()
	res, err := RunCase(c, ReferenceRunner{}, LogprobParityEngine("raw-logits"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("raw-logits engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != 0 {
		t.Fatalf("expected first divergence at token 0 (every row is un-normalized), got %+v", d)
	}
	if !strings.Contains(fb.Detail, "raw logit") {
		t.Errorf("detail should classify the defect as raw logits; got %q", fb.Detail)
	}
	// Token text is again untouched: only the numeric surface is corrupt.
	if fb.Engine.Text != c.Reference.Text {
		t.Fatalf("normalization defect must not change token text: engine %q, reference %q", fb.Engine.Text, c.Reference.Text)
	}
}

// TestLogprobParityCoverageGapFails is the alignment rule-0 witness: a trace
// missing a logprob row for one of its tokens cannot be aligned, and the
// oracle fails it at the first uncovered index instead of judging only the
// rows that happen to exist.
func TestLogprobParityCoverageGapFails(t *testing.T) {
	c := LogprobParityCase()
	eng := lpFaithfulTrace(c.Prompt, c.Params.MaxTokens)
	eng.Logits = eng.Logits[:len(eng.Logits)-1] // drop the last row: token left unscored
	v := LogprobParity{}.Judge(c.Reference, eng, c)
	if v.Pass {
		t.Fatalf("engine with a missing logprob row must not pass; detail %q", v.Detail)
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != len(eng.Logits) {
		t.Fatalf("expected divergence at first uncovered index %d, got %+v", len(eng.Logits), v.FirstDivergence)
	}
	if !strings.Contains(v.Detail, "coverage") {
		t.Errorf("detail should report broken coverage; got %q", v.Detail)
	}
}
