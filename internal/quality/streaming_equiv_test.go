package quality

import (
	"strings"
	"testing"
)

// strmTestSteps is enough tokens for the decode to cross the defect boundary
// (token index strmBoundaryIndex) with tokens to spare on both sides.
const strmTestSteps = 8

// TestStreamingEquivFaithfulStreamPasses is the happy path: a faithful
// streaming engine's concatenated flushes equal the non-streaming reference in
// tokens AND assembled text, so the oracle passes and no failure bundle is
// emitted.
func TestStreamingEquivFaithfulStreamPasses(t *testing.T) {
	c := strmStreamingCase(42, strmTestSteps)
	res, err := RunCase(c, strmFullRunner{}, strmStreamingEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful stream should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean stream must not carry a failure bundle: %+v", res.FailureBundle)
	}
	// The equivalence is literal, not merely judged: the streamed trace is
	// token-for-token and byte-for-byte the non-streaming output.
	ref, eng := res.Provenance.Reference, res.Provenance.Engine
	if strings.Join(ref.Tokens, "\x00") != strings.Join(eng.Tokens, "\x00") {
		t.Fatalf("streamed tokens differ from non-streaming: ref %v eng %v", ref.Tokens, eng.Tokens)
	}
	if ref.Text != eng.Text {
		t.Fatalf("streamed text differs from non-streaming: ref %q eng %q", ref.Text, eng.Text)
	}
}

// TestStreamingEquivDropAtBoundaryFailsAtIndex is the dropped-token witness: a
// flush that loses the first token of the defect chunk fails at EXACTLY the
// boundary token index, with the reference and engine tokens reported there.
// Adjacent-distinct decoding guarantees the shifted token cannot coincide.
func TestStreamingEquivDropAtBoundaryFailsAtIndex(t *testing.T) {
	c := strmStreamingCase(42, strmTestSteps)
	res, err := RunCase(c, strmFullRunner{}, strmStreamingEngine("drop-boundary"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("drop-boundary engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing stream must carry a failure bundle")
	}
	if fb.FailingOracle != "streaming-equivalence" {
		t.Errorf("first failing oracle = %q, want streaming-equivalence", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != strmBoundaryIndex {
		t.Fatalf("expected first divergence at boundary token %d, got %+v", strmBoundaryIndex, d)
	}
	ref := c.Reference.Tokens
	if d.Reference != ref[strmBoundaryIndex] {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, ref[strmBoundaryIndex])
	}
	// Dropping the boundary token shifts the next token into its place.
	if d.Engine != ref[strmBoundaryIndex+1] {
		t.Errorf("divergence engine token = %q, want shifted %q", d.Engine, ref[strmBoundaryIndex+1])
	}
	if got, want := len(fb.Engine.Tokens), len(ref)-1; got != want {
		t.Errorf("dropped-token stream has %d tokens, want %d", got, want)
	}
}

// TestStreamingEquivDupAtBoundaryFailsAtIndex is the duplicated-token witness:
// a flush that re-emits the last already-delivered token fails at exactly the
// boundary index, where the engine repeated the previous token instead of
// advancing.
func TestStreamingEquivDupAtBoundaryFailsAtIndex(t *testing.T) {
	c := strmStreamingCase(42, strmTestSteps)
	res, err := RunCase(c, strmFullRunner{}, strmStreamingEngine("dup-boundary"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("dup-boundary engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing stream must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != strmBoundaryIndex {
		t.Fatalf("expected first divergence at boundary token %d, got %+v", strmBoundaryIndex, d)
	}
	ref := c.Reference.Tokens
	if d.Reference != ref[strmBoundaryIndex] {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, ref[strmBoundaryIndex])
	}
	// The duplicated token is the last token of the previous flush.
	if d.Engine != ref[strmBoundaryIndex-1] {
		t.Errorf("divergence engine token = %q, want duplicated %q", d.Engine, ref[strmBoundaryIndex-1])
	}
	if got, want := len(fb.Engine.Tokens), len(ref)+1; got != want {
		t.Errorf("duplicated-token stream has %d tokens, want %d", got, want)
	}
}

// TestStreamingEquivGlueDefectFailsOnText is the text-assembly witness: every
// token is delivered, but the flush handoff glues two chunks' text with no
// separator — the token rung passes and the TEXT rung fails, proving the
// oracle checks the assembled text and not only the token sequence.
func TestStreamingEquivGlueDefectFailsOnText(t *testing.T) {
	c := strmStreamingCase(42, strmTestSteps)
	eng, err := strmStreamingEngine("glue-boundary").Run(c)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	// Precondition: the token stream is intact — only the text is corrupted.
	ref := c.Reference
	if strings.Join(ref.Tokens, "\x00") != strings.Join(eng.Tokens, "\x00") {
		t.Fatalf("glue defect must leave tokens intact: ref %v eng %v", ref.Tokens, eng.Tokens)
	}
	if ref.Text == eng.Text {
		t.Fatal("glue defect should corrupt the assembled text")
	}
	v := strmEquivalence{}.Judge(ref, eng, c)
	if v.Pass {
		t.Fatalf("glued text must not pass: %+v", v)
	}
	if !strings.Contains(v.Detail, "assembled text") {
		t.Errorf("detail should name the text-assembly defect, got %q", v.Detail)
	}
	// And through the full harness the case fails with the same oracle named.
	res, err := RunCase(c, strmFullRunner{}, strmStreamingEngine("glue-boundary"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("glue-boundary engine must not pass; got %s", Explain(res))
	}
	if res.FailureBundle == nil || res.FailureBundle.FailingOracle != "streaming-equivalence" {
		t.Fatalf("failure bundle should name streaming-equivalence, got %+v", res.FailureBundle)
	}
}

// TestStreamingEquivDecodeAdjacentDistinct pins the property the boundary
// localization relies on: the deterministic decode never emits the same token
// twice in a row, so a dropped or duplicated boundary token is guaranteed to
// differ from the reference at exactly the boundary index.
func TestStreamingEquivDecodeAdjacentDistinct(t *testing.T) {
	for _, seed := range []int64{0, 1, 42, 1337} {
		tr := strmDecode(seed, 64)
		for i := 1; i < len(tr.Tokens); i++ {
			if tr.Tokens[i] == tr.Tokens[i-1] {
				t.Fatalf("seed %d: tokens %d and %d are both %q; adjacent tokens must differ",
					seed, i-1, i, tr.Tokens[i])
			}
		}
	}
}
