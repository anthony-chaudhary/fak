package quality

import (
	"fmt"
	"strings"
	"testing"
)

// kindsTestTrace builds a deterministic n-token trace: token i is "tok<i>",
// with the first `flips` positions FROM THE END replaced by "alt<i>" so a
// mutant trace disagrees with the clean one at exactly `flips` known positions
// and nowhere else.
func kindsTestTrace(n, flips int) Trace {
	toks := make([]string, n)
	for i := 0; i < n; i++ {
		if i >= n-flips {
			toks[i] = fmt.Sprintf("alt%d", i)
		} else {
			toks[i] = fmt.Sprintf("tok%d", i)
		}
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// kindsTestCase wraps a reference trace in a valid case naming the given
// oracles, so RunCase's admission gate is exercised alongside the verdicts.
func kindsTestCase(ref Trace, oracles ...string) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "oracle-kinds-test",
		Version:   1,
		Prompt:    "Reproduce the reference sequence exactly.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: len(ref.Tokens)},
		Reference: ref,
		Oracles:   oracles,
	}
}

// TestKindsRegistered proves both oracles registered under their taxonomy
// kinds: cases resolve them by name through the shared registry, and each
// reports the kind the taxonomy assigns it.
func TestKindsRegistered(t *testing.T) {
	os, err := Lookup([]string{"exact-match", "statistical-agreement"})
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got := os[0].Kind(); got != "exact" {
		t.Errorf("exact-match kind = %q, want %q", got, "exact")
	}
	if got := os[1].Kind(); got != "statistical" {
		t.Errorf("statistical-agreement kind = %q, want %q", got, "statistical")
	}
}

// TestKindsExactMatchIdenticalPasses is the exact oracle's happy path, run
// through the full spine: a byte-identical engine passes both new oracles (a
// perfect trace is also perfect agreement) and emits no failure bundle.
func TestKindsExactMatchIdenticalPasses(t *testing.T) {
	ref := kindsTestTrace(8, 0)
	c := kindsTestCase(ref, "exact-match", "statistical-agreement")
	res, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Label: "engine-clean", Trace: ref}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("identical engine trace must pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	for _, v := range res.Verdicts {
		if v.FirstDivergence != nil {
			t.Errorf("passing verdict %q must not carry a divergence: %+v", v.Oracle, v.FirstDivergence)
		}
	}
}

// TestKindsExactMatchSingleFlipFailsAtIndex is the exact oracle's localized
// defect witness: flipping ONE token makes the oracle fail with the first
// divergence pinned to exactly that index, carrying both traces' tokens there.
func TestKindsExactMatchSingleFlipFailsAtIndex(t *testing.T) {
	const flipAt = 2
	ref := kindsTestTrace(8, 0)
	eng := kindsTestTrace(8, 0)
	eng.Tokens[flipAt] = "flipped"
	eng.Text = strings.Join(eng.Tokens, " ")

	c := kindsTestCase(ref, "exact-match")
	res, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Label: "engine-flip", Trace: eng}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("single-flip engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "exact-match" || fb.FailingKind != "exact" {
		t.Errorf("failing oracle/kind = %q/%q, want exact-match/exact", fb.FailingOracle, fb.FailingKind)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != flipAt {
		t.Fatalf("expected first divergence at index %d, got %+v", flipAt, d)
	}
	if d.Reference != ref.Tokens[flipAt] || d.Engine != "flipped" {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ref.Tokens[flipAt], "flipped")
	}
}

// TestKindsExactMatchLengthAndTextDivergence covers the two non-token exact
// defects: a truncated engine fails at the first missing index, and equal
// tokens with different assembled text fail on the text bytes (the
// detokenizer-bug class greedy-token-diff cannot see).
func TestKindsExactMatchLengthAndTextDivergence(t *testing.T) {
	ref := kindsTestTrace(8, 0)

	short := Trace{Tokens: ref.Tokens[:5], Text: strings.Join(ref.Tokens[:5], " ")}
	v := kindsExactMatch{}.Judge(ref, short, kindsTestCase(ref, "exact-match"))
	if v.Pass {
		t.Fatal("truncated engine trace must not pass exact-match")
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != 5 {
		t.Fatalf("expected length divergence at index 5, got %+v", v.FirstDivergence)
	}
	if v.FirstDivergence.Engine != "<end>" {
		t.Errorf("engine side of a truncation divergence = %q, want %q", v.FirstDivergence.Engine, "<end>")
	}

	textBug := Trace{Tokens: ref.Tokens, Text: strings.ToUpper(ref.Text)}
	v = kindsExactMatch{}.Judge(ref, textBug, kindsTestCase(ref, "exact-match"))
	if v.Pass {
		t.Fatal("equal tokens with different assembled text must not pass exact-match")
	}
	if !strings.Contains(v.Detail, "byte") {
		t.Errorf("text divergence detail should localize a byte, got %q", v.Detail)
	}
}

// TestKindsStatisticalAgreementHighAgreementPasses is the statistical oracle's
// happy path: 198/200 paired agreements (rate 0.99) has a CI lower bound
// comfortably above the 0.95 threshold and passes — even though the SAME
// traces would fail exact-match, which is the taxonomy's point: kinds differ
// in what a defect is, not just in how it is reported.
func TestKindsStatisticalAgreementHighAgreementPasses(t *testing.T) {
	ref := kindsTestTrace(200, 0)
	eng := kindsTestTrace(200, 2)
	c := kindsTestCase(ref, "statistical-agreement")
	c.Rubric.MinScore = 0.95

	res, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Label: "engine-noisy-ok", Trace: eng}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("0.99 agreement over 200 samples must clear a 0.95 threshold; got %s", Explain(res))
	}
	v := res.Verdicts[0]
	if v.Score != 0.99 {
		t.Errorf("verdict score = %v, want 0.99", v.Score)
	}
	// The pass is a bounded claim: the lower bound itself clears the threshold.
	if _, lo, _ := kindsAgreementCI(198, 200); lo < 0.95 {
		t.Errorf("CI lower bound %v should clear 0.95 for this fixture", lo)
	}
	if exact := (kindsExactMatch{}).Judge(ref, eng, c); exact.Pass {
		t.Error("the same 2-flip trace must FAIL exact-match: the kinds are not interchangeable")
	}
}

// TestKindsStatisticalAgreementLowAgreementFails is the statistical defect
// witness: 160/200 agreements (rate 0.80) against a 0.95 threshold fails, and
// the failure is statistically unambiguous — the whole 95% CI lies below the
// threshold, verified independently here, and the verdict's detail carries the
// bound that decided it.
func TestKindsStatisticalAgreementLowAgreementFails(t *testing.T) {
	ref := kindsTestTrace(200, 0)
	eng := kindsTestTrace(200, 40)
	c := kindsTestCase(ref, "statistical-agreement")
	c.Rubric.MinScore = 0.95

	p, lo, hi := kindsAgreementCI(160, 200)
	if p != 0.80 {
		t.Fatalf("fixture agreement rate = %v, want 0.80", p)
	}
	if hi >= 0.95 {
		t.Fatalf("fixture CI [%v, %v] must exclude the 0.95 threshold entirely", lo, hi)
	}

	res, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Label: "engine-drifted", Trace: eng}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("0.80 agreement must not clear a 0.95 threshold; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "statistical-agreement" || fb.FailingKind != "statistical" {
		t.Errorf("failing oracle/kind = %q/%q, want statistical-agreement/statistical", fb.FailingOracle, fb.FailingKind)
	}
	v := res.Verdicts[0]
	if v.Score != 0.80 {
		t.Errorf("verdict score = %v, want 0.80", v.Score)
	}
	if !strings.Contains(v.Detail, "lower bound") || !strings.Contains(v.Detail, "threshold") {
		t.Errorf("failure detail should name the deciding bound and threshold, got %q", v.Detail)
	}
}

// TestKindsStatisticalAgreementFailsClosed covers the two conservative edges:
// zero paired samples cannot be bounded and is a refusal (not a pass), and a
// marginal rate whose CI straddles the threshold fails — the LOWER bound
// gates, not the point estimate.
func TestKindsStatisticalAgreementFailsClosed(t *testing.T) {
	c := kindsTestCase(Trace{}, "statistical-agreement")
	v := kindsStatisticalAgreement{}.Judge(Trace{}, Trace{}, c)
	if v.Pass {
		t.Fatal("zero paired samples must not pass: an unmeasured case is not green")
	}

	// 39/40 = 0.975 is ABOVE the 0.95 threshold, but over only 40 samples the
	// CI lower bound (~0.926) is not — so the verdict is a fail.
	ref := kindsTestTrace(40, 0)
	eng := kindsTestTrace(40, 1)
	if _, lo, _ := kindsAgreementCI(39, 40); lo >= 0.95 {
		t.Fatalf("fixture lower bound %v should straddle the 0.95 threshold", lo)
	}
	c = kindsTestCase(ref, "statistical-agreement")
	c.Rubric.MinScore = 0.95
	v = kindsStatisticalAgreement{}.Judge(ref, eng, c)
	if v.Pass {
		t.Fatal("a point estimate above threshold with a lower bound below it must fail")
	}
}
