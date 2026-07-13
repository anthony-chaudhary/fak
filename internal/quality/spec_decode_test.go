package quality

import (
	"strings"
	"testing"
)

// TestSpecDecodeFaithfulEngineMatchesTargetOnly is the exactness-preservation
// contract on the happy path: a speculative engine with the faithful
// accept/reject rule (rejections fall back to the target token) emits a stream
// IDENTICAL to target-only decode, and the parity oracle passes. The test first
// proves the run is not vacuous — the draft genuinely disagrees with the target
// at specFirstDriftStep, so the faithful pass exercised the reject-fallback
// path, not just trivial agreement.
func TestSpecDecodeFaithfulEngineMatchesTargetOnly(t *testing.T) {
	const seed = 42
	c := SpecDecodeCase(seed)

	// Non-vacuity: the draft must actually propose a token the target rejects
	// somewhere inside the decode window, or the parity pass proves nothing
	// about the accept rule.
	if specFirstDriftStep >= c.Params.MaxTokens {
		t.Fatalf("first drift step %d outside the decode window %d; case exercises no rejection",
			specFirstDriftStep, c.Params.MaxTokens)
	}
	if specDraftToken(seed, specFirstDriftStep) == specTargetToken(seed, specFirstDriftStep) {
		t.Fatalf("draft must disagree with target at step %d for the rejection path to be exercised",
			specFirstDriftStep)
	}

	res, err := RunCase(c, ReferenceRunner{}, SpecDecodeEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful speculative engine must match target-only decode; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean speculative run must not carry a failure bundle: %+v", res.FailureBundle)
	}

	// Token-for-token identity, checked independently of the oracle: the
	// speculative trace equals target-only decode at every position.
	eng := res.Provenance.Engine.Tokens
	ref := res.Provenance.Reference.Tokens
	if len(eng) != len(ref) {
		t.Fatalf("faithful speculative decode emitted %d tokens, target-only %d", len(eng), len(ref))
	}
	for i := range ref {
		if eng[i] != ref[i] {
			t.Fatalf("faithful speculative decode differs at token %d: engine %q, target-only %q",
				i, eng[i], ref[i])
		}
	}
}

// TestSpecDecodeLenientAcceptFailsAtFirstKeptDraft is the injected-defect
// witness: an accept rule that KEEPS a draft token the target would reject
// departs from target-only decode at exactly the first drift step, and the
// parity oracle pins the first divergence there — reference carrying the target
// token, engine carrying the wrongly-kept draft proposal.
func TestSpecDecodeLenientAcceptFailsAtFirstKeptDraft(t *testing.T) {
	const seed = 42
	c := SpecDecodeCase(seed)
	res, err := RunCase(c, ReferenceRunner{}, SpecDecodeEngine("lenient-accept"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("lenient-accept engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing speculative run must carry a failure bundle")
	}
	if fb.FailingOracle != "speculative-parity" {
		t.Errorf("first failing oracle = %q, want speculative-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != specFirstDriftStep {
		t.Fatalf("expected first divergence at the first drift step %d, got %+v", specFirstDriftStep, d)
	}
	if want := specTargetToken(seed, specFirstDriftStep); d.Reference != want {
		t.Errorf("divergence reference token = %q, want target token %q", d.Reference, want)
	}
	if want := specDraftToken(seed, specFirstDriftStep); d.Engine != want {
		t.Errorf("divergence engine token = %q, want wrongly-kept draft token %q", d.Engine, want)
	}
	if !strings.Contains(fb.Detail, "accept rule") {
		t.Errorf("detail should name the accept rule as the defect surface; got %q", fb.Detail)
	}
	// The prefix BEFORE the wrongly-kept token matched: localization did real
	// work — the defect is pinned to a step, not smeared over the whole run.
	for i := 0; i < specFirstDriftStep; i++ {
		if fb.Engine.Tokens[i] != fb.Reference.Tokens[i] {
			t.Fatalf("prefix token %d should match before the first kept draft; engine %q, reference %q",
				i, fb.Engine.Tokens[i], fb.Reference.Tokens[i])
		}
	}
}

// TestSpecDecodeParityLengthDivergence covers the truncation face of the parity
// contract directly at the oracle: a speculative path that stops short of the
// target-only stream (e.g. a discarded block never re-decoded) is a parity
// failure localized at the first missing position, even when every emitted
// token individually matched.
func TestSpecDecodeParityLengthDivergence(t *testing.T) {
	const seed = 7
	c := SpecDecodeCase(seed)
	ref := specTargetDecode(seed, c.Params.MaxTokens)
	short := Trace{Tokens: ref.Tokens[:len(ref.Tokens)-2], Text: strings.Join(ref.Tokens[:len(ref.Tokens)-2], " ")}
	v := SpecDecodeParity{}.Judge(ref, short, c)
	if v.Pass {
		t.Fatal("a truncated speculative stream must not judge as parity")
	}
	d := v.FirstDivergence
	if d == nil || d.Index != len(ref.Tokens)-2 {
		t.Fatalf("expected length divergence at index %d, got %+v", len(ref.Tokens)-2, d)
	}
	if d.Engine != "<end>" {
		t.Errorf("engine side of a truncation divergence = %q, want <end>", d.Engine)
	}
}
