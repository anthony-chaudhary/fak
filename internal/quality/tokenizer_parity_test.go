package quality

import (
	"strings"
	"testing"
)

// tokParityIndexOf locates the first occurrence of id in toks, failing the test
// if it is absent — the reference must actually carry the special token whose
// corruption a defect test is about to localize.
func tokParityIndexOf(t *testing.T, toks []string, id string) int {
	t.Helper()
	i := tokParityFirst(toks, id)
	if i < 0 {
		t.Fatalf("reference rendering does not contain id %s: %v", id, toks)
	}
	return i
}

// tokParityRunDefect runs the parity case against an engine with the given
// injected defect and returns the (required) failure bundle.
func tokParityRunDefect(t *testing.T, defect string) *FailureBundle {
	t.Helper()
	c := TokenizerParityCase()
	res, err := RunCase(c, ReferenceRunner{}, TokenizerParityEngine(defect), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase(%s): %v", defect, err)
	}
	if res.Pass {
		t.Fatalf("engine with injected %q defect must not pass; got %s", defect, Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatalf("failing %q run must carry a failure bundle", defect)
	}
	if fb.FailingOracle != "tokenizer-parity" {
		t.Errorf("first failing oracle = %q, want tokenizer-parity", fb.FailingOracle)
	}
	return fb
}

// TestTokenizerParityFaithfulPasses is the happy path: a faithful engine
// renders the message list to the exact pinned id sequence and the oracle
// passes with no failure bundle. It also pins the structural contract that the
// special tokens are IN-BAND in the compared surface — BOS first, EOS last,
// both role markers and both end-of-turn markers present — so a green here is a
// green over the special tokens, not just the content words.
func TestTokenizerParityFaithfulPasses(t *testing.T) {
	c := TokenizerParityCase()
	ref := c.Reference.Tokens
	if ref[0] != tokParityID(tokParityBOSID) {
		t.Fatalf("reference must start with <bos>; got %s", ref[0])
	}
	if ref[len(ref)-1] != tokParityID(tokParityEOSID) {
		t.Fatalf("reference must end with <eos>; got %s", ref[len(ref)-1])
	}
	for _, id := range []int{tokParitySystemID, tokParityUserID, tokParityEOTID} {
		tokParityIndexOf(t, ref, tokParityID(id))
	}

	res, err := RunCase(c, ReferenceRunner{}, TokenizerParityEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful template engine should render id-exact; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean rendering must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if got := strings.Join(res.Provenance.Engine.Tokens, " "); got != strings.Join(ref, " ") {
		t.Fatalf("faithful engine ids differ from reference:\n  ref: %v\n  eng: %v", ref, res.Provenance.Engine.Tokens)
	}
}

// TestTokenizerParityMissingBOSFailsAtZero: an engine that never prepends BOS
// diverges at id 0 — the reference has <bos> there, the engine has the system
// role marker — and the Detail names the special token by its marker.
func TestTokenizerParityMissingBOSFailsAtZero(t *testing.T) {
	fb := tokParityRunDefect(t, tokParityDefectMissingBOS)
	d := fb.FirstDivergence
	if d == nil || d.Index != 0 {
		t.Fatalf("missing-BOS must diverge at id 0, got %+v", d)
	}
	if d.Reference != tokParityID(tokParityBOSID) {
		t.Errorf("divergence reference id = %s, want <bos> id %s", d.Reference, tokParityID(tokParityBOSID))
	}
	if d.Engine != tokParityID(tokParitySystemID) {
		t.Errorf("divergence engine id = %s, want the shifted <|system|> id %s", d.Engine, tokParityID(tokParitySystemID))
	}
	if !strings.Contains(fb.Detail, "<bos>") {
		t.Errorf("detail should name the <bos> marker; got %q", fb.Detail)
	}
}

// TestTokenizerParityDoubleBOSFailsAtOne: the double-BOS bug leaves id 0 intact
// (both traces start with <bos>) and first diverges at id 1, where the engine
// repeats <bos> and the reference has the system role marker — proving the
// localization points at the duplicate, not merely at "the sequence differs".
func TestTokenizerParityDoubleBOSFailsAtOne(t *testing.T) {
	fb := tokParityRunDefect(t, tokParityDefectDoubleBOS)
	d := fb.FirstDivergence
	if d == nil || d.Index != 1 {
		t.Fatalf("double-BOS must first diverge at id 1, got %+v", d)
	}
	if d.Reference != tokParityID(tokParitySystemID) {
		t.Errorf("divergence reference id = %s, want <|system|> id %s", d.Reference, tokParityID(tokParitySystemID))
	}
	if d.Engine != tokParityID(tokParityBOSID) {
		t.Errorf("divergence engine id = %s, want the duplicated <bos> id %s", d.Engine, tokParityID(tokParityBOSID))
	}
}

// TestTokenizerParityWrongRoleMarkerFails: rendering the user turn under the
// assistant marker diverges exactly at the user role-marker position, with both
// role ids reported.
func TestTokenizerParityWrongRoleMarkerFails(t *testing.T) {
	c := TokenizerParityCase()
	want := tokParityIndexOf(t, c.Reference.Tokens, tokParityID(tokParityUserID))
	fb := tokParityRunDefect(t, tokParityDefectWrongRoleMarker)
	d := fb.FirstDivergence
	if d == nil || d.Index != want {
		t.Fatalf("wrong role marker must diverge at the user-marker id %d, got %+v", want, d)
	}
	if d.Reference != tokParityID(tokParityUserID) || d.Engine != tokParityID(tokParityAssistantID) {
		t.Errorf("divergence ids = ref %s eng %s, want ref %s (<|user|>) eng %s (<|assistant|>)",
			d.Reference, d.Engine, tokParityID(tokParityUserID), tokParityID(tokParityAssistantID))
	}
}

// TestTokenizerParityDroppedRoleMarkerFails: dropping the user marker shifts
// the user content left, so the first divergence is at the marker's position —
// reference <|user|>, engine the first content word id of the user turn.
func TestTokenizerParityDroppedRoleMarkerFails(t *testing.T) {
	c := TokenizerParityCase()
	want := tokParityIndexOf(t, c.Reference.Tokens, tokParityID(tokParityUserID))
	fb := tokParityRunDefect(t, tokParityDefectDroppedMarker)
	d := fb.FirstDivergence
	if d == nil || d.Index != want {
		t.Fatalf("dropped role marker must diverge at the user-marker id %d, got %+v", want, d)
	}
	if d.Reference != tokParityID(tokParityUserID) {
		t.Errorf("divergence reference id = %s, want <|user|> id %s", d.Reference, tokParityID(tokParityUserID))
	}
	if wantEng := tokParityID(tokParityWordID("summarize")); d.Engine != wantEng {
		t.Errorf("divergence engine id = %s, want shifted first user word id %s", d.Engine, wantEng)
	}
}

// TestTokenizerParityWrongEOTIDFails: a stale special-token table (every
// <|eot|> rendered under a legacy id) first diverges at the FIRST end-of-turn
// position, and the Detail names the <|eot|> marker the engine got wrong.
func TestTokenizerParityWrongEOTIDFails(t *testing.T) {
	c := TokenizerParityCase()
	want := tokParityIndexOf(t, c.Reference.Tokens, tokParityID(tokParityEOTID))
	fb := tokParityRunDefect(t, tokParityDefectWrongEOTID)
	d := fb.FirstDivergence
	if d == nil || d.Index != want {
		t.Fatalf("wrong special-token id must first diverge at the first <|eot|> id %d, got %+v", want, d)
	}
	if d.Reference != tokParityID(tokParityEOTID) || d.Engine != tokParityID(tokParityLegacyEOTID) {
		t.Errorf("divergence ids = ref %s eng %s, want ref %s (<|eot|>) eng %s (legacy)",
			d.Reference, d.Engine, tokParityID(tokParityEOTID), tokParityID(tokParityLegacyEOTID))
	}
	if !strings.Contains(fb.Detail, "<|eot|>") {
		t.Errorf("detail should name the <|eot|> marker; got %q", fb.Detail)
	}
}
