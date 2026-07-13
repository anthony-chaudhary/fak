package quality

import (
	"strings"
	"testing"
)

// uniBytes builds a token fragment from raw bytes, keeping the fixtures free of
// escape-sequence ambiguity: uniBytes(0xE4, 0xB8) is the first two bytes of the
// three-byte CJK codepoint U+4E16.
func uniBytes(b ...byte) string { return string(b) }

// uniCJK is U+4E16 (three UTF-8 bytes E4 B8 96), the codepoint the split
// fixture cuts across a token boundary.
var uniCJK = string(rune(0x4E16))

// uniCheck is U+2713 (three UTF-8 bytes E2 9C 93), the codepoint the
// byte-fallback fixture reassembles from three "<0xNN>" tokens.
var uniCheck = string(rune(0x2713))

// uniSplitCase is the split-codepoint fixture (#4533): the three-byte U+4E16 is
// cut across tokens 1 and 2 — token 1 ends with its first two bytes, token 2
// begins with its last byte. The reference text is the full decode.
func uniSplitCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "incremental-unicode-split-codepoint",
		Version: 1,
		Prompt:  "Stream-detokenize the greeting byte fragment by byte fragment.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 8},
		Reference: Trace{
			Tokens: []string{"He", uniBytes(0xE4, 0xB8), uniBytes(0x96) + "llo", "!"},
			Text:   "He" + uniCJK + "llo!",
		},
		Oracles: []string{"incremental-unicode"},
	}
}

// uniSplitEngine mirrors the DemoEngine defect-injection pattern: "" replays a
// faithful incremental decoder that buffers the two lead bytes across the
// token-1 boundary; "eager-flush" flushes them as U+FFFD at that boundary
// instead of waiting for the final byte (the stray continuation byte on token 2
// then decays to a third U+FFFD).
func uniSplitEngine(defect string) ScriptedRunner {
	ref := uniSplitCase().Reference
	switch defect {
	case "eager-flush":
		return ScriptedRunner{
			Label: "engine-eager-flush",
			Trace: Trace{
				Tokens: ref.Tokens,
				Text:   "He" + uniReplacement + uniReplacement + uniReplacement + "llo!",
			},
		}
	default:
		return ScriptedRunner{Label: "engine-clean", Trace: ref}
	}
}

// uniFallbackCase is the byte-fallback fixture (#4533): U+2713 arrives as three
// SentencePiece-style raw-byte tokens that a faithful incremental decoder
// buffers and reassembles into one rune.
func uniFallbackCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "incremental-unicode-byte-fallback",
		Version: 1,
		Prompt:  "Stream-detokenize a byte-fallback check mark.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 8},
		Reference: Trace{
			Tokens: []string{"Ok ", "<0xE2>", "<0x9C>", "<0x93>", "!"},
			Text:   "Ok " + uniCheck + "!",
		},
		Oracles: []string{"incremental-unicode"},
	}
}

// uniFallbackEngine: "" reassembles the fallback bytes correctly;
// "drop-continuation" loses the final continuation byte <0x93>, so the buffered
// lead bytes never complete and the check mark never materializes.
func uniFallbackEngine(defect string) ScriptedRunner {
	ref := uniFallbackCase().Reference
	switch defect {
	case "drop-continuation":
		return ScriptedRunner{
			Label: "engine-drop-continuation",
			Trace: Trace{Tokens: ref.Tokens, Text: "Ok !"},
		}
	default:
		return ScriptedRunner{Label: "engine-clean", Trace: ref}
	}
}

// TestIncrementalUnicodeSplitCleanPasses proves the faithful path: a three-byte
// codepoint split across two tokens decodes to exactly the reference text when
// the engine buffers the partial sequence across the boundary.
func TestIncrementalUnicodeSplitCleanPasses(t *testing.T) {
	c := uniSplitCase()
	res, err := RunCase(c, ReferenceRunner{}, uniSplitEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful split-codepoint decode should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if d := res.Verdicts[0].Detail; !strings.Contains(d, "1 split boundar") {
		t.Errorf("pass detail should count the buffered split boundary; got %q", d)
	}
}

// TestIncrementalUnicodeEagerFlushFails proves the eager-flush gate: an engine
// that flushes the two buffered lead bytes as U+FFFD at the token-1 boundary
// (instead of buffering until the codepoint completes) fails AT token 1, with
// the bad boundary described.
func TestIncrementalUnicodeEagerFlushFails(t *testing.T) {
	c := uniSplitCase()
	res, err := RunCase(c, ReferenceRunner{}, uniSplitEngine("eager-flush"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("U+FFFD at a boundary that should have been buffered must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "incremental-unicode" || fb.FailingKind != "differential" {
		t.Errorf("failing oracle = %q (%s), want incremental-unicode (differential)", fb.FailingOracle, fb.FailingKind)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 1 {
		t.Fatalf("expected first divergence at token 1 (the token ending mid-codepoint), got %+v", fb.FirstDivergence)
	}
	if !strings.HasPrefix(fb.FirstDivergence.Engine, uniReplacement) {
		t.Errorf("engine side of the divergence should show the replacement char; got %q", fb.FirstDivergence.Engine)
	}
	if !strings.HasPrefix(fb.FirstDivergence.Reference, uniCJK) {
		t.Errorf("reference side of the divergence should show the completed codepoint; got %q", fb.FirstDivergence.Reference)
	}
	if !strings.Contains(fb.Detail, "U+FFFD") || !strings.Contains(fb.Detail, "buffering") {
		t.Errorf("detail must describe the bad boundary (U+FFFD instead of buffering); got %q", fb.Detail)
	}
}

// TestIncrementalUnicodeByteFallbackCleanPasses proves byte-fallback
// reassembly: three "<0xNN>" raw-byte tokens buffered across two boundaries
// decode to the single reference codepoint.
func TestIncrementalUnicodeByteFallbackCleanPasses(t *testing.T) {
	c := uniFallbackCase()
	res, err := RunCase(c, ReferenceRunner{}, uniFallbackEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful byte-fallback reassembly should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if d := res.Verdicts[0].Detail; !strings.Contains(d, "2 split boundar") {
		t.Errorf("pass detail should count both buffered fallback boundaries; got %q", d)
	}
}

// TestIncrementalUnicodeDroppedContinuationFails proves the dropped-byte gate:
// an engine that loses the final continuation byte fails at token 3 — the
// token whose byte should have completed the codepoint — with the reference
// side showing the rune that never materialized.
func TestIncrementalUnicodeDroppedContinuationFails(t *testing.T) {
	c := uniFallbackCase()
	res, err := RunCase(c, ReferenceRunner{}, uniFallbackEngine("drop-continuation"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("dropped continuation byte must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "incremental-unicode" {
		t.Errorf("failing oracle = %q, want incremental-unicode", fb.FailingOracle)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 3 {
		t.Fatalf("expected first divergence at token 3 (the dropped continuation byte), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != uniCheck {
		t.Errorf("reference side should show the codepoint that never materialized; got %q", fb.FirstDivergence.Reference)
	}
	if fb.FirstDivergence.Engine != "!" {
		t.Errorf("engine side should show the text that replaced it; got %q", fb.FirstDivergence.Engine)
	}
	if !strings.Contains(fb.Detail, "token 3") {
		t.Errorf("detail must localize the divergence to token 3; got %q", fb.Detail)
	}
}
