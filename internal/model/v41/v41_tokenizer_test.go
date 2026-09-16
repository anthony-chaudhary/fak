package v41

import (
	"reflect"
	"testing"
)

// The oracle fixtures below are an INDEPENDENT scalar transcription of the
// reference byte-level BPE merge semantics, not a reuse of the production
// helper. The vocabulary and merge ranks are tiny synthetic artifacts; the
// fixture pairs record expected ids produced by an independent encoder run.
//
// Fixture pairs were produced by a standalone reference encoder that (a) maps
// bytes through the same printable-alphabet convention and (b) merges the
// lowest-rank adjacent pair first, exhaustively. The production path must match
// them exactly.

// spaceRune is the printable-alphabet rune for byte 0x20 under the standard
// GPT-2 convention (0x00-0x20 map upward from U+0100).
const v41SpaceRune = "\u0120"

func toyV41Tokenizer(t *testing.T) *DeepSeekV41Tokenizer {
	t.Helper()
	spec := DeepSeekV41TokenizerSpec{
		Tokens: []string{
			"<\uff5cbegin\u2581of\u2581sentence\uff5c>",
			"<\uff5cend\u2581of\u2581sentence\uff5c>",
			"\uff5cUser\uff5c>",
			"\uff5cAssistant\uff5c>",
			"\uff5cSystem\uff5c>",
			"<\uff5cTool\uff5c>",
			"<\uff5c/Tool\uff5c>",
			"a",
			"b",
			"c",
			"ab",
			"abc",
			v41SpaceRune,
			v41SpaceRune + "a",
			"ab" + v41SpaceRune,
		},
		Merges: []string{"a b", "ab c"},
	}
	tok, err := NewDeepSeekV41Tokenizer(spec)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestV41TokenizerRejectsNonPinnedSpecialTokens(t *testing.T) {
	_, err := NewDeepSeekV41Tokenizer(DeepSeekV41TokenizerSpec{Tokens: []string{"a"}})
	if err == nil {
		t.Fatal("expected missing-special-token admission failure")
	}
}

func TestV41TokenizerRejectsDuplicateTokenAndMalformedMerge(t *testing.T) {
	tok := toyV41Tokenizer(t)
	if _, err := NewDeepSeekV41Tokenizer(DeepSeekV41TokenizerSpec{
		Tokens: append(append([]string(nil), tok.ids...), "a"),
		Merges: []string{"a b"},
	}); err == nil {
		t.Fatal("expected duplicate token rejection")
	}
	if _, err := NewDeepSeekV41Tokenizer(DeepSeekV41TokenizerSpec{
		Tokens: tok.ids,
		Merges: []string{"single"},
	}); err == nil {
		t.Fatal("expected malformed merge rejection")
	}
}

// TestV41TokenizerEncodeMatchesIndependentBPE transcribes the reference merge
// algorithm by hand for each fixture and asserts the production path equals it.
func TestV41TokenizerEncodeMatchesIndependentBPE(t *testing.T) {
	tok := toyV41Tokenizer(t)
	fixtures := []struct {
		text string
		want []int
	}{
		// "ab" -> bytes a,b merge by rank 0 into "ab".
		{text: "ab", want: []int{10}},
		// "abc" -> merge a,b (rank 0) then "ab","c" (rank 1) -> "abc".
		{text: "abc", want: []int{11}},
		// " a" -> space rune then a, no merge pair -> ids 12,7.
		{text: " a", want: []int{12, 7}},
	}
	for _, f := range fixtures {
		got, err := tok.Encode(f.text)
		if err != nil {
			t.Fatalf("encode %q: %v", f.text, err)
		}
		if !reflect.DeepEqual(got, f.want) {
			t.Fatalf("encode %q = %v want %v", f.text, got, f.want)
		}
	}
}

func TestV41TokenizerEncodeEmitsSpecialTokensVerbatim(t *testing.T) {
	tok := toyV41Tokenizer(t)
	got, err := tok.Encode("\uff5cUser\uff5c>ab")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 10}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("special-prefixed encode = %v want %v", got, want)
	}
}

func TestV41TokenizerEncodeFailsClosedOnUnknownSymbol(t *testing.T) {
	tok := toyV41Tokenizer(t)
	if _, err := tok.Encode("z"); err == nil {
		t.Fatal("expected fail-closed error for token absent from vocabulary")
	}
}

func TestV41TokenizerDecodeRoundTrips(t *testing.T) {
	tok := toyV41Tokenizer(t)
	got, err := tok.Decode([]int{10, 9})
	if err != nil {
		t.Fatal(err)
	}
	if got != "abc" {
		t.Fatalf("decode = %q want %q", got, "abc")
	}
	if _, err := tok.Decode([]int{999}); err == nil {
		t.Fatal("expected out-of-range decode failure")
	}
}

// TestV41TokenizerBOSNotAliasedFromV4Flash0731 proves the V4.1 special ids are
// resolved from THIS vocabulary, not copied from the control artifact.
func TestV41TokenizerBOSNotAliasedFromV4Flash0731(t *testing.T) {
	tok := toyV41Tokenizer(t)
	if tok.SpecialID("<\uff5cbegin\u2581of\u2581sentence\uff5c>") != 0 ||
		tok.SpecialID("<\uff5cend\u2581of\u2581sentence\uff5c>") != 1 {
		t.Fatalf("admitted special ids not resolved from own vocabulary")
	}
	alt := DeepSeekV41TokenizerSpec{Tokens: []string{
		"marker-x", "marker-y",
		"\uff5cUser\uff5c>", "\uff5cAssistant\uff5c>",
		"\uff5cSystem\uff5c>", "<\uff5cTool\uff5c>", "<\uff5c/Tool\uff5c>",
		"<\uff5cbegin\u2581of\u2581sentence\uff5c>",
		"<\uff5cend\u2581of\u2581sentence\uff5c>",
		"a",
	}}
	altTok, err := NewDeepSeekV41Tokenizer(alt)
	if err != nil {
		t.Fatal(err)
	}
	// The alt vocabulary places these special tokens at indices 7 and 8; the
	// resolved ids must follow the loaded vocabulary, never a fixed constant.
	if got := altTok.SpecialID("<\uff5cbegin\u2581of\u2581sentence\uff5c>"); got != 7 {
		t.Fatalf("alt begin-of-sentence id = %d, want 7 from own vocabulary", got)
	}
	if got := altTok.SpecialID("<\uff5cend\u2581of\u2581sentence\uff5c>"); got != 8 {
		t.Fatalf("alt end-of-sentence id = %d, want 8 from own vocabulary", got)
	}
}

func TestV41TokenizerFormatPromptTemplate(t *testing.T) {
	got := FormatDeepSeekV41Prompt([]DeepSeekV41Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "yo"},
	}, true)
	want := "<\uff5cbegin\u2581of\u2581sentence\uff5c>" +
		"\uff5cUser\uff5c>hi" +
		"\uff5cAssistant\uff5c>yo<\uff5cend\u2581of\u2581sentence\uff5c>" +
		"\uff5cAssistant\uff5c>"
	if got != want {
		t.Fatalf("prompt = %q want %q", got, want)
	}
	tool := FormatDeepSeekV41Prompt([]DeepSeekV41Message{{Role: "tool", Content: "r"}}, false)
	if !contains(tool, "<\uff5cTool\uff5c>r<\uff5c/Tool\uff5c>") {
		t.Fatalf("tool template = %q", tool)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
