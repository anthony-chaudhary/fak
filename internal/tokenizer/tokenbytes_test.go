package tokenizer

import (
	"bytes"
	"testing"
)

// TestTokenBytesKnownIDs pins the id->bytes accessor the guided-decode logit-mask
// adapter (#26) is built on: for each known fixture id, TokenBytes returns the
// exact bytes the token decodes to. It covers the GPT-2 ByteLevel remap (a leading
// ▁/Ġ becomes a real space, Ċ becomes a newline) and the special-token case, where
// the bytes come from the token's literal content, not the byte-level alphabet
// (the analogue of guideddecode's special-name-byte handling, commit 29d1b3055).
func TestTokenBytesKnownIDs(t *testing.T) {
	tok := loadFixtureTokenizer(t)

	cases := []struct {
		name string
		id   int
		want []byte
	}{
		{"plain ascii piece", 504, []byte("The")},
		{"leading-space piece (Ġthe)", 260, []byte(" the")},
		{"leading-space piece (Ġof)", 282, []byte(" of")},
		{"newline piece (Ċ)", 198, []byte("\n")},
		{"digit piece", 34, []byte("2")},
		{"dot piece", 30, []byte(".")},
		// Special tokens decode to the bytes of their literal content, verbatim —
		// NOT through the byte-level rune->byte map.
		{"special <|endoftext|>", 0, []byte("<|endoftext|>")},
		{"special <|im_start|>", 1, []byte("<|im_start|>")},
		{"special <|im_end|>", 2, []byte("<|im_end|>")},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := tok.TokenBytes(c.id)
			if !bytes.Equal(got, c.want) {
				t.Fatalf("TokenBytes(%d) = %q, want %q", c.id, got, c.want)
			}
		})
	}
}

// TestTokenBytesMatchesDecode is the load-bearing invariant: TokenBytes(id) is
// exactly the bytes Decode emits for the single-token slice []int{id}. Proving the
// accessor against the shipped Decode path (rather than a re-derived expectation)
// is what lets the adapter trust it as "the raw bytes this token decodes to".
func TestTokenBytesMatchesDecode(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	for _, id := range []int{0, 1, 2, 30, 34, 35, 36, 198, 260, 282, 314, 378, 504, 3575, 4649, 7042, 38634} {
		text, err := tok.Decode([]int{id})
		if err != nil {
			t.Fatalf("Decode([%d]): %v", id, err)
		}
		if got := tok.TokenBytes(id); string(got) != text {
			t.Fatalf("TokenBytes(%d)=%q != Decode([%d])=%q", id, got, id, text)
		}
	}
}

// TestTokenBytesUndecodable pins the nil contract: an out-of-range id, a sparse
// (missing) vocab slot, and — via a metaspace-free fixture — any non-decodable id
// return nil, which the adapter reads as "not an emittable token" and masks. The
// two error ids mirror Decode's rejects in TestTokenizerStageT2Boundary.
func TestTokenBytesUndecodable(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	if got := tok.TokenBytes(99999); got != nil {
		t.Fatalf("TokenBytes(out-of-range) = %q, want nil", got)
	}
	if got := tok.TokenBytes(123); got != nil {
		t.Fatalf("TokenBytes(missing sparse id) = %q, want nil", got)
	}
	if got := tok.TokenBytes(-1); got != nil {
		t.Fatalf("TokenBytes(-1) = %q, want nil", got)
	}
}
