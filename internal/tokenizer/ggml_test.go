package tokenizer

import (
	"reflect"
	"testing"
)

// TestFromGGMLRoundTrip builds a tiny byte-level BPE tokenizer from in-memory
// GGML arrays and checks encode/decode, special-token handling, and the
// pre-tokenizer dispatch — all without any model file on disk.
func TestFromGGMLRoundTrip(t *testing.T) {
	tokens := []string{"a", "b", "ab", "<|end|>"}
	types := []int32{1, 1, 1, ggmlTypeControl}
	merges := []string{"a b"}

	tok, err := FromGGML(tokens, merges, types, "")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}

	// "ab" merges to the single id 2.
	if got, err := tok.Encode("ab"); err != nil || !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("Encode(ab) = %v, %v; want [2]", got, err)
	}
	// Decode is the inverse.
	if got, err := tok.Decode([]int{2}); err != nil || got != "ab" {
		t.Fatalf("Decode([2]) = %q, %v; want \"ab\"", got, err)
	}
	if got, err := tok.Decode([]int{0, 1}); err != nil || got != "ab" {
		t.Fatalf("Decode([0 1]) = %q, %v; want \"ab\"", got, err)
	}

	// The control token is matched literally on both ends.
	if got, err := tok.Encode("<|end|>"); err != nil || !reflect.DeepEqual(got, []int{3}) {
		t.Fatalf("Encode(<|end|>) = %v, %v; want [3]", got, err)
	}
	if got, err := tok.Decode([]int{3}); err != nil || got != "<|end|>" {
		t.Fatalf("Decode([3]) = %q, %v; want \"<|end|>\"", got, err)
	}
	if sp := tok.SpecialTokens(); sp[3] != "<|end|>" || len(sp) != 1 {
		t.Fatalf("SpecialTokens() = %v; want {3:<|end|>}", sp)
	}
}

// TestFromGGMLMetaspaceGemma proves the gemma hint selects SentencePiece-style
// metaspace tokenization: a space is the ▁ marker (so it attaches to the following
// token), the vocab is literal UTF-8 (no GPT-2 byte-level remap), byte-fallback
// "<0xNN>" tokens decode to their raw byte, and decode maps ▁ back to a space.
func TestFromGGMLMetaspaceGemma(t *testing.T) {
	tokens := []string{"a", "b", "▁", "▁b", "<0x0A>"}
	merges := []string{"▁ b"} // ▁ + b -> ▁b
	tok, err := FromGGML(tokens, merges, nil, "gemma4")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	if !tok.metaspace {
		t.Fatal("gemma4 hint must select metaspace mode")
	}
	// "a b": the space becomes ▁ and merges onto b, so it is NOT a separate token.
	if got, err := tok.Encode("a b"); err != nil || !reflect.DeepEqual(got, []int{0, 3}) {
		t.Fatalf("Encode(a b) = %v, %v; want [0 3]", got, err)
	}
	if got, err := tok.Decode([]int{0, 3}); err != nil || got != "a b" {
		t.Fatalf("Decode([0 3]) = %q, %v; want \"a b\"", got, err)
	}
	// Byte-fallback token decodes to the raw byte it names.
	if got, err := tok.Decode([]int{4}); err != nil || got != "\n" {
		t.Fatalf("Decode([4]) = %q, %v; want newline", got, err)
	}
	// A non-gemma hint stays on the byte-level path.
	if bl, err := FromGGML(tokens, merges, nil, ""); err != nil || bl.metaspace {
		t.Fatalf("default hint must not be metaspace (err=%v)", err)
	}
}

// TestFromGGMLPreTokenizerDispatch proves the pre hint actually switches the
// pre-tokenizer: the Qwen Split attaches a leading non-alnum char to the
// following letter run ("(a" -> one piece -> merged), while the GPT-2 ByteLevel
// default keeps "(" and "a" separate.
func TestFromGGMLPreTokenizerDispatch(t *testing.T) {
	tokens := []string{"(", "a", "(a"}
	merges := []string{"( a"}

	byteLevel, err := FromGGML(tokens, merges, nil, "default")
	if err != nil {
		t.Fatalf("FromGGML(default): %v", err)
	}
	if got, err := byteLevel.Encode("(a"); err != nil || !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("ByteLevel Encode(\"(a\") = %v, %v; want [0 1]", got, err)
	}

	qwen, err := FromGGML(tokens, merges, nil, "qwen2")
	if err != nil {
		t.Fatalf("FromGGML(qwen2): %v", err)
	}
	if got, err := qwen.Encode("(a"); err != nil || !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("Qwen Encode(\"(a\") = %v, %v; want [2]", got, err)
	}

	// An empty pre falls back to ByteLevel; an arch-style "qwen2.5" hint also
	// trips the Qwen path (the caller may pass general.architecture).
	if !isQwenPreTokenizer("qwen2.5") || !isQwenPreTokenizer("qwen35moe") || isQwenPreTokenizer("smollm") {
		t.Fatalf("isQwenPreTokenizer dispatch wrong")
	}
}

// TestFromGGMLSpecialFallback marks "<|...|>" tokens special when token_type is
// absent, so a conversion that dropped token_type still stops on ChatML markers.
func TestFromGGMLSpecialFallback(t *testing.T) {
	tokens := []string{"a", "b", "ab", "<|im_end|>"}
	merges := []string{"a b"}
	tok, err := FromGGML(tokens, merges, nil /* no token_type */, "qwen2")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	if sp := tok.SpecialTokens(); sp[3] != "<|im_end|>" {
		t.Fatalf("expected <|im_end|> special via fallback, got %v", sp)
	}
}

func TestFromGGMLErrors(t *testing.T) {
	if _, err := FromGGML(nil, []string{"a b"}, nil, ""); err == nil {
		t.Fatal("want error on empty vocab")
	}
	if _, err := FromGGML([]string{"a", "b"}, nil, nil, ""); err == nil {
		t.Fatal("want error on no merges (non-BPE)")
	}
	if _, err := FromGGML([]string{"a", "b"}, []string{"a b"}, []int32{1}, ""); err == nil {
		t.Fatal("want error on token_type/vocab length mismatch")
	}
	if _, err := FromGGML([]string{"a", "b"}, []string{"ab"}, nil, ""); err == nil {
		t.Fatal("want error on malformed merge (no space)")
	}
}
