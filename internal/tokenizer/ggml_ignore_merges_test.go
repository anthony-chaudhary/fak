package tokenizer

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
// Estimate only: tiny in-memory vocabulary cases; not measured.
func TestGGMLGLMWholePieceBeforeMerges(t *testing.T) {
	tokens := []string{"a", "b", "c", "ab", "abc", "<|special|>", "Ġ", "Ġabc"}
	merges := []string{"a b"}
	for _, pre := range []string{"glm4", "GLM4", "glm5", "GLM5"} {
		t.Run(pre, func(t *testing.T) {
			tok, err := FromGGML(tokens, merges, nil, pre)
			if err != nil {
				t.Fatal(err)
			}
			if tok.preTokKind != preTokGLM4 {
				t.Fatalf("pre-tokenizer = %v, want GLM4", tok.preTokKind)
			}
			for _, tc := range []struct {
				text string
				want []int
			}{
				{"abc", []int{4}},
				{"abca", []int{3, 2, 0}}, // No whole piece: retain ranked BPE.
				{" abc", []int{7}},       // Look up byte-level encoded vocabulary, not raw text.
				{"abc<|special|>abc", []int{4, 5, 4}},
			} {
				got, err := tok.Encode(tc.text)
				if err != nil || !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("Encode(%q) = %v, %v; want %v", tc.text, got, err, tc.want)
				}
			}
		})
	}
	for _, pre := range []string{"chatglm-bpe", "chatglm", "glm-4", "default", "qwen2"} {
		t.Run("unchanged_"+pre, func(t *testing.T) {
			tok, err := FromGGML(tokens, merges, nil, pre)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tok.Encode("abc")
			if err != nil || !reflect.DeepEqual(got, []int{3, 2}) {
				t.Fatalf("Encode = %v, %v; want ordinary merges [3 2]", got, err)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
// Estimate only: small canonical digests and JSON fixture; not measured.
func TestGGMLGLMWholePieceIdentity(t *testing.T) {
	makeTok := func(pre string) *Tokenizer {
		t.Helper()
		tok, err := FromGGML([]string{"a", "b", "c", "ab", "abc"}, []string{"a b"}, nil, pre)
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	glm4, glm5, legacy := makeTok("glm4"), makeTok("glm5"), makeTok("chatglm-bpe")
	if glm4.Identity() != glm5.Identity() {
		t.Fatal("equivalent GLM4/5 operational state has different identities")
	}
	if glm4.Identity() == legacy.Identity() {
		t.Fatal("whole-piece preference was omitted from operational identity")
	}
	// Both use exactly the same split, vocabulary and merges. Disabling the new
	// behavior before the first Identity call must recover the legacy digest.
	disabled := makeTok("glm4")
	disabled.ignoreMerges = false
	if disabled.Identity() != legacy.Identity() {
		t.Fatal("false-mode identity changed")
	}
	jsonTok, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},"merges":["a b"]},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	got, err := jsonTok.Encode("abc")
	if err != nil || !reflect.DeepEqual(got, []int{3, 2}) {
		t.Fatalf("JSON constructor changed: %v, %v", got, err)
	}
}
