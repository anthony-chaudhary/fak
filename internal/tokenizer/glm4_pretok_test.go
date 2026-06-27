package tokenizer

import (
	"reflect"
	"testing"
)

// TestGLM4PreTokenizerDetection pins that GLM-5.2's tokenizer.ggml.pre=glm4 routes to the GLM-4
// pre-tokenizer, not the GPT-2 ByteLevel default (which produced garbage GLM-5.2 output).
func TestGLM4PreTokenizerDetection(t *testing.T) {
	for _, pre := range []string{"glm4", "GLM4", "glm-4", "chatglm", "chatglm-bpe"} {
		if !isGLM4PreTokenizer(pre) {
			t.Errorf("isGLM4PreTokenizer(%q) = false, want true", pre)
		}
	}
	for _, pre := range []string{"gpt2", "qwen2", "llama-bpe", ""} {
		if isGLM4PreTokenizer(pre) {
			t.Errorf("isGLM4PreTokenizer(%q) = true, want false", pre)
		}
	}
}

// TestGLM4PreTokenizerDigitGrouping pins the ONE regex difference from Qwen2: GLM-4 groups digits
// in runs of 1-3 (\p{N}{1,3}), Qwen2 emits one digit at a time (\p{N}). Everything else is shared.
func TestGLM4PreTokenizerDigitGrouping(t *testing.T) {
	cases := []struct {
		in       string
		wantGLM4 []string
		wantQwen []string
	}{
		{"123", []string{"123"}, []string{"1", "2", "3"}},
		{"1234", []string{"123", "4"}, []string{"1", "2", "3", "4"}},
		{"a12b", []string{"a", "12", "b"}, []string{"a", "1", "2", "b"}},
		{"007", []string{"007"}, []string{"0", "0", "7"}},
	}
	for _, tc := range cases {
		g := preTokenizeGLM4(tc.in)
		if !reflect.DeepEqual(g, tc.wantGLM4) {
			t.Errorf("preTokenizeGLM4(%q) = %q, want %q", tc.in, g, tc.wantGLM4)
		}
		q := preTokenizeQwen(tc.in)
		if !reflect.DeepEqual(q, tc.wantQwen) {
			t.Errorf("preTokenizeQwen(%q) = %q, want %q (control)", tc.in, q, tc.wantQwen)
		}
	}
}

// TestGLM4PreTokenizerDiffersFromByteLevel pins that GLM-4 is NOT the GPT-2 ByteLevel default — the
// exact mismatch that mis-tokenized GLM-5.2. On a string with leading punctuation + a contraction,
// the GPT-2 and GLM-4/Qwen families split differently.
func TestGLM4PreTokenizerDiffersFromByteLevel(t *testing.T) {
	// "(can't" — GPT-2 ByteLevel splits "(" then " ?\p{L}+" cannot attach the leading "(" to a
	// following letter run, while the GLM-4/Qwen family attaches a single non-alnum to the letters
	// ("[^\r\n\p{L}\p{N}]?\p{L}+" -> "(can"). So the two families MUST differ here.
	in := "(can't do"
	bl := preTokenizeByteLevel(in)
	g4 := preTokenizeGLM4(in)
	if reflect.DeepEqual(bl, g4) {
		t.Fatalf("GLM-4 pre-tokenizer must differ from GPT-2 ByteLevel on %q (both = %q) — the routing fix would be a no-op", in, bl)
	}
	// GLM-4 attaches the leading "(" to "can": the first token is "(can".
	if len(g4) == 0 || g4[0] != "(can" {
		t.Errorf("preTokenizeGLM4(%q)[0] = %q, want \"(can\"", in, firstOr(g4, ""))
	}
}

func firstOr(s []string, d string) string {
	if len(s) > 0 {
		return s[0]
	}
	return d
}
