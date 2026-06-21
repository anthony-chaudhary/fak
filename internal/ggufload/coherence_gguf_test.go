package ggufload

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// TestGGUFChatCoherence is the end-to-end correctness gate that the math-only kernel
// tests could never catch: it loads a REAL GGUF through the exact path the chat demo
// uses (LoadModelQuant + the embedded tokenizer), greedily decodes a ChatML prompt, and
// asserts the model actually ANSWERS — i.e. the forward pass is coherent, not just
// arithmetically self-consistent. It is the regression test for the rotary-unpermute bug
// that made every GGUF Qwen2.5 chat emit context-blind garbage while llama.cpp on the
// same file answered correctly.
//
// Model discovery (skips cleanly when no weights are present, so it never blocks CI on a
// box without models):
//   - FAK_TEST_GGUF=/path/to/model.gguf  (explicit; any Qwen2.5 instruct Q8/Q4 GGUF), or
//   - the standard cache: ~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf
func TestGGUFChatCoherence(t *testing.T) {
	path := os.Getenv("FAK_TEST_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		cand := filepath.Join(home, ".cache", "fak-models", "gguf", "Qwen2.5-1.5B-Instruct.Q8_0.gguf")
		if _, err := os.Stat(cand); err == nil {
			path = cand
		}
	}
	if path == "" {
		t.Skip("no GGUF available: set FAK_TEST_GGUF=/path/to/Qwen2.5-*.gguf to run the chat-coherence gate")
	}

	// Embedded tokenizer — the same offline path the demo relies on.
	f, err := Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Skipf("%s has no embedded tokenizer", filepath.Base(path))
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}

	// Stop tokens, declared by the model (mirrors cmd/simpledemo).
	stops := map[int]bool{}
	for id, content := range tok.SpecialTokens() {
		if content == "<|im_end|>" || content == "<|endoftext|>" {
			stops[id] = true
		}
	}
	if v, ok := f.Uint64("tokenizer.ggml.eos_token_id"); ok {
		stops[int(v)] = true
	}

	m, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}

	const prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	s := m.NewSession()
	s.Quant = true
	logits := s.Prefill(ids)

	var gen []int
	for i := 0; i < 24; i++ {
		next := argmaxF32(logits)
		if stops[next] {
			break
		}
		gen = append(gen, next)
		logits = s.Step(next)
	}
	out, err := tok.Decode(gen)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// The greedy answer to "capital of France" must contain "Paris". Under the rotary
	// bug this was punctuation/space/special-token garbage with no "Paris" anywhere.
	if !strings.Contains(strings.ToLower(out), "paris") {
		t.Fatalf("incoherent GGUF chat (rotary/forward regression): greedy answer = %q; want it to contain \"Paris\".\n"+
			"model=%s — compare against `llama-cli -m <same.gguf> -p \"What is the capital of France?\" -st --temp 0`",
			out, filepath.Base(path))
	}
	t.Logf("coherent: %q (model=%s)", out, filepath.Base(path))
}

func argmaxF32(v []float32) int {
	best, idx := float32(-math.MaxFloat32), 0
	for i, x := range v {
		if x > best {
			best, idx = x, i
		}
	}
	return idx
}
