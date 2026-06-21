package main

import (
	"bufio"
	"io"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestLooksDegenerate pins the issue #91 detector: the exact failure modes the bug
// report observed must be flagged, and ordinary coherent replies must not be. This
// runs everywhere (no model needed) and is the always-on half of the greedy
// non-degeneracy guard.
func TestLooksDegenerate(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
	}{
		// --- issue #91 failure modes: MUST be flagged ---
		{"sampling loop digits", "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2", true},
		{"greedy role loop", strings.Repeat(".assistant", 8), true},
		{"single word loop", strings.Repeat("hello ", 12), true},
		{"newline role loop", strings.Repeat("<|im_start|>assistant\n", 4), true},

		// --- coherent replies: MUST NOT be flagged ---
		{"arithmetic answer", "2+2 equals 4.", false},
		{"identity answer", "I'm a large language model created to be helpful, harmless, and honest. How can I help you today?", false},
		{"short curt answer", "Blue", false},
		{"list answer", "Here are three colors: red, green, and blue. Each is a primary or secondary color.", false},
		{"empty", "", false},
		{"whitespace only", "   \n  ", false},
		// A sentence that happens to repeat a common word a few times is still fine.
		{"natural mild repeat", "The cat sat on the mat and then the cat looked at the other cat nearby.", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := looksDegenerate(tc.text); got != tc.want {
				t.Fatalf("looksDegenerate(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

// TestGreedyNonDegenerate is the end-to-end guard for issue #91: it drives the
// real demo path — build the ChatML prompt, encode, greedy-decode through
// decodeReply, decode to text — on a recommended GGUF and asserts the reply is
// coherent (non-empty, non-degenerate) and deterministic across reruns.
//
// It needs a real model, so it is gated on FAK_SIMPLEDEMO_GGUF and skips otherwise,
// keeping `go test ./...` fast and offline. Run it against the recommended model:
//
//	FAK_SIMPLEDEMO_GGUF="$HOME/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf" \
//	    go test ./cmd/simpledemo/ -run TestGreedyNonDegenerate -v
func TestGreedyNonDegenerate(t *testing.T) {
	ggufPath := os.Getenv("FAK_SIMPLEDEMO_GGUF")
	if ggufPath == "" {
		t.Skip("set FAK_SIMPLEDEMO_GGUF=<path to a .gguf> to run the greedy non-degeneracy guard")
	}
	if _, err := os.Stat(ggufPath); err != nil {
		t.Fatalf("FAK_SIMPLEDEMO_GGUF=%q: %v", ggufPath, err)
	}

	// Load through the same loader the demo uses (embedded tokenizer, no download).
	m, tok, _, _, err := loadModel(ggufPath, "", false, true)
	if err != nil {
		t.Fatalf("loadModel(%q): %v", ggufPath, err)
	}
	stops := stopTokenIDs(tok, ggufPath)

	// greedy runs one reply with temperature 0 through the exact production path.
	greedy := func(userMsg string) (string, int) {
		prompt := buildPrompt("You are a helpful assistant. Keep answers short and clear.", nil, userMsg)
		ids, err := tok.Encode(prompt)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		session := m.NewSession()
		session.Quant = true
		logits := session.Prefill(ids)
		out := bufio.NewWriter(io.Discard)
		rng := rand.New(rand.NewSource(1)) // unused at temp 0 (greedy argmax)
		return decodeReply(session, tok, logits, stops, 48, 0, rng, out)
	}

	const userMsg = "What is 2+2?"
	reply, gen := greedy(userMsg)
	if gen == 0 || strings.TrimSpace(reply) == "" {
		t.Fatalf("greedy produced no output (gen=%d, reply=%q)", gen, reply)
	}
	if looksDegenerate(reply) {
		t.Fatalf("greedy reply is degenerate (issue #91 regression): %q", reply)
	}

	// Greedy is deterministic: a structural decode bug would show up as drift.
	reply2, _ := greedy(userMsg)
	if reply != reply2 {
		t.Fatalf("greedy not deterministic across reruns:\n  run1=%q\n  run2=%q", reply, reply2)
	}
	t.Logf("greedy reply (%d tok): %q", gen, reply)
}
