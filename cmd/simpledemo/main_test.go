package main

import (
	"bufio"
	"io"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestModelDownloadDerivesURL pins the filename→URL derivation that makes a missing
// "-gguf <name>" auto-download: mradermacher names quants with a DOT and ships one
// repo per model, but users type a dash. Both must resolve to the real file.
func TestModelDownloadDerivesURL(t *testing.T) {
	cases := []struct {
		in        string
		canonical string
		repoFrag  string
		ok        bool
	}{
		// dash-typed (as in the bug report) → canonical dot filename + per-size repo
		{"Qwen2.5-1.5B-Instruct-Q8_0.gguf", "Qwen2.5-1.5B-Instruct.Q8_0.gguf", "mradermacher/Qwen2.5-1.5B-Instruct-GGUF", true},
		// already canonical (dot) is preserved
		{"Qwen2.5-0.5B-Instruct.Q8_0.gguf", "Qwen2.5-0.5B-Instruct.Q8_0.gguf", "mradermacher/Qwen2.5-0.5B-Instruct-GGUF", true},
		// full path + a K-quant, basename only is used
		{"/home/u/Downloads/Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-3B-Instruct.Q4_K_M.gguf", "mradermacher/Qwen2.5-3B-Instruct-GGUF", true},
		// not derivable → ok=false (caller shows a friendly error instead)
		{"not-a-model.txt", "", "", false},
		{"model-without-quant.gguf", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			canonical, urls, ok := modelDownload(tc.in)
			if ok != tc.ok {
				t.Fatalf("modelDownload(%q) ok = %v, want %v", tc.in, ok, tc.ok)
			}
			if !tc.ok {
				return
			}
			if canonical != tc.canonical {
				t.Errorf("canonical = %q, want %q", canonical, tc.canonical)
			}
			if len(urls) < 2 {
				t.Fatalf("want primary + mirror URLs, got %v", urls)
			}
			if !strings.HasPrefix(urls[0], "https://huggingface.co/"+tc.repoFrag+"/resolve/main/") {
				t.Errorf("primary URL = %q, want repo %q", urls[0], tc.repoFrag)
			}
			if !strings.HasSuffix(urls[0], tc.canonical) {
				t.Errorf("primary URL = %q, want suffix %q", urls[0], tc.canonical)
			}
		})
	}
}

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
		full, genIDs := decodeReply(session, tok, logits, stops, 48, 0, rng, out)
		return full, len(genIDs)
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

// TestIncrementalBlocksMatchFullPrompt locks the assumption the multi-turn KV prefix reuse
// rests on: feeding ChatML blocks one at a time tokenizes IDENTICALLY to encoding the whole
// prompt at once. It holds because <|im_start|>/<|im_end|> are atomic special tokens that
// Encode splits on before any BPE merge — so a merge can never cross a turn boundary. If
// that ever stopped holding, the reused KV prefix would silently diverge from a fresh
// prefill (a wrong-but-plausible reply), so this guards it directly.
//
// It needs only the tokenizer (no weights), read straight from the GGUF, so it is gated on
// FAK_SIMPLEDEMO_GGUF and skips otherwise — fast and offline by default.
func TestIncrementalBlocksMatchFullPrompt(t *testing.T) {
	ggufPath := os.Getenv("FAK_SIMPLEDEMO_GGUF")
	if ggufPath == "" {
		t.Skip("set FAK_SIMPLEDEMO_GGUF=<path to a .gguf> to run the ChatML block-equivalence guard")
	}
	tok, err := embeddedTokenizer(ggufPath)
	if err != nil {
		t.Fatalf("embeddedTokenizer(%q): %v", ggufPath, err)
	}
	const sys = "You are a helpful assistant. Keep answers short and clear."
	const msg = "What is 2+2?"

	enc := func(s string) []int {
		ids, err := tok.Encode(s)
		if err != nil {
			t.Fatalf("Encode(%q): %v", s, err)
		}
		return ids
	}

	// Turn 1 of the incremental loop (systemBlock+userBlock) must equal a fresh single-turn
	// prompt — the path the proven greedy guard exercises.
	inc := enc(systemBlock(sys) + userBlock(msg))
	full := enc(buildPrompt(sys, nil, msg))
	if !slicesEqualInt(inc, full) {
		t.Fatalf("turn-1 incremental != full prompt:\n  incremental=%v\n  full       =%v", inc, full)
	}

	// Block independence across the system→user boundary: encode(A)++encode(B) == encode(A+B).
	// This is exactly what later turns rely on when they feed only the new userBlock.
	a := enc(systemBlock(sys))
	b := enc(userBlock(msg))
	joined := append(append([]int{}, a...), b...)
	if !slicesEqualInt(joined, inc) {
		t.Fatalf("block-independent encoding diverges at the ChatML boundary:\n  A++B=%v\n  A+B =%v", joined, inc)
	}
}

func slicesEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
