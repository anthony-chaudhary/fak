package tokenizer

import (
	"bufio"
	"encoding/base64"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// TestQwenOracleGolden cross-validates the Qwen pre-tokenizer + BPE against the
// llama.cpp tokenizer (llama-tokenize) on the Qwen2.5 vocab — a second, independent
// oracle alongside the HF SmolLM2 corpus in tokenizer_test.go. The golden ids were
// produced by `llama-tokenize -m qwen2.5-1.5b-instruct-q8_0.gguf -p <line>` over a
// diverse corpus (ascii/code/json/cjk/contractions/whitespace), stored as
// base64(text)\tspace-separated-ids. It loads the real Qwen2.5 tokenizer.json from
// FAK_TOKENIZER_DIR (default ~/.cache/fak-models/tokenizers/qwen2.5) and skips if
// unavailable, keeping CI green on machines without the model cache.
func TestQwenOracleGolden(t *testing.T) {
	dir := os.Getenv("FAK_TOKENIZER_DIR")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home dir")
		}
		dir = filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
	}
	tjson := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(tjson); err != nil {
		t.Skipf("Qwen2.5 reference tokenizer not present at %s (skipping llama.cpp oracle gate)", tjson)
	}
	tok, err := LoadJSON(tjson)
	if err != nil {
		t.Fatalf("LoadJSON(%s): %v", tjson, err)
	}

	f, err := os.Open(filepath.Join("testdata", "qwen25_golden.tsv"))
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	n := 0
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		tab := strings.IndexByte(line, '\t')
		if tab < 0 {
			t.Fatalf("malformed golden line: %q", line)
		}
		raw, err := base64.StdEncoding.DecodeString(line[:tab])
		if err != nil {
			t.Fatalf("bad base64: %v", err)
		}
		var want []int
		for _, f := range strings.Fields(line[tab+1:]) {
			id, err := strconv.Atoi(f)
			if err != nil {
				t.Fatalf("bad id %q: %v", f, err)
			}
			want = append(want, id)
		}
		got, err := tok.Encode(string(raw))
		if err != nil {
			t.Fatalf("Encode(%q): %v", string(raw), err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("llama.cpp oracle mismatch for %q\n got=%v\nwant=%v", string(raw), got, want)
		}
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("golden empty")
	}
	t.Logf("llama.cpp oracle gate: %d lines byte-exact (Qwen2.5)", n)
}

func TestOptionalQwen36ChatMLPromptMatchesLlamaCpp(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	dir := os.Getenv("FAK_QWEN36_TOKENIZER_DIR")
	if dir == "" {
		dir = filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen3.6")
	}
	tjson := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(tjson); err != nil {
		t.Skipf("Qwen3.6 tokenizer not present at %s", tjson)
	}
	tok, err := LoadJSON(tjson)
	if err != nil {
		t.Fatalf("LoadJSON(%s): %v", tjson, err)
	}

	prompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nSay OK.<|im_end|>\n" +
		"<|im_start|>assistant\n"
	want := []int{
		248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
		248045, 846, 198, 44240, 10092, 13, 248046, 198, 248045, 74455, 198,
	}
	got, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Qwen3.6 ChatML ids mismatch\n got=%v\nwant=%v", got, want)
	}
}
