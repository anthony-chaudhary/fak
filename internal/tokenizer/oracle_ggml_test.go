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

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// TestGGMLEmbeddedOracle proves the GGUF-embedded tokenizer path (tokenizer.FromGGML
// fed from a checkpoint's tokenizer.ggml.* arrays) is byte-exact with llama.cpp on
// the same Qwen2.5 golden corpus that gates the tokenizer.json path. This is the
// path the simpledemo relies on when no tokenizer.json is present, so a regression
// here would silently mis-tokenize the demo.
//
// It needs a Qwen2(.5) GGUF on disk (general.architecture == "qwen2"); it searches
// the usual caches and honors FAK_GGUF, and skips when none is found so CI stays
// green on machines without the model.
func TestGGMLEmbeddedOracle(t *testing.T) {
	ggufPath := findQwen2GGUF(t)
	if ggufPath == "" {
		t.Skip("no Qwen2 GGUF found (set FAK_GGUF=<path>); skipping embedded-tokenizer oracle gate")
	}

	f, err := ggufload.Open(ggufPath)
	if err != nil {
		t.Fatalf("open %s: %v", ggufPath, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Skipf("%s has no embedded BPE tokenizer; skipping", ggufPath)
	}
	embedded, err := FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML(%s): %v", ggufPath, err)
	}

	// Optional cross-check against the tokenizer.json path when it is present.
	var fromJSON *Tokenizer
	if dir := tokenizerJSONDir(); dir != "" {
		if tj, err := LoadJSON(filepath.Join(dir, "tokenizer.json")); err == nil {
			fromJSON = tj
		}
	}

	golden, err := os.Open(filepath.Join("testdata", "qwen25_golden.tsv"))
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer golden.Close()

	sc := bufio.NewScanner(golden)
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
		for _, fld := range strings.Fields(line[tab+1:]) {
			id, err := strconv.Atoi(fld)
			if err != nil {
				t.Fatalf("bad id %q: %v", fld, err)
			}
			want = append(want, id)
		}
		got, err := embedded.Encode(string(raw))
		if err != nil {
			t.Fatalf("embedded.Encode(%q): %v", string(raw), err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("embedded-tokenizer llama.cpp mismatch for %q\n got=%v\nwant=%v", string(raw), got, want)
		}
		if fromJSON != nil {
			j, err := fromJSON.Encode(string(raw))
			if err != nil {
				t.Fatalf("json.Encode(%q): %v", string(raw), err)
			}
			if !reflect.DeepEqual(got, j) {
				t.Errorf("embedded vs tokenizer.json mismatch for %q\n embedded=%v\n json=%v", string(raw), got, j)
			}
		}
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("golden corpus was empty")
	}
	t.Logf("embedded-tokenizer oracle: %d lines byte-exact vs llama.cpp (model %s)", n, filepath.Base(ggufPath))
}

// tokenizerJSONDir returns the Qwen2.5 tokenizer.json dir (FAK_TOKENIZER_DIR or the
// default cache), or "" if it has no tokenizer.json.
func tokenizerJSONDir() string {
	dir := os.Getenv("FAK_TOKENIZER_DIR")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		return ""
	}
	return dir
}

// findQwen2GGUF locates a Qwen2(.5)-family GGUF whose embedded tokenizer matches
// the Qwen2.5 golden corpus. FAK_GGUF overrides the search. It gates on
// general.architecture == "qwen2" so a Qwen3.6 checkpoint (different vocab) does
// not produce false mismatches against the Qwen2.5 golden ids.
func findQwen2GGUF(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("FAK_GGUF"); p != "" {
		if isQwen2GGUF(p) {
			return p
		}
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	dirs := []string{
		filepath.Join(home, "models"),
		filepath.Join(home, ".cache", "fak-models", "gguf"),
		filepath.Join(home, ".cache", "huggingface", "hub"),
		filepath.Join(home, "Downloads"),
	}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".gguf") {
				if p := filepath.Join(d, e.Name()); isQwen2GGUF(p) {
					return p
				}
			}
			if e.IsDir() {
				sub := filepath.Join(d, e.Name())
				subEntries, _ := os.ReadDir(sub)
				for _, se := range subEntries {
					if strings.HasSuffix(se.Name(), ".gguf") {
						if p := filepath.Join(sub, se.Name()); isQwen2GGUF(p) {
							return p
						}
					}
				}
			}
		}
	}
	return ""
}

func isQwen2GGUF(path string) bool {
	f, err := ggufload.Open(path)
	if err != nil {
		return false
	}
	arch, ok := f.String("general.architecture")
	return ok && arch == "qwen2"
}
