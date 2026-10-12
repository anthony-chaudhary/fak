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

// deepSeekV41Golden is the reference corpus for the DeepSeek-V4.1-Flash GGUF tokenizer
// (vcruz305/DeepSeek-V4.1-Flash-GGUF, tokenizer.ggml.pre=joyai-llm). Each line is
// base64(text) TAB the ids llama-tokenize printed for it, built from
// vcruz305/llama.cpp@5210c7c (runtime/deepseek41) with the Q2_K shard 1 (sha256
// 0bcee934...08ee0), no BOS. The first four lines are the raw-completion probe
// prompts (the reference oracle feeds them as [BOS=0] + these ids), then curated edge
// strings (digit runs, CJK, punctuation, whitespace runs), then seeded random strings
// over the same classes.
const deepSeekV41Golden = "deepseek_v41_golden.tsv"

type goldenLine struct {
	text string
	ids  []int
}

func readGoldenTSV(t *testing.T, name string) []goldenLine {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()
	var out []goldenLine
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		b64, idsField, ok := strings.Cut(line, "\t")
		if !ok {
			t.Fatalf("malformed golden line: %q", line)
		}
		raw, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			t.Fatalf("bad base64: %v", err)
		}
		g := goldenLine{text: string(raw)}
		for _, fld := range strings.Fields(idsField) {
			id, err := strconv.Atoi(fld)
			if err != nil {
				t.Fatalf("bad id %q: %v", fld, err)
			}
			g.ids = append(g.ids, id)
		}
		out = append(out, g)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("golden corpus was empty")
	}
	return out
}

// loadDeepSeekV41Subset rebuilds a FromGGML tokenizer from the committed slice of the
// V4.1 GGUF vocab: every token and merge whose text is a substring of a golden line's
// byte-level form, keeping original ids and merge order. Any split of a golden line
// BPEs only within such substrings, so the slice tokenizes the corpus exactly as the
// full 129k vocab does, whichever pre-tokenizer runs.
func loadDeepSeekV41Subset(t *testing.T) *Tokenizer {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "deepseek_v41_vocab_subset.tsv"))
	if err != nil {
		t.Fatalf("open vocab subset: %v", err)
	}
	defer f.Close()
	var (
		pre        string
		tokens     []string
		tokenTypes []int32
		merges     []string
	)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "#") {
			continue
		}
		fields := strings.Split(sc.Text(), "\t")
		switch fields[0] {
		case "pre":
			pre = fields[1]
		case "n_vocab":
			n, err := strconv.Atoi(fields[1])
			if err != nil {
				t.Fatalf("bad n_vocab: %v", err)
			}
			tokens = make([]string, n)
			tokenTypes = make([]int32, n)
			for i := range tokenTypes {
				tokenTypes[i] = 1
			}
		case "token":
			id, err1 := strconv.Atoi(fields[1])
			typ, err2 := strconv.Atoi(fields[2])
			text, err3 := base64.StdEncoding.DecodeString(fields[3])
			if err1 != nil || err2 != nil || err3 != nil || id >= len(tokens) {
				t.Fatalf("bad token line %q", sc.Text())
			}
			tokens[id] = string(text)
			tokenTypes[id] = int32(typ)
		case "merge":
			text, err := base64.StdEncoding.DecodeString(fields[1])
			if err != nil {
				t.Fatalf("bad merge line %q", sc.Text())
			}
			merges = append(merges, string(text))
		default:
			t.Fatalf("unknown vocab subset record %q", fields[0])
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	tok, err := FromGGML(tokens, merges, tokenTypes, pre)
	if err != nil {
		t.Fatalf("FromGGML(subset): %v", err)
	}
	return tok
}

func assertGoldenEncode(t *testing.T, tok *Tokenizer, golden []goldenLine) {
	t.Helper()
	for _, g := range golden {
		got, err := tok.Encode(g.text)
		if err != nil {
			t.Errorf("Encode(%q): %v", g.text, err)
			continue
		}
		if !reflect.DeepEqual(got, g.ids) {
			t.Errorf("token mismatch vs llama.cpp for %q\n got=%v\nwant=%v", g.text, got, g.ids)
		}
	}
}

// TestDeepSeekV41JoyAIGolden pins DeepSeek-V4.1-Flash prompt ids to the reference
// llama.cpp fork token for token, without the 43 GB shard: the vocab slice carries the
// GGUF's own tokenizer.ggml.pre (joyai-llm), so this is the FromGGML path a served
// prompt takes. Before the joyai-llm hint had its own splitter it fell through to GPT-2
// ByteLevel, which splits "17" into "1","7" and diverges on the fourth probe prompt.
func TestDeepSeekV41JoyAIGolden(t *testing.T) {
	tok := loadDeepSeekV41Subset(t)

	// The reference oracle's probe prompts, with its pieces. The oracle prepends BOS id 0
	// itself; Encode never adds it.
	probes := []struct {
		text   string
		pieces []string
		ids    []int
	}{
		{"The chemical symbol for gold is", []string{"The", " chemical", " symbol", " for", " gold", " is"}, []int{671, 7658, 7380, 362, 8495, 344}},
		{"The capital of France is", []string{"The", " capital", " of", " France", " is"}, []int{671, 6102, 294, 8760, 344}},
		{"Say hello.", []string{"Say", " hello", "."}, []int{63006, 44388, 16}},
		{"What is 17 times 3?", []string{"What", " is", " ", "17", " times", " ", "3", "?"}, []int{3085, 344, 223, 1002, 2734, 223, 21, 33}},
	}
	for _, p := range probes {
		if got := tok.split(p.text); !reflect.DeepEqual(got, p.pieces) {
			t.Errorf("pre-tokenizer split(%q) = %q, want %q", p.text, got, p.pieces)
		}
		got, err := tok.Encode(p.text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", p.text, err)
		}
		if !reflect.DeepEqual(got, p.ids) {
			t.Errorf("probe %q: got=%v want=%v", p.text, got, p.ids)
		}
	}
	assertGoldenEncode(t, tok, readGoldenTSV(t, deepSeekV41Golden))
}

// TestDeepSeekV41GGUFOracle is the same gate against the full embedded vocab, for a
// host that has the checkpoint: FAK_V41_GGUF names shard 1 (or a copy of its header).
func TestDeepSeekV41GGUFOracle(t *testing.T) {
	path := os.Getenv("FAK_V41_GGUF")
	if path == "" {
		t.Skip("FAK_V41_GGUF not set; skipping full-vocab DeepSeek-V4.1 tokenizer oracle")
	}
	f, err := ggufload.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatalf("%s has no embedded BPE tokenizer", path)
	}
	if gt.Pre != "joyai-llm" {
		t.Fatalf("tokenizer.ggml.pre = %q, want joyai-llm", gt.Pre)
	}
	tok, err := FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	assertGoldenEncode(t, tok, readGoldenTSV(t, deepSeekV41Golden))
}
