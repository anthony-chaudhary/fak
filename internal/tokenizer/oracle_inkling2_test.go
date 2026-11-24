package tokenizer

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2Golden is the reference corpus for the Inkling 2 tokenizer: base64(text)
// TAB the ids HF tokenizers 0.22.2 produced from thinkingmachines/Inkling-2-Mini
// tokenizer.json with add_special_tokens=False. It covers camelCase and ALLCAPS
// words, contractions, digit runs, newline runs, CJK, Indic and Thai marks,
// accents, emoji and code.
const inkling2Golden = "inkling2_golden.tsv"

// fak-test:runtime fast est=200ms lane=default
// Estimate only: a 1.7k-token vocab slice and a 46-line corpus; not measured.
func TestInkling2O200kGolden(t *testing.T) {
	tok := loadGGMLSubset(t, "inkling2_vocab_subset.tsv")
	if !tok.ignoreMerges {
		t.Fatal("inkling2 GGUF must prefer whole vocabulary pieces before merges")
	}
	for _, tc := range []struct {
		text   string
		pieces []string
	}{
		{"XMLHttpRequest", []string{"XMLHttp", "Request"}},
		{"HELLOworld", []string{"HELLOworld"}},
		{"DON'T go", []string{"DON'T", " go"}},
		{"2026 x", []string{"202", "6", " x"}},
		{"a/b//\nc", []string{"a", "/b", "//\n", "c"}},
		{"x  \n\n y", []string{"x", "  \n\n", " y"}},
	} {
		if got := tok.split(tc.text); !reflect.DeepEqual(got, tc.pieces) {
			t.Errorf("split(%q) = %q, want %q", tc.text, got, tc.pieces)
		}
	}
	assertGoldenEncode(t, tok, readGoldenTSV(t, inkling2Golden))
}

// fak-test:runtime fast est=50ms lane=default
// Estimate only: one JSON config classification; not measured.
func TestInkling2JSONPreTokenizerIsO200k(t *testing.T) {
	cfg := preTokConfig{Type: "Sequence", Pretokenizers: []preTokConfig{
		{Type: "Split", Pattern: preTokPattern{Regex: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`}},
		{Type: "ByteLevel"},
	}}
	split, _ := resolvePreTokenizer(jsonPreTokKind(cfg))
	want := []string{"XMLHttp", "Request", " DON'T"}
	if got := split("XMLHttpRequest DON'T"); !reflect.DeepEqual(got, want) {
		t.Fatalf("o200k tokenizer.json split = %q, want %q (not the GLM-4 digit-grouping splitter)", got, want)
	}
}

// fak-test:runtime fast est=50ms lane=default
// Estimate only: two tiny in-memory tokenizer.json documents; not measured.
func TestParseJSONMergesAcceptPairForm(t *testing.T) {
	const doc = `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},"merges":%s},"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"}}`
	for _, merges := range []string{`["a b","ab c"]`, `[["a","b"],["ab","c"]]`} {
		tok, err := ParseJSON([]byte(fmt.Sprintf(doc, merges)))
		if err != nil {
			t.Fatalf("ParseJSON(merges=%s): %v", merges, err)
		}
		if got, err := tok.Encode("abc"); err != nil || !reflect.DeepEqual(got, []int{4}) {
			t.Fatalf("merges=%s: Encode(abc) = %v, %v; want [4]", merges, got, err)
		}
	}
	if _, err := ParseJSON([]byte(fmt.Sprintf(doc, `[["a"]]`))); err == nil {
		t.Fatal("a one-element merge pair must be rejected")
	}
}

// TestInkling2GGUFOracle runs the corpus against the full embedded vocab when
// FAK_INKLING2_GGUF names an Inkling-2-Mini GGUF.
func TestInkling2GGUFOracle(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" {
		t.Skip("FAK_INKLING2_GGUF not set; skipping full-vocab Inkling 2 tokenizer oracle")
	}
	f, err := ggufload.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatalf("%s has no embedded BPE tokenizer", path)
	}
	if gt.Pre != "inkling2" {
		t.Fatalf("tokenizer.ggml.pre = %q, want inkling2", gt.Pre)
	}
	tok, err := FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	assertGoldenEncode(t, tok, readGoldenTSV(t, inkling2Golden))
}

// TestInkling2TokenizerJSONOracle runs the corpus through the tokenizer.json
// loader when FAK_INKLING2_TOKENIZER_JSON names the HF file.
func TestInkling2TokenizerJSONOracle(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TOKENIZER_JSON")
	if path == "" {
		t.Skip("FAK_INKLING2_TOKENIZER_JSON not set; skipping Inkling 2 tokenizer.json oracle")
	}
	tok, err := LoadJSON(path)
	if err != nil {
		t.Fatalf("LoadJSON(%s): %v", path, err)
	}
	assertGoldenEncode(t, tok, readGoldenTSV(t, inkling2Golden))
}
