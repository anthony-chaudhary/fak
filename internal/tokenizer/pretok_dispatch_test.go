package tokenizer

import (
	"reflect"
	"testing"
)

// glm4JSONFixture is a minimal GLM-4 tokenizer.json: a BPE model with a ByteLevel decoder
// and a pre_tokenizer Sequence whose Split stage carries GLM-4's digit-triplet grouping
// (\p{N}{1,3}) — the ONE marker that tells a GLM-4 tokenizer.json apart from a Qwen one,
// since both families use an explicit Split. The GLM-4 splitter is a hand-written scanner
// (not regex-driven), so only that marker in the Regex text matters here. Vocab is minimal:
// the test drives the resolved .split function directly, not Encode.
const glm4JSONFixture = `{
  "model": {"type": "BPE", "vocab": {"0":0,"1":1,"2":2,"3":3,"4":4,"a":5,"b":6}, "merges": []},
  "decoder": {"type": "ByteLevel"},
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": " ?[^\\s\\p{L}\\p{N}]+|\\p{N}{1,3}|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"}, "behavior": "Isolated"},
      {"type": "ByteLevel", "add_prefix_space": false}
    ]
  }
}`

// TestPreTokenizerDispatchUnifiedAcrossLoadPaths is the #4265 acceptance proof: a GLM-4
// tokenizer loaded through the JSON path (ParseJSON) and through the GGUF path (FromGGML)
// must resolve to the IDENTICAL pre-tokenizer split and metaspace flag. Before the two
// dispatch sites were unified onto resolvePreTokenizer, the JSON path had no GLM-4 branch
// and silently routed this fixture to the Qwen splitter, so "123" split as 1/2/3 via JSON
// but as one "123" piece via GGML — this test failed on the JSON side (failing-before).
func TestPreTokenizerDispatchUnifiedAcrossLoadPaths(t *testing.T) {
	jsonTok, err := ParseJSON([]byte(glm4JSONFixture))
	if err != nil {
		t.Fatalf("ParseJSON(glm4 fixture): %v", err)
	}
	ggmlTok, err := FromGGML([]string{"0", "1", "2", "3", "4", "a", "b"}, []string{"0 1"}, nil, "glm4")
	if err != nil {
		t.Fatalf("FromGGML(glm4): %v", err)
	}
	// Probes whose split differs by family: digit triplets are grouped by GLM-4 but
	// isolated by Qwen and by GPT-2 ByteLevel; the punctuation-before-letters case
	// separates the explicit-Split families from ByteLevel. Both loads are GLM-4 here,
	// so every probe must split identically across the two paths.
	probes := []string{"123", "1234", "007", "a12b", "(can't do", "  x\n"}
	for _, p := range probes {
		jSplit := jsonTok.split(p)
		gSplit := ggmlTok.split(p)
		if !reflect.DeepEqual(jSplit, gSplit) {
			t.Errorf("split mismatch across load paths for %q: JSON=%q GGML=%q", p, jSplit, gSplit)
		}
	}
	if jsonTok.metaspace != ggmlTok.metaspace {
		t.Errorf("metaspace mismatch across load paths: JSON=%v GGML=%v", jsonTok.metaspace, ggmlTok.metaspace)
	}
	// Pin the resolved family: GLM-4 groups a digit triplet into one piece (the exact
	// behavior the JSON path missed before). A regression to Qwen would yield 1/2/3.
	if got := jsonTok.split("123"); !reflect.DeepEqual(got, []string{"123"}) {
		t.Errorf("JSON GLM-4 split(%q) = %q, want [\"123\"] (routed to the wrong family?)", "123", got)
	}
}

// TestResolvePreTokenizerCoversEveryKind is the exhaustiveness witness: every closed
// preTokKind resolves to a non-nil split function, so a new family added to the kind set
// cannot silently resolve to nil at either loader. resolvePreTokenizer is the single seam
// both ParseJSON and FromGGML now route through, so covering it covers both call sites.
func TestResolvePreTokenizerCoversEveryKind(t *testing.T) {
	for _, kind := range []preTokKind{preTokByteLevel, preTokQwen, preTokGLM4, preTokMetaspace} {
		split, _ := resolvePreTokenizer(kind)
		if split == nil {
			t.Errorf("resolvePreTokenizer(%d) returned a nil split function", kind)
		}
	}
	if _, meta := resolvePreTokenizer(preTokMetaspace); !meta {
		t.Errorf("preTokMetaspace must resolve metaspace=true")
	}
}

// TestPreTokKindMappingParity pins that the two loader-specific kind mappers agree on the
// family for equivalent inputs: a GLM-4 JSON config and the "glm4" GGUF hint both map to
// preTokGLM4; a Qwen config and "qwen2" both map to preTokQwen; a no-Split config and a
// plain hint both map to preTokByteLevel. This agreement is what makes cross-path identity
// hold for every family, not just the GLM-4 case the acceptance test exercises end to end.
func TestPreTokKindMappingParity(t *testing.T) {
	glm4Cfg := preTokConfig{Type: "Sequence", Pretokenizers: []preTokConfig{
		{Type: "Split", Pattern: preTokPattern{Regex: `\p{N}{1,3}`}},
	}}
	qwenCfg := preTokConfig{Type: "Sequence", Pretokenizers: []preTokConfig{
		{Type: "Split", Pattern: preTokPattern{Regex: `\p{N}`}},
	}}
	byteLevelCfg := preTokConfig{Type: "ByteLevel"}

	cases := []struct {
		name string
		json preTokConfig
		ggml string
		want preTokKind
	}{
		{"glm4", glm4Cfg, "glm4", preTokGLM4},
		{"qwen", qwenCfg, "qwen2", preTokQwen},
		{"bytelevel", byteLevelCfg, "gpt2", preTokByteLevel},
	}
	for _, tc := range cases {
		gotJSON := jsonPreTokKind(tc.json)
		gotGGML := ggmlPreTokKind(tc.ggml)
		if gotJSON != tc.want {
			t.Errorf("jsonPreTokKind(%s) = %d, want %d", tc.name, gotJSON, tc.want)
		}
		if gotGGML != tc.want {
			t.Errorf("ggmlPreTokKind(%q) = %d, want %d", tc.ggml, gotGGML, tc.want)
		}
		if gotJSON != gotGGML {
			t.Errorf("%s: JSON path kind %d != GGML path kind %d", tc.name, gotJSON, gotGGML)
		}
	}
}
