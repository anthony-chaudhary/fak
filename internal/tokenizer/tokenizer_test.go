package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestDecodeHFByteLevelFixtures(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	cases := []struct {
		name string
		ids  []int
		want string
	}{
		{
			name: "prompt",
			ids:  []int{504, 3575, 282, 4649, 314},
			want: "The capital of France is",
		},
		{
			name: "leading-space",
			ids:  []int{378, 3575, 282, 4649, 314},
			want: " The capital of France is",
		},
		{
			name: "newlines",
			ids:  []int{7042, 30, 198, 198, 38634, 314, 260, 3575, 282, 4649, 30, 198},
			want: " Paris.\n\nParis is the capital of France.\n",
		},
		{
			name: "specials",
			ids:  []int{1, 504, 198, 2},
			want: "<|im_start|>The\n<|im_end|>",
		},
		{
			name: "digits",
			ids:  []int{34, 35, 36, 198, 37, 38, 39},
			want: "234\n567",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.ids)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if got != tc.want {
				t.Fatalf("Decode() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestTokenizerStageT2Boundary(t *testing.T) {
	tok := loadFixtureTokenizer(t)
	if got := tok.Vocab(); got != 38635 {
		t.Fatalf("Vocab() = %d, want 38635", got)
	}
	specials := tok.SpecialTokens()
	if specials[1] != "<|im_start|>" || specials[2] != "<|im_end|>" {
		t.Fatalf("special tokens = %#v", specials)
	}
	specials[1] = "mutated"
	if got := tok.SpecialTokens()[1]; got != "<|im_start|>" {
		t.Fatalf("SpecialTokens did not return a copy, got %q", got)
	}
	if _, err := tok.Decode([]int{99999}); err == nil {
		t.Fatal("Decode accepted out-of-range id")
	}
	if _, err := tok.Decode([]int{123}); err == nil {
		t.Fatal("Decode accepted missing sparse fixture id")
	}
}

func TestEncodeSmallByteLevelBPEFixture(t *testing.T) {
	tok, err := ParseJSON([]byte(`{
	  "model": {
	    "type": "BPE",
	    "vocab": {
	      "<|im_start|>": 0, "<|im_end|>": 1,
	      "H": 2, "e": 3, "l": 4, "o": 5, "He": 6, "Hel": 7, "Hell": 8, "Hello": 9,
	      "Ġ": 10, "w": 11, "r": 12, "d": 13, "wo": 14, "wor": 15, "worl": 16, "world": 17, "Ġworld": 18,
	      "!": 19, "2": 20, "0": 21, "4": 22
	    },
	    "merges": ["H e", "He l", "Hel l", "Hell o", "w o", "wo r", "wor l", "worl d", "Ġ world"]
	  },
	  "decoder": {"type": "ByteLevel"},
	  "added_tokens": [
	    {"id": 0, "content": "<|im_start|>", "special": true},
	    {"id": 1, "content": "<|im_end|>", "special": true}
	  ]
	}`))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	got, err := tok.Encode("<|im_start|>Hello world! 2024<|im_end|>")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	want := []int{0, 9, 18, 19, 10, 20, 21, 20, 22, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
	round, err := tok.Decode(got)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if round != "<|im_start|>Hello world! 2024<|im_end|>" {
		t.Fatalf("Decode(Encode()) = %q", round)
	}
}

func TestOptionalSmolLM2EncodeMatchesHFCorpus(t *testing.T) {
	path := findSmolLM2TokenizerJSON(t)
	tok, err := LoadJSON(path)
	if err != nil {
		t.Fatalf("LoadJSON: %v", err)
	}
	cases := []struct {
		text string
		ids  []int
	}{
		{"The capital of France is", []int{504, 3575, 282, 4649, 314}},
		{" Paris.\n\nParis is the capital of France.\n", []int{7042, 30, 198, 198, 38634, 314, 260, 3575, 282, 4649, 30, 198}},
		{"<|im_start|>user\nHello, world! 2024\n<|im_end|>", []int{1, 4093, 198, 19556, 28, 905, 17, 216, 34, 32, 34, 36, 198, 2}},
		{"Tabs\tspaces  and\nnewlines", []int{68, 7366, 197, 41123, 216, 284, 198, 2241, 5110}},
		{"naïve café déjà vu", []int{3546, 46494, 37366, 32564, 90, 16739, 386, 101}},
		{"JSON: {\"a\": 1, \"b\": [2,3]}", []int{27936, 42, 9583, 81, 1799, 216, 33, 28, 476, 82, 1799, 933, 34, 28, 35, 28852}},
		{"snake_case and camelCase42", []int{48284, 79, 7544, 284, 32782, 10802, 36, 34}},
		{"don't stop believing", []int{12420, 982, 2853, 16517}},
		{"I'm here; you're there.", []int{57, 5248, 1535, 43, 346, 2316, 665, 30}},
		{"We'll test CAN'T and won’t.", []int{1882, 3060, 1028, 27190, 23, 68, 284, 3763, 417, 100, 30}},
		{"emoji 😀😃 plus © and ™", []int{391, 33777, 40303, 218, 10813, 242, 221, 8055, 11435, 284, 4636, 222, 112}},
		{"CRLF\r\nline two", []int{9088, 60, 54, 201, 198, 1311, 827}},
		{" leading   triple spaces", []int{2899, 256, 20930, 5600}},
		{"math: 3.14159 >= 2^10", []int{8898, 42, 216, 35, 30, 33, 36, 33, 37, 41, 10888, 216, 34, 78, 33, 32}},
		{"中文 mixed العربية", []int{28589, 29184, 6468, 27819, 34966, 20602, 31462, 23339, 43463}},
		{"path C:\\work\\fleet and /tmp/x", []int{1537, 340, 19199, 1637, 76, 86, 43727, 284, 2272, 13560, 31, 104}},
		{"email test@example.com", []int{10349, 1028, 48, 7434, 30, 824}},
	}
	for _, tc := range cases {
		got, err := tok.Encode(tc.text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", tc.text, err)
		}
		if !reflect.DeepEqual(got, tc.ids) {
			t.Fatalf("Encode(%q) = %v, want %v", tc.text, got, tc.ids)
		}
		round, err := tok.Decode(got)
		if err != nil {
			t.Fatalf("Decode(%q ids): %v", tc.text, err)
		}
		if round != tc.text {
			t.Fatalf("Decode(Encode(%q)) = %q", tc.text, round)
		}
	}
}

func TestDecodePreservesSplitUTF8Bytes(t *testing.T) {
	tok, err := ParseJSON([]byte(`{
	  "model": {"type": "BPE", "vocab": {"Ã": 0, "©": 1}},
	  "decoder": {"type": "ByteLevel"}
	}`))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	got, err := tok.Decode([]int{0, 1})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got != "é" {
		t.Fatalf("Decode split UTF-8 = %q, want %q", got, "é")
	}
}

func findSmolLM2TokenizerJSON(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	g := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--HuggingFaceTB--SmolLM2-135M-Instruct", "snapshots", "*", "tokenizer.json")
	matches, _ := filepath.Glob(g)
	if len(matches) == 0 {
		t.Skip("no cached SmolLM2 tokenizer.json")
	}
	return matches[0]
}

func TestParseJSONRejectsUnsupportedShapes(t *testing.T) {
	if _, err := ParseJSON([]byte(`{"model":{"type":"WordPiece","vocab":{"x":0}},"decoder":{"type":"ByteLevel"}}`)); err == nil {
		t.Fatal("accepted unsupported model type")
	}
	if _, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"x":0}},"decoder":{"type":"Replace"}}`)); err == nil {
		t.Fatal("accepted unsupported decoder type")
	}
	if _, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"x":0,"y":0}},"decoder":{"type":"ByteLevel"}}`)); err == nil {
		t.Fatal("accepted duplicate id")
	}
	if _, err := ParseJSON([]byte(`{
	  "model":{"type":"BPE","vocab":{"x":0}},
	  "decoder":{"type":"ByteLevel"},
	  "added_tokens":[{"id":0,"content":"y","special":true}]
	}`)); err == nil {
		t.Fatal("accepted conflicting added token id")
	}
}

func loadFixtureTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := ParseJSON([]byte(fixtureTokenizerJSON))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	return tok
}

// fixtureTokenizerJSON is a minimal HF fast tokenizer.json slice for the SmolLM2
// ids above. Expected Decode strings were captured from tokenizers.Tokenizer.decode
// on HuggingFaceTB/SmolLM2-135M-Instruct with skip_special_tokens=false.
const fixtureTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      ".": 30,
      "2": 34,
      "3": 35,
      "4": 36,
      "5": 37,
      "6": 38,
      "7": 39,
      "Ċ": 198,
      "Ġthe": 260,
      "Ġof": 282,
      "Ġis": 314,
      "ĠThe": 378,
      "The": 504,
      "Ġcapital": 3575,
      "ĠFrance": 4649,
      "ĠParis": 7042,
      "Paris": 38634
    }
  },
  "decoder": {"type": "ByteLevel"},
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "special": true},
    {"id": 1, "content": "<|im_start|>", "special": true},
    {"id": 2, "content": "<|im_end|>", "special": true}
  ]
}`
