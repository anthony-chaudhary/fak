package tokenizer

import (
	"reflect"
	"testing"
)

// ornith_qwen35_test.go — proves the tokenizer leaf carries the Ornith 1.0 (Qwen3.5
// arch, #1026/#1030) specifics WITHOUT a hardcoded template: a 248320-entry vocab and
// a TWO-id end-of-turn set. Ornith's config.json reports vocab_size 248320 and the
// turn-ending control tokens <|endoftext|>=248044 and <|im_end|>=248046 (the two ids a
// chat turn can stop on); <|im_start|>=248045 opens a turn. Those ids/strings are
// pinned by internal/model/config_test.go's qwen3_5 fixture and oracle_qwen_test.go's
// Qwen3.6 ChatML golden (same 248320 vocab family).
//
// The leaf does NOT hardcode any of this: ParseJSON/FromGGML carry whatever vocab size
// and special-token set the MODEL supplies, so the size and the eos set are
// model-driven. These tests assert that flow at the Ornith shape — vocab == 248320 and
// the two eos control tokens each resolve to exactly one special id — over both load
// paths (the GGUF-embedded FromGGML path and the tokenizer.json ParseJSON path), so a
// downstream two-id eos stop check (internal/model Config.IsEOS over EOSTokenIDs) sees
// each eos as a single id rather than byte-splattered text.

// Ornith Qwen3.5 control-token ids (model-driven; see config_test.go qwen3_5 fixture).
const (
	ornithVocabSize = 248320
	ornithEndOfText = 248044 // <|endoftext|> — base eos
	ornithIMStart   = 248045 // <|im_start|> — opens a turn
	ornithIMEnd     = 248046 // <|im_end|>   — ends a turn (second eos)
)

// ornithEOSSet is the TWO-id end-of-turn set Ornith's chat template stops on.
var ornithEOSSet = []int{ornithEndOfText, ornithIMEnd}

// TestOrnithQwen35GGMLVocabAndTwoEOS builds an Ornith-shaped GGML vocab (the
// always-matches-the-model FromGGML path) sized to the real 248320 and marks the two
// turn-ending control tokens CONTROL via token_type — exactly how a real Ornith GGUF
// would. It asserts Vocab()==248320 and that BOTH eos ids resolve to a single special
// id, proving the leaf honors a model-supplied two-id eos set with no hardcoded
// template. The fixture is synthetic (a sparse stand-in vocab, NOT the real Ornith
// weights); only the vocab size and the three control ids/strings mirror Ornith.
func TestOrnithQwen35GGMLVocabAndTwoEOS(t *testing.T) {
	// ggmlTypeNormal is llama.cpp's LLAMA_TOKEN_TYPE_NORMAL (1) — the default,
	// non-special class for the placeholder slots (anything not CONTROL/USER_DEFINED).
	const ggmlTypeNormal int32 = 1
	tokens := make([]string, ornithVocabSize)
	tokenTypes := make([]int32, ornithVocabSize)
	// Default the unused slots to placeholder vocab strings (CONTROL/USER_DEFINED only
	// where we set them); FromGGML keeps the first occurrence of a string in tokenToID,
	// so distinct placeholders keep Encode honest.
	for i := range tokens {
		tokens[i] = "<unused_" + itoa(i) + ">"
		tokenTypes[i] = ggmlTypeNormal
	}
	// A minimal real BPE core so FromGGML's "no merges" guard is satisfied and a normal
	// token encodes — the leaf rejects a merge-less GGUF as non-BPE.
	tokens[72] = "H"
	tokens[73] = "i"
	tokens[74] = "Hi"
	tokens[75] = "u"
	tokens[76] = "s"
	tokens[77] = "e"
	tokens[78] = "r"
	tokens[79] = "Ċ"
	// The three Ornith control tokens at their real ids, typed CONTROL so they are
	// recognized as special from the MODEL's token_type, not a "<|...|>" string guess.
	tokens[ornithEndOfText] = "<|endoftext|>"
	tokens[ornithIMStart] = "<|im_start|>"
	tokens[ornithIMEnd] = "<|im_end|>"
	tokenTypes[ornithEndOfText] = ggmlTypeControl
	tokenTypes[ornithIMStart] = ggmlTypeControl
	tokenTypes[ornithIMEnd] = ggmlTypeControl

	merges := []string{"H i"}

	tok, err := FromGGML(tokens, merges, tokenTypes, "qwen35")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}

	if got := tok.Vocab(); got != ornithVocabSize {
		t.Fatalf("Vocab() = %d, want %d (Ornith 248320 must flow through)", got, ornithVocabSize)
	}

	// Each of the two eos ids must encode to exactly that one special id (a two-id eos
	// set only works if every eos is a single id at the encode layer, not byte-split).
	specials := tok.SpecialTokens()
	wantStr := map[int]string{
		ornithEndOfText: "<|endoftext|>",
		ornithIMEnd:     "<|im_end|>",
	}
	eosSpecialCount := 0
	for _, id := range ornithEOSSet {
		if specials[id] != wantStr[id] {
			t.Fatalf("eos id %d special = %q, want %q (model-driven control token missing)", id, specials[id], wantStr[id])
		}
		eosSpecialCount++
		got, err := tok.Encode(wantStr[id])
		if err != nil {
			t.Fatalf("Encode(%q): %v", wantStr[id], err)
		}
		if !reflect.DeepEqual(got, []int{id}) {
			t.Fatalf("Encode(%q) = %v, want single special id %v", wantStr[id], got, []int{id})
		}
	}
	if eosSpecialCount != 2 {
		t.Fatalf("Ornith eos set size = %d, want 2", eosSpecialCount)
	}

	// A full turn keeps both turn boundaries intact: ...<|im_start|>...<|im_end|>.
	turn := "<|im_start|>user\nHi<|im_end|>"
	ids, err := tok.Encode(turn)
	if err != nil {
		t.Fatalf("Encode(turn): %v", err)
	}
	if ids[0] != ornithIMStart || ids[len(ids)-1] != ornithIMEnd {
		t.Fatalf("turn not framed by control tokens: %v (want first=%d last=%d)", ids, ornithIMStart, ornithIMEnd)
	}
}

// TestOrnithQwen35JSONVocabAndTwoEOS proves the same model-driven flow over the
// tokenizer.json ParseJSON path: the vocab size (the max id + 1) and the two eos
// special ids come from the supplied document, never a hardcoded template. The vocab is
// sparse — only the populated ids carry strings — but the Ornith size 248320 is
// established by the highest control id (248046 -> 248047 slots) plus an explicit
// 248319 anchor so Vocab() == 248320 exactly, matching the GGML path.
func TestOrnithQwen35JSONVocabAndTwoEOS(t *testing.T) {
	doc := `{
	  "model": {
	    "type": "BPE",
	    "vocab": {
	      "H": 72, "i": 73, "Hi": 74,
	      "<|endoftext|>": 248044,
	      "<|im_start|>": 248045,
	      "<|im_end|>": 248046,
	      "<|anchor|>": 248319
	    },
	    "merges": ["H i"]
	  },
	  "decoder": {"type": "ByteLevel"},
	  "added_tokens": [
	    {"id": 248044, "content": "<|endoftext|>", "special": true},
	    {"id": 248046, "content": "<|im_end|>", "special": true},
	    {"id": 248045, "content": "<|im_start|>", "special": true}
	  ]
	}`
	tok, err := ParseJSON([]byte(doc))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}

	if got := tok.Vocab(); got != ornithVocabSize {
		t.Fatalf("Vocab() = %d, want %d", got, ornithVocabSize)
	}

	specials := tok.SpecialTokens()
	// The two eos ids are each a single special id; <|im_start|> is special but is NOT
	// part of the eos stop set — the leaf carries the model's set verbatim.
	for _, id := range ornithEOSSet {
		if _, ok := specials[id]; !ok {
			t.Fatalf("eos id %d not registered as special", id)
		}
	}
	if len(ornithEOSSet) != 2 {
		t.Fatalf("eos set size = %d, want 2", len(ornithEOSSet))
	}
	got, err := tok.Encode("<|endoftext|><|im_end|>")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !reflect.DeepEqual(got, []int{ornithEndOfText, ornithIMEnd}) {
		t.Fatalf("Encode(two-eos) = %v, want %v", got, []int{ornithEndOfText, ornithIMEnd})
	}
}

// itoa is a tiny dependency-free int->decimal helper for building the placeholder
// vocab strings (avoids importing strconv only for the fixture).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
