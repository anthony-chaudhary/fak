package agent

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// byteLevelRunes mirrors tokenizer.byteLevelEncode's alphabet: GPT-2 maps every byte to a
// printable rune. We build a no-merge vocab over all 256 byte-runes so ANY text encodes
// (each byte -> one token), plus the ChatML specials. That isolates the structural prefix
// property from any merge-rank subtlety.
func buildByteVocab() string {
	// Reproduce GPT-2 byte->unicode (same table internal/tokenizer uses).
	bs := []int{}
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := 0xA1; i <= 0xAC; i++ {
		bs = append(bs, i)
	}
	for i := 0xAE; i <= 0xFF; i++ {
		bs = append(bs, i)
	}
	cs := append([]int{}, bs...)
	n := 0
	inBs := func(b int) bool {
		for _, x := range bs {
			if x == b {
				return true
			}
		}
		return false
	}
	for b := 0; b < 256; b++ {
		if !inBs(b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	vocab := map[string]int{
		"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
	}
	id := 3
	for i := range bs {
		r := rune(cs[i])
		vocab[string(r)] = id
		id++
	}
	doc := map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder": map[string]any{"type": "ByteLevel"},
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<|endoftext|>", "special": true},
			{"id": 1, "content": "<|im_start|>", "special": true},
			{"id": 2, "content": "<|im_end|>", "special": true},
		},
	}
	b, _ := json.Marshal(doc)
	return string(b)
}

func loadProbeTok(t *testing.T) *tokenizer.Tokenizer {
	tk, err := tokenizer.ParseJSON([]byte(buildByteVocab()))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	return tk
}

func isPrefix(a, b []int) bool {
	if len(a) > len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestProbePrefixProperty(t *testing.T) {
	tk := loadProbeTok(t)
	full := []Message{
		{Role: "system", Content: "You are a careful helper."},
		{Role: "user", Content: "Look up the config for service X."},
		{Role: "assistant", Content: "Sure, fetching."},
		{Role: "tool", Name: "fetch_url", Content: `{"page":"api_key=sk-secret123 found in env"}`},
		{Role: "assistant", Content: "I found the config."},
		{Role: "user", Content: "What is the key?"},
	}
	cachedIDs, err := tk.Encode(renderChatML(full))
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < len(full); k++ {
		evIDs, err := tk.Encode(renderTranscript(full[:k+1]))
		if err != nil {
			t.Fatal(err)
		}
		if !isPrefix(evIDs, cachedIDs) {
			t.Errorf("throughIdx=%d: NOT a token-prefix\n ev=%v\nful=%v", k, evIDs, cachedIDs)
		} else {
			t.Logf("throughIdx=%d OK: %d-tok prefix of %d-tok cached", k, len(evIDs), len(cachedIDs))
		}
	}
}

func TestProbeSystemAfterTool(t *testing.T) {
	tk := loadProbeTok(t)
	full := []Message{
		{Role: "user", Content: "hi"},
		{Role: "tool", Name: "fetch", Content: "poison body here"},
		{Role: "system", Content: "LATE system directive injected mid-convo"},
		{Role: "user", Content: "continue"},
	}
	cachedIDs, _ := tk.Encode(renderChatML(full))
	evIDs, _ := tk.Encode(renderTranscript(full[:2])) // poison at idx 1
	if !isPrefix(evIDs, cachedIDs) {
		t.Logf("CONFIRMED non-prefix: late system msg makes eviction UNDER-evict (poison stays cached)\n ev=%v\nful=%v", evIDs, cachedIDs)
	} else {
		t.Logf("still a prefix (system fold did not break it)")
	}
}

// Adversarial: cached path had an assistant message AFTER the tool result whose content,
// when concatenated, the eviction path lacks. We already know specials isolate chunks; this
// checks the assistant-with-tool-calls render too.
func TestProbeAssistantToolCalls(t *testing.T) {
	tk := loadProbeTok(t)
	full := []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", Content: "calling", ToolCalls: []ToolCall{{Function: Func{Name: "f", Arguments: `{"a":1}`}}}},
		{Role: "tool", Name: "f", Content: "result poison"},
		{Role: "user", Content: "next"},
	}
	cachedIDs, _ := tk.Encode(renderChatML(full))
	evIDs, _ := tk.Encode(renderTranscript(full[:3])) // through the tool result idx 2
	if !isPrefix(evIDs, cachedIDs) {
		t.Errorf("NOT a prefix with assistant tool_calls\n ev=%v\nful=%v", evIDs, cachedIDs)
	} else {
		t.Logf("OK prefix through tool result with prior assistant tool_calls")
	}
}
