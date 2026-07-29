package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// stopIDsFixture is the smallest tokenizer.json ParseJSON accepts that still declares
// both ChatML terminators AND a third special token that must NOT become a stop.
const stopIDsFixture = `{
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
  "decoder": {"type": "ByteLevel"},
  "added_tokens": [
    {"id": 100, "content": "<|im_end|>", "special": true},
    {"id": 101, "content": "<|endoftext|>", "special": true},
    {"id": 102, "content": "<|fim_pad|>", "special": true}
  ]
}`

// TestStopIDsUnionsChatMLSpecialsAndConfigEOS pins the stop set the in-kernel planner and
// the cmd/fakchat REPL now share. Before they shared it each carried its own copy, and
// nothing in this package witnessed the set at all: a mutant returning an EMPTY stop set
// passed the whole package. A stop token one decoder honours and the other does not ends
// the SAME turn at two different places, so the set is pinned here rather than only
// through a full weighted decode.
func TestStopIDsUnionsChatMLSpecialsAndConfigEOS(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(stopIDsFixture))
	if err != nil {
		t.Fatalf("parse fixture tokenizer: %v", err)
	}

	stops := StopIDs(tok, model.Config{EOSTokenID: 7, EOSTokenIDs: []int{9, 0, -1}})
	for _, id := range []int{100, 101, 7, 9} {
		if !stops[id] {
			t.Fatalf("StopIDs missing id %d: %v", id, stops)
		}
	}
	// Ids <= 0 are "unset", not token 0: a config that omits the field must never halt
	// decode at the first token.
	if stops[0] || stops[-1] {
		t.Fatalf("StopIDs admitted a non-positive id: %v", stops)
	}
	// A special token that is not a ChatML terminator is not a stop.
	if stops[102] {
		t.Fatalf("StopIDs treated a non-terminator special as a stop: %v", stops)
	}
	if len(stops) != 4 {
		t.Fatalf("StopIDs = %v, want exactly the two specials plus the two config eos ids", stops)
	}

	// With no eos declared in the config at all, the tokenizer's specials stand alone.
	bare := StopIDs(tok, model.Config{})
	if len(bare) != 2 || !bare[100] || !bare[101] {
		t.Fatalf("StopIDs(no config eos) = %v, want the two ChatML terminators", bare)
	}
}
