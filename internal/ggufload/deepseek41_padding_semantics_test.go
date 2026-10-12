package ggufload

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// The real Q2_K headers declare tokenizer padding=EOS=1, while vocabulary token
// 2 is the pad marker and HF declares padding=2. The serving tokenizer currently
// ignores tokenizer padding metadata. Engram independently consumes its pad ID
// in compressed-token space; equality with an ordinary token ID is not a binding.
// This exercises tokenizer construction and Engram hashing without table bytes,
// model execution, or a claim of reference-output parity.
// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41HeaderPaddingSemantics(t *testing.T) {
	f := loadV41HeaderFixture(t)
	const tokenizerPadKey = "tokenizer.ggml.padding_token_id"
	const engramPadKey = "deepseek41.engram.pad_id"
	for key, want := range map[string]uint64{
		tokenizerPadKey:               1,
		"tokenizer.ggml.eos_token_id": 1,
		engramPadKey:                  2,
	} {
		if got, ok := f.Uint64(key); !ok || got != want {
			t.Fatalf("real header %s = %d, present=%t; want %d", key, got, ok, want)
		}
	}
	hfRaw, err := os.ReadFile(filepath.Join("..", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var hf struct {
		PadTokenID int `json:"pad_token_id"`
	}
	if err := json.Unmarshal(hfRaw, &hf); err != nil {
		t.Fatal(err)
	}
	if hf.PadTokenID != 2 {
		t.Fatalf("HF pad_token_id = %d, want 2", hf.PadTokenID)
	}

	read := func(header *File) (*GGMLTokenizer, *tokenizer.Tokenizer, model.V41EngramLayout) {
		t.Helper()
		if _, err := header.Config(); err != nil {
			t.Fatalf("header Config: %v", err)
		}
		gt, ok := header.GGMLTokenizer()
		if !ok {
			t.Fatal("real header has no embedded BPE tokenizer")
		}
		tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
		if err != nil {
			t.Fatalf("FromGGML: %v", err)
		}
		layout, err := buildV41EngramLayout(header.DeepSeek41Engram)
		if err != nil {
			t.Fatalf("buildV41EngramLayout: %v", err)
		}
		return gt, tok, layout
	}
	gt, tok, layout := read(f)
	if len(gt.Tokens) <= 2 || gt.Tokens[2] != "<｜▁pad▁｜>" || gt.EOS != 1 || layout.PadID != 2 {
		t.Fatal("real header lost the vocabulary-pad/EOS/Engram-pad discrepancy")
	}
	const prompt = "Hello, 123 世界\n<｜▁pad▁｜><｜end▁of▁sentence｜>"
	encode := func(tok *tokenizer.Tokenizer) []int {
		t.Helper()
		ids, err := tok.Encode(prompt)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if len(ids) < 3 || !slices.Equal(ids[len(ids)-2:], []int{2, 1}) {
			t.Fatalf("encoded prompt = %v, want text followed by pad marker 2 and EOS 1", ids)
		}
		return ids
	}
	ids := encode(tok)
	hash := func(layout model.V41EngramLayout) []uint32 {
		t.Helper()
		state, err := model.NewV41EngramHashState(layout)
		if err != nil {
			t.Fatalf("NewV41EngramHashState: %v", err)
		}
		rows, err := state.Hash(ids, nil)
		if err != nil {
			t.Fatalf("Hash: %v", err)
		}
		if len(rows) == 0 {
			t.Fatal("real header produced no Engram row IDs")
		}
		return rows
	}
	rows := hash(layout)

	// Change only the parsed scalar; all committed header bytes stay untouched.
	changed := *f
	changed.Metadata = maps.Clone(f.Metadata)
	changed.Metadata[tokenizerPadKey] = Value{Type: TypeUint32, Value: uint32(hf.PadTokenID)}
	changedGT, changedTok, changedLayout := read(&changed)
	if !reflect.DeepEqual(gt, changedGT) {
		t.Fatal("tokenizer padding metadata changed the serving tokenizer inputs")
	}
	if got, want := changedTok.Identity(), tok.Identity(); got == "" || got != want {
		t.Fatalf("tokenizer padding metadata changed operational identity: %q != %q", got, want)
	}
	if got := encode(changedTok); !slices.Equal(got, ids) {
		t.Fatalf("tokenizer padding metadata changed encoding: %v != %v", got, ids)
	}
	if !reflect.DeepEqual(changedLayout, layout) || !slices.Equal(hash(changedLayout), rows) {
		t.Fatal("tokenizer padding metadata changed Engram layout or row IDs")
	}

	// Positive control: changing Engram's own compressed-space pad affects the
	// sequence-boundary hashes, with the original tokenizer metadata retained.
	control := *f
	control.Metadata = maps.Clone(f.Metadata)
	control.Metadata[engramPadKey] = Value{Type: TypeUint32, Value: uint32(3)}
	if _, err := control.Config(); err != nil {
		t.Fatalf("Engram pad control Config: %v", err)
	}
	controlLayout, err := buildV41EngramLayout(control.DeepSeek41Engram)
	if err != nil {
		t.Fatalf("Engram pad control layout: %v", err)
	}
	if controlLayout.PadID != 3 || slices.Equal(hash(controlLayout), rows) {
		t.Fatal("changing Engram's compressed-space pad did not change row IDs")
	}
}
