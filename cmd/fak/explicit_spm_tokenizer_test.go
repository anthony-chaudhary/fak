package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=100ms lane=default
func TestExplicitServeSPMTokenizerFileAndDirectory(t *testing.T) {
	pieces := []string{"<unk>", "<s>", "</s>", "a"}
	scores := []float32{0, 0, 0, 0}
	types := []int32{2, 3, 3, 1}
	for i := 0; i < 256; i++ {
		pieces = append(pieces, fmt.Sprintf("<0x%02X>", i))
		scores = append(scores, -100)
		types = append(types, 6)
	}
	doc := map[string]any{"fak_tokenizer_profile": "llama-spm-v1", "normalization": "identity", "tokens": pieces, "scores": scores, "types": types, "options": map[string]any{"bos_id": 1, "eos_id": 2, "add_bos": true, "add_eos": true, "add_space_prefix": false}}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{path, dir} {
		tok, err := loadExplicitServeTokenizer(input)
		if err != nil || !tok.IsSPM() {
			t.Fatalf("%s: %v", input, err)
		}
		ids, err := tok.EncodeSPM("a", true, false)
		if err != nil || !reflect.DeepEqual(ids, []int{1, 3, 2}) {
			t.Fatalf("%v %v", ids, err)
		}
	}
	if tok, err := loadAutomaticServeDirectoryTokenizer(dir); tok != nil || !errors.Is(err, tokenizer.ErrNativeSPMProfile) {
		t.Fatalf("native profile automatically activated: %v", err)
	}
	doc["normalization"] = "NFKC"
	b, err = json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if tok, err := loadExplicitServeTokenizer(dir); tok != nil || !errors.Is(err, tokenizer.ErrNativeSPMProfile) {
		t.Fatalf("recognized profile fell back: %v", err)
	}
	if tok, err := loadAutomaticServeDirectoryTokenizer(dir); tok != nil || !errors.Is(err, tokenizer.ErrNativeSPMProfile) {
		t.Fatalf("invalid auto profile fell back: %v", err)
	}
	legacy := []byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`)
	if err = os.WriteFile(path, legacy, 0600); err != nil {
		t.Fatal(err)
	}
	if tok, err := loadAutomaticServeDirectoryTokenizer(dir); err != nil || tok.IsSPM() {
		t.Fatalf("legacy directory changed: %v", err)
	}
}
