package ggufload

import (
	"errors"
	"maps"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41PublishedTokenizerPolicy(t *testing.T) {
	f := loadV41HeaderFixture(t)
	if _, err := f.Config(); err != nil {
		t.Fatalf("published real header: %v", err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok || len(gt.Tokens) != deepSeek41PublishedVocabSize {
		t.Fatalf("embedded tokenizer tokens=%d present=%t", len(gt.Tokens), ok)
	}
	for id, want := range deepSeek41PublishedTokenIDs {
		if got := gt.Tokens[id]; got != want {
			t.Fatalf("token %d=%q want %q", id, got, want)
		}
	}
	if f.DeepSeek41Engram == nil {
		t.Fatal("missing DeepSeek V4.1 Engram metadata")
	}
	if got := f.DeepSeek41Engram.CompressedVocabSize; got != deepSeek41PublishedCompressedVocabSize {
		t.Fatalf("compressed vocab=%d want %d", got, deepSeek41PublishedCompressedVocabSize)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41PublishedTokenizerPolicyRefusesDrift(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*File)
	}{
		{name: "pad spelling", edit: func(f *File) {
			tokens, _ := f.StringArray("tokenizer.ggml.tokens")
			tokens = slices.Clone(tokens)
			tokens[2] = "<bad-pad>"
			items := make([]Value, len(tokens))
			for i, token := range tokens {
				items[i] = Value{Type: TypeString, Value: token}
			}
			f.Metadata["tokenizer.ggml.tokens"] = Value{Type: TypeArray, Value: items}
		}},
		{name: "compressed size", edit: func(f *File) {
			f.Metadata["deepseek41.engram.compressed_vocab_size"] = Value{Type: TypeUint32, Value: uint32(deepSeek41PublishedCompressedVocabSize - 1)}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := loadV41HeaderFixture(t)
			changed := *base
			changed.Metadata = maps.Clone(base.Metadata)
			tc.edit(&changed)
			_, err := changed.Config()
			if !errors.Is(err, model.ErrV41ConfigAdmission) {
				t.Fatalf("Config error=%v want ErrV41ConfigAdmission", err)
			}
		})
	}
}
