package ggufload

import (
	"errors"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41PublishedTokenizerPolicy(t *testing.T) {
	f := loadV41HeaderFixture(t)
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("published real header: %v", err)
	}
	if cfg.BOSTokenID == nil || *cfg.BOSTokenID != deepSeek41PublishedBOSTokenID || cfg.EOSTokenID != deepSeek41PublishedEOSTokenID {
		t.Fatalf("runtime special ids BOS=%v EOS=%d, want %d/%d",
			cfg.BOSTokenID, cfg.EOSTokenID, deepSeek41PublishedBOSTokenID, deepSeek41PublishedEOSTokenID)
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
		name    string
		wantErr string
		edit    func(*File)
	}{
		{name: "bos id", wantErr: "bos_token_id=1, want 0", edit: func(f *File) {
			f.Metadata["tokenizer.ggml.bos_token_id"] = Value{Type: TypeUint32, Value: uint32(1)}
		}},
		{name: "eos id", wantErr: "eos_token_id=2, want 1", edit: func(f *File) {
			f.Metadata["tokenizer.ggml.eos_token_id"] = Value{Type: TypeUint32, Value: uint32(2)}
		}},
		{name: "missing bos id", wantErr: "missing tokenizer.ggml.bos_token_id=0", edit: func(f *File) {
			delete(f.Metadata, "tokenizer.ggml.bos_token_id")
		}},
		{name: "missing eos id", wantErr: "missing tokenizer.ggml.eos_token_id=1", edit: func(f *File) {
			delete(f.Metadata, "tokenizer.ggml.eos_token_id")
		}},
		{name: "pad spelling", wantErr: "tokenizer token 2", edit: func(f *File) {
			tokens, _ := f.StringArray("tokenizer.ggml.tokens")
			tokens = slices.Clone(tokens)
			tokens[2] = "<bad-pad>"
			items := make([]Value, len(tokens))
			for i, token := range tokens {
				items[i] = Value{Type: TypeString, Value: token}
			}
			f.Metadata["tokenizer.ggml.tokens"] = Value{Type: TypeArray, Value: items}
		}},
		{name: "compressed size", wantErr: "compressed tokenizer size", edit: func(f *File) {
			f.Metadata["deepseek41.engram.compressed_vocab_size"] = Value{Type: TypeUint32, Value: uint32(deepSeek41PublishedCompressedVocabSize - 1)}
		}},
		{name: "map length", wantErr: "token map has 129279 entries", edit: func(f *File) {
			setPublishedTokenMap(t, f, func(tokenMap []int32) []int32 {
				return tokenMap[:len(tokenMap)-1]
			})
		}},
		{name: "map range", wantErr: "outside [0,99092)", edit: func(f *File) {
			setPublishedTokenMap(t, f, func(tokenMap []int32) []int32 {
				tokenMap[0] = deepSeek41PublishedCompressedVocabSize
				return tokenMap
			})
		}},
		{name: "map digest", wantErr: "token map digest=", edit: func(f *File) {
			setPublishedTokenMap(t, f, func(tokenMap []int32) []int32 {
				tokenMap[0] = (tokenMap[0] + 1) % deepSeek41PublishedCompressedVocabSize
				return tokenMap
			})
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
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Config error=%q want branch %q", err, tc.wantErr)
			}
		})
	}
}

func setPublishedTokenMap(t *testing.T, f *File, edit func([]int32) []int32) {
	t.Helper()
	got, ok := f.IntArray("deepseek41.engram.token_map")
	if !ok || len(got) != deepSeek41PublishedVocabSize {
		t.Fatalf("fixture token map entries=%d present=%t", len(got), ok)
	}
	tokenMap := make([]int32, len(got))
	for i, id := range got {
		tokenMap[i] = int32(id)
	}
	f.Metadata["deepseek41.engram.token_map"] = vcruzI32Array(edit(tokenMap))
}
