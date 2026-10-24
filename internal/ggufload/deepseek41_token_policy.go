package ggufload

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	deepSeek41PublishedVocabSize           = 129280
	deepSeek41PublishedCompressedVocabSize = 99092
	deepSeek41PublishedTokenMapSHA256      = "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
)

var deepSeek41PublishedTokenIDs = map[int]string{
	0: "<｜begin▁of▁sentence｜>",
	1: "<｜end▁of▁sentence｜>",
	2: "<｜▁pad▁｜>",
}

func validateDeepSeek41PublishedTokenizer(f *File, cfg *model.Config, eng *DeepSeek41Engram) error {
	if cfg.NumLayers != 40 || cfg.HiddenSize != 5120 || cfg.VocabSize != deepSeek41PublishedVocabSize {
		return nil
	}
	tokens, ok := f.StringArray("tokenizer.ggml.tokens")
	if !ok || len(tokens) != deepSeek41PublishedVocabSize {
		return fmt.Errorf("gguf: deepseek41 published tokenizer has %d tokens, want %d: %w",
			len(tokens), deepSeek41PublishedVocabSize, model.ErrV41ConfigAdmission)
	}
	for id, spelling := range deepSeek41PublishedTokenIDs {
		if tokens[id] != spelling {
			return fmt.Errorf("gguf: deepseek41 tokenizer token %d=%q, want %q: %w",
				id, tokens[id], spelling, model.ErrV41ConfigAdmission)
		}
	}
	if eng.CompressedVocabSize != 0 && eng.CompressedVocabSize != deepSeek41PublishedCompressedVocabSize {
		return fmt.Errorf("gguf: deepseek41 compressed tokenizer size=%d, independently derived %d: %w",
			eng.CompressedVocabSize, deepSeek41PublishedCompressedVocabSize, model.ErrV41ConfigAdmission)
	}
	if len(eng.TokenMap) != len(tokens) {
		return fmt.Errorf("gguf: deepseek41 token map has %d entries, want %d: %w",
			len(eng.TokenMap), len(tokens), model.ErrV41ConfigAdmission)
	}
	h := sha256.New()
	maxID := -1
	var word [4]byte
	for sourceID, compressedID := range eng.TokenMap {
		if compressedID < 0 || compressedID >= deepSeek41PublishedCompressedVocabSize {
			return fmt.Errorf("gguf: deepseek41 token map[%d]=%d outside [0,%d): %w",
				sourceID, compressedID, deepSeek41PublishedCompressedVocabSize, model.ErrV41ConfigAdmission)
		}
		maxID = max(maxID, compressedID)
		binary.LittleEndian.PutUint32(word[:], uint32(compressedID))
		_, _ = h.Write(word[:])
	}
	if maxID+1 != deepSeek41PublishedCompressedVocabSize {
		return fmt.Errorf("gguf: deepseek41 token map derives %d compressed tokens, want %d: %w",
			maxID+1, deepSeek41PublishedCompressedVocabSize, model.ErrV41ConfigAdmission)
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != deepSeek41PublishedTokenMapSHA256 {
		return fmt.Errorf("gguf: deepseek41 token map digest=%s, independently derived %s: %w",
			got, deepSeek41PublishedTokenMapSHA256, model.ErrV41ConfigAdmission)
	}
	if eng.CompressedVocabSize == 0 {
		eng.CompressedVocabSize = maxID + 1
	}
	return nil
}
