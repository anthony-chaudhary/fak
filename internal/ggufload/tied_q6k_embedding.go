package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// tied_q6k_embedding.go — keep a tied Q6_K token table packed (fak#13567).
//
// A dense Qwen3.5-family GGUF with a Q6_K token_embd.weight and no output.weight (e.g.
// Qwen3.5/3.8-4B Q4_K_M: 248320x2560) used to load as a 2.37 GiB f32 gather table plus a
// 0.67 GiB native-Q8 head copy. The tied Q6_K mode retains the 0.49 GiB GGUF payload once
// (model.QuantBuilder.SetTiedQ6KEmbedding): rows are dequantized on demand for the gather and
// the head runs as the resident Q6_K GEMV (Metal when the Q6_K band is device-resident).
//
// The mode is selected automatically when the checkpoint qualifies AND the load retains dense
// Q6_K (denseKQuantRetained), i.e. the selected backend can execute a resident Q6_K head. A
// backend that sends Q6_K to the dequant-to-Q8 path keeps the legacy two-copy layout, and
// WithTiedQ6KEmbeddingResident(false) restores it explicitly.

type tiedQ6KEmbeddingMode uint8

const (
	tiedQ6KEmbeddingAuto tiedQ6KEmbeddingMode = iota
	tiedQ6KEmbeddingOn
	tiedQ6KEmbeddingOff
)

// WithTiedQ6KEmbeddingResident overrides the automatic tied-Q6_K packed embedding selection.
// true requires a qualifying checkpoint (the load fails with the validation reason otherwise);
// false keeps the legacy f32 gather table plus native-Q8 head.
func WithTiedQ6KEmbeddingResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		if enabled {
			o.tiedQ6KEmbeddingMode = tiedQ6KEmbeddingOn
		} else {
			o.tiedQ6KEmbeddingMode = tiedQ6KEmbeddingOff
		}
	}
}

// resolveTiedQ6KEmbedding sets o.residentTiedQ6KEmbedding for this checkpoint. It runs after
// the other packed-embedding options are known so the modes stay mutually exclusive, and it is
// shared by the loader and EstimateQ4KLoadMemoryPlan so admission prices the same layout.
func (s *WeightSource) resolveTiedQ6KEmbedding(cfg model.Config, o *q4kLoadOptions) error {
	o.residentTiedQ6KEmbedding = false
	otherPacked := o.residentQ2KEmbedding || o.residentQ4KEmbedding || o.residentPQ2Embedding
	switch o.tiedQ6KEmbeddingMode {
	case tiedQ6KEmbeddingOff:
		return nil
	case tiedQ6KEmbeddingOn:
		if otherPacked {
			return fmt.Errorf("gguf: tied Q6_K embedding conflicts with requested packed embedding format")
		}
		if err := s.validateTiedQ6KEmbedding(cfg); err != nil {
			return err
		}
		o.residentTiedQ6KEmbedding = true
		return nil
	}
	if otherPacked || !denseKQuantRetained(*o, TensorQ6_K) || o.streamedExperts || o.expertShardSet {
		return nil
	}
	if s.validateTiedQ6KEmbedding(cfg) == nil {
		o.residentTiedQ6KEmbedding = true
	}
	return nil
}

// validateTiedQ6KEmbedding is the tied twin of validateResidentPackedEmbedding: dense
// Qwen3.5-family hybrid, tie_word_embeddings, a Q6_K token_embd.weight of exactly
// vocab x hidden/256 x 210 bytes, and NO distinct output.weight.
func (s *WeightSource) validateTiedQ6KEmbedding(cfg model.Config) error {
	if !cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		return fmt.Errorf("gguf: tied Q6_K embedding requires a dense Qwen3.5-family hybrid model")
	}
	if !cfg.TieWordEmbeddings {
		return fmt.Errorf("gguf: tied Q6_K embedding requires tied word embeddings")
	}
	var embInfo *TensorInfo
	for i := range s.File.Tensors {
		t := &s.File.Tensors[i]
		switch t.Name {
		case "token_embd.weight":
			embInfo = t
		case "output.weight":
			return fmt.Errorf("gguf: tied Q6_K embedding requires no distinct output.weight")
		}
	}
	if embInfo == nil {
		return fmt.Errorf("gguf: token_embd.weight tensor missing")
	}
	if embInfo.Type != TensorQ6_K {
		return fmt.Errorf("gguf: token_embd.weight is %s, want Q6_K", embInfo.Type)
	}
	shape, err := modelShapeFromGGUFDims(embInfo.Name, embInfo.Dims)
	if err != nil {
		return err
	}
	if len(shape) != 2 {
		return fmt.Errorf("gguf: token_embd.weight rank %d != 2", len(shape))
	}
	vocab, hidden := shape[0], shape[1]
	if cfg.HiddenSize != 0 && hidden != cfg.HiddenSize {
		return fmt.Errorf("gguf: token_embd.weight hidden %d != config %d", hidden, cfg.HiddenSize)
	}
	if cfg.VocabSize != 0 && vocab != cfg.VocabSize {
		return fmt.Errorf("gguf: token_embd.weight vocab %d != config %d", vocab, cfg.VocabSize)
	}
	if hidden%qkK != 0 {
		return fmt.Errorf("gguf: token_embd.weight hidden dimension %d is not divisible by %d", hidden, qkK)
	}
	payload, err := tensorPayloadBytes(*embInfo)
	if err != nil {
		return err
	}
	if want := uint64(vocab) * uint64(hidden/qkK) * blockQ6KBytes; payload != want {
		return fmt.Errorf("gguf: token_embd.weight payload bytes %d != expected %d", payload, want)
	}
	return nil
}
