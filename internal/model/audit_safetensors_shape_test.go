package model

import (
	"strings"
	"testing"
)

// Regression for fak#12603: LoadSafetensors accepted a tensor whose header declared
// shape [16 32] F32 (2048 payload bytes) while data_offsets covered only 512 bytes.
// The shortened payload was stored under the declared shape, so a later embedding
// lookup indexed past the decoded storage and panicked.
//
// The loader must reject inconsistent tensor geometry at load time with an error
// that names the tensor, instead of silently accepting a short payload.
func TestLoadSafetensorsRejectsShapePayloadLengthMismatch(t *testing.T) {
	// shape [16,32] F32 implies 16*32*4 = 2048 bytes, but supply only 512 (quarter).
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{16, 32},
			data:  f32TestBytes(sequenceFloats(16*32/4, 0.05)),
		},
		"model.norm.weight": {
			dtype: "F32",
			shape: []int{32},
			data:  f32TestBytes(sequenceFloats(32, 0.25)),
		},
	})
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 16, TieWordEmbeddings: true}

	_, err := LoadSafetensors(path, cfg)
	if err == nil {
		t.Fatal("LoadSafetensors accepted a tensor whose payload length disagrees with its declared shape")
	}
	if !strings.Contains(err.Error(), "embed_tokens.weight") {
		t.Fatalf("error must name the malformed tensor; got: %v", err)
	}
}

// A tensor whose declared geometry agrees with its payload length must still load,
// so the geometry guard rejects only the malformed case it is meant to catch.
func TestLoadSafetensorsAcceptsConsistentShapePayload(t *testing.T) {
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{16, 32},
			data:  f32TestBytes(sequenceFloats(16*32, 0.05)),
		},
		"model.norm.weight": {
			dtype: "F32",
			shape: []int{32},
			data:  f32TestBytes(sequenceFloats(32, 0.25)),
		},
	})
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 16, TieWordEmbeddings: true}

	if _, err := LoadSafetensors(path, cfg); err != nil {
		t.Fatalf("LoadSafetensors rejected a well-formed tensor: %v", err)
	}
}
