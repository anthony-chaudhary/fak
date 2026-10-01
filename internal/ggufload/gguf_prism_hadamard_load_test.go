package ggufload

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestPrismHadamardMalformedRefusedByModelLoad makes the rotation contract
// reachable from the actual quantized GGUF load path. Header-only parser tests
// cannot catch a loader that never calls PrismHadamardMeta and silently serves
// Bonsai-2 weights in the wrong basis.
func TestPrismHadamardMalformedRefusedByModelLoad(t *testing.T) {
	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 11)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "qwen35.embedding_length", 128)
	writeKVUint32(&b, "qwen35.block_count", 1)
	writeKVUint32(&b, "qwen35.attention.head_count", 1)
	writeKVUint32(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32(&b, "qwen35.attention.key_length", 128)
	writeKVUint32(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32(&b, "qwen35.feed_forward_length", 128)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32(&b, "prism.hadamard.version", 1) // deliberately incomplete
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{128, 128}, TensorQ2_0, 0)
	padToAlignment(&b, 32)
	b.Write(make([]byte, 128*34)) // 128 group-128 blocks, all-zero scale
	path := filepath.Join(t.TempDir(), "bonsai2-incomplete-rotation.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadModelQ4K(path)
	if !errors.Is(err, ErrPrismHadamardMalformed) {
		t.Fatalf("LoadModelQ4K error = %v, want ErrPrismHadamardMalformed", err)
	}
}
