package ggufload

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The fixture is the complete, losslessly compressed header of the pinned public
// artifact. Provenance distinguishes its locally computed digest from the upstream
// full-file LFS digest. It contains no weight payload and proves no forward execution.
func TestQwen38UDQ2KXLPinnedHeader(t *testing.T) {
	compressed, err := os.ReadFile(filepath.Join("testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatal(err)
	}
	const wantHash = "1fe82fda85430cca654a156e9ec2915baf460752197013563b426db2581dcc0f"
	if len(raw) != 10996640 || fmt.Sprintf("%x", sha256.Sum256(raw)) != wantHash {
		t.Fatalf("pinned header identity mismatch: bytes=%d sha256=%x", len(raw), sha256.Sum256(raw))
	}
	gg, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read pinned header: %v", err)
	}
	counts := map[TensorType]int{}
	for _, tensor := range gg.Tensors {
		counts[tensor.Type]++
	}
	wantCounts := map[TensorType]int{
		TensorF32: 360, TensorQ2_K: 16, TensorQ3_K: 5, TensorQ4_K: 21,
		TensorQ5_K: 2, TensorQ6_K: 6, TensorQ8_0: 98,
		TensorIQ2_XXS: 48, TensorIQ2_XS: 34, TensorIQ3_XXS: 112,
		TensorIQ1_S: 20, TensorIQ3_S: 57, TensorIQ2_S: 67, TensorIQ4_XS: 19, TensorIQ1_M: 1,
	}
	if len(gg.Tensors) != 866 || !reflect.DeepEqual(counts, wantCounts) {
		t.Fatalf("real tensor inventory = %v (%d tensors), want %v (866 tensors)", counts, len(gg.Tensors), wantCounts)
	}
	quant := ClassifyTensorQuant(gg.Tensors)
	const wantName = "mixed(IQ1_M+IQ1_S+IQ2_S+IQ2_XS+IQ2_XXS+IQ3_S+IQ3_XXS+IQ4_XS+Q2_K+Q3_K+Q4_K+Q5_K+Q6_K+Q8_0)"
	if !quant.Q4KResident || quant.Name != wantName || quant.Inventory != "mixed(F32+"+wantName[len("mixed("):] {
		t.Fatalf("real artifact classification = %+v", quant)
	}
	cfg, err := gg.Config()
	if err != nil {
		t.Fatalf("real artifact config: %v", err)
	}
	if cfg.Name != "Qwen3.8-27B" || cfg.ModelType != "qwen35" || cfg.HiddenSize != 5120 || cfg.NumLayers != 64 || cfg.NumNextNPredictLayers != 1 || cfg.FullAttentionInterval != 4 || cfg.MaxPositionEmbeddings != 262144 || cfg.EOSTokenID != 248046 {
		t.Fatalf("real artifact architecture/config mismatch: %+v", cfg)
	}
	toks, ok := gg.StringArray("tokenizer.ggml.tokens")
	if !ok || cfg.VocabSize != len(toks) || cfg.EOSTokenID >= len(toks) || toks[cfg.EOSTokenID] != "<|im_end|>" {
		t.Fatalf("real artifact tokenizer mismatch: vocab=%d eos=%d", len(toks), cfg.EOSTokenID)
	}
}
