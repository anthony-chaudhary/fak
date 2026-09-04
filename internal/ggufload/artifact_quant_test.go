package ggufload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClassifyTensorQuantUsesActualInventory(t *testing.T) {
	for _, tc := range []struct {
		name      string
		tensors   []TensorInfo
		wantName  string
		wantQ4K   bool
		inventory string
	}{
		{name: "empty", wantName: "unknown", inventory: "unknown"},
		{
			name:     "Q8 artifact with ordinary F32 tensors",
			tensors:  []TensorInfo{{Name: "token_embd.weight", Type: TensorQ8_0}, {Name: "output_norm.weight", Type: TensorF32}},
			wantName: "Q8_0", inventory: "mixed(F32+Q8_0)",
		},
		{
			name:     "Q4_K artifact with supported mixed tensors",
			tensors:  []TensorInfo{{Name: "blk.0.attn_q.weight", Type: TensorQ4_K}, {Name: "blk.0.attn_norm.weight", Type: TensorF32}, {Name: "blk.0.ffn_gate.weight", Type: TensorQ6_K}},
			wantName: "mixed(Q4_K+Q6_K)", inventory: "mixed(F32+Q4_K+Q6_K)", wantQ4K: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyTensorQuant(tc.tensors)
			if got.Name != tc.wantName || got.Inventory != tc.inventory || got.Q4KResident != tc.wantQ4K {
				t.Fatalf("ClassifyTensorQuant() = %#v, want name=%q inventory=%q q4k=%v", got, tc.wantName, tc.inventory, tc.wantQ4K)
			}
		})
	}
}

func TestClassifyQ2XLRealHeader(t *testing.T) {
	path := filepath.Join("..", "..", "_scratch", "qwen38-ud-q2xl", "header.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("real header fixture not present in scratch")
	}
	gg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	quant := ClassifyTensorQuant(gg.Tensors)
	if !quant.Q4KResident {
		t.Errorf("expected Q4KResident = true in Q2_K_XL mixture")
	}
	if !strings.Contains(quant.Name, "Q2_K") || !strings.Contains(quant.Name, "IQ2_XXS") {
		t.Errorf("quant.Name = %q, want Q2_K and IQ2_XXS present", quant.Name)
	}
}
