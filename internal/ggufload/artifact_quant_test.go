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
	if quant.Recipe != "UD-Q2_K_XL" {
		t.Errorf("quant.Recipe = %q, want UD-Q2_K_XL", quant.Recipe)
	}
	if !strings.Contains(quant.Name, "Q2_K") || !strings.Contains(quant.Name, "IQ2_XXS") {
		t.Errorf("quant.Name = %q, want Q2_K and IQ2_XXS present", quant.Name)
	}
	if err := ValidateUDQ2KXL(gg.Tensors); err != nil {
		t.Errorf("ValidateUDQ2KXL = %v, want nil", err)
	}
}

func canonicalUDQ2KXLTensors() []TensorInfo {
	return []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{151936, 4096}, Type: TensorQ8_0},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{4096, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.attn_k.weight", Dims: []uint64{1024, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.attn_v.weight", Dims: []uint64{1024, 4096}, Type: TensorQ6_K},
		{Name: "blk.0.attn_output.weight", Dims: []uint64{4096, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{11008, 4096}, Type: TensorQ2_K},
		{Name: "blk.0.ffn_up.weight", Dims: []uint64{11008, 4096}, Type: TensorQ2_K},
		{Name: "blk.0.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ2_XXS},
		{Name: "blk.1.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ2_XS},
		{Name: "blk.2.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ2_S},
		{Name: "blk.3.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ1_S},
		{Name: "blk.4.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ1_M},
		{Name: "blk.5.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ3_XXS},
		{Name: "blk.6.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorIQ4_XS},
		{Name: "blk.7.attn_q.weight", Dims: []uint64{4096, 4096}, Type: TensorQ5_K},
		{Name: "blk.7.ffn_gate.weight", Dims: []uint64{11008, 4096}, Type: TensorQ4_0},
		{Name: "blk.0.attn_norm.weight", Dims: []uint64{4096}, Type: TensorF32},
		{Name: "blk.0.ffn_norm.weight", Dims: []uint64{4096}, Type: TensorF16},
		{Name: "output.weight", Dims: []uint64{151936, 4096}, Type: TensorQ6_K},
	}
}

func pureQ2KTensors() []TensorInfo {
	return []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{4096, 4096}, Type: TensorQ2_K},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{4096, 4096}, Type: TensorQ2_K},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{11008, 4096}, Type: TensorQ2_K},
		{Name: "output_norm.weight", Dims: []uint64{4096}, Type: TensorF32},
	}
}

func standardQ4KMTensors() []TensorInfo {
	return []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{4096, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{4096, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.attn_v.weight", Dims: []uint64{1024, 4096}, Type: TensorQ6_K},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{11008, 4096}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorQ6_K},
		{Name: "output_norm.weight", Dims: []uint64{4096}, Type: TensorF32},
	}
}

func TestClassifyTensorQuantRecipe(t *testing.T) {
	tests := []struct {
		name       string
		tensors    []TensorInfo
		wantRecipe string
		wantName   string
		wantQ4K    bool
	}{
		{
			name:       "empty",
			tensors:    nil,
			wantRecipe: "",
			wantName:   "unknown",
			wantQ4K:    false,
		},
		{
			name:       "pure Q2_K",
			tensors:    pureQ2KTensors(),
			wantRecipe: "",
			wantName:   "Q2_K",
			wantQ4K:    false,
		},
		{
			name:       "standard Q4_K_M",
			tensors:    standardQ4KMTensors(),
			wantRecipe: "Q4_K_M",
			wantName:   "mixed(Q4_K+Q6_K)",
			wantQ4K:    true,
		},
		{
			name:       "canonical UD-Q2_K_XL",
			tensors:    canonicalUDQ2KXLTensors(),
			wantRecipe: "UD-Q2_K_XL",
			wantName:   "mixed(IQ1_M+IQ1_S+IQ2_S+IQ2_XS+IQ2_XXS+IQ3_XXS+IQ4_XS+Q2_K+Q4_0+Q4_K+Q5_K+Q6_K+Q8_0)",
			wantQ4K:    true,
		},
		{
			name: "predominantly Q2_K in MLP with high-precision attention",
			tensors: []TensorInfo{
				{Name: "blk.0.attn_q.weight", Dims: []uint64{4096, 4096}, Type: TensorQ4_K},
				{Name: "blk.0.attn_v.weight", Dims: []uint64{1024, 4096}, Type: TensorQ6_K},
				{Name: "blk.0.ffn_gate.weight", Dims: []uint64{11008, 4096}, Type: TensorQ2_K},
				{Name: "blk.0.ffn_up.weight", Dims: []uint64{11008, 4096}, Type: TensorQ2_K},
				{Name: "blk.0.ffn_down.weight", Dims: []uint64{4096, 11008}, Type: TensorQ2_K},
				{Name: "output_norm.weight", Dims: []uint64{4096}, Type: TensorF32},
			},
			wantRecipe: "UD-Q2_K_XL",
			wantName:   "mixed(Q2_K+Q4_K+Q6_K)",
			wantQ4K:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyTensorQuant(tc.tensors)
			if got.Recipe != tc.wantRecipe {
				t.Errorf("Recipe = %q, want %q", got.Recipe, tc.wantRecipe)
			}
			if got.Name != tc.wantName {
				t.Errorf("Name = %q, want %q", got.Name, tc.wantName)
			}
			if got.Q4KResident != tc.wantQ4K {
				t.Errorf("Q4KResident = %v, want %v", got.Q4KResident, tc.wantQ4K)
			}
		})
	}
}

func TestUDQ2KXL(t *testing.T) {
	canonical := canonicalUDQ2KXLTensors()
	if !IsUDQ2KXL(canonical) {
		t.Errorf("IsUDQ2KXL(canonical) = false, want true")
	}
	if !IsQ2KXL(canonical) {
		t.Errorf("IsQ2KXL(canonical) = false, want true")
	}
	if err := ValidateUDQ2KXL(canonical); err != nil {
		t.Fatalf("ValidateUDQ2KXL(canonical) = %v, want nil", err)
	}

	pure := pureQ2KTensors()
	if IsUDQ2KXL(pure) {
		t.Errorf("IsUDQ2KXL(pure) = true, want false")
	}
	if IsQ2KXL(pure) {
		t.Errorf("IsQ2KXL(pure) = true, want false")
	}
	if err := ValidateUDQ2KXL(pure); err == nil {
		t.Errorf("ValidateUDQ2KXL(pure) succeeded, want error for pure Q2_K")
	}

	q4km := standardQ4KMTensors()
	if IsUDQ2KXL(q4km) {
		t.Errorf("IsUDQ2KXL(q4km) = true, want false")
	}
	if err := ValidateUDQ2KXL(q4km); err == nil {
		t.Errorf("ValidateUDQ2KXL(q4km) succeeded, want error for missing Q2_K")
	}
}

func TestUDQ2KXLValidationFailures(t *testing.T) {
	tests := []struct {
		name        string
		tensors     []TensorInfo
		errContains string
	}{
		{
			name:        "empty tensor slice",
			tensors:     nil,
			errContains: "empty tensor inventory",
		},
		{
			name: "constituent tensor with empty name",
			tensors: []TensorInfo{
				{Name: "", Dims: []uint64{256}, Type: TensorQ2_K},
			},
			errContains: "empty name",
		},
		{
			name: "constituent tensor with empty dims",
			tensors: []TensorInfo{
				{Name: "bad.nodims", Dims: []uint64{}, Type: TensorQ2_K},
			},
			errContains: "empty dims",
		},
		{
			name: "constituent tensor with zero dimension",
			tensors: []TensorInfo{
				{Name: "bad.zerodim", Dims: []uint64{256, 0}, Type: TensorQ2_K},
			},
			errContains: "zero dim",
		},
		{
			name: "unadmitted constituent type Q3_K",
			tensors: append(canonicalUDQ2KXLTensors(), TensorInfo{
				Name: "extra.q3k", Dims: []uint64{256}, Type: TensorQ3_K,
			}),
			errContains: "unadmitted",
		},
		{
			name: "truncated mixture missing Q2_K",
			tensors: func() []TensorInfo {
				var filtered []TensorInfo
				for _, ti := range canonicalUDQ2KXLTensors() {
					if ti.Type != TensorQ2_K {
						filtered = append(filtered, ti)
					}
				}
				return filtered
			}(),
			errContains: "TensorQ2_K",
		},
		{
			name: "truncated mixture missing high-precision boundary",
			tensors: []TensorInfo{
				{Name: "blk.0.ffn_gate.weight", Dims: []uint64{256, 256}, Type: TensorQ2_K},
				{Name: "blk.0.ffn_down.weight", Dims: []uint64{256, 256}, Type: TensorIQ2_XXS},
			},
			errContains: "high-precision",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateUDQ2KXL(tc.tensors)
			if err == nil {
				t.Fatalf("ValidateUDQ2KXL() = nil, want error containing %q", tc.errContains)
			}
			if !strings.Contains(err.Error(), tc.errContains) {
				t.Fatalf("ValidateUDQ2KXL() error = %q, want substring %q", err.Error(), tc.errContains)
			}
		})
	}
}
