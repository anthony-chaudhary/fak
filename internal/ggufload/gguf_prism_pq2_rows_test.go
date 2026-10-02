package ggufload

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Packed PQ2 row normalization must be the same permutation as the existing
// decoded F32 Qwen3.5 source normalization. Every row gets a distinct ternary
// four-trit marker, so an identity or wrong-head permutation cannot pass.
func TestPrismPQ2PackedRowsMatchFloatCanonicalNormalization(t *testing.T) {
	cfg := model.Config{
		ModelType: "qwen35", HiddenSize: 128, NumHeads: 2, NumKVHeads: 1,
		HeadDim: 4, AttnOutputGate: true, LayerTypes: []string{"linear_attention", "full_attention"},
		LinearKeyHeadDim: 4, LinearValueHeadDim: 4,
		LinearNumKeyHeads: 2, LinearNumValueHeads: 4,
	}
	for _, tc := range []struct {
		name string
		rows int
	}{
		{"model.layers.0.linear_attn.in_proj_qkv.weight", 32},
		{"model.layers.0.linear_attn.in_proj_z.weight", 16},
		{"model.layers.1.self_attn.q_proj.weight", 16},
		{"model.layers.1.self_attn.k_proj.weight", 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, tc.rows*blockPQ2_0Bytes)
			for row := 0; row < tc.rows; row++ {
				block := raw[row*blockPQ2_0Bytes : (row+1)*blockPQ2_0Bytes]
				binary.LittleEndian.PutUint16(block[:2], 0x3c00) // exact f16 1.0
				for i := 2; i < len(block); i++ {
					block[i] = 0x55 // four zero trits
				}
				marker := row + 1
				block[2] = byte(marker%3) | byte((marker/3)%3)<<2 |
					byte((marker/9)%3)<<4 | byte((marker/27)%3)<<6
			}
			source := make([]float32, tc.rows*128)
			dequantQ2_0Scalar(source, raw)
			want, err := normalizeCanonicalTensorData(tc.name, source, cfg)
			if err != nil {
				t.Fatalf("F32 normalize: %v", err)
			}
			packed, err := normalizePrismPQ2Rows(tc.name, []int{tc.rows, 128}, raw, cfg)
			if err != nil {
				t.Fatalf("PQ2 normalize: %v", err)
			}
			if bytes.Equal(packed, raw) {
				t.Fatal("test geometry did not exercise a row permutation")
			}
			got := make([]float32, len(want))
			dequantQ2_0Scalar(got, packed)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("decoded packed value[%d] = %g, F32 canonical = %g", i, got[i], want[i])
				}
			}
		})
	}
}

// Prism's pinned llama-graph.cpp perm_rep maps tiled GDN V heads
// [k0r0,k1r0,k0r1,k1r1] to grouped [k0r0,k0r1,k1r0,k1r1] before H/sign.
// Fak's GDN writes core[h*headDim] with kh=h/rep, i.e. the grouped order.
// The PQ2 QKV/gate row normalizer must make the source values agree with that
// core, while a grouped ssm_out must keep its columns as stored.
func TestPrismPQ2GroupedGDNValueOrderMatchesCore(t *testing.T) {
	const (
		nK, nV, headDim = 2, 4, 32
		width           = 128
	)
	cfg := model.Config{
		ModelType: "qwen35", HiddenSize: width, NumLayers: 1,
		LayerTypes:       []string{"linear_attention"},
		LinearKeyHeadDim: headDim, LinearValueHeadDim: headDim,
		LinearNumKeyHeads: nK, LinearNumValueHeads: nV,
	}
	gateName := "model.layers.0.linear_attn.in_proj_z.weight"
	gateRaw := make([]byte, nV*headDim*blockPQ2_0Bytes)
	for sourceHead := 0; sourceHead < nV; sourceHead++ {
		for lane := 0; lane < headDim; lane++ {
			block := gateRaw[(sourceHead*headDim+lane)*blockPQ2_0Bytes : (sourceHead*headDim+lane+1)*blockPQ2_0Bytes]
			// Exact f16 powers of two label the tiled source heads [1,2,4,8].
			binary.LittleEndian.PutUint16(block[:2], 0x3c00+uint16(sourceHead)<<10)
			for i := 2; i < len(block); i++ {
				block[i] = 0x55
			}
			block[2] = 0x56 // first trit = +scale, rest zero
		}
	}
	gateGrouped, err := normalizePrismPQ2Rows(gateName, []int{nV * headDim, width}, gateRaw, cfg)
	if err != nil {
		t.Fatal(err)
	}
	gateValues := make([]float32, nV*headDim*width)
	dequantQ2_0Scalar(gateValues, gateGrouped)
	wantGrouped := []float32{1, 4, 2, 8}
	for h, want := range wantGrouped {
		if got := gateValues[h*headDim*width]; got != want {
			t.Fatalf("grouped GDN head %d marker = %g, want %g from upstream perm_rep", h, got, want)
		}
	}

	// A grouped PQ2 ssm_out row selects grouped head 1. Reordering its
	// columns as stock unrotated GGUF does would select tiled head 1 instead.
	outName := "model.layers.0.linear_attn.out_proj.weight"
	if !prismPQ2ResidentEligible(cfg, outName, true) || prismPQ2ResidentEligible(cfg, outName, false) {
		t.Fatal("ssm_out PQ2 residency must require declared grouped V")
	}
	outRaw := make([]byte, width*blockPQ2_0Bytes)
	block := outRaw[:blockPQ2_0Bytes]
	binary.LittleEndian.PutUint16(block[:2], 0x3c00)
	for i := 2; i < len(block); i++ {
		block[i] = 0x55
	}
	block[2+headDim/4] = 0x56 // select grouped head 1, lane 0
	outPacked, err := normalizePrismPQ2Rows(outName, []int{width, width}, outRaw, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(outPacked, outRaw) {
		t.Fatal("grouped ssm_out columns were permuted a second time")
	}
	weight := make([]float32, width)
	dequantQ2_0Scalar(weight, outPacked[:blockPQ2_0Bytes])
	groupedCore := make([]float32, width)
	tiledCore := make([]float32, width)
	for h := 0; h < nV; h++ {
		groupedCore[h*headDim] = wantGrouped[h]
		tiledCore[h*headDim] = float32(uint32(1) << uint(h))
	}
	dot := func(x []float32) float32 {
		var y float32
		for i, v := range weight {
			y += v * x[i]
		}
		return y
	}
	if got := dot(groupedCore); got != 4 {
		t.Fatalf("grouped Fak GDN core -> ssm_out = %g, want 4", got)
	}
	if got := dot(tiledCore); got != 2 {
		t.Fatalf("tiled source GDN core unexpectedly agrees: got %g, want 2", got)
	}
}
