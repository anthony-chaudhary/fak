package kvcache

import (
	"bytes"
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestKVLayoutDescCrossModelAccounting(t *testing.T) {
	cases := []struct {
		name      string
		desc      LayoutDesc
		wantRatio float64
	}{
		{
			name: "qwen3.8",
			desc: LayoutDesc{
				Arch:      "qwen",
				Precision: PrecisionNVFP4,
				Layers:    64,
				KVHeads:   8,
				HeadDim:   128,
				Layout:    LayoutContiguous,
			},
			wantRatio: 1024.0 / 144.0,
		},
		{
			name: "gemma",
			desc: LayoutDesc{
				Arch:       "gemma",
				Precision:  PrecisionNVFP4,
				Layers:     46,
				KVHeads:    4,
				HeadDim:    256,
				Layout:     LayoutPaged,
				PageTokens: 16,
			},
			wantRatio: 2048.0 / 288.0,
		},
		{
			name: "llama",
			desc: LayoutDesc{
				Arch:      "llama",
				Precision: PrecisionNVFP4,
				Layers:    32,
				KVHeads:   8,
				HeadDim:   128,
				Layout:    LayoutContiguous,
			},
			wantRatio: 1024.0 / 144.0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.desc.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
			gotRatio := tc.desc.CompressionRatioVsFP32()
			if math.Abs(gotRatio-tc.wantRatio) > 1e-6 {
				t.Fatalf("CompressionRatioVsFP32 = %.6f, want %.6f", gotRatio, tc.wantRatio)
			}
			if gotRatio < 7.0 {
				t.Fatalf("expected >7x compression vs FP32, got %.3fx", gotRatio)
			}
			if r16 := tc.desc.CompressionRatioVsFP16(); r16 < 3.5 {
				t.Fatalf("expected >3.5x compression vs FP16, got %.3fx", r16)
			}
		})
	}

	badHeadDim := LayoutDesc{
		Arch:      "qwen",
		Precision: PrecisionNVFP4,
		Layers:    8,
		KVHeads:   4,
		HeadDim:   24,
	}
	if err := badHeadDim.Validate(); !errors.Is(err, ErrInvalidLayout) {
		t.Fatalf("expected ErrInvalidLayout for head_dim=24, got %v", err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNVFP4Block16RoundTripAndOverflow(t *testing.T) {
	const (
		tokens  = 3
		kvHeads = 2
		headDim = 32
	)
	src := make([]float32, tokens*kvHeads*headDim)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.37)) * 12.0
	}
	// Zero out one block to verify sub-threshold zero encoding.
	for i := 0; i < NVFP4BlockSize; i++ {
		src[i] = 0
	}

	codes, scales, overflow, err := QuantizeBlock16NVFP4(src, kvHeads, headDim, tokens)
	if err != nil {
		t.Fatalf("QuantizeBlock16NVFP4: %v", err)
	}
	if overflow {
		t.Fatal("unexpected overflow on bounded inputs")
	}
	if scales[0] != 0 {
		t.Fatalf("zero block scale = 0x%02x, want 0x00", scales[0])
	}

	decoded, err := DequantizeBlock16NVFP4(codes, scales, kvHeads, headDim, tokens)
	if err != nil {
		t.Fatalf("DequantizeBlock16NVFP4: %v", err)
	}
	for i := 0; i < NVFP4BlockSize; i++ {
		if decoded[i] != 0 {
			t.Fatalf("zero block decoded[%d] = %v, want 0", i, decoded[i])
		}
	}

	var dot, normA, normB float64
	for i := range src {
		dot += float64(src[i]) * float64(decoded[i])
		normA += float64(src[i]) * float64(src[i])
		normB += float64(decoded[i]) * float64(decoded[i])
	}
	cos := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if cos < 0.985 {
		t.Fatalf("NVFP4 block-16 cosine = %.5f, want >= 0.985", cos)
	}

	// Overflow trigger: > 2688.0
	overSrc := make([]float32, kvHeads*headDim)
	overSrc[0] = 3000.0
	_, _, overFlag, err := QuantizeBlock16NVFP4(overSrc, kvHeads, headDim, 1)
	if err != nil {
		t.Fatalf("QuantizeBlock16NVFP4 overflow case: %v", err)
	}
	if !overFlag {
		t.Fatal("expected overflow=true for absmax 3000 > 2688")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestKVSnapshotRoundTripAndTamperDetection(t *testing.T) {
	desc := LayoutDesc{
		Arch:       "qwen",
		Precision:  PrecisionNVFP4,
		Layers:     2,
		KVHeads:    2,
		HeadDim:    32,
		Capacity:   64,
		Layout:     LayoutPaged,
		PageTokens: 16,
	}
	tokens := []int{101, 202, 303, 404}
	positions := []int{0, 1, 2, 3}

	layers := make([]SnapshotLayer, desc.Layers)
	for l := 0; l < desc.Layers; l++ {
		kSrc := make([]float32, len(tokens)*desc.KVHeads*desc.HeadDim)
		vSrc := make([]float32, len(tokens)*desc.KVHeads*desc.HeadDim)
		for i := range kSrc {
			kSrc[i] = float32((l+1)*10+i%17) * 0.25
			vSrc[i] = float32((l+1)*5-(i%13)) * 0.5
		}
		kc, ks, overK, err := QuantizeBlock16NVFP4(kSrc, desc.KVHeads, desc.HeadDim, len(tokens))
		if err != nil || overK {
			t.Fatalf("Quantize K layer %d: err=%v over=%v", l, err, overK)
		}
		vc, vs, overV, err := QuantizeBlock16NVFP4(vSrc, desc.KVHeads, desc.HeadDim, len(tokens))
		if err != nil || overV {
			t.Fatalf("Quantize V layer %d: err=%v over=%v", l, err, overV)
		}
		layers[l] = SnapshotLayer{
			Index:   l,
			KCodes:  kc,
			VCodes:  vc,
			KScales: ks,
			VScales: vs,
		}
	}

	snap := &Snapshot{
		Layout:    desc,
		Tokens:    tokens,
		Positions: positions,
		Layers:    layers,
	}
	wire, err := EncodeSnapshot(snap)
	if err != nil {
		t.Fatalf("EncodeSnapshot: %v", err)
	}

	got, err := DecodeSnapshot(wire, &desc)
	if err != nil {
		t.Fatalf("DecodeSnapshot: %v", err)
	}
	if len(got.Layers) != len(snap.Layers) || len(got.Tokens) != len(tokens) {
		t.Fatalf("decoded shape mismatch: layers=%d tokens=%d", len(got.Layers), len(got.Tokens))
	}
	for l := range got.Layers {
		if !bytes.Equal(got.Layers[l].KCodes, snap.Layers[l].KCodes) ||
			!bytes.Equal(got.Layers[l].KScales, snap.Layers[l].KScales) ||
			!bytes.Equal(got.Layers[l].VCodes, snap.Layers[l].VCodes) ||
			!bytes.Equal(got.Layers[l].VScales, snap.Layers[l].VScales) {
			t.Fatalf("layer %d NVFP4 payload mismatch after round-trip", l)
		}
	}

	// Incompatible arch must fail with ErrSnapshotIncompatible.
	wrongArch := desc
	wrongArch.Arch = "llama"
	if _, err := DecodeSnapshot(wire, &wrongArch); !errors.Is(err, ErrSnapshotIncompatible) {
		t.Fatalf("expected ErrSnapshotIncompatible, got %v", err)
	}

	// Bit flip in payload must fail SHA-256 verification with ErrSnapshotCorrupt.
	tampered := append([]byte(nil), wire...)
	tampered[len(tampered)-1] ^= 0xFF
	if _, err := DecodeSnapshot(tampered, &desc); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("expected ErrSnapshotCorrupt on tampered payload, got %v", err)
	}
}
