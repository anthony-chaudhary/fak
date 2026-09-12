package compute

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

// iq2_xxs_test.go — independent witness for the IQ2_XXS GPU compute dtype and its
// cpu-ref backend path (quant_iq2_xxs.go). It mirrors quant_q2k_test.go: metadata,
// a MatMul/BatchedMatMul parity check against an independently dequantized f32
// weight matrix, invalid-storage rejection, and bounds panics.

func TestIQ2XXS(t *testing.T) {
	be := Default()

	// (a) Metadata on a valid [out,in] payload.
	const out, in = 4, 512
	metaRaw := make([]byte, out*(in/iq2xxsSuper)*iq2xxsSuperBlock)
	meta := NewIQ2XXS(be, []int{out, in}, metaRaw)
	if meta.Dtype != IQ2_XXS {
		t.Fatalf("expected dtype IQ2_XXS (%s), got %s", IQ2_XXS, meta.Dtype)
	}
	if len(meta.Shape) != 2 || meta.Shape[0] != out || meta.Shape[1] != in {
		t.Fatalf("expected shape [%d, %d], got %v", out, in, meta.Shape)
	}
	if meta.Quant == nil || meta.Quant.Block != iq2xxsSuper || meta.Quant.Bits != 2 {
		t.Fatalf("unexpected quant spec: %+v", meta.Quant)
	}

	// (b) GPU/backend parity against an independent f32 dequant reference.
	const outB, inB = 5, 512
	rng := rand.New(rand.NewSource(12907))
	raw := make([]byte, outB*(inB/iq2xxsSuper)*iq2xxsSuperBlock)
	rng.Read(raw)
	// Keep the f16 d scale finite and modest (0x3000 ~= small positive) so no
	// NaN/Inf scale poisons the dot, mirroring the Q2_K test fixture.
	for b := 0; b < len(raw); b += iq2xxsSuperBlock {
		binary.LittleEndian.PutUint16(raw[b:], 0x3000)
	}

	wf := make([]float32, outB*inB)
	block := make([]float32, iq2xxsSuper)
	for o := 0; o < outB; o++ {
		for b := 0; b < inB/iq2xxsSuper; b++ {
			off := (o*(inB/iq2xxsSuper) + b) * iq2xxsSuperBlock
			iq2xxsDequantSuperBlock(block, raw[off:off+iq2xxsSuperBlock])
			copy(wf[o*inB+b*iq2xxsSuper:], block)
		}
	}

	x := make([]float32, inB)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	hx := be.Upload(NewF32(be, []int{inB}, x), F32)
	wIQ := NewIQ2XXS(be, []int{outB, inB}, raw)
	wF32 := NewF32(be, []int{outB, inB}, wf)

	got := be.Read(be.MatMul(wIQ, hx))
	want := be.Read(be.MatMul(wF32, hx))
	if c := cosineC(got, want); c < 0.999999 {
		t.Fatalf("MatMul cosine %.9f < 0.999999", c)
	}

	const P = 3
	X := make([]float32, P*inB)
	for i := range X {
		X[i] = rng.Float32()*2 - 1
	}
	hX := be.Upload(NewF32(be, []int{P, inB}, X), F32)
	gotBatch := be.Read(be.BatchedMatMul(wIQ, hX, P))
	wantBatch := be.Read(be.BatchedMatMul(wF32, hX, P))
	if c := cosineC(gotBatch, wantBatch); c < 0.999999 {
		t.Fatalf("BatchedMatMul cosine %.9f < 0.999999", c)
	}

	// (c) Quantized + String tags.
	if !IQ2_XXS.Quantized() {
		t.Fatalf("expected IQ2_XXS.Quantized() true")
	}
	if s := IQ2_XXS.String(); s != "iq2_xxs" {
		t.Fatalf("expected IQ2_XXS.String()==\"iq2_xxs\", got %q", s)
	}
	if b := IQ2_XXS.Bytes(); b != 1 {
		t.Fatalf("expected IQ2_XXS.Bytes()==1, got %d", b)
	}
}

func TestNewIQ2XXSRejectsInvalidStorage(t *testing.T) {
	be := Default()
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		shape []int
		raw   []byte
	}{
		{
			name:  "zero_rows",
			shape: []int{0, iq2xxsSuper},
			raw:   nil,
		},
		{
			name:  "zero_columns",
			shape: []int{1, 0},
			raw:   nil,
		},
		{
			name:  "negative_dimensions",
			shape: []int{-1, iq2xxsSuper},
			raw:   nil,
		},
		{
			name:  "overflow",
			shape: []int{maxInt, iq2xxsSuper},
			raw:   nil,
		},
		{
			name:  "partial_block",
			shape: []int{1, iq2xxsSuper - 1},
			raw:   make([]byte, iq2xxsSuperBlock),
		},
		{
			name:  "short_storage",
			shape: []int{1, iq2xxsSuper},
			raw:   make([]byte, iq2xxsSuperBlock-1),
		},
		{
			name:  "long_storage",
			shape: []int{1, iq2xxsSuper},
			raw:   make([]byte, iq2xxsSuperBlock+1),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s: expected panic, got nil", tc.name)
				}
			}()
			NewIQ2XXS(be, tc.shape, tc.raw)
		})
	}
}

func TestIQ2XXSDequantSuperBlockBounds(t *testing.T) {
	dst := make([]float32, iq2xxsSuper)
	for _, blkLen := range []int{0, 1, 65} {
		t.Run(fmt.Sprintf("short_blk_%d", blkLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for blk len %d, got nil", blkLen)
				}
				if r != "compute: short IQ2_XXS super-block" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			iq2xxsDequantSuperBlock(dst, make([]byte, blkLen))
		})
	}

	blk := make([]byte, iq2xxsSuperBlock)
	for _, dstLen := range []int{0, 255} {
		t.Run(fmt.Sprintf("short_dst_%d", dstLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for dst len %d, got nil", dstLen)
				}
				if r != "compute: short destination for IQ2_XXS dequant" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			iq2xxsDequantSuperBlock(make([]float32, dstLen), blk)
		})
	}
}

func TestIQ2XXSRowDotBounds(t *testing.T) {
	x := make([]float32, iq2xxsSuper)
	scratch := make([]float32, iq2xxsSuper)

	for _, rawLen := range []int{1, 65, 67, 100} {
		t.Run(fmt.Sprintf("misaligned_raw_%d", rawLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for raw len %d, got nil", rawLen)
				}
				if r != "compute: misaligned IQ2_XXS raw weight row" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			iq2xxsRowDot(make([]byte, rawLen), x, scratch)
		})
	}

	raw := make([]byte, iq2xxsSuperBlock)
	for _, xLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_x_%d", xLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for x len %d, got nil", xLen)
				}
				if r != "compute: short input vector for IQ2_XXS row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			iq2xxsRowDot(raw, make([]float32, xLen), scratch)
		})
	}

	for _, scratchLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_scratch_%d", scratchLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for scratch len %d, got nil", scratchLen)
				}
				if r != "compute: short scratch buffer for IQ2_XXS row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			iq2xxsRowDot(raw, x, make([]float32, scratchLen))
		})
	}
}

// TestIQ2XXSDequantGoldenVector pins iq2xxsDequantSuperBlock against the model
// package's canonical DequantIQ2XXS output for a fixed deterministic block. This
// is the NON-circular dequant witness the MatMul parity test cannot provide: the
// expected values below were generated once by internal/model.DequantIQ2XXS
// (llama.cpp dequantize_row_iq2_xxs, independently pinned by the model package's
// iq2xxs_grid golden hash 05826b5d...) on the same 66 raw bytes, so a wrong grid
// index, sign decode, or scale formula in the compute copy fails here even though
// it would cancel out of a self-referential CPU-vs-CPU comparison.
func TestIQ2XXSDequantGoldenVector(t *testing.T) {
	raw := make([]byte, iq2xxsSuperBlock)
	for i := 0; i < iq2xxsSuperBlock; i++ {
		raw[i] = byte((i*37 + 11) & 0xff)
	}
	raw[0], raw[1] = 0x00, 0x30 // f16 d = 0x3000
	got := make([]float32, iq2xxsSuper)
	iq2xxsDequantSuperBlock(got, raw)
	want := []float32{-7.390625, 7.390625, 4.296875, -4.296875, 4.296875, -1.375, -4.296875, 1.375, -1.375, 1.375, -1.375, -4.296875, -1.375, 4.296875, 7.390625, 1.375, 4.296875, 7.390625, -4.296875, -7.390625, 1.375, 4.296875, -1.375, -4.296875, -1.375, 4.296875, 1.375, 1.375, 1.375, 1.375, -7.390625, 4.296875, -6.640625, 2.125, 2.125, 6.640625, -6.640625, 6.640625, 11.421875, 2.125, 2.125, 2.125, -2.125, -2.125, 11.421875, -6.640625, -2.125, 6.640625, 2.125, 11.421875, -11.421875, -6.640625, 2.125, -2.125, -11.421875, 6.640625, 2.125, -6.640625, 11.421875, 2.125, 2.125, 2.125, 6.640625, -11.421875, -2.625, 2.625, 8.203125, -2.625, -2.625, -14.109375, 2.625, 8.203125, 2.625, 14.109375, -8.203125, -2.625, -14.109375, -2.625, 14.109375, 8.203125, -14.109375, 8.203125, -8.203125, -2.625, 8.203125, 2.625, 8.203125, -14.109375, 14.109375, 2.625, -14.109375, 14.109375, 2.625, 2.625, -2.625, 2.625, -10.546875, 10.546875, 3.375, 3.375, 3.375, -10.546875, -18.140625, -10.546875, 3.375, 3.375, -3.375, -3.375, 3.375, 10.546875, 10.546875, 18.140625, 3.375, -3.375, -10.546875, -3.375, 10.546875, -3.375, 3.375, 3.375, -3.375, 18.140625, -3.375, 3.375, 3.375, 18.140625, 3.375, 3.375, -12.109375, 3.875, 12.109375, -12.109375, 12.109375, 12.109375, 12.109375, 20.828125, -3.875, 12.109375, -3.875, -20.828125, -12.109375, 3.875, -3.875, -3.875, 12.109375, -3.875, -3.875, -12.109375, 3.875, 20.828125, -3.875, 3.875, 3.875, -12.109375, -3.875, 3.875, 3.875, 20.828125, -12.109375, -3.875, -3.359375, 0.625, 0.625, 0.625, -3.359375, -0.625, 0.625, -0.625, -3.359375, 0.625, -1.953125, -1.953125, 0.625, -3.359375, 0.625, 0.625, -1.953125, -0.625, -0.625, -3.359375, 0.625, -3.359375, -1.953125, 0.625, -3.359375, -1.953125, -1.953125, 1.953125, 0.625, 0.625, 0.625, -1.953125, -1.125, 1.125, 1.125, -3.515625, -3.515625, 6.046875, -1.125, 1.125, -1.125, 1.125, -6.046875, -6.046875, -3.515625, -6.046875, -3.515625, 1.125, -1.125, -3.515625, -1.125, -6.046875, 1.125, 1.125, 1.125, 3.515625, -1.125, 6.046875, 1.125, 6.046875, 1.125, 1.125, -3.515625, 3.515625, -10.078125, 1.875, 1.875, 1.875, 1.875, 1.875, 10.078125, -1.875, 1.875, 1.875, -10.078125, -1.875, 5.859375, 1.875, -1.875, -5.859375, 1.875, 5.859375, -10.078125, -10.078125, 5.859375, -1.875, 5.859375, -5.859375, 1.875, -5.859375, 1.875, 1.875, 1.875, 5.859375, 1.875, -10.078125}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("golden mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
