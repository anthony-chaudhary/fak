package compute

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

func TestNewQ2KMetadata(t *testing.T) {
	be := Default()
	const out, in = 4, 512
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	tensor := NewQ2K(be, []int{out, in}, raw)

	if tensor.Dtype != Q2_K {
		t.Fatalf("expected dtype Q2_K (%s), got %s", Q2_K, tensor.Dtype)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != out || tensor.Shape[1] != in {
		t.Fatalf("expected shape [%d, %d], got %v", out, in, tensor.Shape)
	}
	if tensor.Quant == nil || tensor.Quant.Block != q2kSuper || tensor.Quant.Bits != 2 {
		t.Fatalf("unexpected quant spec: %+v", tensor.Quant)
	}
}

func TestCPURefQ2KMatMulMatchesIndependentDequant(t *testing.T) {
	const out, in = 5, 512
	rng := rand.New(rand.NewSource(4843))
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	rng.Read(raw)

	// Keep both scales finite and modest while preserving random codes/scales.
	for b := 0; b < len(raw); b += q2kSuperBlock {
		dm := q2kSuper/16 + q2kSuper/4
		binary.LittleEndian.PutUint16(raw[b+dm:], 0x3000)
		binary.LittleEndian.PutUint16(raw[b+dm+2:], 0x2c00)
	}

	wf := make([]float32, out*in)
	block := make([]float32, q2kSuper)
	for o := 0; o < out; o++ {
		for b := 0; b < in/q2kSuper; b++ {
			off := (o*(in/q2kSuper) + b) * q2kSuperBlock
			q2kDequantSuperBlock(block, raw[off:off+q2kSuperBlock])
			copy(wf[o*in+b*q2kSuper:], block)
		}
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	be := Default()
	hx := be.Upload(NewF32(be, []int{in}, x), F32)
	wQ2 := NewQ2K(be, []int{out, in}, raw)
	wF32 := NewF32(be, []int{out, in}, wf)

	got := be.Read(be.MatMul(wQ2, hx))
	want := be.Read(be.MatMul(wF32, hx))
	if c := cosineC(got, want); c < 0.999999 {
		t.Fatalf("MatMul cosine %.9f < 0.999999", c)
	}

	// Also verify BatchedMatMul (prefill GEMM)
	const P = 3
	X := make([]float32, P*in)
	for i := range X {
		X[i] = rng.Float32()*2 - 1
	}
	hX := be.Upload(NewF32(be, []int{P, in}, X), F32)
	gotBatch := be.Read(be.BatchedMatMul(wQ2, hX, P))
	wantBatch := be.Read(be.BatchedMatMul(wF32, hX, P))
	if c := cosineC(gotBatch, wantBatch); c < 0.999999 {
		t.Fatalf("BatchedMatMul cosine %.9f < 0.999999", c)
	}
}

func TestNewQ2KRejectsInvalidStorage(t *testing.T) {
	be := Default()
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		shape []int
		raw   []byte
	}{
		{
			name:  "zero_rows",
			shape: []int{0, q2kSuper},
			raw:   nil,
		},
		{
			name:  "zero_columns",
			shape: []int{1, 0},
			raw:   nil,
		},
		{
			name:  "negative_dimensions",
			shape: []int{-1, q2kSuper},
			raw:   nil,
		},
		{
			name:  "overflow",
			shape: []int{maxInt, q2kSuper},
			raw:   nil,
		},
		{
			name:  "partial_block",
			shape: []int{1, q2kSuper - 1},
			raw:   make([]byte, q2kSuperBlock),
		},
		{
			name:  "short_storage",
			shape: []int{1, q2kSuper},
			raw:   make([]byte, q2kSuperBlock - 1),
		},
		{
			name:  "long_storage",
			shape: []int{1, q2kSuper},
			raw:   make([]byte, q2kSuperBlock + 1),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s: expected panic, got nil", tc.name)
				}
			}()
			NewQ2K(be, tc.shape, tc.raw)
		})
	}
}

func TestQ2KDequantSuperBlockBounds(t *testing.T) {
	dst := make([]float32, q2kSuper)
	for _, blkLen := range []int{0, 15, 79, 81, 83} {
		t.Run(fmt.Sprintf("short_blk_%d", blkLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for blk len %d, got nil", blkLen)
				}
				if r != "compute: short Q2_K super-block" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q2kDequantSuperBlock(dst, make([]byte, blkLen))
		})
	}

	blk := make([]byte, q2kSuperBlock)
	for _, dstLen := range []int{0, 255} {
		t.Run(fmt.Sprintf("short_dst_%d", dstLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for dst len %d, got nil", dstLen)
				}
				if r != "compute: short destination for Q2_K dequant" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q2kDequantSuperBlock(make([]float32, dstLen), blk)
		})
	}
}

func TestQ2KRowDotBounds(t *testing.T) {
	x := make([]float32, q2kSuper)
	scratch := make([]float32, q2kSuper)

	for _, rawLen := range []int{42, 83, 85, 100} {
		t.Run(fmt.Sprintf("misaligned_raw_%d", rawLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for raw len %d, got nil", rawLen)
				}
				if r != "compute: misaligned Q2_K raw weight row" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q2kRowDot(make([]byte, rawLen), x, scratch)
		})
	}

	raw := make([]byte, q2kSuperBlock)
	for _, xLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_x_%d", xLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for x len %d, got nil", xLen)
				}
				if r != "compute: short input vector for Q2_K row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q2kRowDot(raw, make([]float32, xLen), scratch)
		})
	}

	for _, scratchLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_scratch_%d", scratchLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for scratch len %d, got nil", scratchLen)
				}
				if r != "compute: short scratch buffer for Q2_K row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q2kRowDot(raw, x, make([]float32, scratchLen))
		})
	}
}
