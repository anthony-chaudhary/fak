package compute

import (
	"encoding/binary"
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
