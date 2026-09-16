package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// q3kReferenceBlock builds a deterministic, finite Q3_K super-block whose f16 super-scale
// d is modest (2^-6), so the decoded weights stay small and the parity comparison is not
// dominated by long-tail magnitudes.
func q3kReferenceBlock(seed int) []byte {
	blk := make([]byte, q3kSuperBlock)
	for i := 0; i < q3kSuperBlock-2; i++ {
		blk[i] = byte((i*29 + seed*7) & 0xff)
	}
	binary.LittleEndian.PutUint16(blk[q3kSuperBlock-2:], 0x2800) // d = 2^-6
	return blk
}

func TestNewQ3KMetadata(t *testing.T) {
	be := Default()
	const out, in = 4, 512
	raw := make([]byte, out*(in/q3kSuper)*q3kSuperBlock)
	tensor := NewQ3K(be, []int{out, in}, raw)

	if tensor.Dtype != Q3_K {
		t.Fatalf("expected dtype Q3_K (%s), got %s", Q3_K, tensor.Dtype)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != out || tensor.Shape[1] != in {
		t.Fatalf("expected shape [%d, %d], got %v", out, in, tensor.Shape)
	}
	if tensor.Quant == nil || tensor.Quant.Block != q3kSuper || tensor.Quant.Bits != 3 {
		t.Fatalf("unexpected quant spec: %+v", tensor.Quant)
	}
}

func TestNewQ3KRejectsInvalidStorage(t *testing.T) {
	be := Default()
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		shape []int
		raw   []byte
	}{
		{"zero_rows", []int{0, q3kSuper}, nil},
		{"zero_columns", []int{1, 0}, nil},
		{"negative_dimensions", []int{-1, q3kSuper}, nil},
		{"overflow", []int{maxInt, q3kSuper}, nil},
		{"partial_block", []int{1, q3kSuper - 1}, make([]byte, q3kSuperBlock)},
		{"short_storage", []int{1, q3kSuper}, make([]byte, q3kSuperBlock-1)},
		{"long_storage", []int{1, q3kSuper}, make([]byte, q3kSuperBlock+1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s: expected panic, got nil", tc.name)
				}
			}()
			NewQ3K(be, tc.shape, tc.raw)
		})
	}
}

// TestQ3KDequantSuperBlockNonzeroDistance is the in-package sanity check: the decoder is
// deterministic and not a constant map (two different blocks must decode differently, and
// the output must contain nonzero weights), so a later byte-parity test cannot pass on a
// degenerate all-zero decoder.
func TestQ3KDequantSuperBlockNonzeroDistance(t *testing.T) {
	a := make([]float32, q3kSuper)
	b := make([]float32, q3kSuper)
	q3kDequantSuperBlock(a, q3kReferenceBlock(1))
	q3kDequantSuperBlock(b, q3kReferenceBlock(2))

	if a[0] == 0 && a[100] == 0 && a[255] == 0 {
		t.Fatal("decoded block is all-zero at sampled offsets; fixture is degenerate")
	}
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("two distinct Q3_K blocks decoded identically; decoder ignores its input")
	}
}

func TestQ3KDequantSuperBlockBounds(t *testing.T) {
	dst := make([]float32, q3kSuper)
	for _, blkLen := range []int{0, 15, 108, 109} {
		t.Run(fmt.Sprintf("short_blk_%d", blkLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for blk len %d, got nil", blkLen)
				}
				if r != "compute: short Q3_K super-block" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q3kDequantSuperBlock(dst, make([]byte, blkLen))
		})
	}
	blk := make([]byte, q3kSuperBlock)
	for _, dstLen := range []int{0, 255} {
		t.Run(fmt.Sprintf("short_dst_%d", dstLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for dst len %d, got nil", dstLen)
				}
				if r != "compute: short destination for Q3_K dequant" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q3kDequantSuperBlock(make([]float32, dstLen), blk)
		})
	}
}

func TestQ3KRowDotBounds(t *testing.T) {
	x := make([]float32, q3kSuper)
	scratch := make([]float32, q3kSuper)

	for _, rawLen := range []int{42, 109, 111, 221} {
		t.Run(fmt.Sprintf("misaligned_raw_%d", rawLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for raw len %d, got nil", rawLen)
				}
				if r != "compute: misaligned Q3_K raw weight row" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q3kRowDot(make([]byte, rawLen), x, scratch)
		})
	}
	raw := make([]byte, q3kSuperBlock)
	for _, xLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_x_%d", xLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for x len %d, got nil", xLen)
				}
				if r != "compute: short input vector for Q3_K row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q3kRowDot(raw, make([]float32, xLen), scratch)
		})
	}
	for _, scratchLen := range []int{0, 100, 255} {
		t.Run(fmt.Sprintf("short_scratch_%d", scratchLen), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for scratch len %d, got nil", scratchLen)
				}
				if r != "compute: short scratch buffer for Q3_K row dot" {
					t.Fatalf("unexpected panic message: %v", r)
				}
			}()
			q3kRowDot(raw, x, make([]float32, scratchLen))
		})
	}
}

// TestCPURefQ3KMatMulMatchesIndependentDequant proves the cpu-ref Q3_K MatMul and
// BatchedMatMul agree with a straight dequant-then-f32-matmul over the same bytes.
func TestCPURefQ3KMatMulMatchesIndependentDequant(t *testing.T) {
	const out, in = 5, 512
	rng := rand.New(rand.NewSource(31493))
	raw := make([]byte, out*(in/q3kSuper)*q3kSuperBlock)
	for b := 0; b < len(raw); b += q3kSuperBlock {
		copy(raw[b:b+q3kSuperBlock], q3kReferenceBlock(rng.Intn(1<<20)))
	}

	wf := make([]float32, out*in)
	block := make([]float32, q3kSuper)
	for o := 0; o < out; o++ {
		for b := 0; b < in/q3kSuper; b++ {
			off := (o*(in/q3kSuper) + b) * q3kSuperBlock
			q3kDequantSuperBlock(block, raw[off:off+q3kSuperBlock])
			copy(wf[o*in+b*q3kSuper:], block)
		}
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	be := Default()
	hx := be.Upload(NewF32(be, []int{in}, x), F32)
	wQ3 := NewQ3K(be, []int{out, in}, raw)
	wF32 := NewF32(be, []int{out, in}, wf)

	got := be.Read(be.MatMul(wQ3, hx))
	want := be.Read(be.MatMul(wF32, hx))
	if c := cosineC(got, want); c < 0.999999 {
		t.Fatalf("MatMul cosine %.9f < 0.999999", c)
	}

	const P = 3
	X := make([]float32, P*in)
	for i := range X {
		X[i] = rng.Float32()*2 - 1
	}
	hX := be.Upload(NewF32(be, []int{P, in}, X), F32)
	gotBatch := be.Read(be.BatchedMatMul(wQ3, hX, P))
	wantBatch := be.Read(be.BatchedMatMul(wF32, hX, P))
	if c := cosineC(gotBatch, wantBatch); c < 0.999999 {
		t.Fatalf("BatchedMatMul cosine %.9f < 0.999999", c)
	}
}

// TestQ3KDeviceBackendDeclinesCleanly proves Q3_K has no device kernel: the cpuBackend
// advertises it (host reference), while a backend that does NOT carry a Q3_K device arm
// reports false through SupportsDeviceWeightDtype, so a probing caller declines before
// hitting a panic default.
func TestQ3KDeviceBackendDeclinesCleanly(t *testing.T) {
	cpu := Default()
	if !BackendSupportsDeviceWeightDtype(cpu, Q3_K) {
		t.Fatal("cpuBackend must advertise its Q3_K host reference MatMul")
	}
	// A backend that does not implement the probe interface must be reported false.
	if BackendSupportsDeviceWeightDtype(q3kNoProbeBackend{Backend: cpu}, Q3_K) {
		t.Fatal("a backend without a Q3_K device arm must decline via the probe")
	}
}

// q3kNoProbeBackend embeds a Backend but deliberately does not implement
// kquantDeviceKernel, standing in for a device backend with no Q3_K kernel.
type q3kNoProbeBackend struct {
	Backend
}

// q3kUpstreamDigest is the SHA-256 of the 1024 little-endian f32 values emitted by the
// actual llama.cpp dequantize_row_q3_K C function at
// ggml-org/llama.cpp@6fe74980162af0ed5e559870d5deccafaa034e7c (ggml/src/ggml-quants.c,
// MIT), compiled with cc -O0 -ffp-contract=off. The fixture is the SAME one the model
// package pins in TestQ3KIQ3SNativeReference: input bytes (i*37+11)%256 over four
// 110-byte blocks, each fp16 scale replaced by 0x3c00. It is an EXTERNAL oracle -- the
// compute decoder below never produced it -- so agreement is byte-identity with the C
// reference, not self-consistency.
const q3kUpstreamDigest = "0a00e55eace95e14b27bd3699aff2378beeee5338838f2ccd1a3e8b578f210b6"

// TestQ3KReferenceVectorParity is the leaf's acceptance witness (fak#13149): the compute
// Q3_K reference dequant reproduces the committed upstream llama.cpp decoder output
// byte-for-byte on a nonzero multi-block reference vector, so the duplicated arithmetic
// is proven identical to model.DequantQ3K's oracle rather than merely deterministic.
func TestQ3KReferenceVectorParity(t *testing.T) {
	const rows, cols = 2, 512
	raw := make([]byte, 4*q3kSuperBlock)
	for i := range raw {
		raw[i] = byte(i*37 + 11)
	}
	for b := 0; b < 4; b++ {
		binary.LittleEndian.PutUint16(raw[b*q3kSuperBlock+q3kSuperBlock-2:], 0x3c00)
	}

	decoded := make([]float32, rows*cols)
	for b := 0; b < 4; b++ {
		q3kDequantSuperBlock(decoded[b*q3kSuper:], raw[b*q3kSuperBlock:(b+1)*q3kSuperBlock])
	}

	// Non-degeneracy guard: a nonzero, non-constant decode, so the digest below cannot be
	// satisfied by an all-zero or input-ignoring decoder.
	var nonzero int
	first := decoded[0]
	for _, v := range decoded {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("Q3_K reference decode is all-zero; fixture or decoder is degenerate")
	}
	constant := true
	for _, v := range decoded[1:] {
		if v != first {
			constant = false
			break
		}
	}
	if constant {
		t.Fatal("Q3_K reference decode is constant; decoder ignores its input")
	}

	bits := make([]byte, len(decoded)*4)
	for i, v := range decoded {
		binary.LittleEndian.PutUint32(bits[4*i:], math.Float32bits(v))
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(bits)); got != q3kUpstreamDigest {
		t.Fatalf("compute Q3_K reference digest=%s, want upstream llama.cpp %s", got, q3kUpstreamDigest)
	}
}
