package compute

import (
	"errors"
	"math"
	"testing"
)

func approxEqual(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func sliceApproxEqual(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

func TestBlockDiagonalWHT_Validation(t *testing.T) {
	// Block size <= 0
	for _, bs := range []int{0, -1, -8} {
		v := []float32{1, 2, 3, 4}
		err := BlockDiagonalWHT(v, bs)
		if err == nil {
			t.Fatalf("expected error for blockSize=%d, got nil", bs)
		}
		var target ErrBlockDiagonalWHT
		if !errors.As(err, &target) {
			t.Fatalf("expected ErrBlockDiagonalWHT, got %T", err)
		}
		if !errors.Is(err, ErrBlockDiagonalWHT{}) {
			t.Fatalf("expected errors.Is to match ErrBlockDiagonalWHT")
		}
	}

	// Non-power of two block sizes
	for _, bs := range []int{3, 5, 6, 7, 9, 10, 12, 15} {
		v := make([]float32, bs*2)
		orig := append([]float32(nil), v...)
		err := BlockDiagonalWHT(v, bs)
		if err == nil {
			t.Fatalf("expected error for non-power-of-two blockSize=%d, got nil", bs)
		}
		var target ErrBlockDiagonalWHT
		if !errors.As(err, &target) {
			t.Fatalf("expected ErrBlockDiagonalWHT, got %T", err)
		}
		if !sliceApproxEqual(v, orig, 0) {
			t.Fatalf("vector was mutated despite validation error")
		}
	}

	// Vector length not divisible by blockSize
	{
		v := []float32{1, 2, 3, 4, 5}
		orig := append([]float32(nil), v...)
		err := BlockDiagonalWHT(v, 4)
		if err == nil {
			t.Fatalf("expected error for len(v)%%blockSize != 0, got nil")
		}
		var target ErrBlockDiagonalWHT
		if !errors.As(err, &target) {
			t.Fatalf("expected ErrBlockDiagonalWHT, got %T", err)
		}
		if !sliceApproxEqual(v, orig, 0) {
			t.Fatalf("vector was mutated despite validation error")
		}
	}

	// Empty vector is a valid no-op
	{
		var v []float32
		if err := BlockDiagonalWHT(v, 4); err != nil {
			t.Fatalf("unexpected error for empty vector: %v", err)
		}
	}
}

func TestBlockDiagonalWHT_RoundTripInvolution(t *testing.T) {
	const tol = 1e-4
	cases := []struct {
		blockSize int
		data      []float32
	}{
		{
			blockSize: 1,
			data:      []float32{1.5, -2.0, 3.25, 4.0},
		},
		{
			blockSize: 2,
			data:      []float32{1.0, -2.0, 0.5, 3.5, -4.0, 1.25},
		},
		{
			blockSize: 4,
			data:      []float32{1, 2, 3, 4, -1, -2, -3, -4, 0.5, 1.5, 2.5, 3.5},
		},
		{
			blockSize: 8,
			data: []float32{
				1, -1, 2, -2, 3, -3, 4, -4,
				0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			},
		},
	}

	for _, tc := range cases {
		work := append([]float32(nil), tc.data...)
		if err := BlockDiagonalWHTRotate(work, tc.blockSize); err != nil {
			t.Fatalf("Rotate failed for blockSize %d: %v", tc.blockSize, err)
		}
		if err := BlockDiagonalWHTInverse(work, tc.blockSize); err != nil {
			t.Fatalf("Inverse failed for blockSize %d: %v", tc.blockSize, err)
		}
		if !sliceApproxEqual(work, tc.data, tol) {
			t.Fatalf("round trip failed for blockSize %d: got %v, want %v", tc.blockSize, work, tc.data)
		}
	}
}

func TestBlockDiagonalWHT_KnownValues(t *testing.T) {
	const tol = 1e-5
	invSqrt2 := float32(1.0 / math.Sqrt(2))

	// Block size 2
	{
		vec := []float32{1, 0, 1, 1}
		want := []float32{invSqrt2, invSqrt2, 2 * invSqrt2, 0}
		if err := BlockDiagonalWHT(vec, 2); err != nil {
			t.Fatalf("BlockDiagonalWHT failed: %v", err)
		}
		if !sliceApproxEqual(vec, want, tol) {
			t.Fatalf("blockSize=2: got %v want %v", vec, want)
		}
	}

	// Block size 4
	{
		vec := []float32{1, 0, 0, 0, 1, 2, 3, 4}
		want := []float32{0.5, 0.5, 0.5, 0.5, 5, -1, -2, 0}
		if err := BlockDiagonalWHT(vec, 4); err != nil {
			t.Fatalf("BlockDiagonalWHT failed: %v", err)
		}
		if !sliceApproxEqual(vec, want, tol) {
			t.Fatalf("blockSize=4: got %v want %v", vec, want)
		}
	}
}

func TestBlockDiagonalWHT_BlockIsolation(t *testing.T) {
	// Demonstrates that block 0 transform is strictly independent of block 1 transform.
	// In partial-rotary architectures, RoPE modifies the rotary prefix (block 0).
	// The suffix (block 1) must remain invariant under changes to block 0.
	const blockSize = 4
	base := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	modified := []float32{99, -42, 13, 7, 10, 20, 30, 40} // identical suffix

	transformedBase := append([]float32(nil), base...)
	transformedMod := append([]float32(nil), modified...)

	if err := BlockDiagonalWHT(transformedBase, blockSize); err != nil {
		t.Fatal(err)
	}
	if err := BlockDiagonalWHT(transformedMod, blockSize); err != nil {
		t.Fatal(err)
	}

	// Suffixes must be identical
	if !sliceApproxEqual(transformedBase[blockSize:], transformedMod[blockSize:], 1e-6) {
		t.Fatalf("Block isolation violated: suffix changed when prefix was modified.\nbase: %v\nmod:  %v",
			transformedBase[blockSize:], transformedMod[blockSize:])
	}
}

func TestBlockDiagonalWHT_Isometry(t *testing.T) {
	const tol = 1e-4
	vec := []float32{1.5, -2.2, 3.7, 4.1, -0.8, 1.9, -3.3, 0.4}

	var origNorm float64
	for _, x := range vec {
		origNorm += float64(x * x)
	}
	origNorm = math.Sqrt(origNorm)

	work := append([]float32(nil), vec...)
	if err := BlockDiagonalWHT(work, 4); err != nil {
		t.Fatal(err)
	}

	var transformedNorm float64
	for _, x := range work {
		transformedNorm += float64(x * x)
	}
	transformedNorm = math.Sqrt(transformedNorm)

	if math.Abs(origNorm-transformedNorm) > float64(tol) {
		t.Fatalf("Isometry violated: norm before=%f, after=%f", origNorm, transformedNorm)
	}
}

func TestBlockDiagonalWHTBatch(t *testing.T) {
	const headDim = 8
	const blockSize = 4

	// Validations
	if err := BlockDiagonalWHTBatch([]float32{1, 2, 3}, 0, blockSize); err == nil {
		t.Fatal("expected error for headDim <= 0")
	}
	if err := BlockDiagonalWHTBatch([]float32{1, 2, 3, 4, 5}, headDim, blockSize); err == nil {
		t.Fatal("expected error for len(data) % headDim != 0")
	}
	if err := BlockDiagonalWHTBatch([]float32{1, 2, 3, 4, 5, 6, 7, 8}, headDim, 3); err == nil {
		t.Fatal("expected error for non-power-of-two blockSize")
	}
	if err := BlockDiagonalWHTBatch([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 6, 4); err == nil {
		t.Fatal("expected error for headDim % blockSize != 0")
	}

	// Contiguous batch of 3 vectors of headDim 8
	batch := make([]float32, 24)
	for i := range batch {
		batch[i] = float32(i + 1)
	}

	expected := append([]float32(nil), batch...)
	for v := 0; v < 3; v++ {
		if err := BlockDiagonalWHT(expected[v*headDim:(v+1)*headDim], blockSize); err != nil {
			t.Fatal(err)
		}
	}

	actual := append([]float32(nil), batch...)
	if err := BlockDiagonalWHTBatch(actual, headDim, blockSize); err != nil {
		t.Fatalf("BlockDiagonalWHTBatch failed: %v", err)
	}

	if !sliceApproxEqual(actual, expected, 1e-5) {
		t.Fatalf("Batch mismatch: got %v want %v", actual, expected)
	}
}

func TestDeriveRotaryBlockSize(t *testing.T) {
	// Cases where rotaryDim <= 0 or rotaryDim >= headDim -> returns headDim (if power of 2)
	bs, err := DeriveRotaryBlockSize(128, 0)
	if err != nil || bs != 128 {
		t.Fatalf("expected 128, nil, got %d, %v", bs, err)
	}
	bs, err = DeriveRotaryBlockSize(128, -16)
	if err != nil || bs != 128 {
		t.Fatalf("expected 128, nil, got %d, %v", bs, err)
	}
	bs, err = DeriveRotaryBlockSize(128, 128)
	if err != nil || bs != 128 {
		t.Fatalf("expected 128, nil, got %d, %v", bs, err)
	}
	bs, err = DeriveRotaryBlockSize(128, 256)
	if err != nil || bs != 128 {
		t.Fatalf("expected 128, nil, got %d, %v", bs, err)
	}

	// headDim not a power of two when rotaryDim <= 0 or rotaryDim >= headDim
	if _, err := DeriveRotaryBlockSize(96, 0); err == nil {
		t.Fatal("expected error when headDim is not a power of two")
	}
	if _, err := DeriveRotaryBlockSize(96, 96); err == nil {
		t.Fatal("expected error when headDim is not a power of two")
	}

	// Partial rotary: 0 < rotaryDim < headDim
	bs, err = DeriveRotaryBlockSize(128, 64)
	if err != nil || bs != 64 {
		t.Fatalf("expected 64, nil, got %d, %v", bs, err)
	}
	bs, err = DeriveRotaryBlockSize(192, 64)
	if err != nil || bs != 64 {
		t.Fatalf("expected 64, nil, got %d, %v", bs, err)
	}
	bs, err = DeriveRotaryBlockSize(256, 32)
	if err != nil || bs != 32 {
		t.Fatalf("expected 32, nil, got %d, %v", bs, err)
	}

	// Partial rotary with non-power-of-two rotaryDim
	if _, err := DeriveRotaryBlockSize(128, 96); err == nil {
		t.Fatal("expected error for non-power-of-two rotaryDim")
	}

	// Partial rotary where rotaryDim does not divide headDim
	if _, err := DeriveRotaryBlockSize(100, 32); err == nil {
		t.Fatal("expected error when rotaryDim does not divide headDim")
	}
}
