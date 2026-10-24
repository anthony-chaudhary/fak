//go:build cuda && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

const cudaNVFP4NativeRequireEnv = "FAK_REQUIRE_CUDA_NVFP4_NATIVE"

func cudaNVFP4NativeBlackwellOrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	be := Pick("cuda")
	cb, ok := be.(*cudaBackend)
	if !ok {
		message := "native CUDA NVFP4 backend unavailable"
		if os.Getenv(cudaNVFP4NativeRequireEnv) == "1" {
			t.Fatal(message)
		}
		t.Skip(message)
	}
	switch cb.tier {
	case "sm_100", "sm_103", "sm_120":
		return cb
	default:
		message := "native CUDA NVFP4 requires a qualified Blackwell GPU; found " + cb.tier
		if os.Getenv(cudaNVFP4NativeRequireEnv) == "1" {
			t.Fatal(message)
		}
		t.Skip(message)
	}
	return nil
}

func cudaNVFP4NativeFixture(t *testing.T, be Backend, rows, k int, global float32) Tensor {
	t.Helper()
	packed := make([]byte, rows*k/2)
	scales := make([]byte, rows*(k/NVFP4BlockSize))
	scaleValues := [...]byte{0x30, 0x38, 0x40, 0x48}
	blocks := k / NVFP4BlockSize
	for row := 0; row < rows; row++ {
		for block := 0; block < blocks; block++ {
			inner := block * NVFP4BlockSize
			packed[(row*k+inner)/2] = byte(1 + (row*3+block*5)%7)
			scales[row*blocks+block] = scaleValues[(row+block*3)%len(scaleValues)]
		}
	}
	tensor, err := NewNVFP4(be, []int{rows, k}, packed, scales, global)
	if err != nil {
		t.Fatal(err)
	}
	return tensor
}

// fak-test:runtime integration est=100ms lane=optin
func TestCUDANVFP4NativeSM89RefusesWithoutLtEnqueue(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	if cb.tier != "sm_89" {
		t.Skip("L4-specific unsupported-path witness")
	}
	wHost := cudaNVFP4NativeFixture(t, cb, 4, 32, 0.25)
	aHost := cudaNVFP4NativeFixture(t, cb, 1, 32, 0.5)
	w, a := cb.Upload(wHost, NVFP4), cb.Upload(aHost, NVFP4)
	defer cb.Free(w)
	defer cb.Free(a)

	before := cb.cudaNVFP4LtCallsForTest()
	out, err := MatMulNVFP4(cb, w, a)
	if !errors.Is(err, ErrNVFP4GEMMUnsupported) {
		t.Fatalf("MatMulNVFP4 error = %v, want ErrNVFP4GEMMUnsupported", err)
	}
	if out.buf != nil {
		t.Fatalf("unsupported call returned storage %T", out.buf)
	}
	if after := cb.cudaNVFP4LtCallsForTest(); after != before {
		t.Fatalf("cuBLASLt enqueue count changed on SM89: %d -> %d", before, after)
	}
}

// fak-test:runtime integration est=2s lane=optin
func TestCUDANVFP4NativeScaleTileCrossings(t *testing.T) {
	cb := cudaNVFP4NativeBlackwellOrSkip(t)
	const out, rows, k = 132, 129, 128
	ref := &cpuBackend{}
	wCPU := cudaNVFP4NativeFixture(t, ref, out, k, 0.125)
	aCPU := cudaNVFP4NativeFixture(t, ref, rows, k, 0.375)
	wHost := cudaNVFP4NativeFixture(t, cb, out, k, 0.125)
	aHost := cudaNVFP4NativeFixture(t, cb, rows, k, 0.375)
	w, a := cb.Upload(wHost, NVFP4), cb.Upload(aHost, NVFP4)
	defer cb.Free(w)
	defer cb.Free(a)
	defer cb.Recycle()

	wantTensor, err := MatMulNVFP4(ref, wCPU, aCPU)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.Read(wantTensor)
	before := cb.cudaNVFP4LtCallsForTest()
	gotTensor, err := MatMulNVFP4(cb, w, a)
	if errors.Is(err, ErrNVFP4GEMMUnsupported) && os.Getenv(cudaNVFP4NativeRequireEnv) != "1" {
		t.Skipf("native cuBLASLt NVFP4 unavailable: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	got := cb.Read(gotTensor)
	if after := cb.cudaNVFP4LtCallsForTest(); after != before+1 {
		t.Fatalf("successful cuBLASLt enqueue count = %d -> %d, want exactly one", before, after)
	}
	if len(got) != len(want) {
		t.Fatalf("output length = %d, want %d", len(got), len(want))
	}
	// Bit equality is intentional for this fixture, not a general cuBLASLt
	// reduction-order promise: each K=16 block has one nonzero product, and all
	// E2M1 values, E4M3 scales, and unequal globals are small dyadic numbers, so
	// every partial and the eight-term sum are exactly representable in FP32.
	for i := range want {
		if !nvfp4Finite(got[i]) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("output[%d] = %g (%08x), want CPU oracle %g (%08x)",
				i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}
