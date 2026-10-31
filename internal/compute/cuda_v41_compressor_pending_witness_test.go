//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool && fak_cuda_v41_compressor_pending

package compute

import (
	"errors"
	"math"
	"testing"
)

// These are actual device witnesses for the separate pending qualification
// profile, not substitutes for the unchanged legacy direct-Store suite.
// They are authored but UNRUN until native build and physical GPU admission.
func cudaCompressorPendingUpload(t *testing.T, c *cudaBackend, shape []int, values []float32) Tensor {
	t.Helper()
	v := mkResident(c, shape, values)
	t.Cleanup(func() { c.Free(v) })
	return v
}
func cudaCompressorPendingSelectors(t *testing.T, c *cudaBackend, width int, nonfinite bool) (Tensor, Tensor) {
	hidden := 2 * width
	a, b := make([]float32, width*hidden), make([]float32, width*hidden)
	for d := 0; d < width; d++ {
		a[d*hidden+d] = 1
		b[d*hidden+width+d] = 1
	}
	if nonfinite {
		b[3*hidden+width+3] = float32(math.Inf(-1))
	}
	return cudaCompressorPendingUpload(t, c, []int{width, hidden}, a), cudaCompressorPendingUpload(t, c, []int{width, hidden}, b)
}

// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41CompressorRingMatchesPool(t *testing.T) {
	c := cudaGDNBackend(t)
	if c.SupportsV41CompressorRing() {
		t.Fatal("unqualified pending profile advertised ordinary capability")
	}
	const ratio, width = 4, 8
	kv, sc, err := c.V41CompressorRingAlloc(ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free(kv)
	defer c.Free(sc)
	a, b := cudaCompressorPendingSelectors(t, c, width, false)
	expectedKV, expectedSC := make([]float32, ratio*width), make([]float32, ratio*width)
	before := cudaV41CompressorCompleted()
	var out Tensor
	var deferred []*cudaBuf
	for i := 0; i < 12; i++ {
		slot := i % ratio
		input := make([]float32, 2*width)
		for d := 0; d < width; d++ {
			input[d] = float32(i + d + 1)
			input[width+d] = float32(d%3) / 4
			expectedKV[slot*width+d] = input[d]
			expectedSC[slot*width+d] = input[width+d]
		}
		x := mkResident(c, []int{2 * width}, input)
		out, err = c.V41CompressorRingProjectStore(kv, sc, a, b, x, slot, ratio, width, i == 11)
		if err != nil {
			t.Fatal(err)
		}
		c.Free(x)
		deferred = append(deferred, x.buf.(*cudaBuf))
		if i < 11 && out.Buf() != nil {
			t.Fatal("non-emitting store published output")
		}
		if x.buf.(*cudaBuf).ptr == nil {
			t.Fatal("producer Free retired before Read")
		}
	}
	if cudaV41CompressorCompleted() != before {
		t.Fatal("enqueue mislabeled as completion")
	}
	got, err := DecodeV41CompressorRing(c.Read(out), width)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free(out)
	if cudaV41CompressorCompleted() != before+1 {
		t.Fatal("checked completion counter")
	}
	for _, owner := range deferred {
		if owner.ptr != nil {
			t.Fatal("completed queued Free not honored")
		}
	}
	pooled, err := c.V41CompressorPool(cudaCompressorPendingUpload(t, c, []int{ratio, width}, expectedKV), cudaCompressorPendingUpload(t, c, []int{ratio, width}, expectedSC), ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free(pooled)
	want := c.Read(pooled)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("ring/pool bit mismatch dim %d", i)
		}
	}
}

// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41CompressorRingRefusesNonFinite(t *testing.T) {
	c := cudaGDNBackend(t)
	const ratio, width = 2, 8
	kv, sc, err := c.V41CompressorRingAlloc(ratio, width)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free(kv)
	defer c.Free(sc)
	a, b := cudaCompressorPendingSelectors(t, c, width, true)
	var out Tensor
	for slot := 0; slot < ratio; slot++ {
		input := make([]float32, 2*width)
		for i := range input {
			input[i] = 1
		}
		x := mkResident(c, []int{2 * width}, input)
		out, err = c.V41CompressorRingProjectStore(kv, sc, a, b, x, slot, ratio, width, slot == ratio-1)
		c.Free(x)
		if err != nil {
			t.Fatal(err)
		}
	}
	defer c.Free(out)
	_, err = DecodeV41CompressorRing(c.Read(out), width)
	var typed *V41CompressorRingNonFiniteError
	if !errors.As(err, &typed) || typed.Dim != 3 || typed.Status != V41CompressorRingScoreNonFinite {
		t.Fatalf("nonfinite status: %v", err)
	}
}
