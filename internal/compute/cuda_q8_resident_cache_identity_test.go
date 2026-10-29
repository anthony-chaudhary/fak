package compute

import (
	"os"
	"testing"
)

// fak-test:runtime slow est=60s lane=optin
// Like the geometry witness, this executes the actual upload method with host
// effect spies, not a CUDA device. The bound is the subprocess deadline, not a
// measured latency. All fixtures have four-aligned blocks; geometry admission
// and cache identity are distinct contracts.
func TestCUDAQ8ResidentCacheIdentityHost(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_Q8_RESIDENT_CACHE_IDENTITY_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_Q8_RESIDENT_CACHE_IDENTITY_HOST=1 for the host cache witness")
	}
	runCUDAQ8ResidentHostProbe(t, "^TestCacheIdentity$", cudaQ8CacheIdentityHostCases)
}

const cudaQ8CacheIdentityHostCases = `
func TestCacheIdentity(t *testing.T) {
	for _, change := range []string{"scales", "shape", "block"} {
		t.Run(change, func(t *testing.T) {
			c := &cudaBackend{}
			h := &hostBuf{codes: make([]int8, 64)}
			for i := range h.codes { h.codes[i] = 1 }
			a := Tensor{Dtype: Q8_0, Layout: RowMajor, Shape: []int{2, 32}, Quant: &QuantSpec{Block: 32, Scale: []float32{1, 2}}, buf: h}
			uploadCache = make(map[ucKey]Tensor)
			first := c.uploadQ8Resident(a, h)
			if c.allocs != 1 || c.uploads != 1 || c.accounts != 1 {
				t.Fatalf("first upload did not seed a real resident: %+v", c)
			}
			*c = cudaBackend{}
			h.reads = 0
			same := c.uploadQ8Resident(a, h)
			if same.buf != first.buf || c.allocs != 0 || c.uploads != 0 || c.accounts != 0 || c.bytes != 0 {
				t.Fatalf("identical source lost valid cache hit: %+v", c)
			}

			b := Tensor{Dtype: Q8_0, Layout: RowMajor, Shape: []int{2, 32}, Quant: &QuantSpec{Block: 32, Scale: a.Quant.Scale}, buf: h}
			switch change {
			case "scales": b.Quant.Scale = []float32{3, 4}
			case "shape": b.Shape = []int{1, 64}
			case "block": b.Quant.Block = 16; b.Quant.Scale = []float32{1, 2, 3, 4}
			}
			got := c.uploadQ8Resident(b, h)
			if got.buf == first.buf {
				t.Fatalf("stale cache hit after %s identity changed with the same codes and Numel", change)
			}
			buf, ok := got.buf.(*cudaBuf)
			if !ok || buf == nil || c.allocs != 1 || c.uploads != 1 || c.accounts != 1 || c.bytes != 64+4*len(b.Quant.Scale) {
				t.Fatalf("identity miss did not upload replacement payload: %+v", c)
			}
			if len(got.Shape) != 2 || got.Shape[0] != b.Shape[0] || got.Shape[1] != b.Shape[1] || got.Quant == nil || got.Quant.Block != b.Quant.Block {
				t.Fatal("replacement retained stale geometry or block")
			}
			if len(buf.scales) != len(b.Quant.Scale) || len(buf.codes) != len(h.codes) {
				t.Fatal("replacement retained stale payload lengths")
			}
			for i, v := range b.Quant.Scale { if buf.scales[i] != v { t.Fatal("replacement retained stale scales") } }
			for i, v := range h.codes { if buf.codes[i] != v { t.Fatal("replacement changed codes") } }
			key := ucKey{uintptr(unsafe.Pointer(&h.codes[0])), Q8_0, RowMajor}
			if len(uploadCache) != 1 || uploadCache[key].buf != buf {
				t.Fatal("replacement resident was not published under the existing key")
			}
			*c = cudaBackend{}
			if again := c.uploadQ8Resident(b, h); again.buf != got.buf || c.allocs != 0 || c.uploads != 0 || c.accounts != 0 {
				t.Fatal("replacement source did not become a reusable cache hit")
			}
		})
	}
}
`
