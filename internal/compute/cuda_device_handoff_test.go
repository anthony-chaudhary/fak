//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestCUDAHopBitExactAcrossDevices is the device witness for the band-boundary copy:
// a 200-token Qwen3.8 hidden state (5120 x 200 f32, 4 MiB) crosses every ordered pair of
// visible devices by the auto route and by the forced pinned bounce, bit for bit.
// Pending hardware step: run on the laptop (RTX 4070 + two eGPUs).
//
// fak-test:runtime slow est=5s
func TestCUDAHopBitExactAcrossDevices(t *testing.T) {
	n, err := CUDADeviceCount()
	if err != nil || n < 2 {
		t.Skipf("needs >=2 CUDA devices (have %d, err %v)", n, err)
	}
	rng := rand.New(rand.NewSource(20261025))
	src := make([]float32, 5120*200)
	for i := range src {
		src[i] = math.Float32frombits(rng.Uint32())
	}
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			for _, hop := range []CUDAHop{{}, {ForceBounce: true, StagingBytes: 1 << 20}} {
				got, rec, err := hop.HopF32(src, a, b)
				if err != nil {
					t.Fatalf("%d->%d force=%v: %v", a, b, hop.ForceBounce, err)
				}
				for i := range src {
					if math.Float32bits(got[i]) != math.Float32bits(src[i]) {
						t.Fatalf("%d->%d route %s: word %d differs", a, b, rec.Route, i)
					}
				}
				peer, _ := CUDAPeerAccess(a, b)
				t.Logf("%d->%d peer_access=%v route=%s chunks=%d", a, b, peer, rec.Route, rec.Chunks)
			}
		}
	}
}
