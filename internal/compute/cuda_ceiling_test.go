//go:build cuda && cgo

package compute

import "testing"

// TestCUDABackendDeviceLocalCeiling pins the CUDA stable-ceiling seam on LITERAL backends
// (no live GPU): totalMem is the immutable totalGlobalMem captured once at init, so a struct
// literal stands in for a probed device. It covers the method's own nil/zero-total guard AND
// the two-part discovery end-to-end: CUDA advertises Caps().CapacityProbe unconditionally, so
// DeviceCeilingInfo trusts the method value verbatim - the method's known=false guard is the
// only safety on the degenerate backends.
func TestCUDABackendDeviceLocalCeiling(t *testing.T) {
	cases := []struct {
		name        string
		be          *cudaBackend
		wantCeiling int64
		wantKnown   bool
	}{
		{"probed total", &cudaBackend{name: "cuda", totalMem: 24 << 30}, 24 << 30, true},
		{"zero total", &cudaBackend{name: "cuda"}, 0, false},
		{"nil receiver", nil, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if ceiling, known := tc.be.DeviceLocalCeiling(); known != tc.wantKnown || ceiling != tc.wantCeiling {
				t.Fatalf("DeviceLocalCeiling = (%d, %v), want (%d, %v)", ceiling, known, tc.wantCeiling, tc.wantKnown)
			}
			if ceiling, known := DeviceCeilingInfo(Backend(tc.be)); known != tc.wantKnown || ceiling != tc.wantCeiling {
				t.Fatalf("DeviceCeilingInfo = (%d, %v), want (%d, %v) via two-part discovery", ceiling, known, tc.wantCeiling, tc.wantKnown)
			}
		})
	}
}
