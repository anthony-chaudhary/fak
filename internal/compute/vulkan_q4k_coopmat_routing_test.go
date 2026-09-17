package compute

import "testing"

func TestVulkanQ4KCooperativeMatrixActive(t *testing.T) {
	tests := []struct {
		name            string
		nativeAvailable bool
		tokens          int
		want            bool
	}{
		{name: "no native capability", tokens: 33, want: false},
		{name: "heuristic cannot replace native capability", tokens: 33, want: false},
		{name: "native decode stays scalar", nativeAvailable: true, tokens: 1, want: false},
		{name: "native prefill admits coopmat", nativeAvailable: true, tokens: 2, want: true},
		{name: "native prefill admits coopmat large", nativeAvailable: true, tokens: 33, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := vulkanQ4KCooperativeMatrixActive(test.nativeAvailable, test.tokens); got != test.want {
				t.Fatalf("vulkanQ4KCooperativeMatrixActive(%v, %d) = %v, want %v",
					test.nativeAvailable, test.tokens, got, test.want)
			}
		})
	}
}

// TestVulkanQ4KRoutingTileGeometryIsFilePrivate guards the fak#12177 follow-up
// build break: this file once redeclared the exported VulkanQ4KTileM/N/K names
// that vulkan.go also declares in package compute, so any `-tags vulkan` build
// that includes vulkan.go failed with a redeclaration error. The routing scheme
// now owns file-private q4kRoutingTileM/N; this test pins their values and proves
// the routing grid is derived from THEM, not from the exported production
// geometry, so an accidental re-point or re-export is caught here rather than on
// the device build.
func TestVulkanQ4KRoutingTileGeometryIsFilePrivate(t *testing.T) {
	if q4kRoutingTileM != 32 || q4kRoutingTileN != 32 {
		t.Fatalf("routing tile geometry = (%d, %d), want (32, 32)", q4kRoutingTileM, q4kRoutingTileN)
	}
	// Distinct from the production exported geometry in vulkan.go; if they ever
	// coincide, the two schemes have been conflated and this guard is void.
	if q4kRoutingTileM == 16 && q4kRoutingTileN == 32 {
		t.Fatalf("routing tile geometry (%d, %d) collides with the exported VulkanQ4KTile geometry",
			q4kRoutingTileM, q4kRoutingTileN)
	}
	// out=64, tokens=64 with 32/32 routing tiles => (2, 2, 1); the exported
	// production 16/32 geometry would yield (2, 4, 1) for the same shape.
	gotX, gotY, gotZ := vulkanQ4KDispatchGrid(64, 64, true)
	if gotX != 2 || gotY != 2 || gotZ != 1 {
		t.Fatalf("routing grid(64, 64, coopmat) = (%d, %d, %d), want (2, 2, 1)", gotX, gotY, gotZ)
	}
}

func TestVulkanQ4KDispatchGrid(t *testing.T) {
	tests := []struct {
		name                string
		out, tokens         int
		cooperativeMatrix   bool
		wantX, wantY, wantZ int
	}{
		// The scalar flattening keeps ceil(outDim*tokens/64) workgroups in X and one in Y.
		{name: "scalar decode", out: 65, tokens: 1, wantX: 2, wantY: 1, wantZ: 1},
		{name: "scalar prefill", out: 65, tokens: 33, wantX: 34, wantY: 1, wantZ: 1},
		// The cooperative arm issues a 2D grid tiled on (outDim, tokens).
		{name: "native cooperative boundary", out: 65, tokens: 33, cooperativeMatrix: true, wantX: 3, wantY: 2, wantZ: 1},
		{name: "native cooperative exact tiles", out: 64, tokens: 64, cooperativeMatrix: true, wantX: 2, wantY: 2, wantZ: 1},
		// A single token never takes the cooperative arm even when admitted.
		{name: "single token stays decode", out: 65, tokens: 1, cooperativeMatrix: true, wantX: 2, wantY: 1, wantZ: 1},
		// Degenerate shapes clamp to a valid grid rather than dispatching zero workgroups.
		{name: "invalid shape", out: 0, tokens: 33, cooperativeMatrix: true, wantX: 1, wantY: 1, wantZ: 1},
		{name: "invalid token count", out: 65, tokens: 0, cooperativeMatrix: true, wantX: 1, wantY: 1, wantZ: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotX, gotY, gotZ := vulkanQ4KDispatchGrid(test.out, test.tokens, test.cooperativeMatrix)
			if gotX != test.wantX || gotY != test.wantY || gotZ != test.wantZ {
				t.Fatalf("vulkanQ4KDispatchGrid(%d, %d, %v) = (%d, %d, %d), want (%d, %d, %d)",
					test.out, test.tokens, test.cooperativeMatrix, gotX, gotY, gotZ,
					test.wantX, test.wantY, test.wantZ)
			}
		})
	}
}
