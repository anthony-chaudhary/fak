package compute

import "testing"

func TestVulkanQ8NativeCoopmatAdmission(t *testing.T) {
	tests := []struct {
		name            string
		nativeAvailable bool
		tokens          int
		want            bool
	}{
		{name: "no native capability", tokens: 33, want: false},
		{name: "heuristic cannot replace native capability", tokens: 33, want: false},
		{name: "native decode", nativeAvailable: true, tokens: 1, want: false},
		{name: "native prefill", nativeAvailable: true, tokens: 33, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := vulkanQ8CooperativeMatrixActive(test.nativeAvailable, test.tokens); got != test.want {
				t.Fatalf("vulkanQ8CooperativeMatrixActive(%v, %d) = %v, want %v", test.nativeAvailable, test.tokens, got, test.want)
			}
		})
	}
}

func TestVulkanQ8DispatchGrid(t *testing.T) {
	tests := []struct {
		name                string
		out, tokens         int
		cooperativeMatrix   bool
		wantX, wantY, wantZ int
	}{
		{name: "under dispatch regression", out: 65, tokens: 33, wantX: 9, wantY: 33, wantZ: 1},
		{name: "native cooperative boundary", out: 65, tokens: 33, cooperativeMatrix: true, wantX: 3, wantY: 2, wantZ: 1},
		{name: "single token stays decode", out: 65, tokens: 1, cooperativeMatrix: true, wantX: 9, wantY: 1, wantZ: 1},
		{name: "invalid shape", out: 0, tokens: 33, cooperativeMatrix: true, wantX: 1, wantY: 1, wantZ: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotX, gotY, gotZ := vulkanQ8DispatchGrid(test.out, test.tokens, test.cooperativeMatrix)
			if gotX != test.wantX || gotY != test.wantY || gotZ != test.wantZ {
				t.Fatalf("vulkanQ8DispatchGrid(%d, %d, %v) = (%d, %d, %d), want (%d, %d, %d)",
					test.out, test.tokens, test.cooperativeMatrix, gotX, gotY, gotZ,
					test.wantX, test.wantY, test.wantZ)
			}
		})
	}
}
