package compute

import (
	"testing"
)

func TestVulkanStrix_CoherencyFence(t *testing.T) {
	// SFence should execute without panic or error on all architectures
	VulkanStrixSFence()
	VulkanStrixCoherencyFence()
	VulkanStrixQueueSubmitFence(true)
	VulkanStrixQueueSubmitFence(false)
}

func TestVulkanStrix_Wave32WMMA(t *testing.T) {
	M, N, K := 16, 16, 16
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = 1.0
	}
	for i := range B {
		B[i] = 2.0
	}

	C, telem, err := VulkanStrixWave32WMMA(M, N, K, A, B)
	if err != nil {
		t.Fatalf("VulkanStrixWave32WMMA failed: %v", err)
	}
	if len(C) != M*N {
		t.Fatalf("expected output len %d, got %d", M*N, len(C))
	}
	// 1.0 * 2.0 summed across 16 elements = 32.0
	for i, val := range C {
		if val != 32.0 {
			t.Errorf("C[%d] = %v, want 32.0", i, val)
			break
		}
	}
	if !telem.NativeWave32 {
		t.Errorf("expected NativeWave32=true in telemetry")
	}
	if telem.ThroughputGainPercent < 7.0 {
		t.Errorf("expected throughput gain >= 7.0%%, got %.2f%%", telem.ThroughputGainPercent)
	}
}

func TestVulkanStrix_ArchCheck(t *testing.T) {
	if !VulkanStrixIsGFX1151("gfx1151") {
		t.Errorf("expected true for gfx1151")
	}
	if !VulkanStrixIsGFX1151("AMD Radeon 8060S Graphics") {
		t.Errorf("expected true for Radeon 8060S")
	}
	if VulkanStrixIsGFX1151("sm_90") {
		t.Errorf("expected false for sm_90")
	}
}
