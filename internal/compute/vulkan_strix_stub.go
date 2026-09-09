//go:build !(vulkan && (windows || linux) && cgo)

package compute

import (
	"github.com/anthony-chaudhary/fak/internal/compute/strix"
)

// VulkanStrixSFence issues an SFENCE memory barrier to flush CPU write-combining store buffers.
func VulkanStrixSFence() {
	strix.SFence()
}

// VulkanStrixCoherencyFence issues an SFENCE barrier between CPU host writes and GPU queue submissions.
func VulkanStrixCoherencyFence() {
	strix.SFence()
}

// VulkanStrixQueueSubmitFence issues an SFENCE memory barrier if isStrix is true.
func VulkanStrixQueueSubmitFence(isStrix bool) {
	if isStrix {
		strix.SFence()
	}
}

// VulkanStrixWave32WMMA executes a Wave32 WMMA matmul with store buffer coherency fencing.
func VulkanStrixWave32WMMA(M, N, K int, A, B []float32) ([]float32, strix.WMMATelemetry, error) {
	strix.SFence()
	tiler := strix.NewWave32RetiledMatMul(strix.DefaultWave32WMMAConfig())
	res, telem, err := tiler.MatMul(M, N, K, A, B)
	strix.SFence()
	return res, telem, err
}

// VulkanStrixWave32WMMASupported reports false on non-Vulkan builds.
func VulkanStrixWave32WMMASupported(b Backend) bool {
	return false
}

// VulkanStrixIsGFX1151 reports whether the architecture string indicates Strix Halo silicon.
func VulkanStrixIsGFX1151(arch string) bool {
	return isStrixHaloArch(arch)
}
