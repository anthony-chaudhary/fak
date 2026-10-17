//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"strings"
)

// Q4_K cooperative matrix 2D tile geometry constants on gfx1151 / RDNA 3.5.
const (
	VulkanQ4KTileM = 16 // Token tile dimension
	VulkanQ4KTileN = 32 // Output channel / row tile dimension
	VulkanQ4KTileK = 16 // Reduction K dimension (Wave32 WMMA 16x16x16 primitive)
)

// HasCooperativeMatrix reports whether the Vulkan device supports cooperative matrix instructions
// (e.g. AMD RDNA 3.5 / gfx1151 / Strix Halo). When unsupported, execution falls back to the 1D scalar path.
func (v *vulkanBackend) HasCooperativeMatrix() bool {
	if v == nil {
		return false
	}
	if env := computeEnvValue("FAK_VULKAN_COOPMAT"); env != "" {
		return env == "1" || env == "true"
	}
	lower := strings.ToLower(v.tier)
	return strings.Contains(lower, "gfx1151") ||
		strings.Contains(lower, "strix") ||
		strings.Contains(lower, "radeon 8060s") ||
		strings.Contains(lower, "radeon 8050s") ||
		strings.Contains(lower, "radv")
}

// Q4KMatMul2DDispatchGrid computes 2D workgroup dispatch grid dimensions (GridX, GridY, GridZ)
// for Q4_K matrix multiplication. When cooperative matrix is active and tokens > 1 (prefill),
// it returns a 2D block-tiled grid: (ceil(outDim/TileN), ceil(tokens/TileM), 1).
// When cooperative matrix is unsupported or tokens == 1, it falls back to 1D scalar dispatch:
// (ceil(outDim*tokens/64), 1, 1).
func (v *vulkanBackend) Q4KMatMul2DDispatchGrid(outDim, tokens int) (gridX, gridY, gridZ int) {
	return VulkanQ4KDispatchGrid(outDim, tokens, v.HasCooperativeMatrix())
}

// VulkanQ4KDispatchGrid calculates workgroup grid dimensions for Q4_K matmul under cooperative matrix or scalar fallback.
func VulkanQ4KDispatchGrid(outDim, tokens int, coopMatActive bool) (gridX, gridY, gridZ int) {
	if outDim <= 0 || tokens <= 0 {
		return 1, 1, 1
	}
	if coopMatActive && tokens > 1 {
		gridX = (outDim + VulkanQ4KTileN - 1) / VulkanQ4KTileN
		gridY = (tokens + VulkanQ4KTileM - 1) / VulkanQ4KTileM
		if gridX < 1 {
			gridX = 1
		}
		if gridY < 1 {
			gridY = 1
		}
		return gridX, gridY, 1
	}
	// Fallback 1D scalar dispatch grid
	gridX = (outDim*tokens + 63) / 64
	if gridX < 1 {
		gridX = 1
	}
	return gridX, 1, 1
}

// Q8MatMul2DDispatchGrid computes 2D workgroup dispatch grid dimensions (GridX, GridY, GridZ)
// for Q8_0 matrix multiplication. When cooperative matrix is active and tokens > 1 (prefill),
// it returns a 2D block-tiled grid: (ceil(outDim/TileN), ceil(tokens/TileM), 1).
// When cooperative matrix is unsupported or tokens == 1, it falls back to 1D scalar decode dispatch:
// (ceil(outDim/8), tokens, 1).
func (v *vulkanBackend) Q8MatMul2DDispatchGrid(outDim, tokens int) (gridX, gridY, gridZ int) {
	return VulkanQ8DispatchGrid(outDim, tokens, v.q8CooperativeMatrixActive(tokens))
}

// q8CooperativeMatrixActive binds Q8 2D routing to the capability reported by
// the initialized native backend. Device-name and environment heuristics cannot
// prove that the native cooperative-matrix pipeline is available.
func (v *vulkanBackend) q8CooperativeMatrixActive(tokens int) bool {
	return v != nil && vulkanQ8CooperativeMatrixActive(v.haveCoopmat, tokens)
}

// VulkanQ8DispatchGrid calculates workgroup grid dimensions for Q8_0 matmul under cooperative matrix or scalar fallback.
func VulkanQ8DispatchGrid(outDim, tokens int, coopMatActive bool) (gridX, gridY, gridZ int) {
	return vulkanQ8DispatchGrid(outDim, tokens, coopMatActive)
}

// selectQ2KFusionLocked keeps the new packed-Q2_K gate/up kernel behind an
// explicit candidate arm until its repeated gfx1151 A/B receipt is accepted.
// The fallback remains the established native Vulkan composition; it never
// redirects execution through an external engine.
func (v *vulkanBackend) selectQ2KFusionLocked(P int) bool {
	if P != 1 {
		return false
	}
	return q2kFusionOptIn()
}
