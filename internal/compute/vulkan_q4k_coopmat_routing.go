package compute

// Q4_K cooperative matrix 2D tile geometry constants on gfx1151 / RDNA 3.5.
//
// Q4_K is a 256-element super-block format: each super-block carries 8 sub-blocks
// of 32 weights. The cooperative-matrix micro-tile for the Q4_K precision is
// 16x16x32 (see CoopMatQ4_K in coopmat.go), so the reduction tile K is 32 and the
// spatial tiles mirror the Q8_0 cooperative path.
const (
	VulkanQ4KTileM = 32 // Token tile dimension
	VulkanQ4KTileN = 32 // Output channel / row tile dimension
	VulkanQ4KTileK = 32 // Reduction K dimension (Q4_K sub-block)
)

// vulkanQ4KCooperativeMatrixActive reports whether the Q4_K cooperative-matrix
// path may serve a call. It is a candidate arm only: it requires the native
// cooperative-matrix capability to be present AND a multi-token (prefill) shape,
// because the scalar 1D path already owns decode. A single token stays on the
// decode-optimized scalar/Wave32 path.
func vulkanQ4KCooperativeMatrixActive(nativeAvailable bool, tokens int) bool {
	return nativeAvailable && tokens > 1
}

// vulkanQ4KDispatchGrid returns the 2D workgroup grid for a Q4_K matmul call.
// When the cooperative-matrix candidate arm is active and the call is multi-token,
// it issues a 2D grid (ceil(outDim/TileN), ceil(tokens/TileM), 1); otherwise it
// falls back to the flat 1D grid used by the scalar shader
// (ceil(outDim*tokens/64), 1, 1), preserving the existing decode/prefill
// flattening for every token count.
func vulkanQ4KDispatchGrid(outDim, tokens int, cooperativeMatrixActive bool) (gridX, gridY, gridZ int) {
	if outDim <= 0 || tokens <= 0 {
		return 1, 1, 1
	}
	if cooperativeMatrixActive && tokens > 1 {
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
	gridX = (outDim*tokens + 63) / 64
	if gridX < 1 {
		gridX = 1
	}
	return gridX, 1, 1
}
