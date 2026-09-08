package compute

// Q8_0 cooperative matrix 2D tile geometry constants on gfx1151 / RDNA 3.5.
const (
	VulkanQ8TileM = 32 // Token tile dimension
	VulkanQ8TileN = 32 // Output channel / row tile dimension
	VulkanQ8TileK = 32 // Reduction K dimension (Q8_0 BLOCK=32)
)

func vulkanQ8CooperativeMatrixActive(nativeAvailable bool, tokens int) bool {
	return nativeAvailable && tokens > 1
}

func vulkanQ8DispatchGrid(outDim, tokens int, cooperativeMatrixActive bool) (gridX, gridY, gridZ int) {
	if outDim <= 0 || tokens <= 0 {
		return 1, 1, 1
	}
	if cooperativeMatrixActive && tokens > 1 {
		gridX = (outDim + VulkanQ8TileN - 1) / VulkanQ8TileN
		gridY = (tokens + VulkanQ8TileM - 1) / VulkanQ8TileM
		if gridX < 1 {
			gridX = 1
		}
		if gridY < 1 {
			gridY = 1
		}
		return gridX, gridY, 1
	}
	gridX = (outDim + 7) / 8
	if gridX < 1 {
		gridX = 1
	}
	return gridX, tokens, 1
}
