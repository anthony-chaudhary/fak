package compute

// Q4_K cooperative matrix 2D tile geometry for the routing candidate arm.
//
// These are the block-tiled routing scheme's own tile dimensions, distinct from
// the exported VulkanQ4KTile* geometry in vulkan.go (which the production
// Q4KMatMul2DDispatchGrid path uses). They are kept file-private so the two
// schemes cannot collide on the package-level identifier, as they did when this
// file redeclared the exported names (fak#12177 follow-up build break).
const (
	q4kRoutingTileM = 32 // Token tile dimension
	q4kRoutingTileN = 32 // Output channel / row tile dimension
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
		gridX = (outDim + q4kRoutingTileN - 1) / q4kRoutingTileN
		gridY = (tokens + q4kRoutingTileM - 1) / q4kRoutingTileM
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
