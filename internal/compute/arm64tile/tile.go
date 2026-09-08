//go:build arm64

package arm64tile

//go:noescape
func GemmTile4x4Int8NEON(a0, a1, a2, a3, b0, b1, b2, b3 *int8, c *int32, ldc, k int)

//go:noescape
func GemmTile4x4FP16NEON(a0, a1, a2, a3, b0, b1, b2, b3 *uint16, c *float32, ldc, k int)
