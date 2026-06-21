//go:build !amd64

package model

func fdot3Fast(r0, r1, r2, x []float32) (float32, float32, float32, bool) {
	return 0, 0, 0, false
}
