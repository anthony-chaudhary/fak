//go:build !amd64

package model

func saxpy3Fast(out0, out1, out2, x []float32, a0, a1, a2 float32) bool {
	return false
}
