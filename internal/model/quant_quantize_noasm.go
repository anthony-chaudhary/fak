//go:build !amd64 && !arm64

package model

// quantizeRowQ8 on non-amd64/arm64 is the portable scalar reference.
func quantizeRowQ8(x []float32, q []int8, d []float32, nblk int) {
	quantizeRowQ8scalar(x, q, d, nblk)
}
