//go:build !amd64

package model

// q4kMatRowsRangeAffine has no non-amd64 kernel; the caller falls back to the scalar exact path.
func q4kMatRowsRangeAffine(qt *q4kTensor, x, y []float32, lo, hi int, xsum []float32) bool {
	return false
}
