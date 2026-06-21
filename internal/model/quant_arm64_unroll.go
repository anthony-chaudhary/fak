//go:build arm64

package model

//go:noescape
func qdot8unroll4NEON(qw, qx *int8, dw, dx *float32, nblk int) float32

//go:noescape
func qgemm8row4NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

//go:noescape
func qgemm8tile2x4NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
