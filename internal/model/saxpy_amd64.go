//go:build amd64

package model

//go:noescape
func saxpy3asm(out0, out1, out2, x *float32, a0, a1, a2 float32, n int)

//go:noescape
func saxpy3asm512(out0, out1, out2, x *float32, a0, a1, a2 float32, n int)

func saxpy3Fast(out0, out1, out2, x []float32, a0, a1, a2 float32) bool {
	if qtier < tierAVX2 || len(out0) < 8 {
		return false
	}
	if qtier >= tierAVX512 && len(out0) >= 16 {
		saxpy3asm512(&out0[0], &out1[0], &out2[0], &x[0], a0, a1, a2, len(out0))
		return true
	}
	saxpy3asm(&out0[0], &out1[0], &out2[0], &x[0], a0, a1, a2, len(out0))
	return true
}
