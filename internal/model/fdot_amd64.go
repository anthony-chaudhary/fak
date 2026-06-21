//go:build amd64

package model

import "os"

//go:noescape
func fdot3asm(r0, r1, r2, x *float32, n int) (a, b, c float32)

//go:noescape
func fdot3asm512(r0, r1, r2, x *float32, n int) (a, b, c float32)

var fdot3AVX512 = initFdot3AVX512()

func initFdot3AVX512() bool {
	switch os.Getenv("FAK_FDOT3_AVX512") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

func fdot3Fast(r0, r1, r2, x []float32) (float32, float32, float32, bool) {
	if qtier < tierAVX2 || len(x) < 8 {
		return 0, 0, 0, false
	}
	if fdot3AVX512 && qtier >= tierAVX512 && len(x) >= 16 {
		a, b, c := fdot3asm512(&r0[0], &r1[0], &r2[0], &x[0], len(x))
		return a, b, c, true
	}
	a, b, c := fdot3asm(&r0[0], &r1[0], &r2[0], &x[0], len(x))
	return a, b, c, true
}
