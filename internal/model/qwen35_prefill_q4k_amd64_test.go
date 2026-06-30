//go:build amd64

package model

// prefillQ4KKTol is the K/Kraw tolerance for the hybrid-Q4K prefill-vs-decode parity test on
// amd64. On an AVX2-only host (#1127) the prefill GEMM runs the lanes=8 register-blocked tile
// (qgemm8tile256) while decode runs the per-block-scalar qdot8asm, so the same damped drift is
// marginally larger (measured Kraw≈1.4e-5 at the test seed) and K/Kraw take a still-strict 2e-5.
// On AVX-512 the prefill GEMM (qgemm8tile512) and decode GEMV (qdot8gemv512) share a reduction
// order, so the strict 1e-5 holds. qtier/tierAVX2 are defined in quant_amd64.go.
func prefillQ4KKTol() float64 {
	if qtier == tierAVX2 {
		return 2e-5
	}
	return 1e-5
}
