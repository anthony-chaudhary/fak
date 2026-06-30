//go:build !amd64

package model

// prefillQ4KKTol is the K/Kraw tolerance for the hybrid-Q4K prefill-vs-decode parity test on
// non-amd64 hosts (arm64/noasm). There is no AVX2 register-blocked GEMM tier here, so the
// strict 1e-5 bound holds; the amd64 variant in qwen35_prefill_q4k_amd64_test.go relaxes it to
// 2e-5 only for the AVX2-only tier.
func prefillQ4KKTol() float64 { return 1e-5 }
