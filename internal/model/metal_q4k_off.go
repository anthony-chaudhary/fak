//go:build !(darwin && cgo && fakmetal)

package model

// metal_q4k_off.go — the default (pure-Go) q4_k prefill GEMM dispatch: always the CPU
// q4kGemm. The Metal q4_k path lives in metal_q4k_on.go behind -tags fakmetal, so the shipped
// binary stays one pure-Go artifact (s.MetalQ4K is simply ignored here).

func (s *Session) q4kGemmDispatch(name string, qt *q4kTensor, Xf []float32, P int) []float32 {
	return q4kGemm(qt, Xf, P)
}

// q4kMatRowsDispatch is the decode-GEMV twin: always the CPU q4kMatRows in the pure-Go build.
func (s *Session) q4kMatRowsDispatch(name string, qt *q4kTensor, xf []float32) []float32 {
	return q4kMatRows(qt, xf)
}
