//go:build !(darwin && arm64 && cgo)

package model

// metal_q4k_off.go — the default (pure-Go) q4_k prefill GEMM dispatch: always the CPU
// q4kGemm. The Metal q4_k path lives in metal_q4k_on.go on Apple Silicon+cgo, so non-Metal
// builds stay pure-Go (s.MetalQ4K is simply ignored here).

func (s *Session) q4kGemmDispatch(name string, qt *q4kTensor, Xf []float32, P int) []float32 {
	return q4kGemm(qt, Xf, P)
}

// q4kMatRowsDispatch is the decode-GEMV twin: always the CPU q4kMatRows in the pure-Go build.
func (s *Session) q4kMatRowsDispatch(name string, qt *q4kTensor, xf []float32) []float32 {
	return q4kMatRows(qt, xf)
}

// q4kGroupDispatch always declines in the pure-Go build (no Metal) so mulGroup loops the per-call
// CPU path. The Metal one-command-buffer group path lives in metal_q4k_on.go on Apple Silicon+cgo.
func (s *Session) q4kGroupDispatch(names []string, xf []float32, outs []int) [][]float32 {
	return nil
}

// q4kFusedMLP always declines in the pure-Go build; the fused on-GPU SwiGLU MLP lives in
// metal_q4k_on.go on Apple Silicon+cgo. The caller (denseSwiGLU.apply) uses the per-matmul path.
func (s *Session) q4kFusedMLP(gateName, upName, downName string, x []float32) []float32 {
	return nil
}

// metalQ4KWeights is the stub for the pure-Go build — always returns nil since there's no
// Metal device. The on-Metal implementation (metal_q4k_on.go) uploads all Q4_K projection
// weights upfront to avoid per-call GPU round-trips during prefill (#1113).
func (m *Model) metalQ4KWeights() map[string]bool {
	return nil
}
