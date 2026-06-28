//go:build !(darwin && cgo && fakmetal)

package model

// metal_prefill_hybrid_stub.go — the default build's placeholder for the Metal hybrid-prefill
// twin (#71). The GPU twin is linked only under `-tags fakmetal`; without it, Session.Metal is
// never set true (the benchmark gates it on metalgemm.Available(), which the stub backend reports
// false), so this method is unreachable. It exists only so kv.go's hybrid Metal route compiles
// cgo-free, exactly like metal_prefill_stub.go does for prefillBatchedMetal.
func (s *Session) prefillBatchedMetalQwen35Hybrid(ids []int) []float32 {
	panic("model: Metal hybrid prefill not compiled in (build with -tags fakmetal)")
}
