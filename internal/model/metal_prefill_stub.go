//go:build !(darwin && cgo && fakmetal)

package model

// metal_prefill_stub.go — the default build's placeholder for the Metal prefill twin. The
// GPU path is linked only under `-tags fakmetal`; without it, Session.Metal is never set true
// (the benchmark gates it on metalgemm.Available(), which the stub backend reports false), so
// this method is unreachable. It exists only so kv.go's dispatch compiles cgo-free.
func (s *Session) prefillBatchedMetal(ids []int) []float32 {
	panic("model: Metal prefill not compiled in (build with -tags fakmetal)")
}
