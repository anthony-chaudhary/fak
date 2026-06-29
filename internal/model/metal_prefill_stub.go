//go:build !(darwin && arm64 && cgo)

package model

// metal_prefill_stub.go — the non-Metal build's placeholder for the Metal prefill twin. The
// GPU path is linked only on Apple Silicon with cgo; without it, Session.Metal is never set true
// (the benchmark gates it on metalgemm.Available(), which the stub backend reports false), so
// this method is unreachable. It exists only so kv.go's dispatch compiles cgo-free.
func (s *Session) prefillBatchedMetal(ids []int) []float32 {
	panic("model: Metal prefill not compiled in (requires darwin/arm64 with cgo)")
}

// PrepareMetalResidency reports that no Metal residency work is available in the non-Metal build.
func (m *Model) PrepareMetalResidency(q4k bool) bool { return false }
