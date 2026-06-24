package model

import "github.com/anthony-chaudhary/fak/internal/abi"

// kvBackend adapts a *Session onto abi.KVBackend — the seam the KV-MMU
// (internal/kvmmu) enforces a quarantine verdict through. It exposes EXACTLY the
// four operations the bridge performs on a session's kernel-owned cache: the live
// length, a prefill that returns next-token logits, the re-RoPE / renumber span
// eviction, and the model id used to key the cachemeta entry. Wrapping the session
// behind this interface is what lets kvmmu enforce without importing the concrete
// model type — and lets a remote/zero-copy KV backend (the disaggregated direction)
// substitute itself by re-registering the factory (last-wins), with no kvmmu edit.
type kvBackend struct{ s *Session }

// Len reports the session cache's live position count.
func (b kvBackend) Len() int { return b.s.Cache.Len() }

// Prefill prefills a token span into the session cache and returns next-token logits.
func (b kvBackend) Prefill(ids []int) []float32 { return b.s.Prefill(ids) }

// Evict removes a [from,from+n) span via the proven re-RoPE / renumber primitive and
// returns the number of positions removed.
func (b kvBackend) Evict(from, n int) int { return b.s.Cache.Evict(from, n) }

// ModelID returns the model id the cachemeta cache key uses: the config ModelType,
// falling back to the first architecture string, matching the prior in-line logic.
func (b kvBackend) ModelID() string {
	if b.s == nil || b.s.M == nil {
		return ""
	}
	if id := b.s.M.Cfg.ModelType; id != "" {
		return id
	}
	if len(b.s.M.Cfg.Architectures) > 0 {
		return b.s.M.Cfg.Architectures[0]
	}
	return ""
}

// KVBackend wraps a *Session as an abi.KVBackend (the in-process default the KV-MMU
// enforces through). A nil session yields ok=false so a caller can fail closed.
func KVBackend(s *Session) (abi.KVBackend, bool) {
	if s == nil {
		return nil, false
	}
	return kvBackend{s: s}, true
}

// KVBackendFor is the abi.KVBackendFactory the kernel registers (from
// internal/modelengine's init, the existing model->abi seam): it adapts an
// in-process *Session into an abi.KVBackend and reports ok=false for any other value,
// so a session type the in-process backend does not own fails closed rather than
// being silently mis-enforced.
func KVBackendFor(session any) (abi.KVBackend, bool) {
	s, ok := session.(*Session)
	if !ok {
		return nil, false
	}
	return KVBackend(s)
}
