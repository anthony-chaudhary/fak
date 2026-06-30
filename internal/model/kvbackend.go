package model

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// kvBackend adapts a *Session onto abi.KVBackend — the seam the KV-MMU
// (internal/kvmmu) enforces a quarantine verdict through. It exposes the four local
// operations the bridge performs on a session's kernel-owned cache — the live length,
// a prefill that returns next-token logits, the re-RoPE / renumber span eviction, and
// the model id used to key the cachemeta entry — plus the residency-transfer pair
// (StageSpan / RestoreSpan) the widened seam adds for a span hosted off-box. The
// in-process adapter answers that pair from the local synchronous path: the span is
// already resident, so StageSpan is a no-op OK and RestoreSpan is a typed MISS (this
// backend owns no off-box tier). Wrapping the session behind this interface is what
// lets kvmmu enforce without importing the concrete model type — and lets a
// remote/zero-copy KV backend (the disaggregated direction) substitute itself by
// re-registering the factory (last-wins), with no kvmmu edit.
type kvBackend struct{ s *Session }

// Len reports the session cache's live position count.
func (b kvBackend) Len() int { return b.s.Cache.Len() }

// Prefill prefills a token span into the session cache and returns next-token logits.
func (b kvBackend) Prefill(ids []int) []float32 { return b.s.Prefill(ids) }

// Evict removes a [from,from+n) span via the proven re-RoPE / renumber primitive and
// returns the number of positions removed.
func (b kvBackend) Evict(from, n int) int { return b.s.Cache.Evict(from, n) }

// CanEvict reports the cache's span-eviction verdict: nil for a softmax-KV / GLM-DSA
// cache (eviction supported), and a typed *RecurrentEvictUnsupportedError for a hybrid
// Gated-DeltaNet cache (recurrent state with no per-token journal). The KV-MMU consults
// it BEFORE Evict so a recurrent model surfaces the typed limitation at the seam — skip
// eviction, keep the span resident — instead of letting the convenience Evict panic
// inside the package and drop the served request's connection (#1704). It is additive to
// the abi.KVBackend seam (reachable by type assertion), so a backend that does not
// implement it simply reports "evictable" by absence, exactly as today.
func (b kvBackend) CanEvict() error { return b.s.Cache.CanEvict() }

// StageSpan is the in-process local-synchronous default: the span is already resident
// in the kernel-owned cache, so "staging" it off-box is a no-op that returns OK with
// no bytes moved. The digest addresses the span on a remote tier; the in-process
// backend addresses by position and never faults. A remote / disaggregated KV backend
// overrides this to serialize the fak-owned pre-RoPE Kraw rows off-box.
func (b kvBackend) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}

// RestoreSpan is the in-process local-synchronous default: this backend hosts no
// separate off-box residency tier, so a restore-by-digest is a TYPED MISS — the
// caller is told to recompute rather than the backend silently recomputing or
// hanging. A remote KV backend overrides this to page the span back in from L3.
func (b kvBackend) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest, Reason: "no off-box residency tier (in-process backend)"}, nil
}

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
