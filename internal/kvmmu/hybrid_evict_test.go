package kvmmu_test

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// hybridKV is an abi.KVBackend that reports the typed recurrent-unsupported verdict via
// CanEvict (a Gated-DeltaNet hybrid) and records any Evict it receives. A real hybrid
// KVCache.Evict PANICS the *RecurrentEvictUnsupportedError; the KV-MMU must consult
// CanEvict and SKIP eviction for such a cache instead of calling the panicking Evict.
type hybridKV struct {
	inner  abi.KVBackend
	evicts int
}

func (h *hybridKV) Len() int                    { return h.inner.Len() }
func (h *hybridKV) Prefill(ids []int) []float32 { return h.inner.Prefill(ids) }
func (h *hybridKV) Evict(from, n int) int       { h.evicts++; return h.inner.Evict(from, n) }
func (h *hybridKV) ModelID() string             { return "hybrid-gdn" }
func (h *hybridKV) CanEvict() error {
	return errors.New("model: KVCache.Evict does not support Gated-DeltaNet recurrent state")
}
func (h *hybridKV) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return h.inner.StageSpan(ctx, digest, from, n)
}
func (h *hybridKV) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	return h.inner.RestoreSpan(ctx, digest)
}

// TestHybridCacheSkipsEvictionInsteadOfPanicking is the regression for #1704: a kvmmu
// eviction path over a recurrent (Gated-DeltaNet hybrid) cache — the same path the
// gateway complete() loop drives on the served request — must SKIP the evict (keep the
// span resident, correctness-safe) rather than call the convenience KVCache.Evict that
// PANICS the typed *RecurrentEvictUnsupportedError. Before the fix, `fak serve` of a
// Qwen3.6-27B hybrid decoded the completion and then crashed the handler goroutine in
// this path, dropping the client connection (RemoteDisconnected).
func TestHybridCacheSkipsEvictionInsteadOfPanicking(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(synthCfg())
	s := m.NewSession()
	inner, ok := model.KVBackend(s)
	if !ok {
		t.Fatalf("model.KVBackend(session): ok=false, want true")
	}
	h := &hybridKV{inner: inner}
	c := kvmmu.NewBackendWithGate(h, ctxmmu.New())

	prefix := []int{1, 2, 3, 4, 5}
	poison := []int{10, 11, 12, 13}
	c.Append("sys", "system", prefix)

	// AdmitResult on poison drives the quarantine eviction path. For a hybrid cache it
	// must neither panic nor call the (panicking) Evict.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("kvmmu eviction over a hybrid cache PANICKED (#1704 regression): %v", r)
			}
		}()
		_, _, _ = c.AdmitResult(ctx, "q1", "read_refund_policy", poison, []byte(poisonBody))
	}()

	if h.evicts != 0 {
		t.Fatalf("hybrid cache: Evict called %d times, want 0 — the KV-MMU must skip eviction "+
			"(consult CanEvict), not call the panicking convenience Evict", h.evicts)
	}
}
