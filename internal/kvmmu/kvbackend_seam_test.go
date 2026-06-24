package kvmmu_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// recordingKV is a fake abi.KVBackend that records every Evict it receives and
// fronts a real model.Session for the prefill/len numerics — so the test proves the
// KV-MMU enforces through the registered KVBackend SEAM (the Evict it drove is THIS
// backend's, not a hardcoded concrete model path), while the cache mechanics stay
// real enough for the verdict to be non-vacuous.
type recordingKV struct {
	inner    abi.KVBackend
	evicts   [][2]int // [from,len] of each eviction the bridge requested
	modelTag string
}

func (r *recordingKV) Len() int                    { return r.inner.Len() }
func (r *recordingKV) Prefill(ids []int) []float32 { return r.inner.Prefill(ids) }
func (r *recordingKV) Evict(from, n int) int {
	r.evicts = append(r.evicts, [2]int{from, n})
	return r.inner.Evict(from, n)
}
func (r *recordingKV) ModelID() string { return r.modelTag }

// TestRegisteredKVBackendDrivesEnforcement proves issue #385's inversion: a
// Quarantine verdict drives EVICTION on the abi.KVBackend the Context was given, not
// a hardcoded *model.Session. We wrap a real session in a recording backend, hand it
// to kvmmu via the pure-abi NewBackendWithGate seam, and assert the recorded Evict is
// what the bridge called — and that the registered factory yields the same seam type.
func TestRegisteredKVBackendDrivesEnforcement(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(synthCfg())
	s := m.NewSession()

	inner, ok := model.KVBackend(s)
	if !ok {
		t.Fatalf("model.KVBackend(session): ok=false, want true")
	}
	rec := &recordingKV{inner: inner, modelTag: "recording-kv"}

	// Enforce through the abi.KVBackend seam (NO concrete *model.Session here), with a
	// real ctxmmu gate over real poison bytes.
	c := kvmmu.NewBackendWithGate(rec, ctxmmu.New())

	prefix := []int{1, 2, 3, 4, 5}
	poison := []int{10, 11, 12, 13}
	c.Append("sys", "system", prefix)

	v, evicted, _ := c.AdmitResult(ctx, "q1", "read_refund_policy", poison, []byte(poisonBody))
	if v.Kind != abi.VerdictQuarantine || !evicted {
		t.Fatalf("verdict=%v evicted=%v, want Quarantine+evicted", v.Kind, evicted)
	}
	// The enforcement path drove THIS registered backend's Evict, with the poison span.
	if len(rec.evicts) != 1 {
		t.Fatalf("recorded %d evictions on the registered backend, want exactly 1 (the poison span)", len(rec.evicts))
	}
	if got := rec.evicts[0]; got[0] != len(prefix) || got[1] != len(poison) {
		t.Fatalf("evicted span = [from=%d,len=%d] on the registered backend, want [from=%d,len=%d]",
			got[0], got[1], len(prefix), len(poison))
	}
	// And the cache shrank back to the prefix — enforcement, not just a recorded call.
	if c.CacheLen() != len(prefix) {
		t.Fatalf("cache len after quarantine = %d, want %d", c.CacheLen(), len(prefix))
	}
}

// TestKVBackendFactoryIsRegistered proves the in-process default is wired: the
// modelengine init() registers model.KVBackendFor, so abi.KVBackendFor builds a
// backend for a real session through the kernel's registry (the blank-import default).
func TestKVBackendFactoryIsRegistered(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	s := m.NewSession()
	kv, ok := abi.KVBackendFor(s)
	if !ok {
		t.Fatalf("abi.KVBackendFor(*model.Session): ok=false — the in-process KV backend is not registered")
	}
	if kv.ModelID() != "llama" {
		t.Fatalf("registered backend ModelID = %q, want %q", kv.ModelID(), "llama")
	}
	// A value the in-process factory does not own fails closed.
	if _, ok := abi.KVBackendFor("not a session"); ok {
		t.Fatalf("abi.KVBackendFor with a non-session value: ok=true, want false (fail-closed)")
	}
}
