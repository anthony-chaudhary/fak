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
func (r *recordingKV) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return r.inner.StageSpan(ctx, digest, from, n)
}
func (r *recordingKV) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	return r.inner.RestoreSpan(ctx, digest)
}

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

// stubL3 is a PURE remote / disaggregated abi.KVBackend — it fronts NO model.Session,
// so it proves the KV-MMU enforces against an engine fak does not itself run (#638's
// acceptance: a remote L3 backend with ZERO concrete-model dependency). It models a
// tiny cache by position, records every Evict so the bridge's enforcement is
// witnessed, and tracks a digest->positions residency map for the stage/restore pair.
type stubL3 struct {
	positions int
	evicts    [][2]int
	staged    map[string]int
}

func (s *stubL3) Len() int { return s.positions }
func (s *stubL3) Prefill(ids []int) []float32 {
	s.positions += len(ids)
	return make([]float32, 8) // a non-nil, deterministic logits vector
}
func (s *stubL3) Evict(from, n int) int {
	s.evicts = append(s.evicts, [2]int{from, n})
	if n > s.positions {
		n = s.positions
	}
	s.positions -= n
	return n
}
func (s *stubL3) ModelID() string { return "stub-l3-remote" }
func (s *stubL3) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	if s.staged == nil {
		s.staged = map[string]int{}
	}
	s.staged[digest] = n
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n, BytesMoved: int64(n) * 64}, nil
}
func (s *stubL3) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	if n, ok := s.staged[digest]; ok {
		return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
	}
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest, Reason: "not staged"}, nil
}

// TestRemoteL3BackendEnforcedWithoutModelDependency is #638's headline acceptance: a
// remote/stub L3 backend, registered via abi.RegisterKVBackend, is enforced by
// kvmmu.NewBackend against an L3-resident span with NO concrete-model dependency — and
// its widened residency-transfer pair returns typed outcomes (a restore hit/miss is
// TOLD, never a silent recompute). We restore the in-process default factory after, so
// the global registry is left exactly as modelengine's init set it.
func TestRemoteL3BackendEnforcedWithoutModelDependency(t *testing.T) {
	ctx := context.Background()

	// (1) The pure remote backend registers through the kernel seam the SAME way the
	// in-process default does; KVBackendFor builds it with no *model.Session anywhere.
	abi.RegisterKVBackend(func(any) (abi.KVBackend, bool) { return &stubL3{}, true })
	defer abi.RegisterKVBackend(model.KVBackendFor) // restore the in-process default
	kv, ok := abi.KVBackendFor("anything-non-session")
	if !ok {
		t.Fatalf("KVBackendFor through the registered remote factory: ok=false, want true")
	}
	if kv.ModelID() != "stub-l3-remote" {
		t.Fatalf("registered remote backend ModelID = %q, want stub-l3-remote", kv.ModelID())
	}

	// (2) The residency-transfer pair returns TYPED outcomes (not dense logits): stage a
	// span, restore it (OK), and a restore of an unknown digest is a typed MISS.
	rem := &stubL3{}
	if st, err := rem.StageSpan(ctx, "span-A", 0, 4); err != nil || st.Outcome != abi.KVResidencyOK || st.BytesMoved == 0 {
		t.Fatalf("StageSpan -> %+v err=%v, want OK with bytes moved", st, err)
	}
	if got, err := rem.RestoreSpan(ctx, "span-A"); err != nil || got.Outcome != abi.KVResidencyOK || got.Positions != 4 {
		t.Fatalf("RestoreSpan(staged) -> %+v err=%v, want OK positions=4", got, err)
	}
	if miss, _ := rem.RestoreSpan(ctx, "absent"); miss.Outcome != abi.KVResidencyMiss {
		t.Fatalf("RestoreSpan(absent) -> %+v, want a typed MISS (never a silent recompute)", miss)
	}

	// (3) The KV-MMU enforces a quarantine eviction against this remote backend through
	// the pure-abi NewBackendWithGate seam — no concrete *model.Session in the path.
	c := kvmmu.NewBackendWithGate(rem, ctxmmu.New())
	prefix := []int{1, 2, 3}
	poison := []int{10, 11, 12, 13}
	c.Append("sys", "system", prefix)
	v, evicted, _ := c.AdmitResult(ctx, "q1", "read_refund_policy", poison, []byte(poisonBody))
	if v.Kind != abi.VerdictQuarantine || !evicted {
		t.Fatalf("verdict=%v evicted=%v, want Quarantine+evicted on the remote backend", v.Kind, evicted)
	}
	if len(rem.evicts) != 1 || rem.evicts[0] != [2]int{len(prefix), len(poison)} {
		t.Fatalf("recorded evicts=%v, want exactly the poison span [from=%d,len=%d] on the remote backend",
			rem.evicts, len(prefix), len(poison))
	}
	if c.CacheLen() != len(prefix) {
		t.Fatalf("remote cache len after quarantine = %d, want %d", c.CacheLen(), len(prefix))
	}
}
