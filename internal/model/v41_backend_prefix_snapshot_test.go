package model

// v41_backend_prefix_snapshot_test.go is the fak#13337 acceptance witness: a
// DEVICE (backend) session's V4.1 prefix must be captured and restored with the
// complete continuation contract -- the committed token history, every layer's
// bounded temporal V41AttentionState, the HAL attention KV and its token
// lineage, the backend identity, the epoch and the token-position authority --
// and every mismatch (wrong backend, wrong token count, absent qualification)
// must fail CLOSED without consuming the snapshot.
//
// Before this leaf the V4.1 capability guard refused prefix reuse outright
// (v41_prefix_capability_test.go); fak#13342 made a HOST snapshot carry the
// continuation state. This witness closes the device half: it drives a real
// backend session (the cpu-ref identity qualified in-tree) through the genuine
// NewBackendSessionChecked path, captures mid-decode, restores onto a second
// backend session and requires the restored branch to continue IDENTICALLY to
// the original. A snapshot that dropped history, a layer's window/partial
// group, the HAL KV or the lineage would diverge here.
//
// Independence: the branch is compared against an UNSNAPSHOTTED continuation of
// the same session, so the assertion is on observed logits, not on the snapshot
// agreeing with itself.

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41IdentityRefusingBackend wraps the cpu-ref backend and changes ONLY its
// Name(). It is transparent to every arithmetic and KV operation, so a session
// over it runs the identical computation; the distinct name is what lets the
// witness prove the device-identity guard compares backend IDENTITY rather than
// something the wrapped arithmetic happens to share.
type v41IdentityRefusingBackend struct {
	compute.Backend
	mu   sync.Mutex
	name string
}

func (b *v41IdentityRefusingBackend) Name() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.name
}

// v41BackendSnapshotSession builds a reduced V4.1 session over a real
// compute.Backend and primes its continuation state with a prefill, so
// PrefixSnapshot has a device KV store, a lineage and V4.1 state to carry.
func v41BackendSnapshotSession(t *testing.T, layers int, window []int, prompt []int) *Session {
	t.Helper()
	m := v41DecodeStateModel(t, layers)
	if window != nil {
		m.Cfg.Window = window
	}
	be := compute.Pick("cpu-ref")
	if be == nil {
		t.Skip("cpu-ref backend is not registered in this build")
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	if got := s.Prefill(prompt); len(got) == 0 {
		t.Fatalf("priming prefill of %v returned no logits", prompt)
	}
	if s.v41Forward == nil {
		t.Fatal("V4.1 prefill did not install a continuation state")
	}
	// A backend session always carries a device KV STORE, but the V4.1 assembly
	// is cacheless, so its RESIDENT length is legitimately 0 unless the fixture
	// populates it. The prefix contract is that whatever the store holds is
	// carried; an empty store must round-trip as empty, not be dropped.
	if s.halKV == nil {
		t.Fatal("backend session has no device KV store to snapshot")
	}
	return s
}

// v41BackendSeedHALKV appends an explicit span to the session's device KV store
// so the witness can prove HAL-KV residency is carried across capture/restore
// independently of how much the cacheless assembly happens to write.
func v41BackendSeedHALKV(t *testing.T, s *Session, positions []int) {
	t.Helper()
	if s.halKV == nil || s.Backend == nil {
		t.Fatal("session has no device KV store to seed")
	}
	hd := s.M.Cfg.HeadDim
	w := s.M.Cfg.NumKVHeads * hd
	for _, pos := range positions {
		k := make([]float32, w)
		v := make([]float32, w)
		for j := range k {
			k[j] = float32(pos) + float32(j)*0.25
			v[j] = float32(pos)*0.5 - float32(j)*0.125
		}
		kt := compute.NewF32(s.Backend, []int{len(k)}, k)
		vt := compute.NewF32(s.Backend, []int{len(v)}, v)
		s.halKV.AppendKV(0, kt, kt, vt, pos)
		s.halLineage.append(pos + 1)
	}
	if s.halKV.Len() != len(positions) {
		t.Fatalf("seeded HAL KV len = %d, want %d", s.halKV.Len(), len(positions))
	}
}

// TestV41BackendPrefixSnapshot is the named fak#13337 witness.
func TestV41BackendPrefixSnapshot(t *testing.T) {
	const (
		prompt = 4
	)
	base := []int{1, 3, 5, 7}

	t.Run("counting fixture owns and restores every continuation component", func(t *testing.T) {
		s := v41BackendSnapshotSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		// The reduced V4.1 assembly is cacheless, so no HAL KV is resident from the
		// prefill alone. Seed an explicit device span -- positions that continue the
		// committed history -- so the witness proves HAL-KV residency is carried
		// rather than merely asserting an empty store round-trips.
		seedPos := []int{len(base), len(base) + 1}
		named := append(append([]int(nil), base...), 6, 2)
		v41BackendSeedHALKV(t, s, seedPos)

		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()

		// The device identity and the token authority are the two things a host
		// snapshot cannot supply; both must be present and exact.
		if !snap.hasV41DeviceIdentity || snap.v41DeviceIdentity != s.Backend {
			t.Fatalf("snapshot device identity = %v (present=%t), want the capture backend %v",
				snap.v41DeviceIdentity, snap.hasV41DeviceIdentity, s.Backend)
		}
		if !snap.hasV41Tokens || snap.v41Tokens != len(base) {
			t.Fatalf("snapshot token authority = %d (present=%t), want committed %d",
				snap.v41Tokens, snap.hasV41Tokens, len(base))
		}
		if !snap.v41BackendSnapshotSupported() {
			t.Fatal("snapshot over the qualified backend reports unsupported")
		}
		// Every architecture-specific continuation component the issue names is
		// carried: HAL KV, lineage, history and the per-layer temporal states.
		if snap.halKV == nil || snap.halKV.Len() != len(seedPos) {
			t.Fatalf("snapshot HAL KV len = %v, want resident %d", snap.halKV, len(seedPos))
		}
		if got := len(snap.halLineage.ids); got != len(seedPos) {
			t.Fatalf("snapshot lineage positions = %d, want %d", got, len(seedPos))
		}
		if snap.v41 == nil || len(snap.v41.layers) != 2 {
			t.Fatalf("snapshot V4.1 state = %v, want two temporal layers", snap.v41)
		}

		// Branch mutation isolation: the snapshot's layer state must not alias the
		// live session's.
		if snap.v41.layers[0] == s.v41Forward.layerState(0) {
			t.Fatal("snapshot layer state aliases the live session layer state")
		}
		// The deviceness of the HAL KV must survive: the captured store is a deep,
		// independently-owned clone, not the live session's handle.
		if snap.halKV == s.halKV {
			t.Fatal("snapshot HAL KV aliases the live session's device store")
		}

		// Continue the ORIGINAL and the RESTORED branch over the same suffix and
		// require identical logits. A dropped continuation component diverges here.
		target, err := s.M.NewBackendSessionChecked(s.Backend)
		if err != nil {
			t.Fatalf("restore target construction: %v", err)
		}
		defer target.Close()
		if err := snap.Restore(target); err != nil {
			t.Fatalf("Restore: %v", err)
		}
		want := s.Step(6)
		got := target.Step(6)
		assertV41LogitsClose(t, got, want, "restored-vs-original")
		wantAgain := s.Step(2)
		gotAgain := target.Step(2)
		assertV41LogitsClose(t, gotAgain, wantAgain, "restored-vs-original second step")

		// The restored session's V4.1 token authority is the restored committed
		// history, and the restored device store carries exactly the seeded span
		// with its lineage intact.
		if got := len(target.v41Forward.history); got != len(named) {
			t.Fatalf("restored committed history = %d, want %d", got, len(named))
		}
		if got := target.halKV.Len(); got != len(seedPos) {
			t.Fatalf("restored HAL KV len = %d, want seeded %d", got, len(seedPos))
		}
		if got := len(target.halLineage.ids); got != len(seedPos) {
			t.Fatalf("restored lineage positions = %d, want %d", got, len(seedPos))
		}
		for i, pos := range seedPos {
			if want := uint32(pos + 1); target.halLineage.ids[i] != want {
				t.Fatalf("restored lineage[%d] = %d, want %d", i, target.halLineage.ids[i], want)
			}
		}

		// Restore transferred ownership: a second restore of the same snapshot
		// cannot silently reuse a live branch.
		if snap.v41 != nil {
			t.Fatal("Restore did not clear the snapshot's V4.1 state ownership")
		}
	})

	t.Run("wrong backend is refused and the snapshot survives", func(t *testing.T) {
		s := v41BackendSnapshotSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()

		other := &v41IdentityRefusingBackend{Backend: s.Backend, name: "v41-other-device"}
		target, err := s.M.NewBackendSessionChecked(other)
		if err != nil {
			t.Fatalf("mismatched target construction: %v", err)
		}
		defer target.Close()
		err = snap.Restore(target)
		if err == nil {
			t.Fatal("Restore onto a different backend succeeded, want refusal")
		}
		if snap.v41 == nil {
			t.Fatal("refused restore consumed the snapshot's V4.1 state")
		}
		if !snap.hasV41Tokens {
			t.Fatal("refused restore consumed the snapshot's token authority")
		}
	})

	t.Run("unknown backend identity is refused", func(t *testing.T) {
		s := v41BackendSnapshotSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		// Capture on a backend whose name is NOT the qualified identity. The
		// session still runs (the wrapped arithmetic is unchanged), so this
		// isolates the identity guard from execution.
		s.Backend = &v41IdentityRefusingBackend{Backend: s.Backend, name: "v41-unqualified-device"}
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		if snap.v41BackendSnapshotSupported() {
			t.Fatal("unqualified backend identity reported supported")
		}
		target, err := s.M.NewBackendSessionChecked(s.Backend)
		if err != nil {
			t.Fatalf("target construction: %v", err)
		}
		defer target.Close()
		if err := snap.Restore(target); err == nil {
			t.Fatal("Restore of an unqualified backend snapshot succeeded, want refusal")
		}
	})

	t.Run("token authority mismatch is refused", func(t *testing.T) {
		s := v41BackendSnapshotSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		// Advance the target's committed history so its token authority disagrees
		// with the snapshot's. The refusal must be atomic.
		target, err := s.M.NewBackendSessionChecked(s.Backend)
		if err != nil {
			t.Fatalf("target construction: %v", err)
		}
		defer target.Close()
		target.Prefill([]int{3, 5})
		err = snap.Restore(target)
		if err == nil {
			t.Fatal("Restore onto a divergent target succeeded, want token-authority refusal")
		}
		if snap.v41 == nil {
			t.Fatal("refused restore consumed the snapshot's V4.1 state")
		}
	})

	t.Run("host-only session has no device identity to admit", func(t *testing.T) {
		// A plain host session over the same fixture must keep its established
		// restore behavior and must never claim a backend qualification.
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, base)
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		if snap.hasV41DeviceIdentity {
			t.Fatal("host-only snapshot invented a device identity")
		}
		if snap.v41BackendSnapshotSupported() {
			t.Fatal("host-only snapshot reported a backend qualification")
		}
	})
}

// TestV41BackendPrefixReuseCapabilityIsBackendAware is the fak#13334 witness:
// the backend-aware predicate admits ONLY the qualified complete-snapshot
// backend, while the legacy config-only predicate keeps its blanket V4.1
// refusal and a bare-KV/recompute-only path stays refused.
func TestV41BackendPrefixReuseCapabilityIsBackendAware(t *testing.T) {
	v41 := Config{ModelType: "deepseek_v41"}
	// The legacy config-only predicate must NOT start advertising V4.1.
	if v41.KVPrefixReuseSupported() {
		t.Fatal("legacy KVPrefixReuseSupported now admits V4.1")
	}
	if v41.InKernelBackendPrefixReuseSupported() {
		t.Fatal("legacy InKernelBackendPrefixReuseSupported now admits V4.1")
	}
	cpuRef := compute.Pick("cpu-ref")
	if cpuRef == nil {
		t.Skip("cpu-ref backend is not registered in this build")
	}
	if !v41.InKernelBackendPrefixReuseSupportedFor(cpuRef) {
		t.Fatalf("qualified backend %q was refused for V4.1", cpuRef.Name())
	}
	if v41.InKernelBackendPrefixReuseSupportedFor(nil) {
		t.Fatal("nil backend was admitted")
	}
	unqualified := &v41IdentityRefusingBackend{Backend: cpuRef, name: "v41-unqualified-device"}
	if v41.InKernelBackendPrefixReuseSupportedFor(unqualified) {
		t.Fatal("unqualified backend identity was admitted")
	}
	// Non-V4.1 architectures keep answering through the existing predicate.
	qwen := Config{ModelType: "qwen35", LayerTypes: []string{"linear_attention", "full_attention"}}
	if !qwen.InKernelBackendPrefixReuseSupportedFor(cpuRef) {
		t.Fatal("Qwen3.5 hybrid lost its backend prefix reuse")
	}
	if (Config{ModelType: "llama"}).InKernelBackendPrefixReuseSupportedFor(cpuRef) {
		t.Fatal("a plain dense architecture was admitted to the in-kernel backend contract")
	}
}
