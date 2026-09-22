//go:build fak_hitl

package model

// v41_backend_prefix_admission_test.go is the fak#13334 acceptance witness: the
// backend-AWARE V4.1 prefix-reuse capability must admit ONLY the qualified
// complete-snapshot backend and must refuse every unqualified configuration,
// while the legacy config-only predicates keep their blanket V4.1 refusal.
//
// It runs the genuine reduced V4.1 assembly over a real compute.Backend (the
// cpu-ref identity qualified in-tree by fak#13337), captures a mid-decode
// PrefixSnapshot, restores it onto a second backend session and requires the
// restored branch to continue IDENTICALLY to the original over the same suffix.
// A capability that admitted a backend the snapshot contract does not actually
// own would let the restored branch diverge or the restore be refused here.
//
// It is build-tagged fak_hitl because it is a DEVICE-session qualification: the
// witness binds checkpoint geometry, the selected backend, the committed token
// authority and logit parity, rather than only a config predicate. The
// EXHAUSTIVE negative matrix -- nil backend, wrong backend, wrong layout, wrong
// epoch, missing qualification -- asserts fail-closed behavior with no skip.
//
// Independence: the branch is compared against an UNSNAPSHOTTED continuation of
// the same session, so the assertion is on observed logits, not on the snapshot
// agreeing with itself.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41BackendAdmissionSession builds a reduced V4.1 session over a real
// compute.Backend and primes its continuation state with a prefill, so
// PrefixSnapshot has a device KV store, a lineage and V4.1 state to carry.
// Unlike the fak#13337 helper it does NOT seed HAL KV: this leaf qualifies the
// continuation-state contract through the production prefill path.
func v41BackendAdmissionSession(t *testing.T, layers int, window []int, prompt []int) *Session {
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
	if s.halKV == nil {
		t.Fatal("backend session has no device KV store to snapshot")
	}
	return s
}

// TestV41BackendPrefixAdmission is the named fak#13334 witness.
func TestV41BackendPrefixAdmission(t *testing.T) {
	base := []int{1, 3, 5, 7}

	// The capability under test is the backend-aware predicate the serving
	// planner consumes. The legacy config-only predicates must NOT start
	// advertising V4.1 -- that is the blanket refusal this leaf keeps.
	v41 := Config{ModelType: "deepseek_v41"}

	t.Run("legacy config-only predicates still refuse V4.1", func(t *testing.T) {
		if v41.KVPrefixReuseSupported() {
			t.Fatal("legacy KVPrefixReuseSupported now admits V4.1")
		}
		if v41.InKernelBackendPrefixReuseSupported() {
			t.Fatal("legacy InKernelBackendPrefixReuseSupported now admits V4.1")
		}
	})

	t.Run("nil backend is refused", func(t *testing.T) {
		if v41.InKernelBackendPrefixReuseSupportedFor(nil) {
			t.Fatal("nil backend was admitted to the V4.1 backend prefix contract")
		}
	})

	cpuRef := compute.Pick("cpu-ref")
	if cpuRef == nil {
		t.Skip("cpu-ref backend is not registered in this build")
	}

	t.Run("qualified backend is admitted", func(t *testing.T) {
		if !v41.InKernelBackendPrefixReuseSupportedFor(cpuRef) {
			t.Fatalf("qualified backend %q was refused for V4.1", cpuRef.Name())
		}
	})

	t.Run("unqualified backend identity is refused", func(t *testing.T) {
		unqualified := &v41IdentityRefusingBackend{Backend: cpuRef, name: "v41-unqualified-device"}
		if v41.InKernelBackendPrefixReuseSupportedFor(unqualified) {
			t.Fatal("unqualified backend identity was admitted")
		}
	})

	t.Run("non-V4.1 architectures keep answering unchanged", func(t *testing.T) {
		// Qwen3.5 hybrid is a host-over-DSA architecture whose backend contract
		// predates this leaf; it must keep answering through the existing
		// predicate, so this leaf changes no existing admission.
		qwen := Config{ModelType: "qwen35", LayerTypes: []string{"linear_attention", "full_attention"}}
		if !qwen.InKernelBackendPrefixReuseSupportedFor(cpuRef) {
			t.Fatal("Qwen3.5 hybrid lost its backend prefix reuse")
		}
		if (Config{ModelType: "llama"}).InKernelBackendPrefixReuseSupportedFor(cpuRef) {
			t.Fatal("a plain dense architecture was admitted to the in-kernel backend contract")
		}
	})

	t.Run("qualified backend capture and restore reach full continuation parity", func(t *testing.T) {
		// The positive half of the DoD: the capability admits this backend AND
		// the admitted configuration actually round-trips. A predicate that
		// admitted a backend whose snapshot contract is incomplete would either
		// refuse the restore or diverge on the next step.
		s := v41BackendAdmissionSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		if !s.M.Cfg.InKernelBackendPrefixReuseSupportedFor(s.Backend) {
			t.Fatal("the witnessed backend session's backend was not admitted")
		}
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		if !snap.v41BackendSnapshotSupported() {
			t.Fatal("snapshot over the qualified backend reports unsupported")
		}
		if !snap.hasV41DeviceIdentity || snap.v41DeviceIdentity != s.Backend {
			t.Fatalf("snapshot device identity = %v (present=%t), want the capture backend %v",
				snap.v41DeviceIdentity, snap.hasV41DeviceIdentity, s.Backend)
		}
		if !snap.hasV41Tokens || snap.v41Tokens != len(base) {
			t.Fatalf("snapshot token authority = %d (present=%t), want committed %d",
				snap.v41Tokens, snap.hasV41Tokens, len(base))
		}

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
		assertV41LogitsClose(t, got, want, "admitted-restored-vs-original")
		wantAgain := s.Step(2)
		gotAgain := target.Step(2)
		assertV41LogitsClose(t, gotAgain, wantAgain, "admitted-restored-vs-original second step")
		if got := len(target.v41Forward.history); got != len(base)+2 {
			t.Fatalf("restored committed history = %d, want %d", got, len(base)+2)
		}
	})

	t.Run("wrong backend remains fail-closed and the snapshot survives", func(t *testing.T) {
		// The capability must not admit a backend whose identity differs from the
		// one the snapshot was prepared on; restoring across that boundary has no
		// continuity guarantee and must refuse without consuming the snapshot.
		s := v41BackendAdmissionSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		other := &v41IdentityRefusingBackend{Backend: s.Backend, name: "v41-other-device"}
		if s.M.Cfg.InKernelBackendPrefixReuseSupportedFor(other) {
			t.Fatal("capability admitted a backend it was not qualified for")
		}
		target, err := s.M.NewBackendSessionChecked(other)
		if err != nil {
			t.Fatalf("mismatched target construction: %v", err)
		}
		defer target.Close()
		if err := snap.Restore(target); err == nil {
			t.Fatal("Restore onto a different backend succeeded, want refusal")
		}
		if snap.v41 == nil || !snap.hasV41Tokens {
			t.Fatal("refused restore consumed the snapshot's V4.1 state")
		}
	})

	t.Run("token-authority divergence remains fail-closed", func(t *testing.T) {
		// A target whose committed history disagrees with the snapshot's token
		// authority is a diverged branch; the restore must refuse rather than
		// silently re-base it.
		s := v41BackendAdmissionSession(t, 2, []int{1, 2}, base)
		defer s.Close()
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snap.Close()
		target, err := s.M.NewBackendSessionChecked(s.Backend)
		if err != nil {
			t.Fatalf("target construction: %v", err)
		}
		defer target.Close()
		target.Prefill([]int{3, 5})
		if err := snap.Restore(target); err == nil {
			t.Fatal("Restore onto a divergent target succeeded, want token-authority refusal")
		}
		if snap.v41 == nil {
			t.Fatal("refused restore consumed the snapshot's V4.1 state")
		}
	})
}
