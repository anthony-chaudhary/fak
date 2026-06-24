package abi

import "testing"

// fakeKV is a stand-in abi.KVBackend that records the Evict it received, so a test
// can prove the enforcement path drove THIS registered backend rather than a
// hardcoded one.
type fakeKV struct {
	id          string
	evictFrom   int
	evictN      int
	evictCalled bool
}

func (b *fakeKV) Len() int                    { return 0 }
func (b *fakeKV) Prefill(ids []int) []float32 { return nil }
func (b *fakeKV) Evict(from, n int) int {
	b.evictFrom, b.evictN, b.evictCalled = from, n, true
	return n
}
func (b *fakeKV) ModelID() string { return b.id }

// TestRegisterKVBackendRoundTrips proves the factory the kernel registered is what
// KVBackendFor hands back, and that an unrecognized session value fails CLOSED
// (ok=false) — the read side of the RegisterKVBackend seam.
func TestRegisterKVBackendRoundTrips(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	// No factory registered yet -> fail closed.
	if _, ok := KVBackendFor("anything"); ok {
		t.Fatalf("KVBackendFor with no registered factory: ok=true, want false")
	}

	want := &fakeKV{id: "fake"}
	RegisterKVBackend(func(session any) (KVBackend, bool) {
		if s, ok := session.(string); ok && s == "ok" {
			return want, true
		}
		return nil, false
	})

	got, ok := KVBackendFor("ok")
	if !ok {
		t.Fatalf("KVBackendFor(\"ok\"): ok=false, want true")
	}
	if got != want {
		t.Fatalf("KVBackendFor returned a different backend than the factory built")
	}
	if got.ModelID() != "fake" {
		t.Fatalf("ModelID = %q, want %q", got.ModelID(), "fake")
	}
	// An unrecognized session value fails closed, even with a factory registered.
	if _, ok := KVBackendFor(123); ok {
		t.Fatalf("KVBackendFor with an unrecognized value: ok=true, want false (fail-closed)")
	}
}

// TestRegisterKVBackendLastWins pins the documented last-registration-wins override
// (a remote/zero-copy KV backend re-registering over the in-process default).
func TestRegisterKVBackendLastWins(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	first := &fakeKV{id: "first"}
	second := &fakeKV{id: "second"}
	RegisterKVBackend(func(any) (KVBackend, bool) { return first, true })
	RegisterKVBackend(func(any) (KVBackend, bool) { return second, true })

	got, ok := KVBackendFor(nil)
	if !ok || got.ModelID() != "second" {
		t.Fatalf("last-wins: got ok=%v id=%q, want ok=true id=second", ok, modelIDOf(got))
	}
}

func modelIDOf(b KVBackend) string {
	if b == nil {
		return "<nil>"
	}
	return b.ModelID()
}
