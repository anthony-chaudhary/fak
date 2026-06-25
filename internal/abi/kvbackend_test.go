package abi

import (
	"context"
	"testing"
)

// fakeKV is a stand-in abi.KVBackend that records the Evict it received, so a test
// can prove the enforcement path drove THIS registered backend rather than a
// hardcoded one. It implements the residency-transfer pair as a trivial off-box stub
// (stage OK, restore MISS) so it satisfies the widened seam.
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
func (b *fakeKV) StageSpan(_ context.Context, digest string, _, n int) (KVResidency, error) {
	return KVResidency{Outcome: KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (b *fakeKV) RestoreSpan(_ context.Context, digest string) (KVResidency, error) {
	return KVResidency{Outcome: KVResidencyMiss, Digest: digest}, nil
}

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

// TestKVResidencyOutcomeString pins the stable log/metric names, and that the zero
// value renders "unknown" (it must never read as a successful transfer).
func TestKVResidencyOutcomeString(t *testing.T) {
	cases := map[KVResidencyOutcome]string{
		KVResidencyUnknown: "unknown",
		KVResidencyOK:      "ok",
		KVResidencyMiss:    "miss",
		KVResidencyFault:   "fault",
	}
	for o, want := range cases {
		if got := o.String(); got != want {
			t.Fatalf("KVResidencyOutcome(%d).String() = %q, want %q", o, got, want)
		}
	}
	var zero KVResidencyOutcome // the zero value is Unknown, not OK
	if zero != KVResidencyUnknown {
		t.Fatalf("zero value = %v, want KVResidencyUnknown (fail-closed)", zero)
	}
}

// TestKVResidencyTransferThroughSeam proves a registered backend answers the widened
// residency-transfer pair with typed outcomes (not dense logits): the in-process-style
// stub stages OK and restores MISS, so a remote backend can distinguish them.
func TestKVResidencyTransferThroughSeam(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	RegisterKVBackend(func(any) (KVBackend, bool) { return &fakeKV{id: "fake"}, true })
	kv, ok := KVBackendFor(nil)
	if !ok {
		t.Fatalf("KVBackendFor: ok=false, want true")
	}
	staged, err := kv.StageSpan(context.Background(), "span-A", 0, 4)
	if err != nil || staged.Outcome != KVResidencyOK || staged.Positions != 4 {
		t.Fatalf("StageSpan -> %+v err=%v, want OK positions=4", staged, err)
	}
	restored, err := kv.RestoreSpan(context.Background(), "span-A")
	if err != nil || restored.Outcome != KVResidencyMiss {
		t.Fatalf("RestoreSpan -> %+v err=%v, want a typed MISS (never a silent recompute)", restored, err)
	}
}
