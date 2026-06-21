package main

import "testing"

// TestCausalInvalidationChain runs the full demonstrator under the test suite so
// the causal-invalidation witness (right-sizing plan matrix row 6) is guarded
// against regression, not only when someone runs the binary by hand. Every field
// of the record is an asserted invariant; run() already errors on any violation,
// but we re-check the headline bits here so a future change that silently weakens
// an assertion in run() (e.g. drops the byte-exact compare) is caught.
func TestCausalInvalidationChain(t *testing.T) {
	rec, err := run()
	if err != nil {
		t.Fatalf("causal-invalidation chain failed: %v", err)
	}
	checks := []struct {
		name string
		ok   bool
	}{
		{"w1 hit before write", rec.W1HitBeforeWrite},
		{"w2 hit before write", rec.W2HitBeforeWrite},
		{"w1 served byte-exact (hit==fresh call)", rec.W1ServedByteExact},
		{"exactly the w1 entry evicted by the write", rec.W1EvictedByWrite == 1},
		{"w2 warm after the unrelated write", rec.W2WarmAfterWrite},
		{"w2 byte-identical across the write", rec.W2ByteIdenticalAcross},
		{"w1 misses after its witness refuted", rec.W1MissAfterWrite},
		{"re-admission under refuted witness refused", rec.W1ReadmissionRefused},
		{"unrelated witness evicts 0 local entries", rec.UnrelatedEvicts == 0},
		{"coherence bus broadcast fired", rec.CoherenceBroadcast},
		{"trust epoch advanced on refutation", rec.TrustEpochAdvanced},
		{"byte-exact headline max|Δ|=0", rec.MaxAbsDelta == 0},
	}
	for _, c := range checks {
		if !c.ok {
			t.Errorf("invariant violated: %s", c.name)
		}
	}
}
