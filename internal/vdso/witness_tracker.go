package vdso

import "sync"

// witness_tracker.go — the §A4 (GLM52-HOSTED-CACHE-COHERENCE) revocation SOURCE:
// re-check-on-observation world-change detection.
//
// The vdso revocation set is PUBLISHED — someone calls Revoke(witness) to mark that the
// external world a cached span depended on has moved. WitnessTracker is one such
// publisher, implementing the standard read-consistency model: it remembers the witness
// last observed for each external resource, and when the resource is RE-observed under a
// DIFFERENT witness (a repo read returns a new git SHA, a file's content hash changed),
// it Revoke()s the stale witness — so the next GLM turn breaks the now-stale
// provider-prefix span (cachemeta.ShapeGLMTurnSegmentWitnessed sees Revoked(stale)==true).
//
// Honest scope: this catches every change the agent RE-OBSERVES. A change to a resource
// that is never re-read is not detected here — that needs an out-of-band watcher, which
// is simply another publisher of the same Revoke. This component is the in-band half, and
// it is correct (it never invents a revocation; it only refutes a witness the world
// genuinely contradicted on re-read).
type WitnessTracker struct {
	mu   sync.Mutex
	v    *VDSO
	seen map[string]string // resource id -> witness last observed for it
}

// NewWitnessTracker tracks resource->witness changes against v's revocation set.
func NewWitnessTracker(v *VDSO) *WitnessTracker {
	return &WitnessTracker{v: v, seen: map[string]string{}}
}

// Observe records the witness currently seen for resource. If resource was previously
// observed under a DIFFERENT witness, the PRIOR witness is revoked (the world moved) and
// Observe returns true. A first observation, or a re-observation under the SAME witness,
// revokes nothing and returns false. An empty witness clears tracking for the resource
// without revoking (the resource is no longer witnessed); an empty resource is a no-op.
func (t *WitnessTracker) Observe(resource, witness string) (changed bool) {
	if resource == "" {
		return false
	}
	t.mu.Lock()
	prev, had := t.seen[resource]
	if witness == "" {
		delete(t.seen, resource)
		t.mu.Unlock()
		return false
	}
	t.seen[resource] = witness
	t.mu.Unlock()
	if had && prev != witness {
		t.v.Revoke(prev) // the world moved: refute the stale witness on the coherence bus
		return true
	}
	return false
}

// Witness returns the witness currently tracked for resource (empty if untracked).
func (t *WitnessTracker) Witness(resource string) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.seen[resource]
}

// Revoked reports whether a witness has been refuted on this tracker's revocation bus —
// the predicate a coherence shaper passes to cachemeta.ShapeGLMTurnSegmentWitnessed, so it
// reads the SAME vdso this tracker publishes to.
func (t *WitnessTracker) Revoked(witness string) bool { return t.v.Revoked(witness) }
