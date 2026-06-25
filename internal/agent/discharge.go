package agent

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
)

// discharge.go — sound goal-root discharge (#847, epic #844). A goal is the
// intentional GC root of the context heap (#845 pins it in SessionPlanner.pins()).
// When the goal's stop condition holds, its whole retained sub-graph should drop to
// refcount 0 and be collected — otherwise the working set a finished goal pulled in
// stays resident forever. Pinning a goal (the root, #845) and freeing its working
// set (here) are the two ends of the same object's life, through the same
// abi.CASPinner refcount-by-digest mechanism.
//
// Two soundness rules, both enforced here:
//
//  1. WITNESSED, not model-judged. Discharge gates on a witnessed stop — the
//     `dos hook stop` verdict, NOT the harness /goal Haiku judge alone (which is
//     session-only, in-memory, invisible to the kernel). We model the witness as a
//     StopWitness seam the caller supplies, so the hot loop never shells out to dos;
//     the caller passes the already-obtained verdict. A goal whose stop is NOT
//     witnessed is never discharged.
//
//  2. NO-OP when another live root still holds a span. Spans are refcounted by
//     digest (abi.CASPinner): a span retained by the discharged goal AND by another
//     live root must NOT be freed. Discharge unpins ONLY the spans no other live
//     root holds — the set difference of the goal's spans minus the union of every
//     other live root's spans.
//
// DEFAULT-OFF: nothing in the live RunArm loop calls Discharge. It is the mechanism
// + the two guards; the live wiring (calling it when a goal root's witnessed stop
// fires) is a default-off follow-up, kept separate so this change cannot alter a
// live decision.

// StopWitness reports whether a goal's stop condition has been WITNESSED — the
// `dos hook stop` verdict, not the model's own claim. The caller obtains the verdict
// out of band (the witnessed stop hook) and supplies it here; the hot loop never
// shells out. A nil StopWitness is treated as "not witnessed" (fail-closed: no
// discharge), so an absent witness never frees anything.
type StopWitness interface {
	// Witnessed reports whether the goal identified by goalID has a witnessed stop.
	Witnessed(goalID string) bool
}

// StopWitnessFunc adapts a plain func to a StopWitness.
type StopWitnessFunc func(goalID string) bool

// Witnessed implements StopWitness.
func (f StopWitnessFunc) Witnessed(goalID string) bool {
	if f == nil {
		return false
	}
	return f(goalID)
}

// Root is a live retention root over the context heap: a goal (or any pinned root)
// and the span Refs it keeps resident. Other live roots are passed to Discharge so
// it can compute which of the discharged root's spans are held ONLY by it.
type Root struct {
	ID    string    // the root's id (a goal id / digest)
	Spans []abi.Ref // the span Refs this root keeps pinned (its retained sub-graph)
}

// DischargeResult reports what a discharge did: whether it ran at all (a witnessed
// stop), and which span digests it actually unpinned (the ones no other live root
// held). It is the auditable record a caller can EXPLAIN.
type DischargeResult struct {
	Discharged bool     // true iff the stop was witnessed and discharge ran
	Reason     string   // why it did / did not run
	Unpinned   []string // the span digests actually unpinned (held only by this root)
	Retained   []string // span digests NOT unpinned because another live root holds them
}

// Discharge frees the retained sub-graph of a discharged goal root, soundly. It runs
// ONLY when the stop is witnessed (witness.Witnessed(goal.ID)); otherwise it is a
// no-op with a reason. When it runs, it unpins exactly the spans of `goal` that NO
// other root in `otherRoots` holds — a span shared with another live root is
// RETAINED (the other root still needs it). Unpinning is done through
// abi.UnpinResolved (the CASPinner seam), which is itself refcounted by digest, so
// even the unpin is safe under content-addressed dedup.
//
// This is the discharge END of the same mechanism that PINS a goal as a root: a
// goal's life is "pin the root, do the work, discharge → unpin the working set no
// one else holds".
func Discharge(goal Root, otherRoots []Root, witness StopWitness) DischargeResult {
	if witness == nil || !witness.Witnessed(goal.ID) {
		return DischargeResult{Discharged: false, Reason: "stop not witnessed (dos hook stop); no discharge"}
	}

	// Union of every OTHER live root's span digests — the spans that must survive.
	held := make(map[string]bool)
	for _, r := range otherRoots {
		if r.ID == goal.ID {
			continue // a root never protects its own spans from its own discharge
		}
		for _, s := range r.Spans {
			if d, ok := refDigest(s); ok {
				held[d] = true
			}
		}
	}

	res := DischargeResult{Discharged: true, Reason: "witnessed stop; unpinned spans held only by this root"}
	// Dedup the goal's own spans by digest so a digest shared across the goal's own
	// sub-graph is unpinned once (CASPinner is refcounted, but the audit record
	// should name each freed digest once).
	seen := make(map[string]bool)
	for _, s := range goal.Spans {
		d, ok := refDigest(s)
		if !ok {
			continue // inline refs carry their own bytes; nothing to unpin
		}
		if held[d] {
			if !contains(res.Retained, d) {
				res.Retained = append(res.Retained, d)
			}
			continue // another live root still holds it → must not free
		}
		if seen[d] {
			continue
		}
		seen[d] = true
		abi.UnpinResolved(s) // drop this root's pin; CASPinner frees at the last unpin
		res.Unpinned = append(res.Unpinned, d)
	}
	return res
}

// refDigest returns a Ref's CAS digest IFF the bytes live in the backend store
// (Blob / Region). Inline refs carry their own bytes and are never pinned, so they
// have no digest to free. Mirrors abi.casDigest (unexported there).
func refDigest(r abi.Ref) (string, bool) {
	if (r.Kind == abi.RefBlob || r.Kind == abi.RefRegion) && r.Digest != "" {
		return r.Digest, true
	}
	return "", false
}

// contains reports membership (small slices; linear is fine).
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
