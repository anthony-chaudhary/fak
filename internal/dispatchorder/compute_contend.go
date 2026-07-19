// compute_contend.go — the EXPORTED reuse surface of the compute-claim collision
// machinery (#3269, parent epic #3259): the same class/mode/range fold that
// computeCollision prices fan-out candidates with, factored out so the shared
// compute admission kernel (internal/computeadmit) and any compute placer answer
// with IDENTICAL contention semantics instead of growing a private twin.
package dispatchorder

// ComputeClaimsContend reports whether two compute claims contend for the same
// region: a shared/shared pair may overlap; DIFFERENT resource classes never
// contend (the compute-plane analogue of "different lanes decide on tree
// geometry"); within one class the integer ranges decide, with an empty or
// unparseable range colliding conservatively as unknown blast radius. It is the
// exact fold computeCollision applies (computeCollision calls it), so a placer
// answering through computeadmit.Decide prices contention byte-identically to
// the dispatch fan-out price.
func ComputeClaimsContend(a, b ComputeClaim) bool {
	if computeMode(&a) == "shared" && computeMode(&b) == "shared" {
		return false
	}
	if normClass(a.Class) != normClass(b.Class) {
		return false
	}
	return rangesOverlap(a.Range, b.Range)
}

// ComputeRegionLabel renders a compute claim as the "class:range" evidence label
// the collision graph and RepartitionAdvice rows carry ("*" stands in for an
// unknown range).
func ComputeRegionLabel(c ComputeClaim) string { return regionLabel(&c) }

// RangeWithin reports whether a compute Range string parses (known) and, when it
// does, whether EVERY integer it addresses lies inside the inclusive [lo,hi]
// address space — the taxonomy-membership check an admission kernel runs before
// pricing contention. An empty or unparseable Range returns known=false so the
// caller can fail closed, the discipline ExpertParallelPlan already holds for
// ranks outside its band.
func RangeWithin(r string, lo, hi int64) (within, known bool) {
	ivs, ok := parseRange(r)
	if !ok {
		return false, false
	}
	for _, iv := range ivs {
		if iv.lo < lo || iv.hi > hi {
			return false, true
		}
	}
	return true, true
}
