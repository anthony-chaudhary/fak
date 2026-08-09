package compute

import "math"

// requiredfree.go — the inverse of the headroom arithmetic.
//
// BudgetAfterHeadroom (capacity.go) answers "given this much free memory, how many bytes may a
// plan claim?". Every load-time fit check is that question: fitsWithinReportedMemory admits a plan
// exactly when wantBytes <= BudgetAfterHeadroom(free, headroom).
//
// A PRE-FLIGHT gate asks the mirror image — "given this plan, how much free memory must a device
// show BEFORE we spend twenty minutes building and spawning ranks that will each hit the same wall
// after load?". Answering it by hand (want / (1-headroom), or worse, want * (1+headroom)) puts a
// second copy of the arithmetic somewhere the fit check cannot see, and the two drift: a gate that
// rounds the other way passes a run the in-process check then refuses, which reads as a code
// regression rather than as the environmental refusal it is. That misreading is exactly what
// #4952 was originally filed as.
//
// So the inverse lives here, next to the forward direction, and is defined by it.

// RequiredFreeBytes reports the smallest free-byte reading that would let a plan of wantBytes pass
// the same fit check BudgetAfterHeadroom feeds — i.e. the least free such that
// BudgetAfterHeadroom(free, headroom) >= wantBytes. It is the EXACT boundary, not an estimate:
// free == RequiredFreeBytes(want, h) passes and free == RequiredFreeBytes(want, h)-1 does not.
//
// It mirrors BudgetAfterHeadroom's degenerate cases so the two agree everywhere: a non-positive
// wantBytes needs nothing, and a headroom outside (0,1) leaves the budget untouched, so wantBytes
// free bytes are exactly enough.
func RequiredFreeBytes(wantBytes int64, headroom float64) int64 {
	if wantBytes <= 0 {
		return 0
	}
	if !(headroom > 0 && headroom < 1) {
		return wantBytes
	}
	exact := float64(wantBytes) / (1 - headroom)
	if exact >= math.MaxInt64 {
		return math.MaxInt64
	}
	need := int64(math.Ceil(exact))
	if need < wantBytes {
		need = wantBytes // guard the degenerate/rounded-down case; free must exceed want here
	}
	// Both directions round: the division above and the multiplication inside BudgetAfterHeadroom.
	// Walk the last few bytes so the answer is the boundary the fit check actually draws rather
	// than an approximation of it. The step counts are bounded because past float64's exact
	// integer range (2^53) need+1 is not representable, and an unbounded walk would spin there.
	for i := 0; i < 64 && need < math.MaxInt64 && BudgetAfterHeadroom(need, headroom) < wantBytes; i++ {
		need++
	}
	for i := 0; i < 64 && need > wantBytes && BudgetAfterHeadroom(need-1, headroom) >= wantBytes; i++ {
		need--
	}
	return need
}
