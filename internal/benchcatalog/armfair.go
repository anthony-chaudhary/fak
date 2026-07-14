package benchcatalog

import "fmt"

// WitnessSameTasks is the fairness fence for any two-arm (raw vs fak) benchmark
// ablation, extracted so no surface has to hand-assert it. The whole point of a
// raw-arm/fak-arm ablation is that the delta is attributable to fak BECAUSE both
// arms ran the same problems. That "same problems" guarantee is load-bearing, and
// for it to mean anything it must be a WITNESSED fact derived from what each arm
// actually consumed - not a hardcoded `true` a reader cannot distinguish from a
// silent drift (a filtered task, a reordered slice, a dropped id).
//
// It takes the task ids each arm RECORDED consuming (collected from the arms' own
// per-task outputs, not from a shared input read once) and reports whether they
// match, in order, with a human-readable reason on any mismatch. The contract
// mirrors the reference implementation in internal/livecodebench/fakarm.go:
//
//   - Empty is a MISMATCH, never a silent pass: two arms that recorded nothing have
//     not been witnessed to have run the same problems (`len(rawIDs) > 0 && equal`).
//   - Order matters: a reordered slice is drift the fence must catch, so the
//     comparison is positional, not set-based.
//   - A length difference (a dropped or extra id in one arm) is a mismatch naming
//     the counts; a positional difference names the index and the two ids.
//
// The returned reason is "" exactly when same is true, so callers may surface it
// verbatim next to a failed fence. The function is pure: same inputs, same output,
// no I/O - unit-testable with stub id slices and safe to call from any tier.
func WitnessSameTasks(rawIDs, fakIDs []string) (same bool, reason string) {
	if len(rawIDs) == 0 || len(fakIDs) == 0 {
		return false, fmt.Sprintf("fairness fence unwitnessed: raw arm recorded %d task id(s), fak arm recorded %d - an empty arm cannot be proven to have run the same problems", len(rawIDs), len(fakIDs))
	}
	if len(rawIDs) != len(fakIDs) {
		return false, fmt.Sprintf("task-id drift: raw arm ran %d task(s), fak arm ran %d - the arms did not consume the same problem set", len(rawIDs), len(fakIDs))
	}
	for i := range rawIDs {
		if rawIDs[i] != fakIDs[i] {
			return false, fmt.Sprintf("task-id drift at position %d: raw arm ran %q, fak arm ran %q", i, rawIDs[i], fakIDs[i])
		}
	}
	return true, ""
}
