package accountobs

import "time"

// Admit decides whether a call should skip the account based only on advertised
// quota windows in snapshot. A window can block only when the provider reports
// both zero remaining quota and a reset later than now. Retry-After and response
// status are deliberately not admission signals: they may describe one
// credential in a multi-credential account and must not idle healthy seats.
//
// When several windows block, until is the earliest future reset, when the
// caller should re-evaluate the snapshot. Admit is pure and does not mutate the
// snapshot.
func Admit(snapshot Snapshot, now time.Time) (skip bool, until time.Time) {
	for _, family := range snapshot.Families() {
		if !family.HaveRemaining || family.Remaining != 0 || !family.HaveReset || !family.Reset.After(now) {
			continue
		}
		if !skip || family.Reset.Before(until) {
			skip = true
			until = family.Reset
		}
	}
	return skip, until
}
