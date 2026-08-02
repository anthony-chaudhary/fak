package wiprecon

// reclaim.go — the RECOVERY half of reconciliation (#5480). recon.go CLASSIFIES an
// orphaned checkpoint; this file turns the RECLAIM slice of that classification into a
// worklist an operator (or a scheduler) can act on WHILE IT IS STILL RECLAIM.
//
// Why a worklist and not a state-machine change. RECLAIM is a decaying verdict BY
// CONSTRUCTION and that is correct, not a bug: `Applies` is `git apply --check` against
// the CURRENT working tree, so a fixed checkpoint's verdict is a function of a tree that
// keeps moving. The same ref reads RECLAIM early and QUARANTINE later with no change to
// the ref itself. Widening RECLAIM to make it stick would mean calling a delta
// reclaimable that git says does not apply — trading a crashed session's uncommitted
// work for a slot. The transition table in recon.go is therefore UNCHANGED here; what
// was missing is that nothing ever consumed the RECLAIM rows before they expired.
//
// The decay clock is BASE DRIFT: how far HEAD has advanced past the base the checkpoint
// was captured on. A delta captured against a base the trunk has since moved 300 commits
// past is one conflicting hunk away from QUARANTINE, so it is the row to act on first.
// Ranking exists to make that remaining life VISIBLE BEFORE it runs out — the tool
// previously reported only a snapshot verdict, never a trend.
//
// Pure: no git, no I/O, no clock. The cmd shell resolves drift and age and folds here.

import "sort"

// DriftUnknown is the TrunkDistance of a row whose recorded base could not be resolved
// (an empty StartSHA, or a base object the repo no longer has). It is deliberately NOT
// zero: zero means "captured on today's HEAD, maximum remaining life", which is the one
// reading that would push a genuinely unmeasurable row to the bottom of the queue while
// claiming it is fresh. Reported as unknown, and sorted last (see RankReclaim).
const DriftUnknown = -1

// ReclaimRow is one RECLAIM verdict plus the base-drift facts that say how much life the
// verdict has left. Session/Reason come from the Decision; the rest are resolved by the
// cmd shell from the checkpoint's stamp and the repo.
type ReclaimRow struct {
	Session string `json:"session"`
	Object  string `json:"object"`
	// StartSHA is the base commit the delta was captured against — the anchor
	// TrunkDistance is measured from.
	StartSHA string `json:"start_sha"`
	// AgeHours is the checkpoint's wall-clock age. 0 when the stamp carries no
	// capture time, so an unstamped ref never fakes its way to the top of the queue.
	AgeHours float64 `json:"age_hours"`
	// TrunkDistance is how many commits HEAD has advanced past StartSHA, or
	// DriftUnknown. This is the decay signal: the further the base has drifted, the
	// sooner `git apply --check` stops succeeding and the row becomes QUARANTINE.
	TrunkDistance int      `json:"trunk_distance"`
	Leaves        []string `json:"leaves"`
	Reason        string   `json:"reason"`
}

// Reclaimable filters a decision set to the RECLAIM rows — the recovery queue, the
// complement of the QUARANTINE queue `reconcile --file-ticket` already acts on. Returns
// an empty (non-nil) slice when there are none, so a JSON consumer never sees null.
func Reclaimable(ds []Decision) []Decision {
	out := make([]Decision, 0, len(ds))
	for _, d := range ds {
		if d.Action == ActReclaim {
			out = append(out, d)
		}
	}
	return out
}

// RankReclaim orders the recovery worklist MOST-DECAYED-FIRST: descending base drift,
// then descending age, then session for determinism. Descending drift puts DriftUnknown
// (-1) last by construction — a row whose base cannot be measured has no evidence of
// urgency, and must not displace one with measured drift.
//
// Ranking never drops a row: the fold is total, so a caller acting only on the head of
// the queue still sees the whole tail.
func RankReclaim(rows []ReclaimRow) []ReclaimRow {
	out := make([]ReclaimRow, len(rows))
	copy(out, rows)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.TrunkDistance != b.TrunkDistance {
			return a.TrunkDistance > b.TrunkDistance
		}
		if a.AgeHours != b.AgeHours {
			return a.AgeHours > b.AgeHours
		}
		return a.Session < b.Session
	})
	return out
}
