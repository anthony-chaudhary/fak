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

	// ---- adoption facts (#5998) ----
	// A queue that shows only "this is reclaimable" tells a fleet of successors to race.
	// These fields say WHO, IF ANYONE, already holds it, so the queue can surface the
	// UNOWNED rows as work and the held ones as context. All zero when no receipt exists.

	// AdoptedBy is the successor session holding the adoption receipt; "" = unclaimed.
	AdoptedBy string `json:"adopted_by,omitempty"`
	// AdoptedMine is whether that successor is the caller — the difference between
	// "resume your own interrupted recovery" and "someone else's claim".
	AdoptedMine bool `json:"adopted_mine,omitempty"`
	// AdoptPhase is the receipt's Phase, so a resumed row says what is already done.
	AdoptPhase string `json:"adopt_phase,omitempty"`
	// AdoptExpired is whether the claim lapsed AND its holder is provably gone — the
	// takeover precondition, reported rather than acted on.
	AdoptExpired bool `json:"adopt_expired,omitempty"`
	// Attempts is the receipt's attempt count: the attempt history in one number. A row
	// on its fifth attempt wants an operator, not a sixth automatic try.
	Attempts int `json:"attempts,omitempty"`

	// ---- durability facts ----
	// Replication is LOCAL_ONLY / STALE_REMOTE / REPLICATED for this checkpoint against
	// the mirror, and MirrorFreshness is NEVER_SYNCED / FRESH / STALE for the mirror
	// itself. Both matter to a successor deciding urgency: a LOCAL_ONLY row is one disk
	// away from gone, and a STALE mirror means the replication column is old news.
	Replication     string `json:"replication,omitempty"`
	MirrorFreshness string `json:"mirror_freshness,omitempty"`

	// Argv is the EXACT command that advances this row (without the leading "fak"), or
	// nil when no automatic command may. A queue that names a command it did not derive
	// from the row's own facts is how `--apply`-style phantom flags get printed.
	Argv []string `json:"argv,omitempty"`
}

// Unowned reports a row a successor may act on right now: nobody holds its adoption, the
// holder is this session, or the holder's claim has provably lapsed. A row held by a LIVE
// peer is real work but not YOUR work, and surfacing it as actionable is precisely how two
// successors end up materializing one delta twice.
func (r ReclaimRow) Unowned() bool {
	return r.AdoptedBy == "" || r.AdoptedMine || r.AdoptExpired
}

// AdoptArgv is the exact argv that advances one recovery row, or nil when none may: a
// held row needs the holder to finish or its claim to lapse, and no argv can be honest
// about that. Kept here, in the pure package, so the command string has ONE definition
// that a test can pin — a queue and a renderer that each spell it out separately drift,
// and a printed command that does not exist is worse than no command at all.
func AdoptArgv(r ReclaimRow) []string {
	if !r.Unowned() {
		return nil
	}
	verb := "adopt"
	if r.AdoptedMine {
		verb = "resume"
	}
	return []string{"wip", "reconcile", verb, r.Session}
}

// UnownedReclaim narrows a ranked worklist to the rows a successor may claim. Returns an
// empty (non-nil) slice when there are none, matching Reclaimable's contract.
func UnownedReclaim(rows []ReclaimRow) []ReclaimRow {
	out := make([]ReclaimRow, 0, len(rows))
	for _, r := range rows {
		if r.Unowned() {
			out = append(out, r)
		}
	}
	return out
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

// RankReclaim orders the recovery worklist ACTIONABLE-FIRST, then MOST-DECAYED-FIRST:
// unowned rows ahead of ones a live peer already holds, then descending base drift, then
// descending age, then session for determinism. Descending drift puts DriftUnknown (-1)
// last by construction — a row whose base cannot be measured has no evidence of urgency,
// and must not displace one with measured drift.
//
// Ownership leads the sort rather than joining it because the queue's head is what a
// scheduler acts on: a maximally decayed row that a live successor is already recovering
// is the WRONG head, and putting it there converts one recovery into a race. Before
// adoption existed (#5998) every row was unowned, so this key is a no-op on any caller
// that does not populate it and the drift order it documented is preserved exactly.
//
// Ranking never drops a row: the fold is total, so a caller acting only on the head of
// the queue still sees the whole tail.
func RankReclaim(rows []ReclaimRow) []ReclaimRow {
	out := make([]ReclaimRow, len(rows))
	copy(out, rows)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if au, bu := a.Unowned(), b.Unowned(); au != bu {
			return au
		}
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
