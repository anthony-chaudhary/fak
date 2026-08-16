package wipref

// owner.go — CREATION-TIME ownership of an UNTRACKED path (#3874's blind spot).
//
// THE HOLE THIS FILLS. Every ownership surface in this repo reads a TRACKED delta.
// `fak wip attribute` folds `git diff HEAD --` into hunks and grades each against the
// per-session checkpoints (internal/wipattr); `wip sweep-guard` grades the same hunks
// for a broad `git add`; `wip blocked`, `wip reconcile` and `wip reap --census` all
// classify deltas that git already tracks. A file that a session has just CREATED
// produces no diff hunk at all — `git diff HEAD` is silent about it — so the newest,
// most fragile work in the tree is the one thing none of those surfaces can see. The
// only surface that lists untracked source is treedoctor's land-or-park inventory, and
// it (a) waits out a 60-minute abandonment window before it says anything and (b) has
// no production implementation for its own `OwnerOf` hook, so its owner-liveness rung
// reads empty for every file. Between creation and that window a created path has no
// owner anywhere in the system.
//
// THE EVIDENCE, WHICH ALREADY EXISTS. A checkpoint capture is `read-tree HEAD` +
// `add -A` (wip.go), so it stages UNTRACKED files too: a checkpoint's own delta records
// every untracked path alive at capture time as an ADDITION. That makes "which sessions
// have captured this created path" answerable from refs that are already being minted,
// with no new ledger, no new lease, and no new discipline asked of the author. This
// package owns the fold; the cmd shell runs the `git diff --diff-filter=A` that produces
// the Claims.
//
// WHY IT DOES NOT NAME AN AUTHOR WHEN TWO SESSIONS CAPTURED. Capture is tree-wide BY
// DESIGN — that width is what makes a checkpoint lossless for crash recovery — so a ref
// containing a path proves the capturer had it on disk, never that the capturer wrote it
// (the distinction Stamp.Scope exists to carry, and which 3 of this fleet's 1216 stamps
// actually populate). One claim is therefore a usable owner and two are not, and the fold
// refuses to break the tie: it reports AMBIGUOUS with every claimant named, exactly as
// wipattr.Attribute reports SHARED rather than picking. EarliestBy is carried as a
// LABELLED HINT — the first capturer, which for a path created moments ago is very often
// the author because no peer had yet minted a snapshot containing it — and is never
// promoted into Owner.
//
// EXPIRY IS ALSO THE COST BOUND. A claim counts only while it is fresh (ClaimTTL), which
// is what makes "check in or your claim lapses" a real property rather than a slogan. The
// same bound is why this is cheap: on a fleet with 1216 checkpoint refs only ~31 fall
// inside a 90-minute window, so the shell diffs tens of refs, not thousands.
//
// FAIL-SAFE DIRECTION. The state that matters most is UNCLAIMED — a created path NO fresh
// checkpoint records, which is work that a `git clean`, a broad add, or a reap has no
// evidence belongs to anyone. Nothing here is ever a licence to delete: an expired claim
// says "check-in overdue", not "reap me". Pure — no git, no clock, no I/O.

import (
	"sort"
	"time"
)

// DefaultClaimTTL is how long a checkpoint keeps claiming the paths it captured before
// the claim reads as a lapsed check-in. It is deliberately shorter than treedoctor's
// 60-minute abandonment window and longer than the autocheckpoint cadence, so a session
// that is checkpointing at all stays continuously claimed while a session that stopped
// surfaces as overdue well before any cleanup surface would speak about it.
const DefaultClaimTTL = 90 * time.Minute

// Claim is one checkpoint ref's record that it captured a given path as an ADDITION —
// i.e. the path was untracked-and-present in that session's working tree at capture.
// Live is the SESSION's liveness as the caller's oracle reported it (leaseref), not a
// property of the ref.
type Claim struct {
	Session        string `json:"session"`
	CheckpointedAt int64  `json:"checkpointed_at"` // unix seconds; 0 means unstamped
	Live           bool   `json:"live"`
}

// OwnState is the closed creation-ownership vocabulary. Every path handed to the fold
// lands in exactly one state — the same totality guarantee wipattr.Attribute makes.
type OwnState string

const (
	// OwnUnclaimed: no fresh checkpoint records this created path. It is invisible to
	// every attribution surface in the repo, so a broad add or a clean has no evidence
	// it is anyone's work. This is the finding the whole file exists to surface.
	OwnUnclaimed OwnState = "UNCLAIMED"
	// OwnClaimedLive: exactly one fresh claim, and either its session is live or the
	// claim is inside the TTL. Someone is working here.
	OwnClaimedLive OwnState = "CLAIMED_LIVE"
	// OwnClaimedExpired: exactly one claim, its session is not live, and the claim has
	// aged past the TTL. A check-in (re-checkpoint) is overdue. NOT a reap licence.
	OwnClaimedExpired OwnState = "CLAIMED_EXPIRED"
	// OwnAmbiguous: more than one fresh claim. Capture is tree-wide, so containment
	// cannot separate the author from a peer that merely swept the path into its own
	// snapshot. Every claimant is named and none is chosen.
	OwnAmbiguous OwnState = "AMBIGUOUS"
)

// Ownership is the per-path verdict.
type Ownership struct {
	Path  string   `json:"path"`
	State OwnState `json:"state"`
	// Owner is set ONLY for the two single-claim states. Empty for UNCLAIMED and
	// AMBIGUOUS — an ambiguous path has claimants, not an owner.
	Owner string `json:"owner,omitempty"`
	// Owners lists every claimant (sorted) when AMBIGUOUS.
	Owners []string `json:"owners,omitempty"`
	// EarliestBy is the session whose claim is oldest: for a path created moments ago
	// this is usually the author, because no peer had yet minted a snapshot containing
	// it. A HINT for a human, never a verdict — read Owner for the verdict.
	EarliestBy string `json:"earliest_by,omitempty"`
	// ClaimAgeSeconds is the age of the FRESHEST claim (the most recent check-in), or
	// -1 when there is no claim at all or no claim carried a timestamp.
	ClaimAgeSeconds int64 `json:"claim_age_seconds"`
	// Reason is the closed explanation token, one per state, so a caller routes on it
	// without parsing prose.
	Reason string `json:"reason"`
}

// Reason tokens, one per OwnState.
const (
	ReasonNoFreshClaim   = "NO_FRESH_CLAIM"
	ReasonSessionLive    = "SESSION_LIVE"
	ReasonClaimFresh     = "CLAIM_FRESH"
	ReasonCheckinOverdue = "CHECKIN_OVERDUE"
	ReasonTreeWideTie    = "TREE_WIDE_CAPTURE_TIE"
)

// OwnerOfPath folds one path's claims into a verdict. nowUnix is the caller's clock;
// ttl <= 0 uses DefaultClaimTTL. A claim is FRESH when its session is live OR its
// timestamp is within ttl of now; a claim with no timestamp (CheckpointedAt == 0) is
// fresh only if its session is live, because an unstamped ref carries no evidence of
// recency and guessing "recent" would manufacture an owner out of nothing.
//
// Liveness is the primary rung and freshness the fallback — the ordering lanebeat uses
// — so a live session never loses its claim to a slow clock, and a dead session's claim
// still gets the TTL grace before it is called overdue.
func OwnerOfPath(path string, claims []Claim, nowUnix int64, ttl time.Duration) Ownership {
	if ttl <= 0 {
		ttl = DefaultClaimTTL
	}
	ttlSec := int64(ttl / time.Second)

	var (
		fresh      []Claim
		newestAge  int64 = -1
		earliest   string
		earliestAt int64
	)
	for _, c := range claims {
		if c.Session == "" {
			continue
		}
		age := int64(-1)
		if c.CheckpointedAt > 0 {
			age = nowUnix - c.CheckpointedAt
			if age < 0 {
				age = 0 // a clock-skewed future stamp is "just now", never negative
			}
		}
		if !c.Live && (age < 0 || age > ttlSec) {
			continue // lapsed, or unstamped with no live oracle to vouch for it
		}
		fresh = append(fresh, c)
		if age >= 0 && (newestAge < 0 || age < newestAge) {
			newestAge = age
		}
		if c.CheckpointedAt > 0 && (earliest == "" || c.CheckpointedAt < earliestAt) {
			earliest, earliestAt = c.Session, c.CheckpointedAt
		}
	}

	o := Ownership{Path: path, ClaimAgeSeconds: newestAge, EarliestBy: earliest}
	switch {
	case len(fresh) == 0:
		// No fresh claim. Distinguish "never claimed" from "claim lapsed" so the
		// remedy differs: the first is unattributed creation, the second is an
		// overdue check-in by a named session.
		if lapsed, who, age := soleLapsed(claims, nowUnix); lapsed {
			o.State, o.Owner, o.Reason, o.ClaimAgeSeconds = OwnClaimedExpired, who, ReasonCheckinOverdue, age
			o.EarliestBy = who
			return o
		}
		o.State, o.Reason, o.EarliestBy = OwnUnclaimed, ReasonNoFreshClaim, ""
		return o
	case len(fresh) == 1:
		o.State, o.Owner = OwnClaimedLive, fresh[0].Session
		if fresh[0].Live {
			o.Reason = ReasonSessionLive
		} else {
			o.Reason = ReasonClaimFresh
		}
		return o
	default:
		o.State, o.Reason = OwnAmbiguous, ReasonTreeWideTie
		seen := make(map[string]bool, len(fresh))
		for _, c := range fresh {
			if !seen[c.Session] {
				seen[c.Session] = true
				o.Owners = append(o.Owners, c.Session)
			}
		}
		sort.Strings(o.Owners)
		if len(o.Owners) == 1 {
			// Several checkpoints, one session (a session that checkpointed twice) —
			// not a tie at all. Collapse to the single-owner verdict.
			o.State, o.Owner, o.Owners = OwnClaimedLive, o.Owners[0], nil
			o.Reason = ReasonClaimFresh
			for _, c := range fresh {
				if c.Live {
					o.Reason = ReasonSessionLive
					break
				}
			}
		}
		return o
	}
}

// soleLapsed reports whether every claim on a path belongs to ONE named session whose
// claims have all lapsed — the CLAIMED_EXPIRED case. Two different lapsed sessions stay
// UNCLAIMED: naming one of them as the overdue owner would be the same tree-wide-capture
// guess the fold refuses to make while claims are fresh. The returned age is that of the
// session's most recent claim (the last check-in).
func soleLapsed(claims []Claim, nowUnix int64) (bool, string, int64) {
	who, newest := "", int64(0)
	for _, c := range claims {
		if c.Session == "" || c.CheckpointedAt <= 0 {
			continue
		}
		if who == "" {
			who = c.Session
		} else if who != c.Session {
			return false, "", -1
		}
		if c.CheckpointedAt > newest {
			newest = c.CheckpointedAt
		}
	}
	if who == "" {
		return false, "", -1
	}
	age := nowUnix - newest
	if age < 0 {
		age = 0
	}
	return true, who, age
}

// OwnerReport is the whole-pass result: one verdict per input path plus the counts a
// caller (or an exit code) keys on.
type OwnerReport struct {
	Paths     []Ownership `json:"paths"`
	Unclaimed int         `json:"unclaimed"`
	Live      int         `json:"live"`
	Expired   int         `json:"expired"`
	Ambiguous int         `json:"ambiguous"`
	// ClaimTTLSeconds is the window the pass used, echoed so a report is
	// self-describing — a verdict is meaningless without the TTL that produced it.
	ClaimTTLSeconds int64 `json:"claim_ttl_seconds"`
}

// BuildOwnerReport folds every path's claims into a deterministic report. paths is the
// full set to grade (a path absent from claims is graded UNCLAIMED, not dropped — the
// totality guarantee); claims maps a path to the checkpoints that captured it. Output is
// sorted by path, so two runs over the same inputs are byte-identical.
func BuildOwnerReport(paths []string, claims map[string][]Claim, nowUnix int64, ttl time.Duration) OwnerReport {
	if ttl <= 0 {
		ttl = DefaultClaimTTL
	}
	rep := OwnerReport{ClaimTTLSeconds: int64(ttl / time.Second)}
	sorted := append([]string(nil), paths...)
	sort.Strings(sorted)
	rep.Paths = make([]Ownership, 0, len(sorted))
	for _, p := range sorted {
		o := OwnerOfPath(p, claims[p], nowUnix, ttl)
		switch o.State {
		case OwnUnclaimed:
			rep.Unclaimed++
		case OwnClaimedLive:
			rep.Live++
		case OwnClaimedExpired:
			rep.Expired++
		case OwnAmbiguous:
			rep.Ambiguous++
		}
		rep.Paths = append(rep.Paths, o)
	}
	return rep
}

// Unclaimed filters a report's verdicts to the at-risk set — the created paths no fresh
// checkpoint records. This is what a sweep-guard or a CI gate keys on.
func Unclaimed(rows []Ownership) []Ownership {
	out := make([]Ownership, 0)
	for _, o := range rows {
		if o.State == OwnUnclaimed {
			out = append(out, o)
		}
	}
	return out
}
