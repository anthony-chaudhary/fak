// Package lanebeat decides whether a DOS lane lease may be REFRESHED — the
// writer-side half of the lane-lease heartbeat rung (#5864, epic #750).
//
// THE GAP THIS CLOSES. The kernel ships the beat writer (`dos lease-lane
// heartbeat` -> `dos.lane_lease.heartbeat`) and `_lease_is_dead` reads
// `heartbeat_at` as its PRIMARY liveness evidence, falling back to
// `acquired_at`. Nothing on this fleet ever wrote a HEARTBEAT: measured over
// C:\work\fak\.dos\lane-journal.jsonl at 3584 entries — 129 ACQUIRE / 98
// RELEASE / 94 REFUSE / 3263 ENFORCE / **0 HEARTBEAT**, and 0 of the 31
// structurally-live lease records carry a `heartbeat_at` field at all. So the
// primary rung degenerates to "older than 50 min since ACQUIRE" for every lease
// and the liveness oracle cannot tell a working holder from a crashed one.
//
// WHY THE DECISION IS A PURE PACKAGE. A heartbeat is the one fail-DANGEROUS
// lease op. Release only FORGETS a lease (-> reclaimable: the safe direction);
// a beat REVIVES one. A beat written on behalf of a holder that is already gone
// makes a dead lease look permanently alive — strictly worse than the silent
// status quo, because the TTL backstop that currently does all the work would
// stop firing too. That is the same defect shape as the gap itself with the
// sign flipped: state whose lifetime is decoupled from the condition it
// describes. So the authority to beat is a DECISION with a closed refusal
// vocabulary, separated from the I/O that performs it, and every rung below
// fails CLOSED (no beat) — a lane we cannot prove is being worked simply keeps
// today's TTL-only lifetime.
//
// WHAT AUTHORIZES A BEAT. Not the holder's own say-so — epic #750's whole point
// is that liveness must be read from an oracle, never self-reported. Decide
// takes the SUPERVISOR's structural readings of the process doing the work:
// whether its pid is in the process table (Alive), when its output stream last
// grew (LastOutputAt — bytes the OS recorded as a side effect of real work, not
// a field the agent writes), and when the supervisor itself started it
// (StartedAt). A beat is an attestation by the party that spawned the worker
// and can see it, which is exactly the evidence the kernel's own PID rung wants
// and cannot get: the pid recorded ON the lease is the ephemeral `dos lease-lane
// acquire` subprocess, which exits by design, so every healthy lease probes dead
// from the kernel's side.
//
// THE RESIDUAL, STATED HONESTLY. A beat extends a lease by at most one TTL past
// the last proof of progress. Today a lease lives one TTL past a stamp that
// means nothing (the acquire). The residual window is the SAME size either way;
// what changes is its anchor — from an event that proves nothing to the newest
// oracle-observed byte of work. MaxHold caps the total attestation so a hung
// -but-alive process cannot be attested indefinitely.
package lanebeat

import (
	"sort"
	"strings"
	"time"
)

// DefaultQuietAfter is how long a holder's output stream may stand still before
// this package stops attesting it. A worker that is alive but has emitted
// nothing for this long is not evidently PROGRESSING, and the ladder rung it
// would otherwise prove (alive-and-working) is not the one the evidence
// supports. Deliberately well UNDER the kernel's 50 min default lease TTL: a
// quiet holder falls back to exactly today's TTL-only lifetime, never to a
// shorter one, so this rung can only withhold an extension and never cause an
// eviction.
const DefaultQuietAfter = 15 * time.Minute

// Refusal / admission reasons. Closed vocabulary — a caller records the word,
// never free text, so "why was this lane not beaten" is countable.
const (
	// ReasonBeat is the sole admitting verdict.
	ReasonBeat = "BEAT"
	// ReasonNoLane: the holder is not bound to a lane, so there is nothing to
	// refresh. Never guess a lane from the lease side — that inverts the proof.
	ReasonNoLane = "NO_LANE"
	// ReasonHolderDead: the supervisor's process-table read says the worker is
	// gone. THE load-bearing rung. This is the one that keeps the beat from
	// outliving the work.
	ReasonHolderDead = "HOLDER_DEAD"
	// ReasonHolderQuiet: the pid is alive but its output has not grown within
	// QuietAfter. Alive is not progressing; withhold the extension.
	ReasonHolderQuiet = "HOLDER_QUIET"
	// ReasonHolderPastDeadline: the worker has outlived the budget it was
	// spawned under. A process still running past its own deadline is a hang the
	// supervisor is about to reap, not work in progress — attesting it would let
	// one wedged holder pin a lane forever.
	ReasonHolderPastDeadline = "HOLDER_PAST_DEADLINE"
	// ReasonNoLeaseOnLane: nothing live is held on the holder's lane. Nothing to
	// beat; a beat must never CREATE a lease.
	ReasonNoLeaseOnLane = "NO_LEASE_ON_LANE"
	// ReasonForeignHost: the lease was taken on another box. This host's process
	// table is not evidence about that box's processes, so we have no standing.
	ReasonForeignHost = "FOREIGN_HOST"
	// ReasonUnattributableHolder: the lease records no holder string. The kernel
	// itself refuses to beat such a lease (no owner can authenticate against an
	// absent holder); mirror that refusal here rather than discovering it as a
	// non-zero exit.
	ReasonUnattributableHolder = "UNATTRIBUTABLE_HOLDER"
	// ReasonLeasePredatesHolder: the newest live lease on this lane was acquired
	// BEFORE the worker started, so the worker cannot be the thing that took it.
	// Beating it would revive a stranger's orphan on the strength of an unrelated
	// live process — the false-revival this package exists to prevent.
	ReasonLeasePredatesHolder = "LEASE_PREDATES_HOLDER"
)

// Lease is one structurally-live DOS lane-lease record projected to the fields a
// beat decision needs. AcquiredAt is the record's own `acquired_at`; a zero
// value means the stamp was absent or unparseable, which is treated as
// unprovable rather than as "very old" (see Decide).
type Lease struct {
	Lane       string
	Holder     string
	HostID     string
	LoopTS     string
	AcquiredAt time.Time
}

// Holder is the SUPERVISOR's structural reading of the process doing the work.
// Every field is something fak observed from outside the worker; nothing here is
// reported by the worker about itself.
type Holder struct {
	// Lane the supervisor bound this worker to at spawn.
	Lane string
	// HostID of this box, matched against the lease record's host_id.
	HostID string
	// PID is carried for the caller's records; Alive is the verdict that counts.
	PID int
	// Alive is a process-table read taken at decision time.
	Alive bool
	// StartedAt is when the supervisor spawned the worker.
	StartedAt time.Time
	// LastOutputAt is when the worker's output stream last grew (a log mtime).
	// Zero means unknown, which falls back to StartedAt so a worker that has not
	// yet written anything is still inside its opening quiet window.
	LastOutputAt time.Time
	// MaxHold caps how long after StartedAt this worker may be attested at all.
	// Zero disables the rung (the caller has no budget to enforce).
	MaxHold time.Duration
	// QuietAfter overrides DefaultQuietAfter. Zero uses the default; negative
	// disables the progress rung (alive alone suffices) — an explicit opt-out a
	// caller with no output stream to watch can take.
	QuietAfter time.Duration
}

// Decision is the verdict plus the exact identity the beat must carry. Lane,
// Owner and LoopTS are copied from the MATCHED lease record, never from the
// request: the kernel credits a beat by (loop_ts, lane) identity and
// authenticates it against the recorded holder, so re-deriving any of the three
// would mint a different identity and fold as a no-op against the real lease.
type Decision struct {
	Beat   bool
	Lane   string
	Owner  string
	LoopTS string
	Reason string
}

// Decide answers whether the supervisor may refresh the lane lease held on
// behalf of h, given the structurally-live lease set and the current time.
//
// PURE. No clock, no process table, no I/O — every input is passed in, so the
// dangerous direction (a beat that revives a dead lease) is reachable in a test
// rather than only in production.
//
// The rungs run holder-evidence first and lease-identity second, on purpose: the
// question "is anything still working here" must be settled before the question
// "which record would we stamp", so no amount of lease-side detail can talk its
// way past a dead holder.
func Decide(h Holder, live []Lease, now time.Time) Decision {
	lane := strings.TrimSpace(h.Lane)
	if lane == "" {
		return Decision{Reason: ReasonNoLane}
	}

	// (1) The holder must be there. Nothing below can substitute for this.
	if !h.Alive {
		return Decision{Lane: lane, Reason: ReasonHolderDead}
	}

	// (2) The holder must still be inside the budget it was spawned under. An
	// alive process past its own deadline is a hang awaiting reap; attesting it
	// is how one wedged worker pins a lane indefinitely.
	if h.MaxHold > 0 && !h.StartedAt.IsZero() && now.Sub(h.StartedAt) > h.MaxHold {
		return Decision{Lane: lane, Reason: ReasonHolderPastDeadline}
	}

	// (3) The holder must be evidently PROGRESSING, not merely resident. The
	// witness is the output stream's growth — a side effect of real work that the
	// worker cannot fake by asserting it. Unknown output falls back to the spawn
	// time so a just-started worker is inside its opening window rather than
	// instantly quiet.
	quiet := h.QuietAfter
	if quiet == 0 {
		quiet = DefaultQuietAfter
	}
	if quiet > 0 {
		last := h.LastOutputAt
		if last.IsZero() {
			last = h.StartedAt
		}
		if !last.IsZero() && now.Sub(last) > quiet {
			return Decision{Lane: lane, Reason: ReasonHolderQuiet}
		}
	}

	// (4) Identity. Pick the NEWEST live lease on the lane — the same
	// last-wins rule the kernel's own heartbeat/release use when loop_ts is
	// omitted — so a same-lane sibling cannot be beaten by accident.
	match, ok := newestOnLane(live, lane)
	if !ok {
		return Decision{Lane: lane, Reason: ReasonNoLeaseOnLane}
	}
	if match.HostID != "" && h.HostID != "" && !strings.EqualFold(match.HostID, h.HostID) {
		return Decision{Lane: lane, Reason: ReasonForeignHost}
	}
	holder := strings.TrimSpace(match.Holder)
	if holder == "" {
		return Decision{Lane: lane, Reason: ReasonUnattributableHolder}
	}

	// (5) BINDING. The lease must be attributable to THIS worker, and the only
	// structural evidence available is order: a worker cannot have acquired a
	// lease that already existed when it started. Without this rung any live
	// process on a lane would license a beat on whatever orphan happens to sit
	// there — precisely the false-revival the kernel's holder authentication
	// guards against, re-introduced from the supervisor side.
	//
	// An unparseable/absent acquire stamp is UNPROVABLE, not old: it cannot
	// satisfy the binding, so it refuses here too.
	if match.AcquiredAt.IsZero() || (!h.StartedAt.IsZero() && match.AcquiredAt.Before(h.StartedAt)) {
		return Decision{Lane: lane, Reason: ReasonLeasePredatesHolder}
	}

	return Decision{
		Beat:   true,
		Lane:   match.Lane,
		Owner:  holder,
		LoopTS: match.LoopTS,
		Reason: ReasonBeat,
	}
}

// newestOnLane returns the live lease on lane with the newest AcquiredAt, ties
// broken by LoopTS so the pick is deterministic for records minted in the same
// second.
func newestOnLane(live []Lease, lane string) (Lease, bool) {
	var on []Lease
	for _, l := range live {
		if strings.TrimSpace(l.Lane) == lane {
			on = append(on, l)
		}
	}
	if len(on) == 0 {
		return Lease{}, false
	}
	sort.SliceStable(on, func(i, j int) bool {
		if !on[i].AcquiredAt.Equal(on[j].AcquiredAt) {
			return on[i].AcquiredAt.Before(on[j].AcquiredAt)
		}
		return on[i].LoopTS < on[j].LoopTS
	})
	return on[len(on)-1], true
}
