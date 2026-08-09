package leaseref

// cascade.go makes a worker's registrations LEASE-SCOPED TO ITS IDENTITY (#3372): the moment
// a worker's liveness lease — its session descriptor at refs/fak/locks/session-<id> — is
// provably gone, every lock lease bound to it (Record.SessionID) drops out of the live view
// in ONE step, emitting a RemovalEvent per registration. No heartbeat table to consult, no
// reaper pass to wait for.
//
// THE GAP IT CLOSES. Live and LiveLeases partitioned on Record.Expired alone — the lease's
// OWN TTL, computed from AcquiredAt/RenewedAt and nothing else. So when a worker
// disconnected, its registrations kept refusing a cross-machine peer for the whole remaining
// lease TTL, and with TTLSeconds == 0 ("no expiry") they kept refusing it FOREVER: expired()
// short-circuits false, so such a record is never in Live's expired set and Reap — which
// only deletes what Live called expired — can never collect it. The session-liveness signal
// that knows the worker is dead already existed (liveness.go) but only ADVISED: ClassifyLive
// tagged the lease peer-dead/reclaimable while LiveLeases went on advertising it as live.
// This file folds that signal into the read views the arbiter actually consumes.
//
// KEYED ON POSITIVE DEATH, NOT ON ABSENCE (the load-bearing safety rule, inherited verbatim
// from liveness.go). etcd can cascade on a dropped lease because the substrate OWNS the
// connection; a git ref namespace does not. Here a missing session descriptor means "no
// evidence", and there are two ordinary ways to get one on a LIVE worker: PublishSession is
// documented best-effort and FAIL-OPEN (a publish failure never blocks the session), and the
// ref namespace converges across clones only as a SET — machine B may hold A's lease ref
// while A's session ref has simply not been fetched yet. So a cascade drop fires ONLY when
// the descriptor is PRESENT and positively dead: its heartbeat lapsed past TTL, or it
// published the terminal STOPPED state. An absent descriptor classifies peer-unknown and the
// registration is RETAINED. Treating absence as death would let a peer steal a live worker's
// lane — the precise collision this package exists to make visible.
//
// REMOVAL IS A READ-SIDE DROP, NOT A REF DELETE (and why). The registrations vanish from
// LiveLeases / LiveRegistrations without any `update-ref -d`. A converging delete keyed on a
// LOCALLY-read descriptor would be irreversible and fleet-wide: a clone whose fetch of the
// session ref lagged its fetch of the lease ref, or whose clock skewed past a short heartbeat
// TTL, would delete a LIVE peer's lease — destroying its Generation, so the peer's next Fence
// reads NO_LEASE and it halts-and-reacquires on a false death. Dropping from the view is
// self-correcting (the next scan re-reads the refs); deleting is not. Reclaiming a dropped
// lane still goes through AcquireFenced, whose monotonic generation bump is what actually
// protects against the paused-then-resumed holder. TTL-driven Reap (reap.go) remains the only
// deleter of lease refs.
//
// THE HONEST BOUNDARY, unchanged: this is VISIBILITY. The cascade tells an arbiter which
// lanes a dead worker no longer holds; it does not arbitrate a same-fetch-window race, and it
// does not by itself admit anyone.

import (
	"context"
	"time"
)

// The closed removal-reason vocabulary — the `--json` contract a calling loop routes on, in
// the same shape as the fence's Reason* and liveness's Liveness* families. Both reasons are
// POSITIVE death: there is deliberately no reason for a missing descriptor, because absence
// never cascades.
const (
	// RemovalSessionExpired: the owning session's descriptor is present but its heartbeat
	// lapsed past TTL — the worker stopped renewing, so it is gone.
	RemovalSessionExpired = "session-expired"
	// RemovalSessionStopped: the owning session published the terminal PCB state STOPPED —
	// the worker's own statement that it ended, positive evidence before the TTL lapses.
	RemovalSessionStopped = "session-stopped"
)

// RemovalEvent is the substrate's "this registration went away, and here is exactly why"
// record — the removal notification etcd emits on a lease drop, rendered for a ref store.
// One event per cascaded registration, carrying the lease it removed, the worker identity
// whose death removed it, the closed reason, the evidence sentence naming which comparison
// decided, and the read instant the decision was taken at (so a consumer can tell a fresh
// removal from a replayed one).
type RemovalEvent struct {
	LeaseID   string `json:"lease_id"`   // the registration that vanished (ref basename under refs/fak/locks/)
	SessionID string `json:"session_id"` // the owning session whose death cascaded it
	Node      string `json:"node"`       // the machine the dead holder ran on (NodeUnknown for a legacy holder)
	Reason    string `json:"reason"`     // session-expired | session-stopped
	Evidence  string `json:"evidence"`   // the sentence naming the comparison that proved death
	AtUnix    int64  `json:"at_unix"`    // the read instant this removal was decided at
}

// CascadeDrop is the pure cascade rule: given one registration, the session descriptors
// indexed by id, and now, it reports whether the registration's owning worker is provably
// gone — and, if so, the RemovalEvent that records it. It reads only its inputs (no I/O), so
// a test drives every branch with literal values.
//
// It delegates the decision to ClassifyLiveness with an ANONYMOUS reader (no self session):
// a registration cascades exactly when that rule says LivenessPeerDead. Routing through the
// one existing classifier is deliberate — the fail-closed rule (an unbound record, or a
// binding with no descriptor, is peer-unknown and never reclaimable) is then enforced BY
// CONSTRUCTION here rather than re-derived and liable to drift. Passing "" for selfSession
// is the right posture even for the reader's own lease: if MY session's heartbeat lapsed, my
// registrations are dead too, and the arbiter must not be told otherwise.
func CascadeDrop(rec Record, sessions map[string]SessionDescriptor, now time.Time) (RemovalEvent, bool) {
	class, kind, evidence := ClassifyLiveness(rec, sessions, "", now)
	if class != LivenessPeerDead {
		return RemovalEvent{}, false
	}
	// The reason reads straight off the typed evidence kind (#5484). It used to re-derive
	// the STOPPED test from the descriptor and rely on a comment to keep that in step with
	// ClassifyLiveness's own precedence (terminal state before heartbeat lapse); routing
	// through the kind is the same by-construction argument the delegation above makes, one
	// field deeper. Exactly two kinds yield peer-dead, so this is total.
	reason := RemovalSessionExpired
	if kind == EvidenceTerminalStopped {
		reason = RemovalSessionStopped
	}
	return RemovalEvent{
		LeaseID:   rec.ID,
		SessionID: rec.SessionID,
		Node:      rec.HolderNode(),
		Reason:    reason,
		Evidence:  evidence,
		AtUnix:    now.Unix(),
	}, true
}

// LiveRegistrations is the cascaded projection of the lock-lease namespace: the registrations
// still genuinely held at now, plus one RemovalEvent for each that cascaded out because its
// owning worker is provably gone. It is the read LiveLeases is built on, exposed on its own
// so a caller that wants the REMOVAL side (a fleet monitor, an operator view) gets the events
// rather than just the survivors.
//
// Two filters compose, in order: the own-TTL rule (Live drops a lapsed lease — it is already
// reapable on TTL alone, so it is not a cascade and emits no event), then the session
// cascade. Registrations are id-sorted by List, so the events are deterministic. Neither
// slice is nil, so a JSON encoder emits `[]`.
func (s *Store) LiveRegistrations(ctx context.Context, now time.Time) (live []Record, removed []RemovalEvent, err error) {
	byTTL, _, err := s.Live(ctx, now)
	if err != nil {
		return nil, nil, err
	}
	sessions, err := s.sessionsByID(ctx)
	if err != nil {
		return nil, nil, err
	}
	live = make([]Record, 0, len(byTTL))
	removed = make([]RemovalEvent, 0)
	for _, r := range byTTL {
		if ev, dropped := CascadeDrop(r, sessions, now); dropped {
			removed = append(removed, ev)
			continue
		}
		live = append(live, r)
	}
	return live, removed, nil
}

// sessionsByID reads the session descriptors into the id-keyed map both the cascade rule and
// the liveness classification fold a lease against. One place to change if the session view
// ever gains a filter.
func (s *Store) sessionsByID(ctx context.Context) (map[string]SessionDescriptor, error) {
	sessions, err := s.ListSessions(ctx)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]SessionDescriptor, len(sessions))
	for _, d := range sessions {
		byID[d.ID] = d
	}
	return byID, nil
}
