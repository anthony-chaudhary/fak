package leaseref

// liveness.go classifies each LIVE lock lease by the liveness of its OWNING SESSION
// (#2164): `self | peer-live | peer-dead | peer-unknown`. The problem it closes: a lease
// record's pid names the ACQUIRING process, which dies almost immediately, so a dead pid
// does NOT mean the lane is free — an agent reading pids either steals a live lane
// (collision) or over-conservatively skips a free one (lost parallelism). The session
// descriptor (session.go) is the signal that actually carries liveness: a live guard
// session republishes its ref on every PCB transition, refreshing UpdatedAt against a
// TTL — a heartbeat. Record.SessionID binds a lease to that descriptor, and this file
// folds the two views into one classification.
//
// THE FAIL-CLOSED RULE (load-bearing): a lease is reclaimable ONLY when its owning
// session is POSITIVELY dead — the descriptor exists and either its heartbeat lapsed
// (Expired) or it published the terminal STOPPED state. Absence of evidence is NOT
// death: session publishing is best-effort and fail-open (a publish failure never blocks
// the session), so a lease with no session binding, or a binding with no descriptor,
// classifies peer-unknown — never reclaimable. This is the same conservative posture as
// AcquireFenced's anonymous-live-holder refuse.
//
// THE HONEST BOUNDARY (kept in lockstep with the package doc): classification is a
// READ-SIDE projection over the converged ref namespace — VISIBILITY for an admission
// decision, not the admission itself. Reclaiming a peer-dead lane still goes through
// the fenced acquire (the TTL/generation rules are untouched); this view only tells the
// agent WHICH refusals are worth contesting and which lanes must never be stolen.

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// The closed lease-liveness vocabulary (#2164). String constants in the same shape as
// the fence's Reason* family; they are the --json contract a calling loop routes on.
const (
	// LivenessSelf: the lease's SessionID is the reading agent's own session — its own
	// lane, not a peer's.
	LivenessSelf = "self"
	// LivenessPeerLive: the owning session's descriptor is present and heartbeating
	// (not expired, not STOPPED). NEVER reclaimable — this is the lane-steal the
	// classification exists to prevent.
	LivenessPeerLive = "peer-live"
	// LivenessPeerDead: the owning session is POSITIVELY dead — its descriptor exists
	// and either the heartbeat lapsed past TTL or it published the terminal STOPPED
	// state. The only reclaimable class.
	LivenessPeerDead = "peer-dead"
	// LivenessPeerUnknown: no session binding (a legacy/unbound record) or no
	// descriptor for the bound session. Publishing is best-effort, so absence is not
	// proof of death — fails closed to not-reclaimable.
	LivenessPeerUnknown = "peer-unknown"
)

// sessionStateStopped is the terminal PCB run-state a session may publish before its
// descriptor is removed. A descriptor that still carries it is the session's own
// statement that it stopped — positive evidence of death even before the TTL lapses.
const sessionStateStopped = "STOPPED"

// ClassifiedLease is one live lock lease tagged with its session-liveness class. Record
// is embedded so the JSON row stays the familiar record shape plus the three
// classification fields — an operator or arbiter reads {id, tree_globs, holder,
// session_id, ..., liveness, reclaimable, evidence} in one object.
type ClassifiedLease struct {
	Record
	Liveness    string `json:"liveness"`
	Reclaimable bool   `json:"reclaimable"`
	Evidence    string `json:"evidence"`
}

// ClassifyLiveness is the pure classification rule: given one lease record, the session
// descriptors indexed by id, the reading agent's own session id (empty = anonymous
// reader, nothing classifies self), and now, it returns the liveness class and the
// evidence sentence naming exactly which comparison decided. It reads only its inputs —
// no I/O — so a test drives every branch with literal values.
func ClassifyLiveness(rec Record, sessions map[string]SessionDescriptor, selfSession string, now time.Time) (class, evidence string) {
	if rec.SessionID == "" {
		return LivenessPeerUnknown,
			"lease carries no session_id (legacy/unbound record); absence is not proof of death — not reclaimable"
	}
	if selfSession != "" && rec.SessionID == selfSession {
		return LivenessSelf, fmt.Sprintf("lease session_id %q is this session", rec.SessionID)
	}
	d, ok := sessions[rec.SessionID]
	if !ok {
		return LivenessPeerUnknown, fmt.Sprintf(
			"no session descriptor at refs/fak/locks/session-%s; publishing is best-effort, so absence is not proof of death — not reclaimable",
			rec.SessionID)
	}
	if strings.EqualFold(strings.TrimSpace(d.PCBState), sessionStateStopped) {
		return LivenessPeerDead, fmt.Sprintf(
			"owning session %s published terminal pcb_state=STOPPED (updated_at_unix=%d) — positively dead, reclaimable",
			rec.SessionID, d.UpdatedAt)
	}
	if d.Expired(now) {
		return LivenessPeerDead, fmt.Sprintf(
			"owning session %s stopped heartbeating: now_unix=%d >= updated_at_unix=%d + ttl_seconds=%d — positively dead, reclaimable",
			rec.SessionID, now.Unix(), d.UpdatedAt, d.TTLSecs)
	}
	return LivenessPeerLive, fmt.Sprintf(
		"owning session %s is heartbeating (pcb_state=%s, updated_at_unix=%d, ttl_seconds=%d) — never reclaimable",
		rec.SessionID, d.PCBState, d.UpdatedAt, d.TTLSecs)
}

// ClassifyLive folds the two ref views into the classified projection: the LIVE
// (non-expired) lock leases, each tagged by the liveness of its owning session per
// ClassifyLiveness. Expired leases are excluded — those are already reapable on the TTL
// rule alone; this view adds the signal TTL cannot give, the un-expired lease whose
// owner is provably gone (and its converse, the lane whose owner is heartbeating and
// must not be stolen). The slice is non-nil-and-empty when nothing is live, so a JSON
// encoder emits `[]`.
func (s *Store) ClassifyLive(ctx context.Context, selfSession string, now time.Time) ([]ClassifiedLease, error) {
	live, _, err := s.Live(ctx, now)
	if err != nil {
		return nil, err
	}
	sessions, err := s.ListSessions(ctx)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]SessionDescriptor, len(sessions))
	for _, d := range sessions {
		byID[d.ID] = d
	}
	out := make([]ClassifiedLease, 0, len(live))
	for _, r := range live {
		class, ev := ClassifyLiveness(r, byID, selfSession, now)
		out = append(out, ClassifiedLease{
			Record:      r,
			Liveness:    class,
			Reclaimable: class == LivenessPeerDead,
			Evidence:    ev,
		})
	}
	return out, nil
}
