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
//
// THE DIAGNOSTIC CHANNEL (#5484/#5485, strictly one level ABOVE admission): the
// fail-closed rule is right, and neither addition below re-tunes it — a lease with no
// session_id, and a lease whose bound session has no descriptor, both still classify
// peer-unknown and not-reclaimable exactly as before. What they DID share is that the
// only thing naming which comparison decided was an English sentence, and for
// peer-unknown that sentence separates two causes with OPPOSITE remedies: an empty
// SessionID means the ACQUIRER never bound a session (fix the acquire call site —
// waiting never helps), while a missing descriptor means the acquirer did bind but the
// PUBLISHER is absent or died (fix/start `fak leaseref session-publish`). Different
// owners, different fixes, and a supervising loop that wants to REPAIR rather than
// merely avoid the lane had to pattern-match prose to tell them apart. EvidenceKind is
// the machine-routable companion to the sentence that closes that. SummarizeLiveness
// then counts the SAME kinds across the live set, because the per-lease posture has an
// aggregate blind spot: a fleet where nothing publishes the input this classification
// consumes returns a complete, well-formed array in which every row is an absence of
// evidence, and nothing in the output distinguishes that WIRING DEFECT IN THE OBSERVER
// (which silently invalidates every downstream verdict) from the genuine fleet state
// "these owners really are unclassifiable right now". Coverage is that missing field.

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

// The closed evidence-kind vocabulary (#5484) — one constant per `return` in
// ClassifyLiveness, in that function's own precedence order. Same shape as the Liveness*
// family above and equally part of the --json contract: `liveness` says WHAT the verdict
// is, `evidence_kind` says WHICH comparison produced it, and the Evidence sentence stays
// authoritative for humans. The pair is what makes a REMEDY selectable without parsing
// prose — most sharply for the two peer-unknown kinds, which route to different owners.
const (
	// EvidenceNoBinding: the lease carries no session_id at all, so there was nothing to
	// look up. The ACQUIRER never bound a session — remedy at the acquire call site
	// (`--session`). Waiting changes nothing, ever. Class: peer-unknown.
	EvidenceNoBinding = "no-session-binding"
	// EvidenceSelfSession: the lease's session_id equals the reading agent's own.
	// Class: self.
	EvidenceSelfSession = "self-session"
	// EvidenceNoDescriptor: the lease IS bound, but no descriptor exists for that session
	// id. The acquirer did its part; the PUBLISHER is missing or died before its first
	// publish — remedy is to start/repair `fak leaseref session-publish`. Publishing is
	// best-effort, so this is still not proof of death. Class: peer-unknown.
	EvidenceNoDescriptor = "no-descriptor"
	// EvidenceTerminalStopped: the owning session published the terminal pcb_state=STOPPED
	// — it announced its own exit. Class: peer-dead.
	EvidenceTerminalStopped = "terminal-stopped"
	// EvidenceHeartbeatLapsed: the owning session's descriptor exists but its UpdatedAt +
	// TTL is behind now — it stopped answering rather than announcing. A reaper may
	// reasonably log or rate-limit this differently from a clean STOPPED. Class: peer-dead.
	EvidenceHeartbeatLapsed = "heartbeat-lapsed"
	// EvidenceHeartbeating: the descriptor is present, un-expired and not STOPPED.
	// Class: peer-live.
	EvidenceHeartbeating = "heartbeating"
)

// evidenceKinds is the closed vocabulary in ClassifyLiveness's precedence order — the
// enumeration SummarizeLiveness zero-fills its histogram from, so `by_evidence_kind`
// always carries every kind and a reader never has to know whether a missing key means
// zero or means the field was not computed.
var evidenceKinds = []string{
	EvidenceNoBinding,
	EvidenceSelfSession,
	EvidenceNoDescriptor,
	EvidenceTerminalStopped,
	EvidenceHeartbeatLapsed,
	EvidenceHeartbeating,
}

// livenessClasses is the closed Liveness* vocabulary, likewise for `by_class`.
var livenessClasses = []string{
	LivenessSelf,
	LivenessPeerLive,
	LivenessPeerDead,
	LivenessPeerUnknown,
}

// EvidenceIsPositive splits the evidence vocabulary the way #5485 needs it: true when the
// kind rests on something the reader actually OBSERVED (a session id it matched, a
// descriptor it read), false when it rests on the ABSENCE of an input. Exactly the two
// peer-unknown kinds are absences, which is why this is a predicate over kinds and not a
// second parallel rule over classes — the class partition happens to agree today, but the
// kind is the thing that states the reason, so it is the honest key to count.
//
// Note what this does NOT mean: a positive kind is not a claim the verdict is actionable
// (EvidenceHeartbeating is positive and never reclaimable). It means the verdict was
// derived from a signal that was present.
func EvidenceIsPositive(kind string) bool {
	switch kind {
	case EvidenceSelfSession, EvidenceTerminalStopped, EvidenceHeartbeatLapsed, EvidenceHeartbeating:
		return true
	case EvidenceNoBinding, EvidenceNoDescriptor:
		return false
	default:
		// An unknown kind is not evidence of anything — the same fail-closed posture the
		// per-lease rule takes toward an absent descriptor.
		return false
	}
}

// sessionStateStopped is the terminal PCB run-state a session may publish before its
// descriptor is removed. A descriptor that still carries it is the session's own
// statement that it stopped — positive evidence of death even before the TTL lapses.
const sessionStateStopped = "STOPPED"

// ClassifiedLease is one live lock lease tagged with its session-liveness class. Record
// is embedded so the JSON row stays the familiar record shape plus the four
// classification fields — an operator or arbiter reads {id, tree_globs, holder,
// session_id, ..., liveness, reclaimable, evidence, evidence_kind} in one object.
type ClassifiedLease struct {
	Record
	// Node is the stable machine id from the holder's <node-id>/<session-id>
	// convention (#2304) — the "WHICH MACHINE holds this?" component. A legacy
	// free-form holder classifies NodeUnknown, never an error (nodeident.go).
	Node        string `json:"node"`
	Liveness    string `json:"liveness"`
	Reclaimable bool   `json:"reclaimable"`
	Evidence    string `json:"evidence"`
	// EvidenceKind is the machine-routable companion to Evidence: which comparison
	// decided (the closed Evidence* vocabulary). Strictly additive — `liveness`,
	// `reclaimable` and the sentences are byte-identical to before it existed, so a
	// consumer routing on today's contract is untouched.
	EvidenceKind string `json:"evidence_kind"`
}

// ClassifyLiveness is the pure classification rule: given one lease record, the session
// descriptors indexed by id, the reading agent's own session id (empty = anonymous
// reader, nothing classifies self), and now, it returns the liveness class, the typed
// evidence kind, and the evidence sentence — the last two naming, machine-readably and
// then in English, exactly which comparison decided. It reads only its inputs — no I/O —
// so a test drives every branch with literal values.
//
// The kind is a third RESULT rather than a wrapper or a struct (the shape #5484 left
// open) so there stays exactly one place a branch's verdict is stated: a parallel
// classifier would be a second rule free to drift from this one, which is precisely the
// hazard CascadeDrop already routes through here to avoid. class and evidence keep their
// old values byte for byte; only the arity changed.
func ClassifyLiveness(rec Record, sessions map[string]SessionDescriptor, selfSession string, now time.Time) (class, kind, evidence string) {
	if rec.SessionID == "" {
		return LivenessPeerUnknown, EvidenceNoBinding,
			"lease carries no session_id (legacy/unbound record); absence is not proof of death — not reclaimable"
	}
	if selfSession != "" && rec.SessionID == selfSession {
		return LivenessSelf, EvidenceSelfSession, fmt.Sprintf("lease session_id %q is this session", rec.SessionID)
	}
	d, ok := sessions[rec.SessionID]
	if !ok {
		return LivenessPeerUnknown, EvidenceNoDescriptor, fmt.Sprintf(
			"no session descriptor at refs/fak/locks/session-%s; publishing is best-effort, so absence is not proof of death — not reclaimable",
			rec.SessionID)
	}
	if strings.EqualFold(strings.TrimSpace(d.PCBState), sessionStateStopped) {
		return LivenessPeerDead, EvidenceTerminalStopped, fmt.Sprintf(
			"owning session %s published terminal pcb_state=STOPPED (updated_at_unix=%d) — positively dead, reclaimable",
			rec.SessionID, d.UpdatedAt)
	}
	if d.Expired(now) {
		return LivenessPeerDead, EvidenceHeartbeatLapsed, fmt.Sprintf(
			"owning session %s stopped heartbeating: now_unix=%d >= updated_at_unix=%d + ttl_seconds=%d — positively dead, reclaimable",
			rec.SessionID, now.Unix(), d.UpdatedAt, d.TTLSecs)
	}
	return LivenessPeerLive, EvidenceHeartbeating, fmt.Sprintf(
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
	byID, err := s.sessionsByID(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ClassifiedLease, 0, len(live))
	for _, r := range live {
		class, kind, ev := ClassifyLiveness(r, byID, selfSession, now)
		out = append(out, ClassifiedLease{
			Record:       r,
			Node:         r.HolderNode(),
			Liveness:     class,
			Reclaimable:  class == LivenessPeerDead,
			Evidence:     ev,
			EvidenceKind: kind,
		})
	}
	return out, nil
}

// LivenessSummary is the aggregate companion to a ClassifyLive array (#5485): how much of
// the live set rests on evidence that was actually PRESENT. The per-lease rule is careful
// that absence of evidence never reads as death; without this, absence of evidence across
// the WHOLE set reads as a clean answer, because a fleet in which nothing publishes the
// input still yields a complete, well-formed array of correctly-computed rows.
type LivenessSummary struct {
	// Total is the number of live leases classified — the denominator. Read it FIRST:
	// see Coverage on why zero is not a finding.
	Total int `json:"total"`
	// ByClass and ByEvidenceKind are histograms over the two closed vocabularies, both
	// zero-filled across every member, so a missing key never has to be disambiguated
	// from a zero count.
	ByClass        map[string]int `json:"by_class"`
	ByEvidenceKind map[string]int `json:"by_evidence_kind"`
	// PositiveEvidence is how many rows have an EvidenceIsPositive kind — exposed raw
	// beside the ratio so a caller can do its own arithmetic (and so Total==0 stays
	// visibly 0-of-0 rather than hiding behind a float).
	PositiveEvidence int `json:"positive_evidence"`
	// Coverage is PositiveEvidence/Total: the fraction of the live set whose class rests
	// on an observed input rather than on an absence.
	//
	// HOW TO READ IT. Coverage near 1.0 means the liveness feed is working. A Coverage of
	// exactly 0.0 with Total > 0 is the one that matters: every holder in the fleet is
	// peer-unknown, which is far more often a WIRING DEFECT IN THE OBSERVER — nothing on
	// the write path publishes what this classification consumes — than a fleet that
	// genuinely went unclassifiable all at once. ByEvidenceKind then names the remedy:
	// all EvidenceNoBinding means acquirers are not passing --session, all
	// EvidenceNoDescriptor means `fak leaseref session-publish` is down.
	//
	// Total == 0 reports Coverage 0.0, and that is NOT the wiring signal — an empty live
	// set has nothing to have evidence about, so the ratio is undefined and reported as
	// the zero value. Any caller gating on a coverage FLOOR must require Total > 0 first.
	Coverage float64 `json:"liveness_coverage"`
}

// SummarizeLiveness folds a classified projection into its aggregate. It is a pure
// function over the rows ClassifyLive already returned — not a second read of the ref
// namespace and not a second classification rule: it counts the EvidenceKind each row was
// decided by, so the aggregate can never disagree with the per-lease verdicts it
// summarizes. A nil or empty slice yields a well-formed zero summary with both histograms
// present and fully zero-filled.
func SummarizeLiveness(rows []ClassifiedLease) LivenessSummary {
	s := LivenessSummary{
		Total:          len(rows),
		ByClass:        make(map[string]int, len(livenessClasses)),
		ByEvidenceKind: make(map[string]int, len(evidenceKinds)),
	}
	for _, c := range livenessClasses {
		s.ByClass[c] = 0
	}
	for _, k := range evidenceKinds {
		s.ByEvidenceKind[k] = 0
	}
	for _, r := range rows {
		s.ByClass[r.Liveness]++
		s.ByEvidenceKind[r.EvidenceKind]++
		if EvidenceIsPositive(r.EvidenceKind) {
			s.PositiveEvidence++
		}
	}
	if s.Total > 0 {
		s.Coverage = float64(s.PositiveEvidence) / float64(s.Total)
	}
	return s
}
