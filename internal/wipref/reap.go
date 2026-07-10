package wipref

import "sort"

// This file is the concurrency + retention core of the WIP namespace (#3873). It
// stays PURE — no git I/O — so both decisions are unit-testable without a repo and
// the concurrent-writer invariant can be exercised under `-race`:
//
//   - Reconcile  decides which of two racing checkpoints a single ref converges to
//                (last-writer-wins), so the cmd shell's `git update-ref` OLD-VALUE
//                compare-and-swap loop never lets a STALE snapshot clobber a newer
//                one. It mirrors leaseref's fence CAS discipline (internal/leaseref
//                /fence.go) — read current, decide, swap on the old value, retry on
//                a lost CAS — without holding any lock across the git subprocess.
//   - Reap       folds every live ref against its owner's lifecycle state into a
//                delete/keep verdict over a CLOSED vocabulary, fail-safe by
//                construction: nothing that is not positively landed or cleanly
//                closed is ever deleted, so an unknown or live owner keeps its WIP.

// Reconcile picks the record a checkpoint ref should hold when a candidate write
// meets the value already there, under last-writer-wins ordering. It returns the
// winner and whether the candidate displaces the incumbent (changed == true means
// the caller should attempt the compare-and-swap; changed == false means the
// candidate lost — an equal or newer checkpoint is already anchored, so the caller
// backs off rather than overwrite it).
//
// "Last writer" is ordered by CheckpointedAt (the wall-clock instant the snapshot
// was captured), with a deterministic tie-break on the git object id so two racers
// at the same instant pick the SAME winner regardless of arrival order — a stable
// total order, never a coin flip. An empty incumbent (no ref yet) always yields to
// the candidate. Re-proposing the identical object is a no-op (changed == false).
func Reconcile(current, candidate RefRecord) (RefRecord, bool) {
	if current.Object == "" {
		return candidate, true
	}
	if candidateWins(current, candidate) {
		return candidate, true
	}
	return current, false
}

// candidateWins is Reconcile's strict total order: newer CheckpointedAt wins; on a
// tie the lexically-greater object id wins. Equality on both axes (the same object)
// is NOT a win, so an idempotent retry does not churn the ref.
func candidateWins(cur, cand RefRecord) bool {
	if cand.Stamp.CheckpointedAt != cur.Stamp.CheckpointedAt {
		return cand.Stamp.CheckpointedAt > cur.Stamp.CheckpointedAt
	}
	return cand.Object > cur.Object
}

// OwnerState is the CLOSED set of lifecycle facts about a checkpoint's owning
// session that decide whether its ref may be reaped. The cmd shell resolves each
// live ref to exactly one of these (from git + whatever session liveness it can
// observe); this fold turns them into delete/keep decisions. Every state that is
// not POSITIVELY reapable maps to keep, so the retention decision is fail-safe by
// construction — an unknown or live owner never loses its working-tree snapshot.
type OwnerState string

const (
	// OwnerLive: the session is still running. Its WIP is in active use — never reap.
	OwnerLive OwnerState = "LIVE"
	// OwnerLanded: the checkpoint's delta is already present in HEAD (the owner
	// committed it). The ref is now redundant — reap.
	OwnerLanded OwnerState = "LANDED"
	// OwnerClosedClean: the session ended and left no residual delta to recover — reap.
	OwnerClosedClean OwnerState = "CLOSED_CLEAN"
	// OwnerClosedDirty: the session ended but its delta was never landed. KEEP — this
	// is exactly the snapshot a later recover exists to restore.
	OwnerClosedDirty OwnerState = "CLOSED_DIRTY"
	// OwnerUnknown: no evidence either way. KEEP — the fail-safe default any
	// unresolved or unrecognised owner collapses to.
	OwnerUnknown OwnerState = "UNKNOWN"
)

// ReapAction is the closed decision vocabulary a ReapVerdict carries.
type ReapAction string

const (
	ReapDelete ReapAction = "DELETE"
	ReapKeep   ReapAction = "KEEP"
)

// ReapVerdict is one ref's reap decision plus the owner state and human reason
// behind it, so a `fak wip reap --json` snapshot is self-explaining and auditable.
type ReapVerdict struct {
	Session string     `json:"session"`
	Ref     string     `json:"ref"`
	Object  string     `json:"object"`
	Owner   OwnerState `json:"owner"`
	Action  ReapAction `json:"action"`
	Reason  string     `json:"reason"`
}

// ReapDecision maps one (record, owner-state) pair to a delete/keep verdict. Only
// OwnerLanded and OwnerClosedClean delete; every other state — including any
// unrecognised OwnerState value — keeps and is normalised to OwnerUnknown, so the
// function cannot be talked into deleting by an owner label it does not understand.
func ReapDecision(rec RefRecord, owner OwnerState) ReapVerdict {
	v := ReapVerdict{Session: sessionOf(rec), Ref: rec.Ref, Object: rec.Object, Owner: owner, Action: ReapKeep}
	switch owner {
	case OwnerLanded:
		v.Action, v.Reason = ReapDelete, "delta landed in HEAD"
	case OwnerClosedClean:
		v.Action, v.Reason = ReapDelete, "owner closed with no residual delta"
	case OwnerLive:
		v.Reason = "owner still live"
	case OwnerClosedDirty:
		v.Reason = "owner closed but delta unlanded — recoverable"
	default:
		v.Owner, v.Reason = OwnerUnknown, "owner state unknown — kept (fail-safe)"
	}
	return v
}

// Reap folds every live checkpoint record against its owner's state into a
// deterministic, session-sorted slice of verdicts. A record whose session is
// absent from owners is treated as OwnerUnknown (kept). The output order is stable
// (by session), so a before/after reap snapshot diffs cleanly.
func Reap(recs []RefRecord, owners map[string]OwnerState) []ReapVerdict {
	out := make([]ReapVerdict, 0, len(recs))
	for _, r := range recs {
		owner, ok := owners[sessionOf(r)]
		if !ok {
			owner = OwnerUnknown
		}
		out = append(out, ReapDecision(r, owner))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Session < out[j].Session })
	return out
}

// sessionOf recovers a record's session id from its stamp, falling back to the ref
// name when the stamp is missing — the same identity rule the status Fold uses.
func sessionOf(rec RefRecord) string {
	if rec.Stamp.SessionID != "" {
		return rec.Stamp.SessionID
	}
	return SessionFromRef(rec.Ref)
}
