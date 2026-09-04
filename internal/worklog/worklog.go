// Package worklog is the unified agent-work change feed (#3172): the "outbox
// insight" applied to agent work. Instead of leaving the high-value facts about
// a unit of agent work scattered across separate signals — a commit in git, a
// diff-witnessed verdict from the audit path, the lease epoch it ran under, a
// later verdict flip — worklog folds them into ONE append-only, cursor-drained
// feed a consumer tails by offset, exactly like the coherence bus
// (internal/gateway/coherence.go) does for cache mutations.
//
// It is a change SOURCE for agent work, in CDC terms (see
// docs/explainers/change-data-capture-for-agents.md): a bounded, Seq-ordered,
// principal-scoped log whose delete-shaped events (a verdict flip that refutes a
// prior "done") behave like tombstones a consumer must apply.
//
// Scope fence (this is an epic, #3172): this package ships the RECORD and the
// FEED primitive with its drain/idempotency/retention contract and a fold
// read-model. It does NOT yet wire the producers — the commit hook, the
// dos_commit_audit verdict, and the lease manager that would call Append are
// left for the sessionledger integration (#2392). Nothing here reaches into the
// gateway or adjudicator, so the primitive is testable in isolation.
package worklog

import "sync"

// Kind enumerates the change kinds carried on the unified feed. The set is
// closed: a producer that needs a new kind adds it here so consumers can switch
// exhaustively.
const (
	// KindCommit is a committed unit of agent work: a SHA, its claim (the commit
	// subject), and the diff-witnessed verdict that graded whether the diff did
	// the KIND of thing the claim asserts. This is a create — a new work fact.
	KindCommit = "commit"

	// KindVerdictFlip is a tombstone-shaped change: a verdict that previously read
	// OK now reads otherwise (or vice-versa) for an already-published SHA — the
	// "done" claim was refuted or restored. A consumer applies it like a CDC
	// delete/upsert against its own view of that SHA.
	KindVerdictFlip = "verdict_flip"
)

// WorkChange is one high-value agent-work change on the unified cursor feed. Seq
// is the drain cursor (stamped by the feed on Append, not by the producer).
type WorkChange struct {
	Seq  uint64 `json:"seq"`  // shared feed sequence — the cursor; assigned by Append
	Kind string `json:"kind"` // KindCommit | KindVerdictFlip

	SHA        string `json:"sha"`                   // the committed unit of work
	Claim      string `json:"claim,omitempty"`       // the commit subject / claim being graded
	Lane       string `json:"lane,omitempty"`        // the lane the work touched
	LeaseEpoch uint64 `json:"lease_epoch,omitempty"` // the lease epoch the work ran under

	// Verdict is the dos_commit_audit rung for this SHA — OK | CLAIM_UNWITNESSED |
	// ABSTAIN — and Witness is how it was reached — diff-witnessed | subject-only |
	// abstain. Together they say whether the "done" is evidence-backed or a bare
	// self-report, without the consumer re-running the audit.
	Verdict string `json:"verdict,omitempty"`
	Witness string `json:"witness,omitempty"`

	// PrevVerdict is set only on a KindVerdictFlip: the verdict this change
	// supersedes, so a consumer can detect a "was OK, now unwitnessed" regression.
	PrevVerdict string `json:"prev_verdict,omitempty"`

	// Principal scopes visibility exactly like the coherence bus: a tenant drains
	// its own changes plus principal-less global broadcasts, never a peer's.
	Principal string `json:"principal,omitempty"`
}

// key is the idempotency identity of a change: a commit is keyed by its SHA (a
// re-appended commit is a no-op), a verdict flip by SHA+verdict (the same flip
// replayed is a no-op, but a genuinely new verdict for that SHA is a new event).
func (c WorkChange) key() string {
	if c.Kind == KindVerdictFlip {
		return c.Kind + ":" + c.SHA + ":" + c.Verdict
	}
	return c.Kind + ":" + c.SHA
}

// Feed is a bounded, append-only, cursor-drained log of WorkChanges with the same
// consumer contract as the coherence bus: at-least-once delivery by offset,
// idempotency by change key, and bounded retention (a consumer that falls behind
// the retained window sees a Seq gap and re-syncs to head). The zero value is not
// usable; construct with NewFeed.
//
// Invariant: worklog appending is fail-closed and monotonic; sequence cursors advance strictly.
// Contract: replaying an existing change key dedupes deterministically without publishing duplicate events.
// Guard: tenant-scoped drain strictly prevents cross-tenant visibility leaks across isolated principals.
type Feed struct {
	mu   sync.Mutex
	seq  uint64            // highest Seq ever assigned (monotonic, survives ring eviction)
	ring []WorkChange      // retained window, ascending Seq
	seen map[string]uint64 // idempotency: change key -> the Seq it first got
	cap  int               // max retained entries; <=0 means unbounded
}

// NewFeed constructs a Feed retaining at most capacity entries (<=0 = unbounded).
func NewFeed(capacity int) *Feed {
	return &Feed{seen: make(map[string]uint64), cap: capacity}
}

// Append stamps a change with the next Seq and appends it, returning the stamped
// change and true. If a change with the same key() was already appended, Append
// is a no-op returning the previously-stamped change and false — so a producer
// that replays a commit or a flip cannot double-publish it. Retention is applied
// after the append: the oldest entries beyond cap are evicted from the ring, and
// the idempotency map is pruned to a wider retention window (pruneSeen), so a late
// replay of any change still within that window dedupes while seen stays bounded.
func (f *Feed) Append(c WorkChange) (WorkChange, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()

	k := c.key()
	if prevSeq, ok := f.seen[k]; ok {
		// Already published — return the retained copy if still in the ring, else
		// just the Seq (the payload may have aged out of the window).
		for _, r := range f.ring {
			if r.Seq == prevSeq {
				return r, false
			}
		}
		return WorkChange{Seq: prevSeq, Kind: c.Kind, SHA: c.SHA}, false
	}

	f.seq++
	c.Seq = f.seq
	f.seen[k] = c.Seq
	f.ring = append(f.ring, c)
	if f.cap > 0 && len(f.ring) > f.cap {
		f.ring = f.ring[len(f.ring)-f.cap:]
		f.pruneSeen()
	}
	return c, true
}

// seenRetentionFactor sets how much longer the idempotency map retains a key than
// the payload ring: seen keys survive for seenRetentionFactor× the ring window, so
// a late replay of a change whose PAYLOAD has aged out of the ring still dedupes,
// while the map cannot grow without bound alongside a long-lived feed.
const seenRetentionFactor = 8

// pruneSeen bounds the idempotency map when retention is on (f.cap > 0). Called
// under the lock after the ring is trimmed. seen keys are kept for
// seenRetentionFactor× the ring window; keys older than that (Seq far enough below
// head that no in-window replay can still reference them) are dropped in a batch.
// Every key whose Seq is still in the ring is always retained — the cutoff sits
// well below the ring's oldest Seq — so dedup of a replay of any change still in
// the window is never weakened; only replays of very old, long-evicted changes may
// lapse, which is the same bounded-retention tradeoff the ring itself makes.
func (f *Feed) pruneSeen() {
	seenCap := f.cap * seenRetentionFactor
	// Prune with a full-ring of slack so the sweep is amortized O(1) per Append:
	// let seen grow one ring beyond seenCap before compacting it back down.
	if len(f.seen) <= seenCap+f.cap {
		return
	}
	// Guarded by the length check above: len(seen) distinct keys means seq has
	// advanced past seenCap+cap, so this subtraction cannot underflow.
	cutoff := f.seq - uint64(seenCap)
	for k, s := range f.seen {
		if s <= cutoff {
			delete(f.seen, k)
		}
	}
}

// Drain returns every retained change with Seq > sinceSeq that is VISIBLE to the
// requesting principal, plus the highest Seq the feed has ever assigned (the next
// cursor). Visibility matches the coherence bus: an empty principal (single-tenant
// / admin) sees everything; a tenant sees its own changes plus principal-less
// global broadcasts. sinceSeq==0 drains all retained.
//
// The returned cursor is the feed head, NOT the max Seq in the returned slice —
// so a consumer that advances to it and finds the next Drain empty knows it is
// caught up, and a consumer whose sinceSeq is below the retained window sees the
// gap (returned[0].Seq > sinceSeq+1) and must re-sync.
func (f *Feed) Drain(principal string, sinceSeq uint64) ([]WorkChange, uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	out := make([]WorkChange, 0, len(f.ring))
	for _, c := range f.ring {
		if c.Seq > sinceSeq && visibleTo(c, principal) {
			out = append(out, c)
		}
	}
	return out, f.seq
}

// visibleTo reports whether a draining principal may see a change. An empty
// drainer principal (single-tenant / admin) sees everything; a tenant sees
// principal-less (global) changes and its own.
func visibleTo(c WorkChange, principal string) bool {
	return principal == "" || c.Principal == "" || c.Principal == principal
}

// FoldLatestVerdict is a CQRS read-model over the feed: it replays changes in Seq
// order and returns the latest verdict per SHA. This is how a consumer answers
// "what is the current standing of this committed work?" from the log alone —
// a verdict flip supersedes the earlier commit verdict, so a SHA that was OK and
// later refuted folds to its refuted verdict.
func FoldLatestVerdict(changes []WorkChange) map[string]string {
	out := make(map[string]string, len(changes))
	for _, c := range changes {
		if c.Verdict != "" {
			out[c.SHA] = c.Verdict
		}
	}
	return out
}
