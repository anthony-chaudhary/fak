package agent

// compact_join.go — the per-fire event-join key and its 1:1 resolution (#2788). Parent epic #2783
// (cache-value netting/attribution); sibling of compact_receipt.go (#2787, the per-fire receipt).
//
// compact_receipt.go decomposed the session's opaque compaction aggregate into one receipt per
// fire, but its OBSERVED provider fields could only be stamped by a caller that happened to hold
// BOTH the CompactOutcome and the provider usage block in the same call frame. That works for the
// inline gateway path and nowhere else: a receipt journaled at fire time and a usage record
// journaled at response time land in two independently collected streams with no shared
// coordinate, so per-fire netting degrades back to the aggregate the receipts exist to decompose.
//
// This file supplies the shared coordinate: CompactJoinKey, the (turn sequence, monotonic ts)
// pair a fire shares with the provider usage record for the turn it affected. The turn sequence
// says WHICH request the fire rewrote; the monotonic timestamp disambiguates re-fires of the same
// turn (a retry compacts the same turn sequence again at a later monotonic reading) and is immune
// to wall-clock steps. Stamped on both halves at emission time, the key makes the join computable
// AFTER the fact: ResolveCompactJoin takes the two streams and correlates fire to usage 1:1 —
// a fire's WITNESSED shed lands beside the SAME turn's OBSERVED cache_read / cache_creation
// without a second ledger and without trusting either stream about the other.
//
// Like the receipt, this is the PURE primitive: no I/O, no clock (callers supply the monotonic
// reading they already hold), no upward import. The durable emission of stamped receipts and the
// gateway's live keying land via the leased internal/cachevaluereport/** lane; this file is the
// key type, the stamp, and the resolution invariant they build on.

// CompactJoinKey is the event-join key one compaction fire shares with the provider usage record
// for the turn it affected (#2788). Two coordinates, both caller-stamped at emission:
//
//   - TurnSeq: the 1-based sequence of the turn (request) the fire rewrote — the same counter a
//     gateway caller already threads into CompactOptions.CurrentTurn. It answers WHICH turn.
//   - MonotonicTSNano: a monotonic-clock reading (nanoseconds) taken when the fire was attempted.
//     It answers WHICH ATTEMPT when the same turn is compacted more than once (a retry re-fires
//     the same TurnSeq at a strictly later reading) and cannot be perturbed by wall-clock steps.
//     It is an ORDER anchor, not a wall-clock time — readers must not render it as a date.
//
// The zero key means UNSTAMPED: a byte-level caller with no turn context leaves it zero, and
// ResolveCompactJoin passes such receipts through unjoined rather than inventing a coordinate.
type CompactJoinKey struct {
	TurnSeq         uint64 `json:"turn_seq"`
	MonotonicTSNano int64  `json:"monotonic_ts_nano"`
}

// IsZero reports whether the key is unstamped — no turn coordinate was known at emission. An
// unstamped key is not an error: it is the honest state of a byte-level receipt, and the
// resolution counts it apart (Unstamped) instead of treating it as a failed join.
func (k CompactJoinKey) IsZero() bool {
	return k == CompactJoinKey{}
}

// WithJoinKey stamps the event-join key onto the receipt, returning the updated copy. A caller
// that knows the turn coordinate (the gateway, which holds CurrentTurn and a monotonic reading)
// calls this at fire time; a byte-level caller leaves the key zero. Value receiver, like
// WithObservedUsage: the original receipt is unchanged, and the stamp never touches the
// WITNESSED fields, so it cannot perturb the ReconcileShed invariant.
func (r CompactReceipt) WithJoinKey(k CompactJoinKey) CompactReceipt {
	r.JoinKey = k
	return r
}

// CompactTurnUsage is the provider-side half of the join: the OBSERVED cache_read /
// cache_creation one turn's provider response reported, keyed by the same CompactJoinKey the
// turn's fire receipt carries. The token fields are provider-relayed, never WITNESSED by fak —
// the resolution copies them onto the matched receipt's Observed* fields verbatim and asserts
// nothing about them.
type CompactTurnUsage struct {
	Key                 CompactJoinKey `json:"key"`
	CacheReadTokens     uint64         `json:"cache_read_tokens"`
	CacheCreationTokens uint64         `json:"cache_creation_tokens"`
}

// CompactJoinResolution is the outcome of one ResolveCompactJoin pass. Joined preserves the
// input receipts in order — matched ones returned with the OBSERVED usage stamped, the rest
// returned verbatim — so the WITNESSED shed sum (and therefore ReconcileShed) is invariant
// across resolution. The counters are the join-health verdict:
//
//   - Unstamped: receipts with a zero key (byte-level, no turn context). Not joinable, not an
//     error.
//   - Unmatched: receipts whose stamped key found NO usage record — a fire whose turn's usage
//     went unrecorded. Left unstamped rather than guessed.
//   - Ambiguous: receipts whose key appears more than once on EITHER side — the 1:1 guarantee
//     is broken for that key, so the resolution refuses to pick a winner and stamps nothing.
//
// A clean join is Unmatched == 0 && Ambiguous == 0: every stamped fire resolved to exactly one
// provider usage record.
type CompactJoinResolution struct {
	Joined    []CompactReceipt
	Unstamped int
	Unmatched int
	Ambiguous int
}

// ResolveCompactJoin correlates per-fire receipts with per-turn provider usage records by their
// shared CompactJoinKey, 1:1 — the #2788 resolution. For each receipt whose key matches exactly
// one usage record (and is itself carried by exactly one receipt), the record's OBSERVED
// cache_read / cache_creation are stamped onto the returned copy via WithObservedUsage. Any key
// duplicated on either side is refused as Ambiguous — a 1:1 join must not silently pick among
// candidates — and unstamped-key receipts pass through counted as Unstamped. Pure: no I/O, no
// mutation of either input slice's elements.
func ResolveCompactJoin(receipts []CompactReceipt, usage []CompactTurnUsage) CompactJoinResolution {
	// Count key multiplicity on both sides first: ambiguity is a property of the key, and every
	// receipt carrying an over-represented key must be refused, not just the second one seen.
	receiptKeyCount := make(map[CompactJoinKey]int, len(receipts))
	for _, r := range receipts {
		if !r.JoinKey.IsZero() {
			receiptKeyCount[r.JoinKey]++
		}
	}
	usageKeyCount := make(map[CompactJoinKey]int, len(usage))
	usageByKey := make(map[CompactJoinKey]CompactTurnUsage, len(usage))
	for _, u := range usage {
		if u.Key.IsZero() {
			// A zero-key usage record has no coordinate to join on; it can never match a stamped
			// receipt (stamped keys are nonzero by definition) so it is simply not indexed.
			continue
		}
		usageKeyCount[u.Key]++
		usageByKey[u.Key] = u
	}

	res := CompactJoinResolution{Joined: make([]CompactReceipt, 0, len(receipts))}
	for _, r := range receipts {
		switch {
		case r.JoinKey.IsZero():
			res.Unstamped++
			res.Joined = append(res.Joined, r)
		case receiptKeyCount[r.JoinKey] > 1 || usageKeyCount[r.JoinKey] > 1:
			res.Ambiguous++
			res.Joined = append(res.Joined, r)
		case usageKeyCount[r.JoinKey] == 0:
			res.Unmatched++
			res.Joined = append(res.Joined, r)
		default:
			u := usageByKey[r.JoinKey]
			res.Joined = append(res.Joined, r.WithObservedUsage(u.CacheReadTokens, u.CacheCreationTokens))
		}
	}
	return res
}
