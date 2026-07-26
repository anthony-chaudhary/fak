package agent

import (
	"encoding/json"
	"testing"
)

// TestCompactJoinKeyStamp proves the key is zero (unstamped) on a byte-level receipt, that
// WithJoinKey stamps it without perturbing the WITNESSED fields or mutating the receiver, and
// that the stamped key round-trips through the receipt's JSON shape.
func TestCompactJoinKeyStamp(t *testing.T) {
	r := NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 2, ShedTokens: 800})
	if !r.JoinKey.IsZero() {
		t.Fatalf("a byte-level receipt must leave the join key zero, got %+v", r.JoinKey)
	}

	key := CompactJoinKey{TurnSeq: 7, MonotonicTSNano: 123456789}
	stamped := r.WithJoinKey(key)
	if stamped.JoinKey != key || stamped.JoinKey.IsZero() {
		t.Fatalf("WithJoinKey = %+v, want key %+v", stamped.JoinKey, key)
	}
	if stamped.ShedTokens != 800 || stamped.DroppedTurns != 2 || !stamped.Fired {
		t.Fatalf("stamping the join key perturbed the WITNESSED fields: %+v", stamped)
	}
	// The original is unchanged (value receiver) — receipts are append-only, not mutated in place.
	if !r.JoinKey.IsZero() {
		t.Fatalf("WithJoinKey must not mutate the receiver, got %+v", r.JoinKey)
	}

	// The key survives the receipt's JSON round-trip — it is part of the durable row shape.
	raw, err := json.Marshal(stamped)
	if err != nil {
		t.Fatalf("marshal stamped receipt: %v", err)
	}
	var back CompactReceipt
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal stamped receipt: %v", err)
	}
	if back.JoinKey != key {
		t.Fatalf("join key did not round-trip: got %+v, want %+v", back.JoinKey, key)
	}
}

// TestCompactJoinKeyZero pins the unstamped semantics: only the all-zero key reads as zero. A
// half-stamped key (either coordinate set) is a real, joinable key — TurnSeq 0 with a monotonic
// reading is degenerate but must not be silently treated as "no coordinate".
func TestCompactJoinKeyZero(t *testing.T) {
	if !(CompactJoinKey{}).IsZero() {
		t.Fatal("zero-value key must read as unstamped")
	}
	if (CompactJoinKey{TurnSeq: 1}).IsZero() || (CompactJoinKey{MonotonicTSNano: 1}).IsZero() {
		t.Fatal("a key with either coordinate set is stamped, not zero")
	}
}

// TestResolveCompactJoinOneToOne is the happy path: every stamped fire finds exactly one usage
// record, the OBSERVED provider tokens land on the matched receipts, and the WITNESSED shed sum
// (ReconcileShed) is invariant across resolution.
func TestResolveCompactJoinOneToOne(t *testing.T) {
	k1 := CompactJoinKey{TurnSeq: 3, MonotonicTSNano: 100}
	k2 := CompactJoinKey{TurnSeq: 5, MonotonicTSNano: 250}
	receipts := []CompactReceipt{
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 2, ShedTokens: 700}).WithJoinKey(k1),
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 300}).WithJoinKey(k2),
	}
	usage := []CompactTurnUsage{
		{Key: k2, CacheReadTokens: 2048, CacheCreationTokens: 64},
		{Key: k1, CacheReadTokens: 4096, CacheCreationTokens: 128},
	}

	res := ResolveCompactJoin(receipts, usage)
	if res.Unstamped != 0 || res.Unmatched != 0 || res.Ambiguous != 0 {
		t.Fatalf("clean 1:1 join must have zero counters, got %+v", res)
	}
	if len(res.Joined) != 2 {
		t.Fatalf("Joined must preserve every input receipt, got %d", len(res.Joined))
	}
	if res.Joined[0].ObservedCacheReadTokens != 4096 || res.Joined[0].ObservedCacheCreationTokens != 128 {
		t.Fatalf("receipt[0] must carry k1's usage, got %+v", res.Joined[0])
	}
	if res.Joined[1].ObservedCacheReadTokens != 2048 || res.Joined[1].ObservedCacheCreationTokens != 64 {
		t.Fatalf("receipt[1] must carry k2's usage, got %+v", res.Joined[1])
	}
	// Resolution stamps OBSERVED fields only — the WITNESSED reconciliation is untouched.
	if !ReconcileShed(res.Joined, 1000) {
		t.Fatalf("joined receipts must still reconcile the aggregate shed, sum=%d", SumReceiptShed(res.Joined))
	}
	// Inputs are not mutated: the original receipts still carry zero OBSERVED usage.
	if receipts[0].ObservedCacheReadTokens != 0 || receipts[1].ObservedCacheReadTokens != 0 {
		t.Fatal("ResolveCompactJoin must not mutate the input receipts")
	}
}

// TestResolveCompactJoinUnstampedAndUnmatched proves the two non-error / error edges: a zero-key
// receipt passes through counted as Unstamped (byte-level, not joinable), and a stamped receipt
// whose turn has no usage record is counted Unmatched and left unstamped rather than guessed.
func TestResolveCompactJoinUnstampedAndUnmatched(t *testing.T) {
	stampedKey := CompactJoinKey{TurnSeq: 9, MonotonicTSNano: 400}
	receipts := []CompactReceipt{
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonUnderBudget}), // bail, unstamped
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 500}).WithJoinKey(stampedKey),
	}
	res := ResolveCompactJoin(receipts, nil)
	if res.Unstamped != 1 || res.Unmatched != 1 || res.Ambiguous != 0 {
		t.Fatalf("want 1 unstamped + 1 unmatched, got %+v", res)
	}
	if len(res.Joined) != 2 {
		t.Fatalf("Joined must preserve every input receipt, got %d", len(res.Joined))
	}
	for i, r := range res.Joined {
		if r.ObservedCacheReadTokens != 0 || r.ObservedCacheCreationTokens != 0 {
			t.Fatalf("receipt[%d] found no usage record and must stay unstamped, got %+v", i, r)
		}
	}
	// A zero-key usage record has no coordinate: it must never match a stamped receipt.
	res = ResolveCompactJoin(receipts, []CompactTurnUsage{{CacheReadTokens: 999}})
	if res.Unmatched != 1 || res.Joined[1].ObservedCacheReadTokens != 0 {
		t.Fatalf("zero-key usage must not join to anything, got %+v", res)
	}
}

// TestResolveCompactJoinAmbiguous proves the 1:1 refusal on both sides: a key carried by two
// usage records, or by two receipts, breaks the 1:1 guarantee, so every receipt involved is
// counted Ambiguous and stamped with nothing — the resolution never picks a winner.
func TestResolveCompactJoinAmbiguous(t *testing.T) {
	dup := CompactJoinKey{TurnSeq: 4, MonotonicTSNano: 100}

	// Duplicate on the usage side.
	receipts := []CompactReceipt{
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 100}).WithJoinKey(dup),
	}
	usage := []CompactTurnUsage{
		{Key: dup, CacheReadTokens: 1},
		{Key: dup, CacheReadTokens: 2},
	}
	res := ResolveCompactJoin(receipts, usage)
	if res.Ambiguous != 1 || res.Joined[0].ObservedCacheReadTokens != 0 {
		t.Fatalf("duplicate usage key must refuse the join, got %+v", res)
	}

	// Duplicate on the receipt side: BOTH carriers are ambiguous, not just the second one seen.
	receipts = []CompactReceipt{
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 100}).WithJoinKey(dup),
		NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 200}).WithJoinKey(dup),
	}
	res = ResolveCompactJoin(receipts, []CompactTurnUsage{{Key: dup, CacheReadTokens: 3}})
	if res.Ambiguous != 2 {
		t.Fatalf("every carrier of a duplicated receipt key is ambiguous, got %+v", res)
	}
	for i, r := range res.Joined {
		if r.ObservedCacheReadTokens != 0 {
			t.Fatalf("ambiguous receipt[%d] must not be stamped, got %+v", i, r)
		}
	}
}
