package model

import "testing"

// TestCreditLedgerReserveRefusedPastGrant proves the core backpressure property:
// a reserve that would exceed the granted cumulative ceiling is refused and
// leaves the ledger untouched, while a reserve that fits succeeds.
func TestCreditLedgerReserveRefusedPastGrant(t *testing.T) {
	l := NewCreditLedger()
	key := CreditKey{Receiver: "decode0", Sender: "prefill0", QoS: "gold"}

	if !l.Grant(key, 1, 1, 10) {
		t.Fatalf("first grant should advance the ceiling")
	}
	if got := l.Available(key); got != 10 {
		t.Fatalf("Available after grant=10: got %d", got)
	}
	// A reserve past the ceiling is refused and reserves nothing.
	if l.TryReserve(key, 11) {
		t.Fatalf("reserve of 11 past a grant of 10 must be refused")
	}
	if got := l.Available(key); got != 10 {
		t.Fatalf("refused reserve must not change Available: got %d", got)
	}
	// A reserve that fits succeeds and lowers Available.
	if !l.TryReserve(key, 6) {
		t.Fatalf("reserve of 6 within a grant of 10 must succeed")
	}
	if got := l.Available(key); got != 4 {
		t.Fatalf("Available after reserving 6 of 10: got %d", got)
	}
	// The remaining 4 fit; 5 does not.
	if l.TryReserve(key, 5) {
		t.Fatalf("reserve of 5 with 4 available must be refused")
	}
	if !l.TryReserve(key, 4) {
		t.Fatalf("reserve of the exact remaining 4 must succeed")
	}
	if got := l.Available(key); got != 0 {
		t.Fatalf("Available after exhausting the grant: got %d", got)
	}
}

// TestCreditGrantMonotonicAndDeduped proves grants are idempotent under replay:
// a stale/lower cumulative grant and a replayed (epoch,seq) are ignored, and only
// a strictly-newer, higher cumulative grant raises the ceiling.
func TestCreditGrantMonotonicAndDeduped(t *testing.T) {
	l := NewCreditLedger()
	key := CreditKey{Receiver: "decode1", Sender: "prefill1", QoS: "silver"}

	if !l.Grant(key, 2, 5, 100) {
		t.Fatalf("initial grant should apply")
	}
	// Replay of the same (epoch,seq) is a no-op even with a higher cumulative.
	if l.Grant(key, 2, 5, 999) {
		t.Fatalf("replayed (epoch,seq) must be ignored")
	}
	if got := l.Available(key); got != 100 {
		t.Fatalf("replayed grant must not change the ceiling: got %d", got)
	}
	// An older (epoch,seq) is ignored even if newer within its own epoch numbering.
	if l.Grant(key, 1, 9999, 500) {
		t.Fatalf("older epoch grant must be ignored")
	}
	if got := l.Available(key); got != 100 {
		t.Fatalf("older-epoch grant must not change the ceiling: got %d", got)
	}
	// A strictly-newer (epoch,seq) but LOWER cumulative never lowers the ceiling.
	if l.Grant(key, 3, 1, 40) {
		t.Fatalf("newer grant with a lower cumulative must not advance the ceiling")
	}
	if got := l.Available(key); got != 100 {
		t.Fatalf("monotonic ceiling must not drop to a lower cumulative: got %d", got)
	}
	// A strictly-newer (epoch,seq) with a higher cumulative raises the ceiling.
	if !l.Grant(key, 3, 2, 150) {
		t.Fatalf("newer grant with a higher cumulative must advance the ceiling")
	}
	if got := l.Available(key); got != 150 {
		t.Fatalf("ceiling should advance to 150: got %d", got)
	}
	// seq breaks ties within the same epoch: newer seq wins.
	if !l.Grant(key, 3, 3, 175) {
		t.Fatalf("same-epoch newer seq with higher cumulative must advance")
	}
	if got := l.Available(key); got != 175 {
		t.Fatalf("ceiling should advance to 175 via seq: got %d", got)
	}
}

// TestCreditReserveRollbackRestoresExactly proves a reserve followed by a
// rollback restores Available to exactly its pre-reserve value, and that rollback
// is clamped against underflow.
func TestCreditReserveRollbackRestoresExactly(t *testing.T) {
	l := NewCreditLedger()
	key := CreditKey{Receiver: "decode2", Sender: "prefill2", QoS: "bronze"}

	l.Grant(key, 1, 1, 20)
	before := l.Available(key)
	if before != 20 {
		t.Fatalf("Available before reserve=20: got %d", before)
	}
	if !l.TryReserve(key, 12) {
		t.Fatalf("reserve of 12 within 20 must succeed")
	}
	if got := l.Available(key); got != 8 {
		t.Fatalf("Available after reserving 12: got %d", got)
	}
	l.Rollback(key, 12)
	if got := l.Available(key); got != before {
		t.Fatalf("rollback must restore Available exactly to %d: got %d", before, got)
	}
	// A double / oversized rollback is clamped and never overflows Available.
	l.Rollback(key, 1000)
	if got := l.Available(key); got != before {
		t.Fatalf("clamped rollback must not exceed the ceiling: got %d", got)
	}
}

// TestCreditConsumeSpendsPermanently proves Consume moves credit out of the
// available pool for good (unlike Rollback), is clamped to the outstanding
// reservation, and is deterministic across tuples.
func TestCreditConsumeSpendsPermanently(t *testing.T) {
	l := NewCreditLedger()
	key := CreditKey{Receiver: "decode3", Sender: "prefill3", QoS: "gold"}

	l.Grant(key, 1, 1, 30)
	if !l.TryReserve(key, 25) {
		t.Fatalf("reserve of 25 within 30 must succeed")
	}
	// Consuming more than reserved is clamped to the outstanding reservation.
	if spent := l.Consume(key, 40); spent != 25 {
		t.Fatalf("consume clamped to the reserved 25: got %d", spent)
	}
	snap := l.Snapshot(key)
	if snap.Consumed != 25 {
		t.Fatalf("Consumed should be 25: got %d", snap.Consumed)
	}
	if snap.Reserved != 0 {
		t.Fatalf("Reserved should return to 0 after consume: got %d", snap.Reserved)
	}
	// Consumed credit does NOT return to Available: only the un-reserved 5 remains.
	if snap.Available != 5 {
		t.Fatalf("Available after consuming 25 of 30: got %d", snap.Available)
	}
	// A fresh grant raising the ceiling makes room again beyond the spent credit.
	if !l.Grant(key, 2, 1, 60) {
		t.Fatalf("raising the ceiling to 60 must advance")
	}
	if got := l.Available(key); got != 35 {
		t.Fatalf("Available = 60 granted - 25 consumed: got %d", got)
	}
}

// TestCreditTuplesAreIsolated proves throttling one (receiver,sender,qos) tuple
// never touches another: distinct keys keep independent tallies.
func TestCreditBackpressureTuplesIsolated(t *testing.T) {
	l := NewCreditLedger()
	gold := CreditKey{Receiver: "r", Sender: "s", QoS: "gold"}
	bulk := CreditKey{Receiver: "r", Sender: "s", QoS: "bulk"}

	l.Grant(gold, 1, 1, 10)
	l.Grant(bulk, 1, 1, 3)

	// Exhaust the bulk class entirely.
	if !l.TryReserve(bulk, 3) {
		t.Fatalf("reserve of the full bulk grant must succeed")
	}
	if l.TryReserve(bulk, 1) {
		t.Fatalf("bulk class is exhausted; further reserve must be refused")
	}
	// The gold class is untouched by bulk's exhaustion.
	if got := l.Available(gold); got != 10 {
		t.Fatalf("gold class must be unaffected by bulk backpressure: got %d", got)
	}
	if !l.TryReserve(gold, 10) {
		t.Fatalf("gold class still has its full grant available")
	}
}
