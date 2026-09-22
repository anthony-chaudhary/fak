package dosadapter

import (
	"errors"
	"testing"
	"time"
)

// TestArbitrateTracksLiveLeaseSet is the regression for fak-private#1619: a stale
// live-lease snapshot must not make dos_arbitrate disagree with the CLI. Re-deriving
// the set after the CLI reaps its last intersecting dead holder must flip the verdict
// from refuse to acquire, proving arbitration tracks the live set.
func TestArbitrateTracksLiveLeaseSet(t *testing.T) {
	now := time.Date(2026, 9, 21, 10, 0, 0, 0, time.UTC)
	client := NewAdapterClient(
		WithClock(func() time.Time { return now }),
		WithIssuer("fak/test-runner"),
	)

	deadHolder := LeaseRequest{
		ID:       "held-native-rocm-model-test",
		Lane:     "native-rocm-model-test",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/model/**"},
	}

	req := LeaseRequest{
		ID:       "req-model",
		Lane:     "model",
		LaneKind: "cluster",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/model/**"},
	}

	// (a) Pre-reap set: the dead holder still intersects lane "model" -> REFUSE.
	staleSet := DeriveLiveLeaseSet(
		[]LeaseRequest{deadHolder},
		now.Add(-10*time.Minute),
		7,
		LeaseSourceSnapshot,
	)
	outcome, err := client.ArbitrateSet(req, staleSet, now)
	if err == nil {
		t.Fatalf("ArbitrateSet() pre-reap expected refusal error, got nil")
	}
	if !errors.Is(err, ErrArbitrationRefused) {
		t.Errorf("ArbitrateSet() pre-reap error = %v, want ErrArbitrationRefused", err)
	}
	if outcome.Outcome != OutcomeRefuse {
		t.Errorf("ArbitrateSet() pre-reap outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
	}
	if outcome.Reason.Token != ReasonCollisionRisk {
		t.Errorf("ArbitrateSet() pre-reap token = %q, want %q", outcome.Reason.Token, ReasonCollisionRisk)
	}

	// (b) Simulate the CLI reap: re-derive with the dead lease REMOVED and generation+1.
	liveSet := DeriveLiveLeaseSet(
		[]LeaseRequest{},
		now,
		staleSet.Generation+1,
		LeaseSourceLiveJournal,
	)
	outcome, err = client.ArbitrateSet(req, liveSet, now)
	if err != nil {
		t.Fatalf("ArbitrateSet() post-reap unexpected error: %v", err)
	}
	if outcome.Outcome != OutcomeAcquire {
		t.Errorf("ArbitrateSet() post-reap outcome = %q, want %q", outcome.Outcome, OutcomeAcquire)
	}
	if outcome.LeaseSource != LeaseSourceLiveJournal {
		t.Errorf("ArbitrateSet() post-reap source = %q, want %q", outcome.LeaseSource, LeaseSourceLiveJournal)
	}
	if outcome.LeaseGeneration != staleSet.Generation+1 {
		t.Errorf("ArbitrateSet() post-reap generation = %d, want %d", outcome.LeaseGeneration, staleSet.Generation+1)
	}
}

// TestArbitrationOutcomeCarriesLeaseProvenance asserts the outcome exposes the deciding
// set's Source/Generation/ObservedUnix, and that they change after a re-derive.
func TestArbitrationOutcomeCarriesLeaseProvenance(t *testing.T) {
	now := time.Date(2026, 9, 21, 11, 0, 0, 0, time.UTC)
	client := NewAdapterClient(WithClock(func() time.Time { return now }))

	req := LeaseRequest{
		ID:       "req-provenance",
		Lane:     "docs",
		LockMode: LockModeExclusive,
		Tree:     []string{"docs/**"},
	}

	first := DeriveLiveLeaseSet(nil, now.Add(-time.Minute), 3, LeaseSourceCaller)
	out1, err := client.ArbitrateSet(req, first, now)
	if err != nil {
		t.Fatalf("ArbitrateSet() first unexpected error: %v", err)
	}
	if out1.LeaseSource != LeaseSourceCaller {
		t.Errorf("outcome LeaseSource = %q, want %q", out1.LeaseSource, LeaseSourceCaller)
	}
	if out1.LeaseGeneration != 3 {
		t.Errorf("outcome LeaseGeneration = %d, want 3", out1.LeaseGeneration)
	}
	if out1.LeaseObservedUnix != first.ObservedUnix {
		t.Errorf("outcome LeaseObservedUnix = %d, want %d", out1.LeaseObservedUnix, first.ObservedUnix)
	}

	second := DeriveLiveLeaseSet(nil, now, 4, LeaseSourceLiveJournal)
	out2, err := client.ArbitrateSet(req, second, now)
	if err != nil {
		t.Fatalf("ArbitrateSet() second unexpected error: %v", err)
	}
	if out2.LeaseGeneration == out1.LeaseGeneration {
		t.Errorf("outcome LeaseGeneration did not change after re-derive: %d", out2.LeaseGeneration)
	}
	if out2.LeaseObservedUnix == out1.LeaseObservedUnix {
		t.Errorf("outcome LeaseObservedUnix did not change after re-derive: %d", out2.LeaseObservedUnix)
	}
	if out2.LeaseSource == out1.LeaseSource {
		t.Errorf("outcome LeaseSource did not change after re-derive: %q", out2.LeaseSource)
	}
}

// TestStaleSnapshotRefuses asserts a caller can observe staleness and choose to re-derive.
func TestStaleSnapshotRefuses(t *testing.T) {
	now := time.Date(2026, 9, 21, 12, 0, 0, 0, time.UTC)
	maxAge := 5 * time.Minute

	fresh := NewLiveLeaseSet(nil, now.Add(-time.Minute), 9, LeaseSourceLiveJournal)
	if fresh.Stale(now, maxAge) {
		t.Errorf("fresh set reported stale: %s", fresh.Describe())
	}

	old := NewLiveLeaseSet(nil, now.Add(-30*time.Minute), 9, LeaseSourceSnapshot)
	if !old.Stale(now, maxAge) {
		t.Errorf("old set not reported stale: %s", old.Describe())
	}

	unattributed := NewLiveLeaseSet(nil, time.Time{}, 0, LeaseSourceUnreadable)
	if !unattributed.Stale(now, maxAge) {
		t.Errorf("zero-observed set must fail closed as stale: %s", unattributed.Describe())
	}

	// The adjudicator still decides against the stale rows (fail-closed): a conflicting
	// dead holder in a stale snapshot must keep refusing until the set is re-derived.
	client := NewAdapterClient(WithClock(func() time.Time { return now }))
	holder := LeaseRequest{
		ID:       "held-native-rocm-model-test",
		Lane:     "native-rocm-model-test",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/model/**"},
	}
	staleSet := DeriveLiveLeaseSet([]LeaseRequest{holder}, time.Unix(old.ObservedUnix, 0), old.Generation, LeaseSourceSnapshot)
	outcome, err := client.ArbitrateSet(LeaseRequest{
		ID:       "req-model-stale",
		Lane:     "model",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/model/**"},
	}, staleSet, now)
	if !errors.Is(err, ErrArbitrationRefused) {
		t.Fatalf("stale snapshot adjudication error = %v, want ErrArbitrationRefused", err)
	}
	if outcome.Outcome != OutcomeRefuse {
		t.Errorf("stale snapshot outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
	}
}

// TestArbitrateStampsInMemoryGeneration asserts the in-memory path names its own view
// and never fabricates freshness: each register/release bumps the observed generation.
func TestArbitrateStampsInMemoryGeneration(t *testing.T) {
	client := NewAdapterClient()

	req := LeaseRequest{
		ID:       "in-memory-1",
		Lane:     "alpha",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/alpha/**"},
	}
	if err := client.RegisterLease(req); err != nil {
		t.Fatalf("RegisterLease() unexpected error: %v", err)
	}
	out1, err := client.Arbitrate(LeaseRequest{
		ID:       "in-memory-2",
		Lane:     "beta",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/beta/**"},
	})
	if err != nil {
		t.Fatalf("Arbitrate() unexpected error: %v", err)
	}
	if out1.LeaseSource != LeaseSourceInMemory {
		t.Errorf("Arbitrate() LeaseSource = %q, want %q", out1.LeaseSource, LeaseSourceInMemory)
	}
	if out1.LeaseGeneration != 1 {
		t.Errorf("Arbitrate() LeaseGeneration = %d, want 1 after one register", out1.LeaseGeneration)
	}

	if !client.ReleaseLease("in-memory-1") {
		t.Fatalf("ReleaseLease() = false, want true")
	}
	out2, err := client.Arbitrate(LeaseRequest{
		ID:       "in-memory-3",
		Lane:     "beta",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/beta/**"},
	})
	if err != nil {
		t.Fatalf("Arbitrate() unexpected error: %v", err)
	}
	if out2.LeaseGeneration != 2 {
		t.Errorf("Arbitrate() LeaseGeneration = %d, want 2 after register+release", out2.LeaseGeneration)
	}
}
