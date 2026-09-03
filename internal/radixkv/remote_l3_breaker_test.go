package radixkv

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type countingSnapshotStore struct {
	mu       sync.Mutex
	data     map[string][]byte
	getCalls int
	putCalls int
	getErr   error
}

func (s *countingSnapshotStore) Put(_ context.Context, key string, payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data == nil {
		s.data = map[string][]byte{}
	}
	s.data[key] = append([]byte(nil), payload...)
	s.putCalls++
	return nil
}

func (s *countingSnapshotStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.getCalls++
	if s.getErr != nil {
		return nil, false, s.getErr
	}
	payload, ok := s.data[key]
	return append([]byte(nil), payload...), ok, nil
}

func (s *countingSnapshotStore) GetCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getCalls
}

func (s *countingSnapshotStore) SetGetErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.getErr = err
}

func TestRemoteL3BreakerDefaultConfig(t *testing.T) {
	b := NewRemoteL3Breaker(BreakerConfig{})
	if b.State() != BreakerClosed {
		t.Fatalf("initial state = %v, want %v", b.State(), BreakerClosed)
	}
	stats := b.Stats()
	if stats.FaultThreshold != DefaultBreakerFaultThreshold {
		t.Fatalf("fault threshold = %d, want %d", stats.FaultThreshold, DefaultBreakerFaultThreshold)
	}
	if stats.Cooldown != DefaultBreakerCooldown {
		t.Fatalf("cooldown = %v, want %v", stats.Cooldown, DefaultBreakerCooldown)
	}
}

func TestRemoteL3BreakerClosedAllowsReads(t *testing.T) {
	b := NewRemoteL3Breaker(DefaultBreakerConfig())
	for i := 0; i < 10; i++ {
		allowed, isProbe := b.Allow()
		if !allowed || isProbe {
			t.Fatalf("Allow() on closed breaker = (%v, %v), want (true, false)", allowed, isProbe)
		}
	}
	if b.OpenSkips() != 0 {
		t.Fatalf("open skips = %d, want 0", b.OpenSkips())
	}
}

func TestRemoteL3BreakerTripsAtThreshold(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	const threshold = 3
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: threshold,
		Cooldown:       10 * time.Second,
		Now:            func() time.Time { return now },
	})

	backendErr := errors.New("remote storage 503 unavailable")

	// Faults 1 and 2: consecutive faults increment, breaker remains Closed.
	for i := 1; i < threshold; i++ {
		b.RecordResult(backendErr, false)
		if got := b.State(); got != BreakerClosed {
			t.Fatalf("fault %d: state = %v, want %v", i, got, BreakerClosed)
		}
		if got := b.ConsecutiveFaults(); got != i {
			t.Fatalf("fault %d: consecutive faults = %d, want %d", i, got, i)
		}
		if got := b.TotalFaults(); got != i {
			t.Fatalf("fault %d: total faults = %d, want %d", i, got, i)
		}
		allowed, isProbe := b.Allow()
		if !allowed || isProbe {
			t.Fatalf("fault %d: Allow() = (%v, %v), want (true, false)", i, allowed, isProbe)
		}
	}

	// Fault 3 reaches threshold -> trips breaker to Open.
	b.RecordResult(backendErr, false)
	if got := b.State(); got != BreakerOpen {
		t.Fatalf("at threshold: state = %v, want %v", got, BreakerOpen)
	}
	if got := b.ConsecutiveFaults(); got != threshold {
		t.Fatalf("at threshold: consecutive faults = %d, want %d", got, threshold)
	}
	if got := b.TotalFaults(); got != threshold {
		t.Fatalf("at threshold: total faults = %d, want %d", got, threshold)
	}
	if got := b.OpenedAt(); got != now {
		t.Fatalf("openedAt = %v, want %v", got, now)
	}

	// While Open, Allow returns false and counts open skips.
	allowed, isProbe := b.Allow()
	if allowed || isProbe {
		t.Fatalf("Allow() while Open = (%v, %v), want (false, false)", allowed, isProbe)
	}
	if got := b.OpenSkips(); got != 1 {
		t.Fatalf("openSkips = %d, want 1", got)
	}

	// Consecutive reset behavior: an intermittent success resets consecutive faults.
	b2 := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: 3,
		Cooldown:       10 * time.Second,
		Now:            func() time.Time { return now },
	})
	b2.RecordResult(backendErr, false)
	b2.RecordResult(backendErr, false)
	if b2.ConsecutiveFaults() != 2 {
		t.Fatalf("b2 consecutive faults = %d, want 2", b2.ConsecutiveFaults())
	}
	// Intermittent success resets consecutive faults
	b2.RecordResult(nil, false)
	if b2.ConsecutiveFaults() != 0 {
		t.Fatalf("b2 consecutive faults after success = %d, want 0", b2.ConsecutiveFaults())
	}
	if b2.TotalFaults() != 2 {
		t.Fatalf("b2 total faults after success = %d, want 2", b2.TotalFaults())
	}
	if b2.State() != BreakerClosed {
		t.Fatalf("b2 state after success = %v, want %v", b2.State(), BreakerClosed)
	}
	// Requires another full threshold (3 faults) to trip
	b2.RecordResult(backendErr, false)
	b2.RecordResult(backendErr, false)
	if b2.State() != BreakerClosed {
		t.Fatalf("b2 prematurely opened: state = %v", b2.State())
	}
	b2.RecordResult(backendErr, false)
	if b2.State() != BreakerOpen {
		t.Fatalf("b2 state after 3 more faults = %v, want %v", b2.State(), BreakerOpen)
	}
	if b2.ConsecutiveFaults() != 3 || b2.TotalFaults() != 5 {
		t.Fatalf("b2 faults: consecutive=%d total=%d, want cons=3 total=5", b2.ConsecutiveFaults(), b2.TotalFaults())
	}
}

func TestRemoteL3BreakerOpenSkipsStoreIO(t *testing.T) {
	cfg := remoteL3TestConfig()
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	store := &countingSnapshotStore{}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
	if err := tree.ConfigureRemoteSnapshotStore(store, "synthetic-l3-test", be, m.Cfg); err != nil {
		t.Fatal(err)
	}

	currentTime := time.Date(2026, 9, 3, 10, 0, 0, 0, time.UTC)
	breaker := tree.RemoteL3Breaker()
	breaker.SetClock(func() time.Time { return currentTime })

	ids := []int{10, 20, 30}
	digest := insertRemoteL3Snapshot(t, tree, m, be, ids)
	if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage remote: %+v", got)
	}
	if tree.EvictHotSnapshot(digest) != len(ids) {
		t.Fatal("hot owner was not removed")
	}

	// Capture node snapshot metadata before tripping breaker
	ns, nFound := tree.findSnapshotByDigestNS(digest)
	if nFound == nil || nFound.remoteSnapshot == nil {
		t.Fatalf("remote snapshot reference missing before trip: ns=%q", ns)
	}
	origRef := *nFound.remoteSnapshot
	if origRef.digest != digest || origRef.tokens != len(ids) || origRef.bytes <= 0 {
		t.Fatalf("unexpected original remoteSnapshotRef: %+v", origRef)
	}

	// Trip the breaker to Open state by recording faults up to threshold
	backendErr := errors.New("backend connection refused")
	for i := 0; i < breaker.faultThresholdLocked(); i++ {
		breaker.RecordResult(backendErr, false)
	}
	if breaker.State() != BreakerOpen {
		t.Fatalf("breaker state = %v, want %v", breaker.State(), BreakerOpen)
	}

	callsBefore := store.GetCalls()
	skipsBefore := breaker.OpenSkips()

	// 1. Attempt LookupSnapshotTieredContext while breaker is Open within cooldown.
	n, snap, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), ids)
	tree.Done(n)
	if snap != nil {
		snap.Close()
		t.Fatal("expected nil snapshot on breaker skip")
	}
	if err != nil {
		t.Fatalf("expected nil error on breaker skip (treated as clean miss), got: %v", err)
	}
	if tier != SnapshotTierMiss {
		t.Fatalf("tier on breaker skip = %q, want %q", tier, SnapshotTierMiss)
	}
	if matched != 0 {
		t.Fatalf("matched on breaker skip = %d, want 0", matched)
	}

	// Verify store.Get was NOT invoked
	if got := store.GetCalls(); got != callsBefore {
		t.Fatalf("store.Get was invoked during Open state: calls before=%d, after=%d", callsBefore, got)
	}

	// Verify skips are counted
	if got := breaker.OpenSkips(); got != skipsBefore+1 {
		t.Fatalf("breaker OpenSkips = %d, want %d", got, skipsBefore+1)
	}
	if got := tree.Stats().L3BreakerOpenSkips; got != breaker.OpenSkips() {
		t.Fatalf("tree.Stats().L3BreakerOpenSkips = %d, want %d", got, breaker.OpenSkips())
	}

	// Verify node snapshot metadata is preserved
	ns2, nFound2 := tree.findSnapshotByDigestNS(digest)
	if nFound2 == nil || nFound2.remoteSnapshot == nil {
		t.Fatalf("node snapshot metadata was cleared/wiped on breaker skip: ns=%q", ns2)
	}
	if *nFound2.remoteSnapshot != origRef {
		t.Fatalf("node snapshot metadata mutated: got %+v, want %+v", *nFound2.remoteSnapshot, origRef)
	}

	// 2. Attempt RestoreSnapshotFromRemote while Open
	xfer := tree.RestoreSnapshotFromRemote(context.Background(), digest)
	if xfer.Outcome != SnapshotTransferMiss {
		t.Fatalf("RestoreSnapshotFromRemote outcome = %v, want %v", xfer.Outcome, SnapshotTransferMiss)
	}
	// Verify store.Get still NOT invoked
	if got := store.GetCalls(); got != callsBefore {
		t.Fatalf("store.Get was invoked during RestoreSnapshotFromRemote: calls before=%d, after=%d", callsBefore, got)
	}
	// Verify skips counted again
	if got := breaker.OpenSkips(); got != skipsBefore+2 {
		t.Fatalf("breaker OpenSkips after restore attempt = %d, want %d", got, skipsBefore+2)
	}
	// Verify node snapshot metadata is still preserved
	if *nFound2.remoteSnapshot != origRef {
		t.Fatalf("node snapshot metadata mutated after restore attempt: got %+v, want %+v", *nFound2.remoteSnapshot, origRef)
	}
}

func TestRemoteL3BreakerIgnoresContextCancellation(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	const threshold = 3
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: threshold,
		Cooldown:       10 * time.Second,
		Now:            func() time.Time { return now },
	})

	// Direct context.Canceled
	for i := 0; i < 5; i++ {
		b.RecordResult(context.Canceled, false)
	}
	if b.ConsecutiveFaults() != 0 || b.TotalFaults() != 0 || b.State() != BreakerClosed {
		t.Fatalf("direct context.Canceled mutated breaker: cons=%d total=%d state=%v",
			b.ConsecutiveFaults(), b.TotalFaults(), b.State())
	}

	// Wrapped context.Canceled
	wrapped := fmt.Errorf("read remote snapshot: %w", context.Canceled)
	for i := 0; i < 5; i++ {
		b.RecordResult(wrapped, false)
	}
	if b.ConsecutiveFaults() != 0 || b.TotalFaults() != 0 || b.State() != BreakerClosed {
		t.Fatalf("wrapped context.Canceled mutated breaker: cons=%d total=%d state=%v",
			b.ConsecutiveFaults(), b.TotalFaults(), b.State())
	}

	// Accumulate faults just below threshold
	backendErr := errors.New("backend 500 error")
	b.RecordResult(backendErr, false)
	b.RecordResult(backendErr, false)
	if b.ConsecutiveFaults() != 2 || b.TotalFaults() != 2 || b.State() != BreakerClosed {
		t.Fatalf("pre-cancellation state: cons=%d total=%d state=%v",
			b.ConsecutiveFaults(), b.TotalFaults(), b.State())
	}

	// Repeated context.Canceled when faults are already accumulated does not increment faults or trip
	for i := 0; i < 5; i++ {
		b.RecordResult(context.Canceled, false)
		b.RecordResult(wrapped, false)
	}
	if b.ConsecutiveFaults() != 2 || b.TotalFaults() != 2 || b.State() != BreakerClosed {
		t.Fatalf("cancellation mutated accumulated faults: cons=%d total=%d state=%v",
			b.ConsecutiveFaults(), b.TotalFaults(), b.State())
	}

	// Cancellation during HalfOpen probe: reverts to Open without penalty
	b.RecordResult(backendErr, false) // 3rd fault -> trips to Open
	if b.State() != BreakerOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerOpen)
	}

	// Advance clock past cooldown and admit probe
	now = now.Add(10 * time.Second)
	allowed, isProbe := b.Allow()
	if !allowed || !isProbe || b.State() != BreakerHalfOpen {
		t.Fatalf("probe admission failed: allowed=%v isProbe=%v state=%v", allowed, isProbe, b.State())
	}

	// Probe is canceled by caller
	b.RecordResult(context.Canceled, true)
	if b.State() != BreakerOpen {
		t.Fatalf("state after canceled probe = %v, want %v", b.State(), BreakerOpen)
	}
	if b.ProbeFailures() != 0 {
		t.Fatalf("probeFailures after canceled probe = %d, want 0", b.ProbeFailures())
	}
	if b.ConsecutiveFaults() != 3 {
		t.Fatalf("consecutiveFaults after canceled probe = %d, want 3", b.ConsecutiveFaults())
	}

	// Immediate next Allow() still admits another probe without waiting for restarted cooldown
	allowed2, isProbe2 := b.Allow()
	if !allowed2 || !isProbe2 || b.State() != BreakerHalfOpen {
		t.Fatalf("retry probe after cancel: allowed=%v isProbe=%v state=%v", allowed2, isProbe2, b.State())
	}

	// Tree integration verification
	cfg := remoteL3TestConfig()
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	store := &countingSnapshotStore{}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
	if err := tree.ConfigureRemoteSnapshotStore(store, "synthetic-l3-test", be, m.Cfg); err != nil {
		t.Fatal(err)
	}
	treeBreaker := tree.RemoteL3Breaker()
	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()
	store.SetGetErr(cancelCtx.Err())

	ids := []int{1, 2, 3}
	digest := insertRemoteL3Snapshot(t, tree, m, be, ids)
	if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage remote: %+v", got)
	}
	if tree.EvictHotSnapshot(digest) != len(ids) {
		t.Fatal("hot owner was not removed")
	}

	for i := 0; i < 10; i++ {
		n, snap, _, _, _ := tree.LookupSnapshotTieredContext(cancelCtx, ids)
		tree.Done(n)
		if snap != nil {
			snap.Close()
		}
	}
	if treeBreaker.State() != BreakerClosed {
		t.Fatalf("tree breaker state = %v, want %v", treeBreaker.State(), BreakerClosed)
	}
	if treeBreaker.ConsecutiveFaults() != 0 || treeBreaker.TotalFaults() != 0 {
		t.Fatalf("tree breaker faults on cancelled ctx: cons=%d total=%d",
			treeBreaker.ConsecutiveFaults(), treeBreaker.TotalFaults())
	}
}

func TestRemoteL3BreakerHalfOpenSingleProbe(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	cooldown := 30 * time.Second
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: 2,
		Cooldown:       cooldown,
		Now:            func() time.Time { return now },
	})

	backendErr := errors.New("backend 500")
	b.RecordResult(backendErr, false)
	b.RecordResult(backendErr, false)
	if b.State() != BreakerOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerOpen)
	}

	// Before cooldown: calls rejected
	now = now.Add(cooldown - time.Second)
	allowed, isProbe := b.Allow()
	if allowed || isProbe {
		t.Fatalf("Allow() before cooldown = (%v, %v), want (false, false)", allowed, isProbe)
	}
	if b.OpenSkips() != 1 {
		t.Fatalf("openSkips = %d, want 1", b.OpenSkips())
	}

	// Advance past cooldown
	now = now.Add(2 * time.Second) // cooldown + 1s

	// Concurrently invoke Allow() from 50 goroutines
	const concurrency = 50
	var wg sync.WaitGroup
	start := make(chan struct{})

	type callResult struct {
		allowed bool
		isProbe bool
	}
	results := make([]callResult, concurrency)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-start
			a, p := b.Allow()
			results[idx] = callResult{allowed: a, isProbe: p}
		}(i)
	}

	close(start)
	wg.Wait()

	probeCount := 0
	rejectedCount := 0
	otherCount := 0
	for _, res := range results {
		if res.allowed && res.isProbe {
			probeCount++
		} else if !res.allowed && !res.isProbe {
			rejectedCount++
		} else {
			otherCount++
		}
	}

	if probeCount != 1 {
		t.Fatalf("admitted probes = %d, want exactly 1", probeCount)
	}
	if rejectedCount != concurrency-1 {
		t.Fatalf("rejected requests = %d, want %d", rejectedCount, concurrency-1)
	}
	if otherCount != 0 {
		t.Fatalf("unexpected call results: count = %d", otherCount)
	}
	if b.State() != BreakerHalfOpen {
		t.Fatalf("breaker state = %v, want %v", b.State(), BreakerHalfOpen)
	}
	if b.ProbesAttempted() != 1 {
		t.Fatalf("probesAttempted = %d, want 1", b.ProbesAttempted())
	}

	// Subsequent sequential call while in HalfOpen is also rejected
	subsequentAllowed, subsequentProbe := b.Allow()
	if subsequentAllowed || subsequentProbe {
		t.Fatalf("subsequent Allow() during HalfOpen = (%v, %v), want (false, false)",
			subsequentAllowed, subsequentProbe)
	}
	if b.OpenSkips() != 1+rejectedCount+1 {
		t.Fatalf("total openSkips = %d, want %d", b.OpenSkips(), 1+rejectedCount+1)
	}
}

func TestRemoteL3BreakerProbeRecovery(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	cooldown := 20 * time.Second
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: 3,
		Cooldown:       cooldown,
		Now:            func() time.Time { return now },
	})

	backendErr := errors.New("backend 503 service unavailable")
	for i := 0; i < 3; i++ {
		b.RecordResult(backendErr, false)
	}
	if b.State() != BreakerOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerOpen)
	}
	if b.ConsecutiveFaults() != 3 || b.TotalFaults() != 3 {
		t.Fatalf("faults: cons=%d total=%d, want 3, 3", b.ConsecutiveFaults(), b.TotalFaults())
	}

	// Advance clock past cooldown
	now = now.Add(cooldown + time.Second)

	// Probe is admitted
	allowed, isProbe := b.Allow()
	if !allowed || !isProbe {
		t.Fatalf("Allow() = (%v, %v), want (true, true)", allowed, isProbe)
	}
	if b.State() != BreakerHalfOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerHalfOpen)
	}
	if b.ProbesAttempted() != 1 {
		t.Fatalf("probesAttempted = %d, want 1", b.ProbesAttempted())
	}

	// Successful probe result
	b.RecordResult(nil, true)

	// Transitions to Closed and resets fault counters
	if got := b.State(); got != BreakerClosed {
		t.Fatalf("state after probe recovery = %v, want %v", got, BreakerClosed)
	}
	if got := b.ConsecutiveFaults(); got != 0 {
		t.Fatalf("consecutiveFaults after recovery = %d, want 0", got)
	}
	if got := b.TotalFaults(); got != 3 {
		t.Fatalf("totalFaults after recovery = %d, want 3 (cumulative)", got)
	}
	if got := b.ProbeRecoveries(); got != 1 {
		t.Fatalf("probeRecoveries = %d, want 1", got)
	}
	if got := b.ProbeFailures(); got != 0 {
		t.Fatalf("probeFailures = %d, want 0", got)
	}
	if !b.OpenedAt().IsZero() {
		t.Fatalf("openedAt after recovery = %v, want zero", b.OpenedAt())
	}

	// Normal closed requests are allowed
	for i := 0; i < 5; i++ {
		allowNorm, probeNorm := b.Allow()
		if !allowNorm || probeNorm {
			t.Fatalf("Allow() #%d on closed breaker = (%v, %v), want (true, false)", i, allowNorm, probeNorm)
		}
	}

	// Verify Stats snapshot matches
	stats := b.Stats()
	if stats.State != BreakerClosed || stats.ConsecutiveFaults != 0 || stats.TotalFaults != 3 ||
		stats.ProbesAttempted != 1 || stats.ProbeRecoveries != 1 || stats.ProbeFailures != 0 || !stats.OpenedAt.IsZero() {
		t.Fatalf("unexpected stats snapshot: %+v", stats)
	}
}

func TestRemoteL3BreakerProbeFailureReopens(t *testing.T) {
	t0 := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	cooldown := 25 * time.Second
	currentTime := t0
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: 2,
		Cooldown:       cooldown,
		Now:            func() time.Time { return currentTime },
	})

	backendErr := errors.New("storage gateway 502")
	b.RecordResult(backendErr, false)
	b.RecordResult(backendErr, false)
	if b.State() != BreakerOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerOpen)
	}
	if b.OpenedAt() != t0 {
		t.Fatalf("openedAt = %v, want %v", b.OpenedAt(), t0)
	}

	// Advance past cooldown to admit probe
	t1 := t0.Add(cooldown + 5*time.Second)
	currentTime = t1

	allowed, isProbe := b.Allow()
	if !allowed || !isProbe {
		t.Fatalf("Allow() probe = (%v, %v), want (true, true)", allowed, isProbe)
	}
	if b.State() != BreakerHalfOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerHalfOpen)
	}

	// Probe fails
	probeErr := errors.New("probe connection refused")
	b.RecordResult(probeErr, true)

	// Transitions back to Open, records failure, resets cooldown timer
	if got := b.State(); got != BreakerOpen {
		t.Fatalf("state after probe failure = %v, want %v", got, BreakerOpen)
	}
	if got := b.ProbeFailures(); got != 1 {
		t.Fatalf("probeFailures = %d, want 1", got)
	}
	if got := b.ProbeRecoveries(); got != 0 {
		t.Fatalf("probeRecoveries = %d, want 0", got)
	}
	if got := b.OpenedAt(); got != t1 {
		t.Fatalf("openedAt after failure = %v, want %v (cooldown reset to probe failure time)", got, t1)
	}
	if got := b.ConsecutiveFaults(); got != 3 {
		t.Fatalf("consecutiveFaults = %d, want 3", got)
	}
	if got := b.TotalFaults(); got != 3 {
		t.Fatalf("totalFaults = %d, want 3", got)
	}

	// Cooldown timer reset to t1:
	// At t1 + 10s (within restarted cooldown): requests rejected
	currentTime = t1.Add(10 * time.Second)
	a1, p1 := b.Allow()
	if a1 || p1 {
		t.Fatalf("Allow() at t1+10s = (%v, %v), want (false, false)", a1, p1)
	}

	// At t1 + cooldown - 1s: still rejected
	currentTime = t1.Add(cooldown - time.Second)
	a2, p2 := b.Allow()
	if a2 || p2 {
		t.Fatalf("Allow() at t1+cooldown-1s = (%v, %v), want (false, false)", a2, p2)
	}

	// At t1 + cooldown: new probe is admitted
	currentTime = t1.Add(cooldown)
	a3, p3 := b.Allow()
	if !a3 || !p3 {
		t.Fatalf("Allow() at t1+cooldown = (%v, %v), want (true, true)", a3, p3)
	}
	if b.State() != BreakerHalfOpen {
		t.Fatalf("state = %v, want %v", b.State(), BreakerHalfOpen)
	}
	if b.ProbesAttempted() != 2 {
		t.Fatalf("probesAttempted = %d, want 2", b.ProbesAttempted())
	}

	// Second probe succeeds and recovers breaker
	b.RecordResult(nil, true)
	if b.State() != BreakerClosed {
		t.Fatalf("state after second probe = %v, want %v", b.State(), BreakerClosed)
	}
	if b.ConsecutiveFaults() != 0 {
		t.Fatalf("consecutiveFaults = %d, want 0", b.ConsecutiveFaults())
	}
	if b.ProbeRecoveries() != 1 {
		t.Fatalf("probeRecoveries = %d, want 1", b.ProbeRecoveries())
	}
	if b.ProbeFailures() != 1 {
		t.Fatalf("probeFailures = %d, want 1", b.ProbeFailures())
	}
}

func TestRemoteL3BreakerConcurrentAccess(t *testing.T) {
	now := time.Now()
	b := NewRemoteL3Breaker(BreakerConfig{
		FaultThreshold: 5,
		Cooldown:       5 * time.Millisecond,
		Now:            func() time.Time { return now },
	})

	var wg sync.WaitGroup
	workers := 16
	iterations := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				allowed, isProbe := b.Allow()
				if allowed {
					if j%3 == 0 {
						b.RecordResult(errors.New("simulated error"), isProbe)
					} else {
						b.RecordResult(nil, isProbe)
					}
				}
				_ = b.Stats()
			}
		}(i)
	}

	wg.Wait()
	stats := b.Stats()
	if stats.TotalFaults < 0 || stats.OpenSkips < 0 {
		t.Fatalf("invalid stats: %+v", stats)
	}
}
