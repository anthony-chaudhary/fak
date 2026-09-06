package cacheobs

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"
)

// deterministicWorkload returns a fixed sequence of valid and rejected tier
// observations, interleaved with turn observations, to verify reproducible
// accounting across runs.
func deterministicWorkload() ([]TierAccess, []struct {
	prompt int
	reused int
}) {
	invalid := []TierAccess{
		{Tier: CacheTier(-1), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: CacheTier(99), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: numCacheTiers, Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: TierOp(-1), Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: TierOp(7), Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: numTierOps, Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(42), Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: numTierOutcomes, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(-1)},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(99)},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: numBackendClasses},
		{
			Tier: CacheTier(255), Op: TierOp(255), Outcome: TierOutcome(255), Backend: BackendClass(255),
			Bytes: math.MaxInt64, BytesKnown: true, Latency: time.Hour, LatencyKnown: true,
		},
	}
	turns := []struct {
		prompt int
		reused int
	}{
		{prompt: 1000, reused: 800},
		{prompt: 500, reused: 300},
		{prompt: 600, reused: 200},
	}
	return invalid, turns
}

func runDeterministicTrial() (*Observer, Stats, TierReport, []byte, error) {
	o := New()
	invalid, turns := deterministicWorkload()

	// 1. Initial valid tier observation
	o.ObserveTier(TierAccess{
		Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory,
		Bytes: 4096, BytesKnown: true, Latency: 200 * time.Microsecond, LatencyKnown: true,
	})

	// 2. Rejected tier accesses
	for _, a := range invalid {
		o.ObserveTier(a)
	}

	// 3. Valid observations across other tiers and outcomes
	o.ObserveTier(TierAccess{
		Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeMiss, Backend: BackendMemory,
		Bytes: 0, BytesKnown: true, Latency: 100 * time.Microsecond, LatencyKnown: true,
	})
	o.ObserveTier(TierAccess{
		Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote,
		Bytes: 65536, BytesKnown: true, Latency: 10 * time.Millisecond, LatencyKnown: true,
	})
	o.ObserveTier(TierAccess{
		Tier: TierSharedStore, Op: OpWrite, Outcome: OutcomeHit, Backend: BackendDisk,
		Bytes: 32768, BytesKnown: true, Latency: 5 * time.Millisecond, LatencyKnown: true,
	})
	o.ObserveTier(TierAccess{
		Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeError, Backend: BackendRemote,
		Bytes: 0, BytesKnown: false, Latency: 25 * time.Millisecond, LatencyKnown: true,
	})
	o.ObserveTier(TierAccess{
		Tier: TierProviderManaged, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote,
	})

	// 4. Interleaved turn observations (each turn emits a local prefix read)
	for _, tr := range turns {
		o.Observe(tr.prompt, tr.reused)
	}

	stats := o.Snapshot()
	report := o.TierSnapshot()
	rawJSON, err := json.Marshal(report)
	return o, stats, report, rawJSON, err
}

// TestCacheObs_Determinism proves that observation accounting across repeated
// independent runs produces byte-identical JSON and strictly identical structs,
// confirming that rejected tier observations do not introduce non-determinism,
// leak into tier rows, or corrupt aggregate accounting (#10096).
func TestCacheObs_Determinism(t *testing.T) {
	const trials = 50
	const wantRejected = 13
	const wantValidRequests = 9

	_, baseStats, baseReport, baseJSON, err := runDeterministicTrial()
	if err != nil {
		t.Fatalf("marshal trial 0: %v", err)
	}

	if baseStats.RejectedTierAccesses != wantRejected {
		t.Fatalf("stats rejected tier accesses = %d, want %d", baseStats.RejectedTierAccesses, wantRejected)
	}
	if baseReport.RejectedTierAccesses != wantRejected {
		t.Fatalf("report rejected tier accesses = %d, want %d", baseReport.RejectedTierAccesses, wantRejected)
	}
	if baseReport.Total.Requests != wantValidRequests {
		t.Fatalf("total requests = %d, want %d", baseReport.Total.Requests, wantValidRequests)
	}

	// Verify all trials match the baseline identically.
	for trial := 1; trial < trials; trial++ {
		_, stats, report, rawJSON, trialErr := runDeterministicTrial()
		if trialErr != nil {
			t.Fatalf("marshal trial %d: %v", trial, trialErr)
		}
		if !bytes.Equal(rawJSON, baseJSON) {
			t.Fatalf("trial %d produced different JSON:\ngot:  %s\nwant: %s", trial, rawJSON, baseJSON)
		}
		if !reflect.DeepEqual(stats, baseStats) {
			t.Fatalf("trial %d produced different Stats:\ngot:  %+v\nwant: %+v", trial, stats, baseStats)
		}
		if !reflect.DeepEqual(report, baseReport) {
			t.Fatalf("trial %d produced different TierReport:\ngot:  %+v\nwant: %+v", trial, report, baseReport)
		}
	}
}

// TestDeterminism is an alias ensuring standard determinism test discovery.
func TestDeterminism(t *testing.T) {
	TestCacheObs_Determinism(t)
}

// TestCacheObs_ConcurrentRaceWitness verifies that concurrent observation accounting
// under heavy contention — mixing valid tier accesses, rejected tier accesses, turn
// observations, and concurrent snapshot reads — is race-free and accounts every
// rejected and admitted access with exact precision (#10096).
func TestCacheObs_ConcurrentRaceWitness(t *testing.T) {
	const workers = 16
	const perWorker = 100
	const invalidPerTurn = 3

	o := New()
	start := make(chan struct{})
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			<-start
			for i := 0; i < perWorker; i++ {
				// 1. Rejected tier observations with varied invalid dimensions
				o.ObserveTier(TierAccess{
					Tier: CacheTier(-1 - workerID), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory,
				})
				o.ObserveTier(TierAccess{
					Tier: TierLocalPrefix, Op: TierOp(99 + i), Outcome: OutcomeHit, Backend: BackendMemory,
				})
				o.ObserveTier(TierAccess{
					Tier: numCacheTiers, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendClass(42),
				})

				// 2. Valid tier observations
				o.ObserveTier(TierAccess{
					Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote,
					Bytes: 100, BytesKnown: true, Latency: 50 * time.Microsecond, LatencyKnown: true,
				})
				o.ObserveTier(TierAccess{
					Tier: TierLocalPrefix, Op: OpWrite, Outcome: OutcomeMiss, Backend: BackendMemory,
				})

				// 3. Turn observation (books 1 TierLocalPrefix hit)
				o.Observe(200, 100)

				// 4. Concurrent snapshot reads under write contention
				_ = o.Snapshot()
				_ = o.TierSnapshot()
			}
		}(w)
	}

	close(start)
	wg.Wait()

	wantRejected := uint64(workers * perWorker * invalidPerTurn)
	wantTurns := uint64(workers * perWorker)
	wantSharedRequests := uint64(workers * perWorker)
	wantSharedBytes := uint64(workers * perWorker * 100)
	wantSharedLatency := uint64(workers*perWorker) * uint64(50*time.Microsecond)
	wantLocalRequests := uint64(workers * perWorker * 2) // 1 from turn hit + 1 from direct write miss

	stats := o.Snapshot()
	report := o.TierSnapshot()

	if stats.RejectedTierAccesses != wantRejected {
		t.Fatalf("stats rejected accesses = %d, want %d", stats.RejectedTierAccesses, wantRejected)
	}
	if report.RejectedTierAccesses != wantRejected {
		t.Fatalf("report rejected accesses = %d, want %d", report.RejectedTierAccesses, wantRejected)
	}
	if stats.Turns != wantTurns {
		t.Fatalf("stats turns = %d, want %d", stats.Turns, wantTurns)
	}

	shared, ok := report.Tier(TierSharedStore)
	if !ok {
		t.Fatal("shared_store row missing")
	}
	if shared.Requests != wantSharedRequests || shared.Hits != wantSharedRequests {
		t.Fatalf("shared requests/hits = (%d, %d), want (%d, %d)", shared.Requests, shared.Hits, wantSharedRequests, wantSharedRequests)
	}
	if shared.Bytes != wantSharedBytes || shared.LatencyNanos != wantSharedLatency {
		t.Fatalf("shared bytes/latency = (%d, %d), want (%d, %d)", shared.Bytes, shared.LatencyNanos, wantSharedBytes, wantSharedLatency)
	}

	local, ok := report.Tier(TierLocalPrefix)
	if !ok {
		t.Fatal("local_prefix row missing")
	}
	if local.Requests != wantLocalRequests || local.Hits != wantTurns || local.Misses != wantTurns {
		t.Fatalf("local requests/hits/misses = (%d, %d, %d), want (%d, %d, %d)",
			local.Requests, local.Hits, local.Misses, wantLocalRequests, wantTurns, wantTurns)
	}

	var sumRequests, sumHits, sumMisses uint64
	for _, ts := range report.Tiers {
		sumRequests += ts.Requests
		sumHits += ts.Hits
		sumMisses += ts.Misses
	}
	if sumRequests != report.Total.Requests || sumHits != report.Total.Hits || sumMisses != report.Total.Misses {
		t.Fatalf("per-tier sums disagree with total: sum=(%d,%d,%d) total=(%d,%d,%d)",
			sumRequests, sumHits, sumMisses, report.Total.Requests, report.Total.Hits, report.Total.Misses)
	}
	if report.Total.Requests != wantSharedRequests+wantLocalRequests {
		t.Fatalf("total requests = %d, want %d", report.Total.Requests, wantSharedRequests+wantLocalRequests)
	}
}
