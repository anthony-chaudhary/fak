package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// SubagentReportPayload simulates an artifact report sent by a worker.
type SubagentReportPayload struct {
	SubagentID string            `json:"subagent_id"`
	Topic      string            `json:"topic"`
	Epoch      uint64            `json:"epoch"`
	Content    string            `json:"content"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Timestamp  string            `json:"timestamp"`
}

func TestBlackboard_SubagentSynthesis(t *testing.T) {
	bb := ctxmmu.NewBlackboard()
	coord := NewCoordinatorSynthesizer(bb)

	const numSubagents = 5
	const topic = "task:research:subagents"

	for i := 1; i <= numSubagents; i++ {
		subagentID := fmt.Sprintf("subagent-%d", i)
		content := []byte(fmt.Sprintf("findings-from-%s\n", subagentID))
		meta := map[string]string{"role": "researcher", "index": fmt.Sprintf("%d", i)}
		_, ref, err := PublishSubagentPayload(bb, topic, content, 1, subagentID, meta)
		if err != nil {
			t.Fatalf("PublishSubagentPayload for %s failed: %v", subagentID, err)
		}
		if ref == nil {
			t.Fatalf("expected non-nil ref for %s", subagentID)
		}
	}

	// Coordinator reads artifact references directly from the blackboard
	refs, err := coord.Collect(topic)
	if err != nil {
		t.Fatalf("coord.Collect failed: %v", err)
	}
	if len(refs) != numSubagents {
		t.Fatalf("expected %d refs, got %d", numSubagents, len(refs))
	}

	// Coordinator aggregates multiple subagent references zero-copy
	synthRef, err := AggregateSubagentRefs(refs, map[string]string{"synthesizer": "coordinator-1"})
	if err != nil {
		t.Fatalf("AggregateSubagentRefs failed: %v", err)
	}
	if synthRef == nil {
		t.Fatal("expected non-nil synthesized ref")
	}
	if synthRef.Kind != abi.RefRegion {
		t.Fatalf("expected RefRegion kind for synthesized report, got %v", synthRef.Kind)
	}

	expectedLen := int64(len("findings-from-subagent-1\n") * numSubagents)
	if synthRef.Len != expectedLen {
		t.Fatalf("expected total length %d, got %d", expectedLen, synthRef.Len)
	}

	// Verify zero-copy unpacking
	resolvedRefs, ok := ResolveSynthesizedRefs(synthRef)
	if !ok {
		t.Fatal("ResolveSynthesizedRefs failed to resolve composite report")
	}
	if len(resolvedRefs) != numSubagents {
		t.Fatalf("expected %d resolved refs, got %d", numSubagents, len(resolvedRefs))
	}

	// Verify materialized byte output
	materialized, err := MaterializeReportBytes(context.Background(), synthRef)
	if err != nil {
		t.Fatalf("MaterializeReportBytes failed: %v", err)
	}
	if int64(len(materialized)) != expectedLen {
		t.Fatalf("expected %d materialized bytes, got %d", expectedLen, len(materialized))
	}

	// Verify SynthesizeTopic publishes to new topic
	pubID, pubRef, err := coord.Synthesize(topic, "task:research:final", 1, map[string]string{"status": "finalized"})
	if err != nil {
		t.Fatalf("Synthesize failed: %v", err)
	}
	if pubID == "" || pubRef == nil {
		t.Fatal("Synthesize returned empty id or nil ref")
	}
	finalEntries := bb.Subscribe("task:research:final")
	if len(finalEntries) != 1 {
		t.Fatalf("expected 1 entry in final topic, got %d", len(finalEntries))
	}
}

func TestBlackboard_ConcurrentSimulatedWorkers(t *testing.T) {
	bb := ctxmmu.NewBlackboard()
	coord := NewCoordinatorSynthesizer(bb)
	const numWorkers = 20
	const topic = "topic:concurrent:workers"

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(workerIdx int) {
			defer wg.Done()
			workerID := fmt.Sprintf("worker-%02d", workerIdx)
			payload := []byte(fmt.Sprintf("payload-%s", workerID))
			_, _, err := PublishSubagentPayload(bb, topic, payload, 1, workerID, map[string]string{"worker": workerID})
			if err != nil {
				t.Errorf("worker %s publish failed: %v", workerID, err)
			}
		}(i)
	}

	wg.Wait()

	refs, err := coord.Collect(topic)
	if err != nil {
		t.Fatalf("coord.Collect failed: %v", err)
	}
	if len(refs) != numWorkers {
		t.Fatalf("expected %d subagent refs, got %d", numWorkers, len(refs))
	}

	reportRef, err := AggregateSubagentRefs(refs, nil)
	if err != nil {
		t.Fatalf("AggregateSubagentRefs failed: %v", err)
	}
	if reportRef == nil {
		t.Fatal("expected non-nil reportRef")
	}

	resolved, ok := ResolveSynthesizedRefs(reportRef)
	if !ok || len(resolved) != numWorkers {
		t.Fatalf("expected %d resolved refs, got %d", numWorkers, len(resolved))
	}
}

const (
	bbLatencyWorkers      = 20
	bbLatencyOpsPerWorker = 500
	bbLatencyTotalOps     = bbLatencyWorkers * bbLatencyOpsPerWorker
	// bbLatencyTrials bounds the sampling envelope: the comparison is repeated
	// this many times and the fastest trial of each path is used, so one
	// host-load spike on a shared CI/desktop host cannot decide the verdict.
	bbLatencyTrials = 9
	// bbLatencyMinSpeedup is the acceptance floor applied to the ratio of the
	// fastest JSON trial to the fastest Blackboard trial: it preserves the
	// original "Blackboard zero-copy is faster than JSON" invariant. Normalizing
	// by the comparator's own best trial cancels host-load drift common to both
	// paths; an 80-run characterization of the fastest-of-9 ratio on this host
	// measured min 1.07x, median 1.73x. The regression-sensitivity proof is NOT
	// this floor (a fixed threshold cannot survive host-load tail) but the
	// dedicated control test below, which slows Blackboard deliberately and must
	// fail the same comparison.
	bbLatencyMinSpeedup = 1.0
)

// measureJSONLatency runs the JSON serialize/parse comparator for one trial and
// returns its wall-clock duration for totalOps operations.
func measureJSONLatency() time.Duration {
	start := time.Now()
	var wg sync.WaitGroup
	wg.Add(bbLatencyWorkers)
	for w := 0; w < bbLatencyWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			idStr := fmt.Sprintf("worker-%d", workerID)
			for i := 0; i < bbLatencyOpsPerWorker; i++ {
				report := SubagentReportPayload{
					SubagentID: idStr,
					Topic:      "benchmark:json",
					Epoch:      1,
					Content:    "simulated artifact content from worker",
					Metadata:   map[string]string{"key": "value"},
					Timestamp:  "2026-09-05T12:00:00Z",
				}
				data, err := json.Marshal(report)
				if err != nil {
					panic(err)
				}
				var parsed SubagentReportPayload
				if err := json.Unmarshal(data, &parsed); err != nil {
					panic(err)
				}
			}
		}(w)
	}
	wg.Wait()
	return time.Since(start)
}

// measureBlackboardLatency runs the Blackboard zero-copy pointer-sharing path
// for one trial and returns its wall-clock duration for totalOps operations.
// When slow is true it adds a deliberate per-operation delay so the acceptance
// rule can be proven regression-sensitive.
func measureBlackboardLatency(slow bool) time.Duration {
	bb := ctxmmu.NewBlackboard()
	sharedRef := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: []byte("simulated artifact content from worker"),
		Len:    38,
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}

	start := time.Now()
	var wg sync.WaitGroup
	wg.Add(bbLatencyWorkers)
	for w := 0; w < bbLatencyWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			topic := fmt.Sprintf("bench:worker-%d", workerID)
			for i := 0; i < bbLatencyOpsPerWorker; i++ {
				if slow {
					time.Sleep(5 * time.Microsecond)
				}
				id, err := bb.Publish(topic, sharedRef, 1, nil)
				if err != nil {
					panic(err)
				}
				entry, ok := bb.Lookup(id)
				if !ok || entry == nil || entry.Ref == nil {
					panic(fmt.Sprintf("bb.Lookup failed for %s", id))
				}
				_ = entry.Ref.Inline
			}
		}(w)
	}
	wg.Wait()
	return time.Since(start)
}

// minDuration returns the smallest of the supplied durations. Host contention
// only adds time to a trial, so the minimum is the least-contended estimate of
// a path's intrinsic capability and is the stable regression signal.
func minDuration(ds []time.Duration) time.Duration {
	min := ds[0]
	for _, d := range ds[1:] {
		if d < min {
			min = d
		}
	}
	return min
}

func TestBlackboard_ZeroCopyVsJSONLatency(t *testing.T) {
	// Sampling envelope: bbLatencyTrials independent trials of the same
	// concurrent workload; the fastest trial of each path is compared. The
	// minimum cancels host-load contention (which only adds time), so the test
	// is stable under representative suite load while a real slow-path
	// regression (fastest Blackboard trial no longer beating fastest JSON) still
	// fails.
	jsonDurations := make([]time.Duration, bbLatencyTrials)
	bbDurations := make([]time.Duration, bbLatencyTrials)
	for trial := 0; trial < bbLatencyTrials; trial++ {
		jsonDurations[trial] = measureJSONLatency()
		bbDurations[trial] = measureBlackboardLatency(false)
	}

	minJSON := minDuration(jsonDurations)
	minBB := minDuration(bbDurations)
	minJSONLatency := minJSON / time.Duration(bbLatencyTotalOps)
	minBBLatency := minBB / time.Duration(bbLatencyTotalOps)
	speedup := float64(minJSON) / float64(minBB)

	t.Logf("=== %d Simulated Concurrent Workers Latency Comparison (fastest of %d trials) ===",
		bbLatencyWorkers, bbLatencyTrials)
	t.Logf("Total Operations: %d (%d workers x %d ops)", bbLatencyTotalOps, bbLatencyWorkers, bbLatencyOpsPerWorker)
	t.Logf("JSON Serialize/Parse Latency (fastest): %v/op (total %v)", minJSONLatency, minJSON)
	t.Logf("Blackboard Zero-Copy Latency (fastest): %v/op (total %v)", minBBLatency, minBB)
	t.Logf("Blackboard Performance Speedup (fastest-of-%d): %.2fx", bbLatencyTrials, speedup)

	// Regression-sensitive, host-load-normalized acceptance rule: the fastest
	// Blackboard trial must remain faster than the fastest JSON trial. The
	// ratio, not a fixed ns threshold, is the signal.
	if speedup < bbLatencyMinSpeedup {
		t.Errorf("expected Blackboard zero-copy to be at least %.2fx faster than JSON, got %.2fx (blackboard %v vs json %v)",
			bbLatencyMinSpeedup, speedup, minBB, minJSON)
	}
}

// TestBlackboard_LatencyGuardRejectsRegression proves the acceptance rule above
// is regression-sensitive: a deliberately slowed Blackboard path must fail the
// same speedup comparison that the healthy path passes.
func TestBlackboard_LatencyGuardRejectsRegression(t *testing.T) {
	jsonDur := measureJSONLatency()
	slowBB := measureBlackboardLatency(true)
	speedup := float64(jsonDur) / float64(slowBB)

	if speedup >= bbLatencyMinSpeedup {
		t.Fatalf("guard failed to reject an intentional slow-path regression: speedup %.2fx >= floor %.1fx",
			speedup, bbLatencyMinSpeedup)
	}
	t.Logf("intentional slow-path control correctly rejected: speedup %.2fx < floor %.1fx", speedup, bbLatencyMinSpeedup)
}

func BenchmarkJSONSerialization_20Workers(b *testing.B) {
	b.SetParallelism(20)
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		workerID := 1
		for pb.Next() {
			report := SubagentReportPayload{
				SubagentID: fmt.Sprintf("subagent-%d", workerID),
				Topic:      "bench:json",
				Epoch:      1,
				Content:    "synthesized report payload data for benchmark testing",
				Metadata:   map[string]string{"type": "audit", "tier": "pdp"},
				Timestamp:  "2026-09-05T12:00:00Z",
			}
			bytes, err := json.Marshal(report)
			if err != nil {
				b.Fatal(err)
			}
			var parsed SubagentReportPayload
			if err := json.Unmarshal(bytes, &parsed); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkBlackboardZeroCopy_20Workers(b *testing.B) {
	bb := ctxmmu.NewBlackboard()
	payload := []byte("synthesized report payload data for benchmark testing")
	ref := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload,
		Len:    int64(len(payload)),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}

	var workerCounter int64
	b.SetParallelism(20)
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		wID := atomic.AddInt64(&workerCounter, 1)
		topic := fmt.Sprintf("bench:topic:%d", wID)
		for pb.Next() {
			id, err := bb.Publish(topic, ref, 1, nil)
			if err != nil {
				b.Fatal(err)
			}
			entry, ok := bb.Lookup(id)
			if !ok || entry == nil {
				b.Fatal("lookup failed")
			}
			_ = entry.Ref.Inline
		}
	})
}

func BenchmarkBlackboardAggregation_20Workers(b *testing.B) {
	payload := []byte("subagent piece")
	ref := &abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))}
	refs := make([]*abi.Ref, 20)
	for i := range refs {
		refs[i] = ref
	}

	b.SetParallelism(20)
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			synth, err := AggregateSubagentRefs(refs, nil)
			if err != nil {
				b.Fatal(err)
			}
			resolved, ok := ResolveSynthesizedRefs(synth)
			if !ok || len(resolved) != 20 {
				b.Fatal("resolve failed")
			}
		}
	})
}
