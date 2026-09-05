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

func TestBlackboard_ZeroCopyVsJSONLatency(t *testing.T) {
	const numWorkers = 20
	const opsPerWorker = 500
	const totalOps = numWorkers * opsPerWorker

	// 1. Measure JSON serialization/parsing overhead
	jsonStart := time.Now()
	var jsonWg sync.WaitGroup
	jsonWg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer jsonWg.Done()
			idStr := fmt.Sprintf("worker-%d", workerID)
			for i := 0; i < opsPerWorker; i++ {
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
					t.Errorf("json.Marshal failed: %v", err)
					return
				}
				var parsed SubagentReportPayload
				if err := json.Unmarshal(data, &parsed); err != nil {
					t.Errorf("json.Unmarshal failed: %v", err)
					return
				}
			}
		}(w)
	}
	jsonWg.Wait()
	jsonDuration := time.Since(jsonStart)
	jsonLatencyPerOp := jsonDuration / time.Duration(totalOps)

	// 2. Measure Blackboard zero-copy pointer sharing
	bb := ctxmmu.NewBlackboard()
	bbStart := time.Now()
	var bbWg sync.WaitGroup
	bbWg.Add(numWorkers)

	sharedRef := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: []byte("simulated artifact content from worker"),
		Len:    38,
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer bbWg.Done()
			topic := fmt.Sprintf("bench:worker-%d", workerID)
			for i := 0; i < opsPerWorker; i++ {
				id, err := bb.Publish(topic, sharedRef, 1, nil)
				if err != nil {
					t.Errorf("bb.Publish failed: %v", err)
					return
				}
				entry, ok := bb.Lookup(id)
				if !ok || entry == nil || entry.Ref == nil {
					t.Errorf("bb.Lookup failed for %s", id)
					return
				}
				_ = entry.Ref.Inline
			}
		}(w)
	}
	bbWg.Wait()
	bbDuration := time.Since(bbStart)
	bbLatencyPerOp := bbDuration / time.Duration(totalOps)

	speedup := float64(jsonDuration) / float64(bbDuration)

	t.Logf("=== 20 Simulated Concurrent Workers Latency Comparison ===")
	t.Logf("Total Operations: %d (%d workers x %d ops)", totalOps, numWorkers, opsPerWorker)
	t.Logf("JSON Serialize/Parse Latency: %v/op (total %v)", jsonLatencyPerOp, jsonDuration)
	t.Logf("Blackboard Zero-Copy Latency: %v/op (total %v)", bbLatencyPerOp, bbDuration)
	t.Logf("Blackboard Performance Speedup: %.2fx", speedup)

	// Verify sub-microsecond latency (< 1 µs / 1000ns per operation)
	if bbLatencyPerOp >= time.Microsecond {
		t.Errorf("expected Blackboard zero-copy latency to be sub-microsecond (< 1µs), got %v", bbLatencyPerOp)
	}

	// Verify Blackboard is substantially faster than JSON
	if bbDuration >= jsonDuration {
		t.Errorf("expected Blackboard zero-copy (%v) to be faster than JSON (%v)", bbDuration, jsonDuration)
	}
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
