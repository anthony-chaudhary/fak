package agent

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// fak-test:runtime fast est=100ms
func TestPrefillChunkRequestObservation(t *testing.T) {
	previous := enginestep.Default
	recorder := enginestep.New(32)
	enginestep.Default = recorder
	t.Cleanup(func() { enginestep.Default = previous })
	planner := qwenQ4KPrefillPlanner(namedPrefillBackend{Backend: compute.Default(), name: "vulkan"})
	planner.qwenQ4KPrefillChunkTokens = 128
	ctx := enginestep.WithRequestObservation(context.Background(), "request-one", "load-one")
	session := &cancelPrefillSession{}
	if _, err := planner.prefillDivergentSuffix(ctx, session, make([]int, 129)); err != nil {
		t.Fatal(err)
	}
	if _, err := planner.prefillDivergentSuffix(ctx, session, make([]int, 7)); err != nil {
		t.Fatal(err)
	}
	snapshot := recorder.Snapshot(32, enginestep.KindPrefillChunk)
	if len(snapshot.Recent) != 3 {
		t.Fatalf("completed chunks=%d", len(snapshot.Recent))
	}
	starts := []uint64{0, 128, 129}
	counts := []int{128, 1, 7}
	for i, row := range snapshot.Recent {
		if row.RequestID != "request-one" || row.LoadID != "load-one" || row.ChunkOrdinal == nil || *row.ChunkOrdinal != uint64(i+1) || row.ComputedTokenStart == nil || *row.ComputedTokenStart != starts[i] || row.Tokens != counts[i] || row.TokenRangeOrigin != "computed_prompt_stream" || row.DurationNS < 0 || row.Seq == 0 || row.AtUnixNano == 0 {
			t.Fatalf("chunk%d=%+v", i, row)
		}
	}
	other := enginestep.WithRequestObservation(context.Background(), "request-two", "load-one")
	recorder.ObservePrefillChunkContext(other, 4, time.Nanosecond)
	last := recorder.Snapshot(1, enginestep.KindPrefillChunk).Recent[0]
	if last.ChunkOrdinal == nil || *last.ChunkOrdinal != 1 || last.ComputedTokenStart == nil || *last.ComputedTokenStart != 0 {
		t.Fatal("request streams share counters")
	}
}

// fak-test:runtime fast est=100ms
func TestPrefillChunkCancellationAndUnknownIdentity(t *testing.T) {
	previous := enginestep.Default
	recorder := enginestep.New(8)
	enginestep.Default = recorder
	t.Cleanup(func() { enginestep.Default = previous })
	planner := qwenQ4KPrefillPlanner(namedPrefillBackend{Backend: compute.Default(), name: "vulkan"})
	planner.qwenQ4KPrefillChunkTokens = 128
	parent := enginestep.WithRequestObservation(context.Background(), "request-cancel", "load-one")
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	session := &cancelPrefillSession{cancel: cancel, cancelAfter: 1}
	if _, err := planner.prefillDivergentSuffix(ctx, session, make([]int, 257)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	snapshot := recorder.Snapshot(8, enginestep.KindPrefillChunk)
	if len(snapshot.Recent) != 1 || snapshot.Recent[0].Tokens != 128 || snapshot.Recent[0].RequestID != "request-cancel" {
		t.Fatalf("partial=%+v", snapshot)
	}
	traceOnly := WithRequestTraceID(context.Background(), "session-not-request")
	if _, err := planner.prefillDivergentSuffix(traceOnly, &cancelPrefillSession{}, make([]int, 5)); err != nil {
		t.Fatal(err)
	}
	last := recorder.Snapshot(1, enginestep.KindPrefillChunk).Recent[0]
	if last.RequestID != "" || last.LoadID != "" || last.ChunkOrdinal != nil || last.ComputedTokenStart != nil || last.TokenRangeOrigin != "" {
		t.Fatal("missing unique carrier fabricated identity")
	}
}

// fak-test:runtime fast est=100ms
func TestPrefillChunkBoundedRingOverwrite(t *testing.T) {
	recorder := enginestep.New(2)
	ctx := enginestep.WithRequestObservation(nil, "request-one", "load-one")
	for i := 0; i < 5; i++ {
		recorder.ObservePrefillChunkContext(ctx, 3, time.Nanosecond)
	}
	snapshot := recorder.Snapshot(512, enginestep.KindPrefillChunk)
	if snapshot.RingCapacity != 2 || snapshot.RingRecordsOverwritten != 3 || len(snapshot.Recent) != 2 || snapshot.Recent[0].Seq != 4 || snapshot.Recent[1].Seq != 5 || *snapshot.Recent[0].ComputedTokenStart != 9 {
		t.Fatalf("snapshot=%+v", snapshot)
	}
	id, load := enginestep.RequestObservationFromContext(ctx)
	if id != "request-one" || load != "load-one" {
		t.Fatal("immutable identity unavailable")
	}
	missingID, missingLoad := enginestep.RequestObservationFromContext(context.Background())
	if missingID != "" || missingLoad != "" {
		t.Fatal("unknown context has identity")
	}
}
