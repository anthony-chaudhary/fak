package engine

import (
	"context"
	"testing"
)

func TestRecurrentLoopedBlockLayerVisits(t *testing.T) {
	shape := LoopedBlockShape{PreludeLayers: 6, LoopLayers: 12, CodaLayers: 6}
	if got := LoopedBlockFromCoreRange(24, 6, 18); got != shape {
		t.Fatalf("LoopedBlockFromCoreRange(24,6,18) = %+v, want %+v", got, shape)
	}
	for _, tc := range []struct{ loops, want int }{{0, 24}, {1, 24}, {3, 48}, {5, 72}} {
		if got := shape.LayerVisits(tc.loops); got != tc.want {
			t.Fatalf("LayerVisits(%d) = %d, want %d", tc.loops, got, tc.want)
		}
	}
	if (LoopedBlockShape{}).Declared() {
		t.Fatal("zero shape must not be declared")
	}
}

func TestRecurrentAdmissionKVBytesFollowLayerVisits(t *testing.T) {
	const bpt = 2400 // one unrolled 24-layer pass: 100 bytes per layer per token
	shape := LoopedBlockShape{PreludeLayers: 6, LoopLayers: 12, CodaLayers: 6}
	for _, tc := range []struct {
		name   string
		shape  LoopedBlockShape
		depth  int
		visits int64
	}{
		{"declared L=1", shape, 1, 24},
		{"declared L=3", shape, 3, 48},
		{"declared L=5", shape, 5, 72},
		{"undeclared L=3 keeps full stack per pass", LoopedBlockShape{}, 3, 72},
		{"undeclared L=5 keeps full stack per pass", LoopedBlockShape{}, 5, 120},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cb, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 1, KVBytesPerToken: bpt, LoopedBlock: tc.shape})
			if err != nil {
				t.Fatalf("NewContinuousBatcher: %v", err)
			}
			defer cb.Close()
			req := &SubagentRequest{PromptTokens: make([]int, 10), TargetTokens: 10, ExecutionDepth: tc.depth}
			want := int64(20) * 100 * tc.visits
			if got := cb.RequiredKVCacheBytes(req); got != want {
				t.Fatalf("RequiredKVCacheBytes = %d, want %d", got, want)
			}
		})
	}
}

func TestRecurrentPassCountIsLoopsPerToken(t *testing.T) {
	cb, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 1})
	if err != nil {
		t.Fatalf("NewContinuousBatcher: %v", err)
	}
	defer cb.Close()
	id, err := cb.Submit(&SubagentRequest{PromptTokens: []int{1, 2}, ExecutionDepth: 3})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := cb.Step(context.Background()); err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
	}
	slot, ok := cb.GetSlot(id)
	if !ok {
		t.Fatalf("GetSlot(%s) missing", id)
	}
	if got := len(slot.GeneratedTokens); got != 3 {
		t.Fatalf("generated %d tokens, want 3", got)
	}
	if got := slot.RecurrentPasses; got != 9 {
		t.Fatalf("RecurrentPasses = %d, want 9 (3 tokens x 3 loops)", got)
	}
}
