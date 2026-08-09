package microagent

import (
	"reflect"
	"testing"
	"time"
)

func TestDeterminismCompatibilityPlanner(t *testing.T) {
	now := time.Unix(1000, 0)
	key := CompatibilityKey{Model: "m", Sampling: "s", Tools: "p", Prefix: "base", Phase: "decode", SequenceBucket: 8}
	items := []CompatibleWork{
		{ID: "b", Key: key, Tokens: 7, Priority: 1, Enqueued: now.Add(-2 * time.Second)},
		{ID: "a", Key: key, Tokens: 3, Priority: 1, Enqueued: now.Add(-3 * time.Second)},
		{ID: "c", Key: CompatibilityKey{Model: "other", Sampling: "s", Tools: "p", Prefix: "base", Phase: "decode", SequenceBucket: 8}, Tokens: 5, Enqueued: now.Add(-time.Second)},
	}
	cfg := CompatibilityConfig{MaxBatch: 2, MaxQueuePerClass: 8, MaxPadding: 1, StarvationAfter: time.Second, Now: now}
	firstBatches, firstStats, err := ComposeCompatible(items, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		batches, stats, err := ComposeCompatible(items, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(batches, firstBatches) || !reflect.DeepEqual(stats, firstStats) {
			t.Fatalf("run %d nondeterministic:\nfirst=%+v %+v\ngot=%+v %+v", i, firstBatches, firstStats, batches, stats)
		}
	}
}
