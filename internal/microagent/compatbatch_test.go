package microagent_test

import (
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"testing"
	"time"
)

func TestCompatibilityPlannerIsolatesKeysBoundsPaddingAndFailsOpen(t *testing.T) {
	now := time.Unix(100, 0)
	a := microagent.CompatibilityKey{Model: "m", Sampling: "t0", Tools: "none", Prefix: "p", Phase: "prefill", SequenceBucket: 128}
	b := a
	b.Tools = "read"
	in := []microagent.CompatibleWork{{ID: "a", Key: a, Tokens: 100, Enqueued: now.Add(-time.Second)}, {ID: "b", Key: a, Tokens: 110, Enqueued: now.Add(-time.Second)}, {ID: "c", Key: b, Tokens: 105, Enqueued: now.Add(-time.Second)}, {ID: "unknown", Tokens: 50, Key: microagent.CompatibilityKey{Model: "m"}}}
	batches, s, err := microagent.ComposeCompatible(in, microagent.CompatibilityConfig{MaxBatch: 4, MaxQueuePerClass: 8, MaxPadding: .1, StarvationAfter: time.Second, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 3 || s.SingletonFallbacks != 1 || s.PaddingTax > .1 {
		t.Fatalf("batches=%+v stats=%+v", batches, s)
	}
	for _, x := range batches {
		for _, id := range x.IDs {
			if id == "c" && len(x.IDs) > 1 {
				t.Fatal("incompatible tool key coalesced")
			}
		}
	}
}
func TestCompatibilityPlannerAgingPreventsStarvationAndCancellationFreesWork(t *testing.T) {
	now := time.Unix(200, 0)
	k := microagent.CompatibilityKey{Model: "m", Sampling: "t0", Tools: "none", Phase: "decode", SequenceBucket: 64}
	in := []microagent.CompatibleWork{{ID: "old", Key: k, Tokens: 10, Priority: 0, Enqueued: now.Add(-10 * time.Second)}, {ID: "new", Key: k, Tokens: 10, Priority: 5, Enqueued: now}, {ID: "cancel", Key: k, Tokens: 10, Priority: 99, Cancelled: true}}
	b, s, err := microagent.ComposeCompatible(in, microagent.CompatibilityConfig{MaxBatch: 1, MaxQueuePerClass: 8, MaxPadding: 0, StarvationAfter: time.Second, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if b[0].IDs[0] != "old" || s.Cancelled != 1 || s.Scheduled != 2 {
		t.Fatalf("batches=%+v stats=%+v", b, s)
	}
}
