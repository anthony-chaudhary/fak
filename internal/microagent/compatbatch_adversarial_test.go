package microagent

import (
	"testing"
	"time"
)

func TestAdversarialCompatibilityInputsStayBounded(t *testing.T) {
	now := time.Unix(100, 0)
	key := CompatibilityKey{Model: "m", Sampling: "s", Tools: "t", Prefix: "p", Phase: "decode", SequenceBucket: 8}
	tests := []struct {
		name                                       string
		work                                       []CompatibleWork
		cfg                                        CompatibilityConfig
		wantRejected, wantCancelled, wantSingleton int
	}{
		{"empty", nil, CompatibilityConfig{MaxBatch: 2, MaxQueuePerClass: 2, MaxPadding: 1, Now: now}, 0, 0, 0},
		{"oversized-queue", []CompatibleWork{{ID: "a", Key: key, Tokens: 1}, {ID: "b", Key: key, Tokens: 1}, {ID: "c", Key: key, Tokens: 1}}, CompatibilityConfig{MaxBatch: 2, MaxQueuePerClass: 2, MaxPadding: 1, Now: now}, 1, 0, 0},
		{"cancelled", []CompatibleWork{{ID: "a", Key: key, Tokens: 1, Cancelled: true}}, CompatibilityConfig{MaxBatch: 2, MaxQueuePerClass: 2, MaxPadding: 1, Now: now}, 0, 1, 0},
		{"incomplete-key", []CompatibleWork{{ID: "a", Key: CompatibilityKey{Model: "m"}, Tokens: 1}}, CompatibilityConfig{MaxBatch: 2, MaxQueuePerClass: 2, MaxPadding: 1, Now: now}, 0, 0, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			batches, stats, err := ComposeCompatible(tc.work, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if stats.Rejected != tc.wantRejected || stats.Cancelled != tc.wantCancelled || stats.SingletonFallbacks != tc.wantSingleton {
				t.Fatalf("stats=%+v batches=%+v", stats, batches)
			}
			for _, b := range batches {
				if len(b.IDs) > tc.cfg.MaxBatch {
					t.Fatalf("batch exceeded cap: %+v", b)
				}
			}
		})
	}
}
