package gateway

import (
	"testing"
)

func TestResolveKVTPTransferStrategyWitness(t *testing.T) {
	// First witness requirements (#9919):
	// 1. Enumerate equal-TP (e.g. 4 -> 4) -> bijective pairing.
	// 2. Enumerate fan-out (e.g. 2 -> 4) -> broadcast/partitioning.
	// 3. Enumerate fan-in (e.g. 4 -> 2) -> gather.
	// 4. Enumerate incompatible layouts (e.g. 3 -> 5) -> fail-closed refusal.
	// 5. Verify every accepted pair is bijective or explicitly broadcast.

	numKVHeads := 8

	// 1. Equal-TP
	t.Run("equal_tp_bijective", func(t *testing.T) {
		rec, err := ResolveKVTPTransferStrategy(4, 4, numKVHeads)
		if err != nil {
			t.Fatalf("ResolveKVTPTransferStrategy failed: %v", err)
		}
		if rec.Mode != KVTPTransferBijective || !rec.IsBijective {
			t.Fatalf("expected bijective pairing, got mode=%s, isBijective=%t", rec.Mode, rec.IsBijective)
		}
		if len(rec.Mappings) != 4 {
			t.Fatalf("expected 4 mappings, got %d", len(rec.Mappings))
		}
		for i, m := range rec.Mappings {
			if m.SrcRank != i || m.DstRank != i {
				t.Fatalf("mapping %d not bijective: %+v", i, m)
			}
			if len(m.Heads) != 2 { // 8 heads / 4 ranks = 2 heads per rank
				t.Fatalf("mapping %d expected 2 heads, got %d", i, len(m.Heads))
			}
		}
	})

	// 2. Fan-out
	t.Run("fan_out_broadcast", func(t *testing.T) {
		rec, err := ResolveKVTPTransferStrategy(2, 4, numKVHeads)
		if err != nil {
			t.Fatalf("ResolveKVTPTransferStrategy failed: %v", err)
		}
		if rec.Mode != KVTPTransferFanOut || !rec.IsBroadcast {
			t.Fatalf("expected fan-out broadcast, got mode=%s, isBroadcast=%t", rec.Mode, rec.IsBroadcast)
		}
		if len(rec.Mappings) != 4 { // 4 destination ranks
			t.Fatalf("expected 4 mappings, got %d", len(rec.Mappings))
		}
	})

	// 3. Fan-in
	t.Run("fan_in_gather", func(t *testing.T) {
		rec, err := ResolveKVTPTransferStrategy(4, 2, numKVHeads)
		if err != nil {
			t.Fatalf("ResolveKVTPTransferStrategy failed: %v", err)
		}
		if rec.Mode != KVTPTransferFanIn {
			t.Fatalf("expected fan-in gather, got mode=%s", rec.Mode)
		}
		if len(rec.Mappings) != 4 { // 4 source ranks send to 2 destination ranks
			t.Fatalf("expected 4 mappings, got %d", len(rec.Mappings))
		}
	})

	// 4. Incompatible group layouts
	t.Run("incompatible_layouts", func(t *testing.T) {
		// Indivisible TP transition: TP=3 to TP=5
		_, err := ResolveKVTPTransferStrategy(3, 5, 15)
		if err == nil {
			t.Fatal("expected error on incompatible TP=3 to TP=5")
		}

		// NumKVHeads not divisible by TP
		_, err = ResolveKVTPTransferStrategy(4, 4, 7) // 7 not divisible by 4
		if err == nil {
			t.Fatal("expected error when numKVHeads is not divisible by TP")
		}
	})
}
