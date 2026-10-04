package model

import "testing"

// TestDenseDecodeKVLimitBoundsByDeviceRoom pins the process-wide device-room bound on a
// new dense decode KV mirror: the per-session budget is further capped by the device
// working set minus the model's resident weights and every other live mirror, and an
// unknown device budget proves nothing fits.
//
// fak-test:runtime fast est=10ms
func TestDenseDecodeKVLimitBoundsByDeviceRoom(t *testing.T) {
	const gib = int64(1) << 30
	frac := metalQ8UploadFraction
	for _, tc := range []struct {
		name                                    string
		budget, device, model, otherKV, wantMax int64
	}{
		{"roomy device: session budget binds", 4 * gib, 128 * gib, 8 * gib, 0, 4 * gib},
		{"tight device: room binds", 4 * gib, 16 * gib, 12 * gib, 0, int64(frac*float64(16*gib)) - 12*gib},
		{"other sessions' mirrors consume the room", 4 * gib, 16 * gib, 5 * gib, 10 * gib, 0},
		{"unknown device budget", 4 * gib, 0, 1 * gib, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := denseDecodeKVLimit(tc.budget, tc.device, tc.model, tc.otherKV)
			if got != tc.wantMax {
				t.Fatalf("denseDecodeKVLimit = %d, want %d", got, tc.wantMax)
			}
		})
	}
}

// TestDenseDecodeKVCapacityShrinksNearLimit pins the mirror sizing: two f32 sides (KPost,
// V) per token, geometric growth while room allows, a smaller final step near the
// limit, and a below-need result once the context no longer fits.
//
// fak-test:runtime fast est=10ms
func TestDenseDecodeKVCapacityShrinksNearLimit(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 64}
	perToken := int64(2 * 4 * 2 * 64 * 4)
	if got := denseDecodeKVCapacity(cfg, 100, 1<<40); got != 356 {
		t.Fatalf("roomy capacity = %d, want need+256 = 356", got)
	}
	if got := denseDecodeKVCapacity(cfg, 1000, 1<<40); got != 2000 {
		t.Fatalf("roomy capacity = %d, want 2x = 2000", got)
	}
	if got := denseDecodeKVCapacity(cfg, 1000, 1200*perToken); got != 1200 {
		t.Fatalf("near-limit capacity = %d, want the 1200 rows that fit", got)
	}
	if got := denseDecodeKVCapacity(cfg, 1000, 999*perToken); got >= 1000 {
		t.Fatalf("over-limit capacity = %d, want < need", got)
	}
}

// TestKVCacheMutationGenerationBumpsOnNonAppend pins the dense decode mirror's validity
// key: appends leave KVCache.mutationGeneration unchanged, and every non-append row
// mutation (Truncate, Evict, RestoreSpan, tree compaction, precision conversion) bumps
// it, so a device mirror keyed on it can never attend over rewritten rows.
//
// fak-test:runtime fast est=50ms
func TestKVCacheMutationGenerationBumpsOnNonAppend(t *testing.T) {
	cfg := denseSpanCfg()
	s := NewSynthetic(cfg).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41, 2, 19, 7, 11, 13})
	c := s.Cache
	bumps := func(label string, mutate func()) {
		t.Helper()
		before := c.mutationGeneration()
		mutate()
		if c.mutationGeneration() == before {
			t.Fatalf("%s did not bump the mutation generation", label)
		}
	}
	// SerializeSpan and tree compaction need the exact prefill lineage, so they run first.
	blob, err := c.SerializeSpan(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	bumps("Evict", func() { c.Evict(2, 3) })
	bumps("RestoreSpan", func() {
		if _, err := c.RestoreSpan(blob); err != nil {
			t.Fatal(err)
		}
	})
	bumps("PruneAndCompactTreeKV", func() {
		if err := PruneAndCompactTreeKV(c, c.Len()-3, []int{0, 1}, 3); err != nil {
			t.Fatal(err)
		}
	})
	gen := c.mutationGeneration()
	s.Step(29)
	if got := c.mutationGeneration(); got != gen {
		t.Fatalf("append bumped the mutation generation %d -> %d", gen, got)
	}
	bumps("Truncate", func() { c.Truncate(c.Len() - 1) })
	bumps("ConvertToPrecision", func() { c.ConvertToPrecision(KVPrecisionQ8_0) })
}
