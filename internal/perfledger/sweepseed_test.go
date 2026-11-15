package perfledger

import (
	"testing"
)

// TestStableSeed verifies the stable sweep seed derivation ported from
// aiperf's _stable_sweep_seed (ai-dynamo/aiperf@ab610ad, Apache-2.0): the
// first four bytes of SHA-256 of the canonical plan identity, read
// big-endian. The expected values are hand-computed with the source's own
// derivation (python3 hashlib.sha256, int.from_bytes(digest[:4], "big"))
// against the canonical identity strings.
func TestStableSeed(t *testing.T) {
	planA := SweepPlan{Axes: []Axis{
		{Name: "concurrency", Values: []string{"1", "8"}},
		{Name: "rate", Values: []string{"10", "20"}},
	}}
	// identity: "concurrency=1,8;rate=10,20"
	const wantA = uint32(0x0fdc8517)

	t.Run("determinism", func(t *testing.T) {
		t.Parallel()
		seed1, err := StableSweepSeed(planA)
		if err != nil {
			t.Fatalf("StableSweepSeed(planA) error: %v", err)
		}
		seed2, err := StableSweepSeed(planA)
		if err != nil {
			t.Fatalf("StableSweepSeed(planA) second call error: %v", err)
		}
		if seed1 != seed2 {
			t.Fatalf("same plan derived different seeds: 0x%08x vs 0x%08x", seed1, seed2)
		}
		if seed1 != wantA {
			t.Fatalf("seed = 0x%08x, want hand-computed source output 0x%08x", seed1, wantA)
		}
	})

	t.Run("axis order invariance", func(t *testing.T) {
		t.Parallel()
		planShuffled := SweepPlan{Axes: []Axis{
			{Name: "rate", Values: []string{"10", "20"}},
			{Name: "concurrency", Values: []string{"1", "8"}},
		}}
		seed, err := StableSweepSeed(planShuffled)
		if err != nil {
			t.Fatalf("StableSweepSeed(shuffled) error: %v", err)
		}
		if seed != wantA {
			t.Fatalf("shuffled axis order derived 0x%08x, want 0x%08x (same plan)", seed, wantA)
		}
	})

	t.Run("different plans differ", func(t *testing.T) {
		t.Parallel()
		planB := SweepPlan{Axes: []Axis{
			{Name: "concurrency", Values: []string{"1", "8"}},
			{Name: "rate", Values: []string{"10", "30"}},
		}}
		// identity: "concurrency=1,8;rate=10,30"
		const wantB = uint32(0xaffc0cab)
		seedB, err := StableSweepSeed(planB)
		if err != nil {
			t.Fatalf("StableSweepSeed(planB) error: %v", err)
		}
		if seedB != wantB {
			t.Fatalf("seed = 0x%08x, want hand-computed source output 0x%08x", seedB, wantB)
		}
		if seedB == wantA {
			t.Fatalf("different plans derived the same seed 0x%08x", seedB)
		}

		planC := SweepPlan{Axes: []Axis{
			{Name: "concurrency", Values: []string{"1", "8"}},
			{Name: "rate", Values: []string{"10", "20"}},
			{Name: "batch", Values: []string{"4"}},
		}}
		// identity: "batch=4;concurrency=1,8;rate=10,20" (axes sorted by name)
		const wantC = uint32(0x99518169)
		seedC, err := StableSweepSeed(planC)
		if err != nil {
			t.Fatalf("StableSweepSeed(planC) error: %v", err)
		}
		if seedC != wantC {
			t.Fatalf("seed = 0x%08x, want hand-computed source output 0x%08x", seedC, wantC)
		}
		if seedC == wantA || seedC == wantB {
			t.Fatalf("different plans derived a colliding seed 0x%08x", seedC)
		}
	})

	t.Run("unseeded plan rejected", func(t *testing.T) {
		t.Parallel()
		if _, err := StableSweepSeed(SweepPlan{}); err == nil {
			t.Fatal("StableSweepSeed(empty plan) = no error, want error (source raises ValueError for a missing identity)")
		}
	})
}
