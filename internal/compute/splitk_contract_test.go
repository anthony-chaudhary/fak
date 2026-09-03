package compute

import (
	"math/rand"
	"testing"
)

func TestSplitKContractWitness(t *testing.T) {
	// First witness requirements (#9935):
	// 1. Repeated-run bit identity for deterministic mode (max drift == 0, BitExactRepeat == true).
	// 2. Bounded error distribution for atomic mode (due to non-associative float addition ordering).
	// 3. Receipt names mode and reduction order explicitly.
	// 4. K-heavy shapes (e.g. M=8, N=8, K=4096, Splits=16).

	m, n, k := 8, 8, 4096
	splits := 16

	rng := rand.New(rand.NewSource(9935))
	A := make([]float32, m*k)
	B := make([]float32, k*n)

	for i := range A {
		A[i] = rng.Float32()*2 - 1
	}
	for i := range B {
		B[i] = rng.Float32()*2 - 1
	}

	// 1. Deterministic Mode: must be 100% bit-exact across runs with different arrival seeds
	t.Run("deterministic_mode_bit_exact", func(t *testing.T) {
		contract := SplitKContract{
			Mode:         SplitKModeDeterministic,
			Splits:       splits,
			M:            m,
			N:            n,
			K:            k,
			ExecutionTag: "test-deterministic-splitk",
		}

		receipt, err := CompareSplitKRuns(A, B, contract, 1, 2)
		if err != nil {
			t.Fatalf("CompareSplitKRuns failed: %v", err)
		}

		if receipt.Mode != SplitKModeDeterministic {
			t.Fatalf("expected mode %v, got %v", SplitKModeDeterministic, receipt.Mode)
		}
		if receipt.ReductionOrder != "sequential-0-to-splits" {
			t.Fatalf("expected sequential reduction order, got %s", receipt.ReductionOrder)
		}
		if !receipt.BitExactRepeat {
			t.Fatalf("deterministic mode failed bit-exact repeat: max drift %v", receipt.MaxRunDrift)
		}
		if receipt.MaxRunDrift != 0.0 {
			t.Fatalf("deterministic mode drift = %v, want 0.0", receipt.MaxRunDrift)
		}
	})

	// 2. Atomic Mode: allows small bounded float non-associative rounding drift
	t.Run("atomic_mode_bounded_drift", func(t *testing.T) {
		contract := SplitKContract{
			Mode:         SplitKModeAtomic,
			Splits:       splits,
			M:            m,
			N:            n,
			K:            k,
			ExecutionTag: "test-atomic-splitk",
		}

		receipt, err := CompareSplitKRuns(A, B, contract, 1, 2)
		if err != nil {
			t.Fatalf("CompareSplitKRuns failed: %v", err)
		}

		if receipt.Mode != SplitKModeAtomic {
			t.Fatalf("expected mode %v, got %v", SplitKModeAtomic, receipt.Mode)
		}
		// In atomic mode with varying addition order, drift should be small but non-zero
		// for high K accumulation, and bounded by machine precision
		if receipt.MaxRunDrift > 1e-3 {
			t.Fatalf("atomic mode drift %v exceeds bound 1e-3", receipt.MaxRunDrift)
		}
	})

	// 3. Fail closed on invalid parameters
	t.Run("fail_closed", func(t *testing.T) {
		badContract := SplitKContract{
			Mode:   "invalid-mode",
			Splits: 4,
			M:      4,
			N:      4,
			K:      16,
		}
		C := make([]float32, 16)
		if _, err := ExecuteSplitKGEMM(make([]float32, 64), make([]float32, 64), C, badContract, 0); err == nil {
			t.Fatal("expected error on invalid mode")
		}

		badSplitsContract := SplitKContract{
			Mode:   SplitKModeDeterministic,
			Splits: 0,
			M:      4,
			N:      4,
			K:      16,
		}
		if _, err := ExecuteSplitKGEMM(make([]float32, 64), make([]float32, 64), C, badSplitsContract, 0); err == nil {
			t.Fatal("expected error on splits <= 0")
		}
	})
}
