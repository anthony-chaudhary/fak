package compute

import (
	"fmt"
	"math"
)

// SplitKReductionMode defines the explicit reduction contract for Split-K matrix multiplication.
type SplitKReductionMode string

const (
	// SplitKModeDeterministic guarantees exact bitwise repeatability across runs
	// by enforcing an ordered, deterministic reduction sequence over splits.
	SplitKModeDeterministic SplitKReductionMode = "deterministic"

	// SplitKModeAtomic uses concurrent atomic accumulation for peak throughput,
	// where floating-point addition order varies across thread completion order.
	SplitKModeAtomic SplitKReductionMode = "atomic"
)

// SplitKContract specifies the parameters and execution mode for a Split-K GEMM.
type SplitKContract struct {
	Mode         SplitKReductionMode `json:"mode"`
	Splits       int                 `json:"splits"`
	M            int                 `json:"m"`
	N            int                 `json:"n"`
	K            int                 `json:"k"`
	ExecutionTag string              `json:"execution_tag"`
}

// SplitKReceipt records the reduction mode, ordering, and repeatability evidence.
type SplitKReceipt struct {
	Mode           SplitKReductionMode `json:"mode"`
	Splits         int                 `json:"splits"`
	ReductionOrder string              `json:"reduction_order"`
	BitExactRepeat bool                `json:"bit_exact_repeat"`
	MaxRunDrift    float32             `json:"max_run_drift"`
}

// ExecuteSplitKGEMM executes matrix multiplication C = A * B using the explicit Split-K contract.
// For K-heavy shapes, K is partitioned into contract.Splits independent reductions.
func ExecuteSplitKGEMM(A, B, C []float32, contract SplitKContract, runSeed int) (SplitKReceipt, error) {
	var receipt SplitKReceipt
	m, n, k := contract.M, contract.N, contract.K
	splits := contract.Splits

	if m <= 0 || n <= 0 || k <= 0 {
		return receipt, fmt.Errorf("dimensions must be positive: M=%d, N=%d, K=%d", m, n, k)
	}
	if splits <= 0 || splits > k {
		return receipt, fmt.Errorf("invalid splits=%d for K=%d", splits, k)
	}
	if contract.Mode != SplitKModeDeterministic && contract.Mode != SplitKModeAtomic {
		return receipt, fmt.Errorf("unknown split-K mode: %q", contract.Mode)
	}
	if len(A) != m*k || len(B) != k*n || len(C) != m*n {
		return receipt, fmt.Errorf("slice length mismatch: len(A)=%d, len(B)=%d, len(C)=%d", len(A), len(B), len(C))
	}

	splitSize := (k + splits - 1) / splits

	// Compute partial sums per split
	partials := make([][]float32, splits)
	for s := 0; s < splits; s++ {
		partials[s] = make([]float32, m*n)
		kStart := s * splitSize
		kEnd := kStart + splitSize
		if kEnd > k {
			kEnd = k
		}

		for r := 0; r < m; r++ {
			rowA := r * k
			rowOut := r * n
			for c := 0; c < n; c++ {
				var dot float64
				for kk := kStart; kk < kEnd; kk++ {
					dot += float64(A[rowA+kk]) * float64(B[kk*n+c])
				}
				partials[s][rowOut+c] = float32(dot)
			}
		}
	}

	// Reduce partials into C
	for i := range C {
		C[i] = 0
	}

	var reductionOrder string
	if contract.Mode == SplitKModeDeterministic {
		reductionOrder = "sequential-0-to-splits"
		// Deterministic ordered reduction: always 0, 1, ..., splits-1
		for s := 0; s < splits; s++ {
			for i := 0; i < m*n; i++ {
				C[i] += partials[s][i]
			}
		}
	} else {
		// Atomic / out-of-order reduction simulation:
		// Order depends on runSeed permutation (simulating race-to-arrive atomic adds)
		reductionOrder = fmt.Sprintf("atomic-unordered-seed-%d", runSeed)
		order := make([]int, splits)
		for s := 0; s < splits; s++ {
			order[s] = s
		}
		// Permute order based on runSeed
		for s := splits - 1; s > 0; s-- {
			j := (runSeed*37 + s*13) % (s + 1)
			order[s], order[j] = order[j], order[s]
		}

		for _, s := range order {
			for i := 0; i < m*n; i++ {
				C[i] += partials[s][i]
			}
		}
	}

	receipt = SplitKReceipt{
		Mode:           contract.Mode,
		Splits:         splits,
		ReductionOrder: reductionOrder,
		BitExactRepeat: contract.Mode == SplitKModeDeterministic,
		MaxRunDrift:    0.0,
	}

	return receipt, nil
}

// CompareSplitKRuns executes two runs with different schedule seeds and evaluates numeric drift.
func CompareSplitKRuns(A, B []float32, contract SplitKContract, seed1, seed2 int) (SplitKReceipt, error) {
	c1 := make([]float32, contract.M*contract.N)
	c2 := make([]float32, contract.M*contract.N)

	rec1, err := ExecuteSplitKGEMM(A, B, c1, contract, seed1)
	if err != nil {
		return rec1, err
	}
	_, err = ExecuteSplitKGEMM(A, B, c2, contract, seed2)
	if err != nil {
		return rec1, err
	}

	var maxDrift float32
	bitExact := true
	for i := range c1 {
		if c1[i] != c2[i] {
			bitExact = false
			drift := float32(math.Abs(float64(c1[i] - c2[i])))
			if drift > maxDrift {
				maxDrift = drift
			}
		}
	}

	rec1.BitExactRepeat = bitExact
	rec1.MaxRunDrift = maxDrift
	return rec1, nil
}
