package compute

import (
	"fmt"
	"math"
)

// BitonicTopKEntry holds a selected value and its original token/kv index.
type BitonicTopKEntry struct {
	Value float32 `json:"value"`
	Index int32   `json:"index"`
}

// BitonicTopKReceipt records operational metrics for the persistent bitonic top-k pass.
type BitonicTopKReceipt struct {
	N           int    `json:"n"`
	K           int    `json:"k"`
	PaddedN     int    `json:"padded_n"`
	Stages      int    `json:"stages"`
	Comparisons int    `json:"comparisons"`
	Strategy    string `json:"strategy"`
}

// nextPowerOfTwo returns the smallest power of two >= n.
func nextPowerOfTwo(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// bitonicCompare returns true if entry A outranks entry B:
// Higher value wins; NaNs are treated as -Inf; ties broken by smaller index.
func bitonicCompare(a, b BitonicTopKEntry) bool {
	aNaN := math.IsNaN(float64(a.Value))
	bNaN := math.IsNaN(float64(b.Value))

	if aNaN && bNaN {
		return a.Index < b.Index
	}
	if aNaN {
		return false // B wins
	}
	if bNaN {
		return true // A wins
	}

	if a.Value != b.Value {
		return a.Value > b.Value
	}
	// Tie-break: smaller index wins
	return a.Index < b.Index
}

// PersistentBitonicTopK selects the top-k entries from up to N elements using a deterministic
// bitonic network. Ideal for sparse indexers where k approaches N and heap-based selection degrades.
func PersistentBitonicTopK(values []float32, k int) ([]BitonicTopKEntry, BitonicTopKReceipt, error) {
	n := len(values)
	var receipt BitonicTopKReceipt

	if n == 0 {
		return nil, receipt, fmt.Errorf("values must not be empty")
	}
	if k <= 0 || k > n {
		return nil, receipt, fmt.Errorf("k=%d must be in range [1, %d]", k, n)
	}

	paddedN := nextPowerOfTwo(n)
	entries := make([]BitonicTopKEntry, paddedN)

	// Populate entries; pad remaining slots with -Inf and index -1
	for i := 0; i < n; i++ {
		entries[i] = BitonicTopKEntry{
			Value: values[i],
			Index: int32(i),
		}
	}
	for i := n; i < paddedN; i++ {
		entries[i] = BitonicTopKEntry{
			Value: float32(math.Inf(-1)),
			Index: int32(i),
		}
	}

	stages := 0
	comparisons := 0

	// Bitonic sorting network: sorts in descending order (highest first)
	for segLen := 2; segLen <= paddedN; segLen <<= 1 {
		for inc := segLen >> 1; inc > 0; inc >>= 1 {
			stages++
			for i := 0; i < paddedN; i++ {
				partner := i ^ inc
				if partner > i {
					comparisons++
					// Direction for descending bitonic sort
					dirDesc := (i & segLen) == 0
					outranks := bitonicCompare(entries[i], entries[partner])

					if (dirDesc && !outranks) || (!dirDesc && outranks) {
						entries[i], entries[partner] = entries[partner], entries[i]
					}
				}
			}
		}
	}

	result := make([]BitonicTopKEntry, k)
	copy(result, entries[:k])

	receipt = BitonicTopKReceipt{
		N:           n,
		K:           k,
		PaddedN:     paddedN,
		Stages:      stages,
		Comparisons: comparisons,
		Strategy:    "bitonic-persistent",
	}

	return result, receipt, nil
}
