//go:build !amd64

package model

import "unsafe"

func hasAVX512() bool {
	return false
}

func scanTokenSubsequenceAVX512(tokens, pattern []int) int {
	return firstTokenSubsequenceScalar(tokens, pattern)
}

func scanTokenSubsequenceI32AVX512(tokens, pattern []int32) int {
	return firstTokenSubsequenceI32Scalar(tokens, pattern)
}

func findContinuationBranchesAVX512(haystack, pattern []int, maxDraft, maxBranches int) [][]int {
	return findContinuationBranchesScalar(haystack, pattern, maxDraft, maxBranches)
}

func draftTreeAVX512(history []int, minMatch, maxMatch, maxDraft, maxBranches int) *CandidateDraftTree {
	return draftTreeScalar(history, minMatch, maxMatch, maxDraft, maxBranches)
}

func scanNextMatchAVX512_I32(tokens *int32, start int, n int, target int32) int {
	if tokens == nil || start < 0 || start >= n {
		return -1
	}
	slice := unsafe.Slice(tokens, n)
	for i := start; i < n; i++ {
		if slice[i] == target {
			return i
		}
	}
	return -1
}

func scanNextMatchAVX512_I64(tokens *int, start int, n int, target int) int {
	if tokens == nil || start < 0 || start >= n {
		return -1
	}
	slice := unsafe.Slice(tokens, n)
	for i := start; i < n; i++ {
		if slice[i] == target {
			return i
		}
	}
	return -1
}
