//go:build amd64

package model

import (
	"os"
	"strings"
)

// External assembly declarations
//
//go:noescape
func scanNextMatchAVX512_I32(tokens *int32, start int, n int, target int32) int

//go:noescape
func scanNextMatchAVX512_I64(tokens *int, start int, n int, target int) int

// hasAVX512 reports whether AVX-512 vector scanning is supported by the CPU
// and operating system, and not disabled by the FAK_NGRAM_AVX512 environment variable.
func hasAVX512() bool {
	if v := os.Getenv("FAK_NGRAM_AVX512"); v != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "0", "false", "off", "no", "disable", "disabled":
			return false
		}
	}
	return detectAVX512()
}

// scanTokenSubsequenceAVX512 finds the first occurrence of pattern in tokens
// using vectorized AVX-512 64-bit integer scanning.
func scanTokenSubsequenceAVX512(tokens, pattern []int) int {
	if !hasAVX512() {
		return firstTokenSubsequenceScalar(tokens, pattern)
	}
	m := len(pattern)
	n := len(tokens)
	if m == 0 {
		return 0
	}
	if m > n {
		return -1
	}
	target := pattern[0]
	searchBound := n - m + 1
	start := 0
	for start < searchBound {
		idx := scanNextMatchAVX512_I64(&tokens[0], start, searchBound, target)
		if idx < 0 {
			return -1
		}
		matched := true
		for i := 1; i < m; i++ {
			if tokens[idx+i] != pattern[i] {
				matched = false
				break
			}
		}
		if matched {
			return idx
		}
		start = idx + 1
	}
	return -1
}

// scanTokenSubsequenceI32AVX512 finds the first occurrence of pattern in tokens
// using vectorized AVX-512 32-bit integer scanning.
func scanTokenSubsequenceI32AVX512(tokens, pattern []int32) int {
	if !hasAVX512() {
		return firstTokenSubsequenceI32Scalar(tokens, pattern)
	}
	m := len(pattern)
	n := len(tokens)
	if m == 0 {
		return 0
	}
	if m > n {
		return -1
	}
	target := pattern[0]
	searchBound := n - m + 1
	start := 0
	for start < searchBound {
		idx := scanNextMatchAVX512_I32(&tokens[0], start, searchBound, target)
		if idx < 0 {
			return -1
		}
		matched := true
		for i := 1; i < m; i++ {
			if tokens[idx+i] != pattern[i] {
				matched = false
				break
			}
		}
		if matched {
			return idx
		}
		start = idx + 1
	}
	return -1
}

// findContinuationBranchesAVX512 extracts up to maxBranches continuation branches from haystack for pattern.
func findContinuationBranchesAVX512(haystack, pattern []int, maxDraft, maxBranches int) [][]int {
	if !hasAVX512() {
		return findContinuationBranchesScalar(haystack, pattern, maxDraft, maxBranches)
	}
	m := len(pattern)
	hLen := len(haystack)
	if m == 0 || hLen <= m || maxDraft <= 0 || maxBranches <= 0 {
		return nil
	}
	searchBound := hLen - m
	target := pattern[0]
	var branches [][]int
	start := 0
	for start < searchBound {
		matchIdx := scanNextMatchAVX512_I64(&haystack[0], start, searchBound, target)
		if matchIdx < 0 {
			break
		}
		matched := true
		for i := 1; i < m; i++ {
			if haystack[matchIdx+i] != pattern[i] {
				matched = false
				break
			}
		}
		if matched {
			contStart := matchIdx + m
			contEnd := contStart + maxDraft
			if contEnd > hLen {
				contEnd = hLen
			}
			if contEnd > contStart {
				branch := make([]int, contEnd-contStart)
				copy(branch, haystack[contStart:contEnd])
				branches = append(branches, branch)
				if len(branches) >= maxBranches {
					break
				}
			}
		}
		start = matchIdx + 1
	}
	return branches
}

// draftTreeAVX512 extracts a candidate draft tree from history using AVX-512 scanning.
func draftTreeAVX512(history []int, minMatch, maxMatch, maxDraft, maxBranches int) *CandidateDraftTree {
	histLen := len(history)
	for n := maxMatch; n >= minMatch; n-- {
		searchBound := histLen - n
		if searchBound <= 0 {
			continue
		}
		pattern := history[histLen-n:]
		target := pattern[0]

		var branches [][]int
		matchCount := 0

		start := 0
		for start < searchBound {
			matchIdx := scanNextMatchAVX512_I64(&history[0], start, searchBound, target)
			if matchIdx < 0 {
				break
			}

			matched := true
			for i := 1; i < n; i++ {
				if history[matchIdx+i] != pattern[i] {
					matched = false
					break
				}
			}
			if !matched {
				start = matchIdx + 1
				continue
			}

			matchCount++
			if len(branches) < maxBranches {
				contStart := matchIdx + n
				contEnd := contStart + maxDraft
				if contEnd > histLen {
					contEnd = histLen
				}
				if contEnd > contStart {
					branch := make([]int, contEnd-contStart)
					copy(branch, history[contStart:contEnd])
					branches = append(branches, branch)
				}
			}
			start = matchIdx + 1
		}

		if matchCount > 0 {
			var firstTokens []int
			if len(branches) > 0 {
				firstTokens = branches[0]
			}
			return &CandidateDraftTree{
				Tokens:      firstTokens,
				Branches:    branches,
				MatchCount:  matchCount,
				MatchLength: n,
			}
		}
	}

	return &CandidateDraftTree{
		Tokens:      nil,
		Branches:    nil,
		MatchCount:  0,
		MatchLength: 0,
	}
}
