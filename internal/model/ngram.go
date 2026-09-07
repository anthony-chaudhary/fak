package model

// NgramDrafter is the model-free native speculative baseline. It finds the longest
// suffix of the committed token history that occurred earlier, then proposes the tokens
// that followed that earlier occurrence. The target still verifies every proposal through
// VerifyForward, so a poor match can waste work but cannot change greedy output.
//
// Enabled is explicit and false by default. This keeps native generation unchanged until
// an operator selects prompt lookup; unlike a co-resident draft model, this proposer owns
// no weights, session, or KV cache.
type NgramDrafter struct {
	Enabled  bool
	MinMatch int
	MaxMatch int
	MaxDraft int
}

const (
	defaultNgramMinMatch    = 3
	defaultNgramMaxMatch    = 8
	defaultNgramMaxDraft    = 4
	defaultNgramMaxBranches = 4
)

// CandidateDraftTree holds speculative continuation candidates extracted from prompt history.
type CandidateDraftTree struct {
	Tokens      []int
	Branches    [][]int
	MatchCount  int
	MatchLength int
}

// DraftTree extracts a candidate draft tree from history proposing multiple continuation
// branches. It uses AVX-512 vector acceleration when available, falling back to scalar execution.
func (d NgramDrafter) DraftTree(history []int, maxBranches int) *CandidateDraftTree {
	if !d.Enabled || len(history) < 2 {
		return nil
	}
	minMatch, maxMatch, maxDraft := d.limits(len(history))
	if maxMatch < minMatch || maxDraft == 0 {
		return nil
	}
	if maxBranches <= 0 {
		maxBranches = defaultNgramMaxBranches
	}
	if hasAVX512() {
		return draftTreeAVX512(history, minMatch, maxMatch, maxDraft, maxBranches)
	}
	return draftTreeScalar(history, minMatch, maxMatch, maxDraft, maxBranches)
}

// DraftTreeScalar extracts a candidate draft tree using scalar execution for reference comparison.
func (d NgramDrafter) DraftTreeScalar(history []int, maxBranches int) *CandidateDraftTree {
	if !d.Enabled || len(history) < 2 {
		return nil
	}
	minMatch, maxMatch, maxDraft := d.limits(len(history))
	if maxMatch < minMatch || maxDraft == 0 {
		return nil
	}
	if maxBranches <= 0 {
		maxBranches = defaultNgramMaxBranches
	}
	return draftTreeScalar(history, minMatch, maxMatch, maxDraft, maxBranches)
}

// DraftTree extracts a candidate draft tree from history using default drafter settings.
func DraftTree(history []int, maxBranches int) *CandidateDraftTree {
	return NgramDrafter{Enabled: true}.DraftTree(history, maxBranches)
}

// DraftTreeScalar extracts a candidate draft tree from history using scalar reference logic.
func DraftTreeScalar(history []int, maxBranches int) *CandidateDraftTree {
	return NgramDrafter{Enabled: true}.DraftTreeScalar(history, maxBranches)
}

// Draft returns a copied proposal, or nil when prompt lookup is disabled or the committed
// history has no repeated suffix with at least one known continuation. Longer suffixes win;
// ties use the earliest occurrence so the result is deterministic.
func (d NgramDrafter) Draft(history []int) []int {
	if !d.Enabled || len(history) < 2 {
		return nil
	}
	minMatch, maxMatch, maxDraft := d.limits(len(history))
	if maxMatch < minMatch || maxDraft == 0 {
		return nil
	}

	// Exclude the final token from the search haystack. A match must end before the
	// current history ends so at least one already-observed continuation token exists.
	haystack := history[:len(history)-1]
	for n := maxMatch; n >= minMatch; n-- {
		pattern := history[len(history)-n:]
		start := firstTokenSubsequence(haystack, pattern)
		if start < 0 {
			continue
		}
		continuation := history[start+n:]
		if len(continuation) > maxDraft {
			continuation = continuation[:maxDraft]
		}
		if len(continuation) == 0 {
			continue
		}
		return append([]int(nil), continuation...)
	}
	return nil
}

func (d NgramDrafter) limits(historyLen int) (minMatch, maxMatch, maxDraft int) {
	minMatch = d.MinMatch
	if minMatch <= 0 {
		minMatch = defaultNgramMinMatch
	}
	maxMatch = d.MaxMatch
	if maxMatch <= 0 {
		maxMatch = defaultNgramMaxMatch
	}
	if maxMatch >= historyLen {
		maxMatch = historyLen - 1
	}
	maxDraft = d.MaxDraft
	if maxDraft < 0 {
		maxDraft = 0
	} else if maxDraft == 0 {
		maxDraft = defaultNgramMaxDraft
	}
	return minMatch, maxMatch, maxDraft
}

// draftTreeScalar is the scalar reference implementation of candidate draft tree extraction.
func draftTreeScalar(history []int, minMatch, maxMatch, maxDraft, maxBranches int) *CandidateDraftTree {
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
			matchIdx := -1
			for i := start; i < searchBound; i++ {
				if history[i] == target {
					matchIdx = i
					break
				}
			}
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

// findContinuationBranchesScalar extracts up to maxBranches continuation branches from haystack for pattern.
func findContinuationBranchesScalar(haystack, pattern []int, maxDraft, maxBranches int) [][]int {
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
		matchIdx := -1
		for i := start; i < searchBound; i++ {
			if haystack[i] == target {
				matchIdx = i
				break
			}
		}
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

// firstTokenSubsequence returns the first occurrence of pattern in tokens.
// It dispatches to scanTokenSubsequenceAVX512 when AVX-512 is available, otherwise
// it uses firstTokenSubsequenceScalar.
func firstTokenSubsequence(tokens, pattern []int) int {
	if hasAVX512() {
		return scanTokenSubsequenceAVX512(tokens, pattern)
	}
	return firstTokenSubsequenceScalar(tokens, pattern)
}

// firstTokenSubsequenceScalar returns the first occurrence of pattern in tokens using the KMP
// prefix table. Prompt lookup runs every decode round, so this keeps the scan linear in the
// context length for each bounded candidate suffix without allocating string keys or maps.
func firstTokenSubsequenceScalar(tokens, pattern []int) int {
	if len(pattern) == 0 {
		return 0
	}
	if len(pattern) > len(tokens) {
		return -1
	}
	lps := make([]int, len(pattern))
	for i, prefix := 1, 0; i < len(pattern); {
		if pattern[i] == pattern[prefix] {
			prefix++
			lps[i] = prefix
			i++
			continue
		}
		if prefix > 0 {
			prefix = lps[prefix-1]
			continue
		}
		i++
	}
	for i, matched := 0, 0; i < len(tokens); {
		if tokens[i] == pattern[matched] {
			i++
			matched++
			if matched == len(pattern) {
				return i - matched
			}
			continue
		}
		if matched > 0 {
			matched = lps[matched-1]
			continue
		}
		i++
	}
	return -1
}

// firstTokenSubsequenceI32 returns the first occurrence of pattern in tokens for int32 slices.
// It dispatches to scanTokenSubsequenceI32AVX512 when AVX-512 is available, otherwise
// it uses firstTokenSubsequenceI32Scalar.
func firstTokenSubsequenceI32(tokens, pattern []int32) int {
	if hasAVX512() {
		return scanTokenSubsequenceI32AVX512(tokens, pattern)
	}
	return firstTokenSubsequenceI32Scalar(tokens, pattern)
}

// firstTokenSubsequenceI32Scalar returns the first occurrence of pattern in tokens for int32 slices
// using the KMP prefix table.
func firstTokenSubsequenceI32Scalar(tokens, pattern []int32) int {
	if len(pattern) == 0 {
		return 0
	}
	if len(pattern) > len(tokens) {
		return -1
	}
	lps := make([]int, len(pattern))
	for i, prefix := 1, 0; i < len(pattern); {
		if pattern[i] == pattern[prefix] {
			prefix++
			lps[i] = prefix
			i++
			continue
		}
		if prefix > 0 {
			prefix = lps[prefix-1]
			continue
		}
		i++
	}
	for i, matched := 0, 0; i < len(tokens); {
		if tokens[i] == pattern[matched] {
			i++
			matched++
			if matched == len(pattern) {
				return i - matched
			}
			continue
		}
		if matched > 0 {
			matched = lps[matched-1]
			continue
		}
		i++
	}
	return -1
}
