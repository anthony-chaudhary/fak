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
	defaultNgramMinMatch = 3
	defaultNgramMaxMatch = 8
	defaultNgramMaxDraft = 4
)

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

// firstTokenSubsequence returns the first occurrence of pattern in tokens using the KMP
// prefix table. Prompt lookup runs every decode round, so this keeps the scan linear in the
// context length for each bounded candidate suffix without allocating string keys or maps.
func firstTokenSubsequence(tokens, pattern []int) int {
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
