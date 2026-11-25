package model

// RepetitionPenaltySanitizer sanitizes repetition penalty states and guards against cyclic loops.
type RepetitionPenaltySanitizer struct {
	FrequencyPenalty float64
	PresencePenalty  float64
	MaxLoopRepeats   int
}

// NewRepetitionPenaltySanitizer creates a sanitizer configured with frequency and presence penalties.
func NewRepetitionPenaltySanitizer(frequencyPenalty, presencePenalty float64) *RepetitionPenaltySanitizer {
	return &RepetitionPenaltySanitizer{
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		MaxLoopRepeats:   2,
	}
}

// ApplyPenalty modifies logits according to the current generation counts.
// logit[t] -= frequencyPenalty * count[t] + presencePenalty * (1 if count[t] > 0 else 0)
func (s *RepetitionPenaltySanitizer) ApplyPenalty(logits []float32, counts []int32) []float32 {
	return s.ApplyPenaltyInto(nil, logits, counts)
}

// ApplyPenaltyInto is ApplyPenalty writing the penalized copy into dst's backing array
// (grown when short) so one scratch row can serve every verify row of a round. When no
// penalty applies it returns logits itself, so a caller reusing the result as scratch must
// first check penalizes.
func (s *RepetitionPenaltySanitizer) ApplyPenaltyInto(dst, logits []float32, counts []int32) []float32 {
	if !s.penalizes(counts) {
		return logits
	}
	eff := append(dst[:0], logits...)
	for tok, c := range counts {
		if tok >= len(eff) || c <= 0 {
			continue
		}
		penalty := s.FrequencyPenalty*float64(c) + s.PresencePenalty
		eff[tok] -= float32(penalty)
	}
	return eff
}

func (s *RepetitionPenaltySanitizer) penalizes(counts []int32) bool {
	return s != nil && (s.FrequencyPenalty != 0 || s.PresencePenalty != 0) && len(counts) > 0
}

// DetectDegenerateLoop checks if the suffix of history exhibits a periodic cyclic loop.
func (s *RepetitionPenaltySanitizer) DetectDegenerateLoop(history []int, minPeriod, maxPeriod, minRepeats int) (bool, int) {
	if len(history) < minPeriod*minRepeats {
		return false, 0
	}
	if minPeriod <= 0 {
		minPeriod = 1
	}
	if maxPeriod <= 0 || maxPeriod > len(history)/minRepeats {
		maxPeriod = len(history) / minRepeats
	}
	if minRepeats <= 1 {
		minRepeats = 2
	}

	n := len(history)
	for p := minPeriod; p <= maxPeriod; p++ {
		match := true
		for r := 1; r < minRepeats; r++ {
			for i := 0; i < p; i++ {
				idx1 := n - 1 - i
				idx2 := n - 1 - r*p - i
				if idx2 < 0 || history[idx1] != history[idx2] {
					match = false
					break
				}
			}
			if !match {
				break
			}
		}
		if match {
			return true, p
		}
	}
	return false, 0
}

// SanitizeProposal inspects a proposed draft and truncates or suppresses proposals that continue degenerate loops.
func (s *RepetitionPenaltySanitizer) SanitizeProposal(proposal DraftProposal, history []int) DraftProposal {
	if s == nil || len(history) == 0 {
		return proposal
	}
	repeats := s.MaxLoopRepeats
	if repeats <= 0 {
		repeats = 2
	}
	hasLoop, period := s.DetectDegenerateLoop(history, 1, 8, repeats)
	if !hasLoop {
		return proposal
	}

	// History already has a degenerate repeating loop with period.
	// Check if proposal continues this exact periodic cycle.
	if len(proposal.Tokens) > 0 {
		expectedCycleToken := history[len(history)-period]
		if proposal.Tokens[0] == expectedCycleToken {
			// Proposal continues degenerate cycle; suppress to break loop
			sanitized := proposal
			sanitized.Tokens = nil
			sanitized.Tree = nil
			sanitized.Mask = nil
			if sanitized.Metadata == nil {
				sanitized.Metadata = make(map[string]any)
			}
			sanitized.Metadata["sanitized_degenerate_loop"] = true
			sanitized.Metadata["detected_period"] = period
			return sanitized
		}
	}

	return proposal
}

// SanitizeCounts guarantees that counts only records committed and accepted tokens,
// ensuring rollback never leaks unaccepted speculative candidate counts.
func (s *RepetitionPenaltySanitizer) SanitizeCounts(counts []int32, committed []int, accepted []int) {
	if len(counts) == 0 {
		return
	}
	for i := range counts {
		counts[i] = 0
	}
	for _, tok := range committed {
		if tok >= 0 && tok < len(counts) {
			counts[tok]++
		}
	}
	for _, tok := range accepted {
		if tok >= 0 && tok < len(counts) {
			counts[tok]++
		}
	}
}
