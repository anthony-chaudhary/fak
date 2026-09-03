package gateway

import (
	"errors"
	"fmt"
	"math"
)

// ErrGrammarMaskWidenedDomain is emitted when a grammar constraint attempts to turn a previously
// impossible (-inf) token into a finite probability, violating domain monotonicity.
var ErrGrammarMaskWidenedDomain = errors.New("gateway: grammar mask violated domain monotonicity by unmasking impossible token")

// GrammarMaskDomainMonotonicityReceipt records validation evidence that grammar application never widens model domain.
type GrammarMaskDomainMonotonicityReceipt struct {
	VocabSize              int  `json:"vocab_size"`
	OriginalInfiniteTokens int  `json:"original_infinite_tokens"`
	FinalInfiniteTokens    int  `json:"final_infinite_tokens"`
	WidenedTokens          int  `json:"widened_tokens"`
	DomainMonotonicityOK   bool `json:"domain_monotonicity_ok"`
}

// ApplyGrammarMaskWithMonotonicity applies an allowed-token grammar filter to logits,
// guaranteeing that tokens that were already -inf (e.g. padded vocabulary tails or EOS restrictions)
// can never become finite candidates under any grammar state.
func ApplyGrammarMaskWithMonotonicity(
	logits []float32,
	allowedTokens map[int]bool,
) (GrammarMaskDomainMonotonicityReceipt, error) {
	var receipt GrammarMaskDomainMonotonicityReceipt
	n := len(logits)
	if n == 0 {
		return receipt, fmt.Errorf("logits must not be empty")
	}

	negInf := float32(math.Inf(-1))

	origInf := make([]bool, n)
	origInfCount := 0
	for i, v := range logits {
		if math.IsInf(float64(v), -1) {
			origInf[i] = true
			origInfCount++
		}
	}

	widened := 0
	for i := range logits {
		isAllowed := allowedTokens[i]
		if !isAllowed {
			logits[i] = negInf
		} else {
			if origInf[i] {
				logits[i] = negInf
			}
		}

		if origInf[i] && !math.IsInf(float64(logits[i]), -1) {
			widened++
		}
	}

	if widened > 0 {
		return receipt, fmt.Errorf("%w: %d tokens resurrected", ErrGrammarMaskWidenedDomain, widened)
	}

	finalInfCount := 0
	for _, v := range logits {
		if math.IsInf(float64(v), -1) {
			finalInfCount++
		}
	}

	receipt = GrammarMaskDomainMonotonicityReceipt{
		VocabSize:              n,
		OriginalInfiniteTokens: origInfCount,
		FinalInfiniteTokens:    finalInfCount,
		WidenedTokens:          0,
		DomainMonotonicityOK:   true,
	}

	return receipt, nil
}
