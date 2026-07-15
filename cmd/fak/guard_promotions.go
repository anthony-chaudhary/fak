package main

import (
	"fmt"
	"io"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

const guardPromotionDefaultThreshold = 3

var guardPromotionState struct {
	sync.Mutex
	ledger    *adjudicator.Ledger
	threshold int
}

func configureGuardPromotionLedger(complain map[string]bool, threshold int) *adjudicator.Ledger {
	guardPromotionState.Lock()
	defer guardPromotionState.Unlock()
	guardPromotionState.ledger = nil
	guardPromotionState.threshold = threshold
	if len(complain) == 0 {
		return nil
	}
	if threshold <= 0 {
		threshold = guardPromotionDefaultThreshold
		guardPromotionState.threshold = threshold
	}
	ledger := adjudicator.NewLedger()
	guardPromotionState.ledger = ledger
	abi.RegisterEmitter(ledger)
	return ledger
}

func renderGuardPromotionOffers(w io.Writer) int {
	guardPromotionState.Lock()
	ledger, threshold := guardPromotionState.ledger, guardPromotionState.threshold
	guardPromotionState.Unlock()
	if ledger == nil {
		return 0
	}
	offers := ledger.Promotable(threshold)
	for _, offer := range offers {
		fmt.Fprintf(w, "fak guard promotion: fak guard allow %s  # %d clean complain admits, 0 hard refusals\n", offer.Tool, offer.CleanEvents)
	}
	return len(offers)
}
