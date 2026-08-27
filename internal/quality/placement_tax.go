package quality

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/placementtax"
)

// PlacementTaxReview is the default analysis contract for any tuning proposal
// that crosses a coherence, device, or host boundary.
type PlacementTaxReview struct {
	Required bool
	Reason   string
	Report   *placementtax.Report
}

// ReviewPlacementTax requires feasibility and a component ledger whenever a
// proposal distributes one matched workload across multiple placement domains.
// Callers may pass nil for a local single-domain proposal.
func ReviewPlacementTax(c *placementtax.Comparison) (PlacementTaxReview, error) {
	if c == nil {
		return PlacementTaxReview{Reason: "single-domain proposal: placement-tax analysis not required"}, nil
	}
	review := PlacementTaxReview{
		Required: true,
		Reason:   "cross-domain proposal: feasibility and compute-placement-tax ledger required",
	}
	report, err := placementtax.Analyze(*c)
	if err != nil {
		return review, fmt.Errorf("compute-placement-tax analysis: %w", err)
	}
	review.Report = &report
	return review, nil
}
