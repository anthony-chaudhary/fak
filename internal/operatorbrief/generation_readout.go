package operatorbrief

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

func generationReadout(rows []milestonereport.GenerationRow) *Generation {
	if len(rows) == 0 {
		return nil
	}
	out := &Generation{}
	var nowOpen, laterTracked, unreadable, debt int
	var hottest GenerationLane
	for _, row := range rows {
		lane := GenerationLane{
			Generation:          row.Generation,
			Tracked:             row.Tracked,
			Measured:            row.Measured,
			Programs:            row.Programs,
			Discrete:            row.Discrete,
			Closed:              row.Closed,
			Total:               row.Total,
			OpenDiscrete:        mathx.MaxInt(0, row.Total-row.Closed),
			OverallPct:          row.OverallPct,
			Errored:             row.Errored,
			DebtScore:           row.DebtScore,
			StaleIssues:         row.StaleIssues,
			MissingWitnesses:    row.MissingWitnesses,
			UnpromotedBets:      row.UnpromotedBets,
			LabelShipMismatches: row.LabelShipMismatches,
			DebtReason:          row.DebtReason,
		}
		lane.PromotionCandidates = generationPromotionCandidates(row)
		lane.BlockedAssumptions = generationBlockedAssumptions(row)
		lane.ShipVelocity = generationShipVelocity(row)
		lane.StaleAge = generationStaleAge(row)
		lane.HeatScore, lane.HeatReason = generationHeat(row, lane)
		out.Lanes = append(out.Lanes, lane)
		if row.Generation == "now" {
			nowOpen = lane.OpenDiscrete
		} else if row.Generation != "unclassified" {
			laterTracked += lane.Tracked
		}
		unreadable += lane.Errored
		debt += lane.DebtScore
		out.PromotionCandidates += lane.PromotionCandidates
		out.BlockedAssumptions += lane.BlockedAssumptions
		if lane.HeatScore > hottest.HeatScore {
			hottest = lane
		}
	}
	out.HottestGeneration = hottest.Generation
	out.Heat = generationHeatSummary(hottest, out.PromotionCandidates, out.BlockedAssumptions)
	switch {
	case unreadable > 0:
		out.Summary = fmt.Sprintf("generation lanes have %d unreadable item(s), debt %d; do not promote or demote from this readout yet", unreadable, debt)
		out.Attention = "repair the unreadable generation lane signal before changing dispatch focus"
	case nowOpen > 0:
		out.Summary = fmt.Sprintf("ship-now lane has %d open discrete item(s); %d later-horizon item(s) stay visible; generation debt %d", nowOpen, laterTracked, debt)
		out.Attention = "delegate from the now lane first; review later lanes only when promotion evidence changes"
	case laterTracked > 0:
		out.Summary = fmt.Sprintf("ship-now lane is clear; %d later-horizon item(s) remain as bets or foundations; generation debt %d", laterTracked, debt)
		out.Attention = "no extra human attention unless a later item asks for promotion into now"
	default:
		out.Summary = "generation lanes are clear or unclassified only"
		out.Attention = "preserve generation labels and project fields; no attention budget needed"
	}
	return out
}

func generationPromotionCandidates(row milestonereport.GenerationRow) int {
	if row.Generation == "now" || row.Generation == "unclassified" {
		return 0
	}
	return row.UnpromotedBets
}

func generationBlockedAssumptions(row milestonereport.GenerationRow) int {
	return row.MissingWitnesses + row.LabelShipMismatches + row.Errored
}

func generationShipVelocity(row milestonereport.GenerationRow) string {
	if row.Total > 0 {
		return fmt.Sprintf("%d/%d shipped", row.Closed, row.Total)
	}
	if row.Programs > 0 && row.Measured > 0 {
		return fmt.Sprintf("%d ongoing measured", row.Measured)
	}
	if row.Measured > 0 {
		return fmt.Sprintf("%d measured", row.Measured)
	}
	return "not measured"
}

func generationStaleAge(row milestonereport.GenerationRow) string {
	if row.StaleIssues == 0 {
		return ""
	}
	return fmt.Sprintf("age not measured; %d stale-risk issue(s)", row.StaleIssues)
}

func generationHeat(row milestonereport.GenerationRow, lane GenerationLane) (int, string) {
	score := lane.DebtScore + lane.OpenDiscrete + 2*lane.BlockedAssumptions + lane.PromotionCandidates
	var parts []string
	if lane.OpenDiscrete > 0 {
		parts = append(parts, fmt.Sprintf("%d open discrete", lane.OpenDiscrete))
	}
	if lane.StaleIssues > 0 {
		parts = append(parts, fmt.Sprintf("%d stale-risk", lane.StaleIssues))
	}
	if lane.BlockedAssumptions > 0 {
		parts = append(parts, fmt.Sprintf("%d blocked assumption", lane.BlockedAssumptions))
	}
	if lane.PromotionCandidates > 0 {
		parts = append(parts, fmt.Sprintf("%d promotion candidate", lane.PromotionCandidates))
	}
	if row.DebtScore > 0 {
		parts = append(parts, fmt.Sprintf("debt %d", row.DebtScore))
	}
	return score, strings.Join(parts, ", ")
}

func generationHeatSummary(hottest GenerationLane, promotionCandidates, blockedAssumptions int) string {
	if hottest.Generation == "" || hottest.HeatScore == 0 {
		return "heat clear; no generation lane needs extra attention"
	}
	return fmt.Sprintf("hottest=%s score=%d; promotion_candidates=%d blocked_assumptions=%d; stale_age=%s",
		hottest.Generation, hottest.HeatScore, promotionCandidates, blockedAssumptions, strmatch.FirstNonBlank(hottest.StaleAge, "no stale-risk issue age to measure"))
}
