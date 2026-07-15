package negframe

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const NegationTaxSchema = "fak-negation-tax-scorecard/1"
const NegationTaxDebtKey = "negation_tax_debt"

// BuildNegationTax folds the declared hot-path corpus into a stable scorecard.
// Debt is broadcast-weighted outstanding tax (fallback + residual), while the
// raw totals remain available for audit.
func BuildNegationTax(corpus []HotPathString) scorecard.Payload {
	var defects []string
	weightedDebt := 0
	mechanical, judgement := 0, 0
	perSurface := make([]map[string]any, 0, len(corpus))
	for _, site := range corpus {
		res := ReframePass(site.Text)
		outstanding := res.VerbatimFallback + res.ResidualNegatives
		weighted := outstanding * site.Tier.Weight()
		mechanical += res.VerbatimFallback
		judgement += res.ResidualNegatives
		weightedDebt += weighted
		perSurface = append(perSurface, map[string]any{"site": site.Name, "tier": site.Tier.String(), "applied": res.Applied, "verbatim_fallback": res.VerbatimFallback, "residual": res.ResidualNegatives, "weighted": weighted})
		if weighted > 0 {
			defects = append(defects, fmt.Sprintf("%s tier=%s residual=%d fallback=%d weighted=%d", site.Name, site.Tier.String(), res.ResidualNegatives, res.VerbatimFallback, weighted))
		}
	}
	kpi := scorecard.KPI{Key: "positive_hot_path", Group: "framing", Score: 100, Detail: fmt.Sprintf("%d weighted outstanding negation tax", weightedDebt), Defects: defects}
	if weightedDebt > 0 {
		kpi.Score = 0
	}
	return scorecard.Fold(NegationTaxSchema, []scorecard.KPI{kpi}, NegationTaxDebtKey, nil, scorecard.Messages{
		Grade: scorecard.GradeStrict, Finding: "hot-path prose residual tax measured", FindingClean: "hot-path prose is positive", NextAction: "pay down highest weighted site", NextActionClean: "hold",
		ExtraCorpus: map[string]any{NegationTaxDebtKey: weightedDebt, "surfaces": len(corpus), "mechanical": mechanical, "judgement": judgement, "weighted_debt": weightedDebt, "per_surface": perSurface},
	})
}
