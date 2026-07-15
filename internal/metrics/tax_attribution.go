package metrics

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// TaxSite is one stable guard-runtime prose site. Prose is supplied at fold
// time and never stored in the attribution result.
type TaxSite struct{ Key, Prose string }

type TaxAttribution struct {
	Site             string `json:"site"`
	Applied          int    `json:"applied"`
	VerbatimFallback int    `json:"verbatim_fallback"`
	Residual         int    `json:"residual"`
	Outstanding      int    `json:"outstanding"`
}

type TaxAttributionReport struct {
	Applied          int              `json:"applied"`
	VerbatimFallback int              `json:"verbatim_fallback"`
	Residual         int              `json:"residual"`
	Outstanding      int              `json:"outstanding"`
	Sites            []TaxAttribution `json:"sites"`
}

// AttributeNegationTax re-derives telemetry from prose bytes and returns the
// highest outstanding (fallback + residual) sites first. limit<=0 means all.
func AttributeNegationTax(sites []TaxSite, limit int) TaxAttributionReport {
	var report TaxAttributionReport
	for _, site := range sites {
		res := negframe.ReframePass(site.Prose)
		row := TaxAttribution{Site: site.Key, Applied: res.Applied, VerbatimFallback: res.VerbatimFallback, Residual: res.ResidualNegatives}
		row.Outstanding = row.VerbatimFallback + row.Residual
		report.Applied += row.Applied
		report.VerbatimFallback += row.VerbatimFallback
		report.Residual += row.Residual
		report.Outstanding += row.Outstanding
		report.Sites = append(report.Sites, row)
	}
	sort.SliceStable(report.Sites, func(i, j int) bool {
		a, b := report.Sites[i], report.Sites[j]
		if a.Outstanding != b.Outstanding {
			return a.Outstanding > b.Outstanding
		}
		if a.VerbatimFallback != b.VerbatimFallback {
			return a.VerbatimFallback > b.VerbatimFallback
		}
		return a.Site < b.Site
	})
	if limit > 0 && len(report.Sites) > limit {
		report.Sites = report.Sites[:limit]
	}
	return report
}
