package negframe

import "strings"

// Residual is a mechanically proven positive restatement of one negated span.
// Judgment-tier negatives never produce one.
type Residual struct {
	Original string `json:"original"`
	Positive string `json:"positive"`
	Applied  int    `json:"applied"`
}

// ResolveResidual applies the same bounded mechanical rules as ReframePass and
// returns ok only when at least one rewrite was admitted with no residual
// judgment-tier negative or verbatim fallback.
func ResolveResidual(text string) (residual Residual, ok bool) {
	result := ReframePass(text)
	if result.Applied == 0 || result.ResidualNegatives != 0 || result.VerbatimFallback != 0 || result.Text == text {
		return Residual{}, false
	}
	positive := strings.TrimSpace(result.Text)
	if positive == "" {
		return Residual{}, false
	}
	return Residual{Original: text, Positive: positive, Applied: result.Applied}, true
}
