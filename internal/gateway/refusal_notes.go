package gateway

import "strings"

// refusalNote is one "actionable half of a refusal" renderer: given a refused
// ToolAdjudication it emits the in-band remedy string for that refusal facet
// (preview-confirm recipe, sanctioned alternative, livelock nudge, ...), or ""
// when the facet does not apply to this adjudication.
//
// The registry below is the single seam every content-channel wire folds over
// (#2750). A NEW refusal kind that carries actionable Meta is surfaced by
// appending its renderer here — NOT by hand-stitching another call at each
// denySummary / adjudicationNote site, which is exactly how reversibilityGateNote,
// remedyNote, and livelockInBandNote each grew a bespoke retrofit.
type refusalNote struct {
	render func(ToolAdjudication) string
	// confirmRecipe marks a non-empty render as a preview-confirm PAUSE rather
	// than a denial, so the trailer that sanctions re-proposing the same call
	// (only reversibilityGateNote today) can be scoped correctly.
	confirmRecipe bool
}

// refusalNotes is the ordered registry of actionable-half renderers. Order is
// load-bearing: it fixes the in-band rendering order (reversibility recipe, then
// the sanctioned alternative, then the livelock nudge) that the wire tests pin.
var refusalNotes = []refusalNote{
	{render: reversibilityGateNote, confirmRecipe: true},
	{render: remedyNote},
	{render: livelockInBandNote},
}

// renderRefusalNotes folds every registered renderer over one refused
// adjudication and returns the space-joined actionable notes plus whether any of
// them was a preview-confirm recipe. Both content-channel call sites (denySummary
// and adjudicationNote) consume this instead of stitching each note by hand.
func renderRefusalNotes(a ToolAdjudication) (notes string, confirmRecipe bool) {
	parts := make([]string, 0, len(refusalNotes))
	for _, n := range refusalNotes {
		if s := n.render(a); s != "" {
			parts = append(parts, s)
			if n.confirmRecipe {
				confirmRecipe = true
			}
		}
	}
	return strings.Join(parts, " "), confirmRecipe
}
