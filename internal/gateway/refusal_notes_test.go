package gateway

import (
	"strings"
	"testing"
)

// The whole point of the refusal-note seam (#2750): a BRAND-NEW refusal kind
// that carries actionable Meta must surface in-band on the content channel by
// registering ONE renderer — with no edit to the denySummary / adjudicationNote
// call sites. Before the seam, every new kind (reversibility, remedy, livelock)
// had to hand-stitch another note call at each site; this test pins that a fresh
// kind now rides the generic fold instead.
func TestRefusalNoteSeamSurfacesNewKindWithoutCallSiteEdit(t *testing.T) {
	saved := refusalNotes
	defer func() { refusalNotes = saved }()
	// Register a new refusal kind whose actionable Meta lives in a Detail key that
	// no existing renderer reads — proving the surfacing is generic, not bespoke.
	refusalNotes = append(append([]refusalNote(nil), saved...), refusalNote{
		render: func(a ToolAdjudication) string {
			if h := a.Verdict.Detail["witness_hint"]; h != "" {
				return "witness required: " + h
			}
			return ""
		},
	})

	adj := ToolAdjudication{
		Tool:     "Bash",
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "DENY",
			Reason:      "NEEDS_WITNESS",
			Disposition: "ESCALATE",
			Detail:      map[string]string{"witness_hint": "attach a failing test"},
		},
	}
	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
	} {
		if !strings.Contains(got, "witness required: attach a failing test") {
			t.Fatalf("%s did not surface the new refusal kind through the seam:\n%s", name, got)
		}
	}
}

// The seam must preserve the pinned rendering ORDER of the existing notes
// (reversibility recipe, then sanctioned alternative) — the wire tests read the
// confirm token adjacent to the recipe, so a reordered fold would be a silent
// regression.
func TestRefusalNoteSeamPreservesOrderAndConfirmRecipe(t *testing.T) {
	adj := reversibilityRefusal()
	adj.Verdict.Detail["remedy"] = "run git push --dry-run"
	notes, recipe := renderRefusalNotes(adj)
	if !recipe {
		t.Fatalf("reversibility refusal did not flag a confirm recipe: %q", notes)
	}
	recipeIdx := strings.Index(notes, "preview-confirm gate")
	fixIdx := strings.Index(notes, "sanctioned alternative: run git push --dry-run")
	if recipeIdx < 0 || fixIdx < 0 {
		t.Fatalf("seam dropped a note: recipe=%d fix=%d in %q", recipeIdx, fixIdx, notes)
	}
	if recipeIdx > fixIdx {
		t.Fatalf("seam reordered notes (recipe must precede the sanctioned alternative):\n%s", notes)
	}
}
