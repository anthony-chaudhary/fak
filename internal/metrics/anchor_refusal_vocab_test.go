package metrics

// anchor_refusal_vocab_test.go — the REAL drift alarm for the anchor-refusal monitor's
// mirrored placement vocabulary.
//
// anchor_refusal.go restates internal/agent's BreakpointReason* values as string literals on
// purpose: metrics sits below agent and must not import it in PRODUCTION, so the vocabulary is
// mirrored rather than referenced. TestAnchorOutcomeClassification pins that map, but it pins it
// against the same literals, which leaves the mirror's one silent failure mode wide open: if the
// agent side changes a reason's VALUE, every live turn carrying it folds to AnchorUnknown — and
// AnchorUnknown, by design, never arms the alarm. The monitor would go quiet on precisely the
// condition it is named for, with every test still green.
//
// The import costs nothing at runtime here, so this file closes the hole from the test tier: it
// binds each class to the agent constant ITSELF. A renamed value fails here instead of silencing
// a live alarm.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestAnchorVocabularyTracksTheAgentConstants pins every real BreakpointReason to the class the
// monitor must give it, and pins the map's own entries back the other way. Both directions
// matter: a reason added on the agent side and never classified here would be absorbed as
// AnchorUnknown (a new bail this monitor has never considered, silently not alarming), and an
// entry left behind for a reason that no longer exists is dead vocabulary the same-literal test
// cannot see.
func TestAnchorVocabularyTracksTheAgentConstants(t *testing.T) {
	want := map[string]AnchorPlacementClass{
		agent.BreakpointReasonNone:         AnchorEarned,
		agent.BreakpointReasonVolatileHead: AnchorRefused,
		agent.BreakpointReasonNoStableHead: AnchorRefused,
		agent.BreakpointReasonSpliceFailed: AnchorRefused,
		agent.BreakpointReasonRedecodeFail: AnchorRefused,
		agent.BreakpointReasonAlreadySet:   AnchorDeferred,
		agent.BreakpointReasonNonJSON:      AnchorInapplicable,
	}
	for outcome, class := range want {
		if got := ClassifyAnchorOutcome(outcome); got != class {
			t.Errorf("ClassifyAnchorOutcome(%q) = %q, want %q — the mirrored vocabulary has drifted from internal/agent",
				outcome, got, class)
		}
	}
	for outcome := range anchorOutcomeClasses {
		if outcome == AnchorOutcomePlaced {
			continue // the metric's own spelling of BreakpointReasonNone; it has no agent constant
		}
		if _, ok := want[outcome]; !ok {
			t.Errorf("anchorOutcomeClasses classifies %q, which is not an agent.BreakpointReason* value — dead vocabulary", outcome)
		}
	}
}
