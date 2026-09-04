package lookahead

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// Invariant: Lookahead distilled lessons must preserve witness rungs and format outputs accurately.
// Guard: Render includes witness rungs and lesson descriptions in output strings.

func TestLookaheadLifecycle(t *testing.T) {
	t.Parallel()

	fact := Lesson{Claim: "lifecycle verified", Kind: KindFact, Rung: trajctl.W3}
	rendered := fact.Render()
	if rendered != "Witnessed (W3): lifecycle verified" {
		t.Fatalf("unexpected render output: %q", rendered)
	}
}
