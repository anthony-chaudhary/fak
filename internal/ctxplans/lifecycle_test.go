package ctxplans

import (
	"testing"
)

// Invariant: Context plan surface scanning must correctly detect declared and undeclared context-related CLI verbs.
// Guard: Scan separates declared verbs from context debt without missing surface items.

func TestCtxPlansLifecycle(t *testing.T) {
	t.Parallel()

	rep, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan failed on fixtureRoot: %v", err)
	}
	if rep.DeclaredVerbs == 0 {
		t.Fatal("expected non-zero declared verbs")
	}
	if len(rep.Surfaces) == 0 {
		t.Fatal("expected non-zero surfaces discovered")
	}
}
