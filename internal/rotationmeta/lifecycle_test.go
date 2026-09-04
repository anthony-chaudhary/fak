package rotationmeta

import (
	"testing"
)

// Invariant: Weight rotation metadata adjudication must distinguish online vs offline transform placements.
// Guard: Adjudicate refuses unsupported fusions and delegates online transformations without runtime support.

func TestRotationMetaLifecycle(t *testing.T) {
	t.Parallel()

	p, ok := PinnedProvenance(RecipeQuaRot, "arxiv:2404.00456v2")
	if !ok {
		t.Fatal("missing pinned provenance for QuaRot")
	}

	d := Descriptor{
		ContractVersion: ContractVersion,
		Recipe:          RecipeQuaRot,
		RecipeVersion:   "arxiv:2404.00456v2",
		Provenance:      p,
		ArtifactFormat:  "safetensors",
		Transforms:      []Transform{{Name: "residual", Placement: PlacementOffline}},
	}
	caps := Capabilities{Recipes: map[Recipe][]string{RecipeQuaRot: {"arxiv:2404.00456v2"}}}

	res := Adjudicate(d, caps)
	if res.Outcome != OutcomeSupported {
		t.Fatalf("expected OutcomeSupported, got %s: %s", res.Outcome, res.Reason)
	}
}
