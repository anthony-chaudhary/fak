package rotationmeta

import "testing"

func fixture(tb testing.TB, recipe Recipe, version string, transforms ...Transform) Descriptor {
	if tb != nil {
		tb.Helper()
	}
	p, ok := PinnedProvenance(recipe, version)
	if !ok {
		if tb != nil {
			tb.Fatalf("missing test provenance for %s %s", recipe, version)
		}
		panic("missing test provenance")
	}
	return Descriptor{ContractVersion: ContractVersion, Recipe: recipe, RecipeVersion: version, Provenance: p, ArtifactFormat: "safetensors", Transforms: transforms}
}

func TestGoldenRecordsDistinguishOnlineAndOfflineRotations(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name         string
		descriptor   Descriptor
		capabilities Capabilities
		want         Outcome
		reason       Reason
	}{
		{"quarot offline is artifact-complete", fixture(t, RecipeQuaRot, "arxiv:2404.00456v2", Transform{Name: "residual", Placement: PlacementOffline}), Capabilities{Recipes: map[Recipe][]string{RecipeQuaRot: {"arxiv:2404.00456v2"}}}, OutcomeSupported, ReasonSupported},
		{"spinquant online delegates without recipe runtime", fixture(t, RecipeSpinQuant, "arxiv:2405.16406v4", Transform{Name: "down-project", Placement: PlacementOnline, Fusion: "hadamard/down-project"}), Capabilities{}, OutcomeDelegate, ReasonRuntimeRequired},
		{"lightrot online refuses unsupported fusion", fixture(t, RecipeLightRot, "arxiv:2607.27704v1", Transform{Name: "activation", Placement: PlacementOnline, Fusion: "lightrot/activation"}), Capabilities{Recipes: map[Recipe][]string{RecipeLightRot: {"arxiv:2607.27704v1"}}}, OutcomeUnsupported, ReasonRuntimeTransformUnavailable},
		{"spinquant online supported when declared", fixture(t, RecipeSpinQuant, "arxiv:2405.16406v4", Transform{Name: "down-project", Placement: PlacementOnline, Fusion: "hadamard/down-project"}), Capabilities{Recipes: map[Recipe][]string{RecipeSpinQuant: {"arxiv:2405.16406v4"}}, Fusions: map[string]bool{"hadamard/down-project": true}}, OutcomeSupported, ReasonSupported},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := Adjudicate(tc.descriptor, tc.capabilities)
			if got.Outcome != tc.want || got.Reason != tc.reason {
				t.Fatalf("got %+v, want %s/%s", got, tc.want, tc.reason)
			}
		})
	}
}

func TestGoldenRecordsRejectMissingTransformMetadata(t *testing.T) {
	t.Parallel()
	d := fixture(t, RecipeQuaRot, "arxiv:2404.00456v2")
	got := Validate(d)
	if got.Outcome != OutcomeUnsupported || got.Reason != ReasonMissingTransform {
		t.Fatalf("got %+v", got)
	}
}

func TestUnknownInputsNeverFallBack(t *testing.T) {
	t.Parallel()
	base := fixture(t, RecipeQuaRot, "arxiv:2404.00456v2", Transform{Name: "residual", Placement: PlacementOffline})
	cases := []struct {
		name    string
		mutate  func(*Descriptor)
		outcome Outcome
		reason  Reason
	}{
		{"contract", func(d *Descriptor) { d.ContractVersion = "rotationmeta/v99" }, OutcomeDelegate, ReasonUnknownContract},
		{"recipe", func(d *Descriptor) { d.Recipe = "novelrot" }, OutcomeDelegate, ReasonUnknownRecipe},
		{"version", func(d *Descriptor) { d.RecipeVersion = "arxiv:2404.00456v99" }, OutcomeDelegate, ReasonUnknownRecipeVersion},
		{"provenance", func(d *Descriptor) { d.Provenance.SHA256 = "" }, OutcomeUnsupported, ReasonMissingProvenance},
		{"placement", func(d *Descriptor) { d.Transforms[0].Placement = "sometimes" }, OutcomeUnsupported, ReasonInvalidPlacement},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			d := base
			d.Transforms = append([]Transform(nil), base.Transforms...)
			tc.mutate(&d)
			got := Validate(d)
			if got.Outcome != tc.outcome || got.Reason != tc.reason {
				t.Fatalf("got %+v, want %s/%s", got, tc.outcome, tc.reason)
			}
		})
	}
}

func TestPinnedProvenanceIsExact(t *testing.T) {
	t.Parallel()
	for recipe, version := range map[Recipe]string{RecipeQuaRot: "arxiv:2404.00456v2", RecipeSpinQuant: "arxiv:2405.16406v4", RecipeLightRot: "arxiv:2607.27704v1"} {
		p, ok := PinnedProvenance(recipe, version)
		if !ok || p.URI == "" || len(p.SHA256) != 64 {
			t.Fatalf("bad pin for %s: %+v", recipe, p)
		}
	}
	if _, ok := PinnedProvenance(RecipeQuaRot, "latest"); ok {
		t.Fatal("floating provenance must not resolve")
	}
}
