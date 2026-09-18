package rotationmeta

import "testing"

// prism-hadamard-g128 (prism-ml/Ternary-Bonsai-2-27B-gguf, sha 6ed5e12b).
//
// The checkpoint folds a blockwise normalized Sylvester Walsh-Hadamard rotation into
// the stored ternary weights and declares the matching activation transform as GGUF
// metadata. The declared sign payload is explicit: sign_widths=[5120,6144,17408],
// sign_values=+/-1 x 28672 (sum of widths). A runtime must either apply that exact
// transform or refuse the file — silently ignoring the rotation is the failure mode
// the contract exists to prevent.

// prismSign is the real declared payload shape (values elided to a valid pattern).
func prismSign() *SignVector {
	s := &SignVector{Widths: []int{5120, 6144, 17408}}
	s.Values = make([]int, 0, 28672)
	for i := 0; i < 28672; i++ {
		if i%3 == 0 {
			s.Values = append(s.Values, -1)
		} else {
			s.Values = append(s.Values, 1)
		}
	}
	return s
}

func prismDescriptor(sign *SignVector) Descriptor {
	p, _ := PinnedProvenance(RecipePrismHadamard, "prism-bonsai2-gguf-v1")
	return Descriptor{
		ContractVersion: ContractVersion,
		Recipe:          RecipePrismHadamard,
		RecipeVersion:   "prism-bonsai2-gguf-v1",
		Provenance:      p,
		ArtifactFormat:  "gguf",
		Transforms:      []Transform{{Name: "activation-last-dim", Placement: PlacementOnline, Fusion: "hadamard/h1024-activation"}},
		Sign:            sign,
	}
}

func TestPrismHadamardSupportedWithDeclaredRecipeAndFusion(t *testing.T) {
	t.Parallel()
	caps := Capabilities{
		Recipes: map[Recipe][]string{RecipePrismHadamard: {"prism-bonsai2-gguf-v1"}},
		Fusions: map[string]bool{"hadamard/h1024-activation": true},
	}
	got := Adjudicate(prismDescriptor(prismSign()), caps)
	if got.Outcome != OutcomeSupported || got.Reason != ReasonSupported {
		t.Fatalf("got %+v, want supported", got)
	}
}

func TestPrismHadamardDelegatesWithoutDeclaredRuntime(t *testing.T) {
	t.Parallel()
	got := Adjudicate(prismDescriptor(prismSign()), Capabilities{})
	if got.Outcome != OutcomeDelegate || got.Reason != ReasonRuntimeRequired {
		t.Fatalf("got %+v, want delegate/runtime_required", got)
	}
}

func TestPrismHadamardRefusesWithoutActivationFusion(t *testing.T) {
	t.Parallel()
	caps := Capabilities{Recipes: map[Recipe][]string{RecipePrismHadamard: {"prism-bonsai2-gguf-v1"}}}
	got := Adjudicate(prismDescriptor(prismSign()), caps)
	if got.Outcome != OutcomeUnsupported || got.Reason != ReasonRuntimeTransformUnavailable {
		t.Fatalf("got %+v, want unsupported/runtime_transform_unavailable", got)
	}
}

// A declared rotation whose sign payload is missing or inconsistent must never be
// treated as ordinary metadata: the whole point of the contract is that ignoring the
// rotation produces garbage, so malformed signs fail closed.
func TestPrismHadamardRefusesMalformedSignPayload(t *testing.T) {
	t.Parallel()
	caps := Capabilities{
		Recipes: map[Recipe][]string{RecipePrismHadamard: {"prism-bonsai2-gguf-v1"}},
		Fusions: map[string]bool{"hadamard/h1024-activation": true},
	}
	cases := []struct {
		name string
		sign *SignVector
	}{
		{"absent", nil},
		{"no widths", &SignVector{Values: []int{1}}},
		{"width sum mismatch", &SignVector{Widths: []int{2, 2}, Values: []int{1, 1, 1}}},
		{"non binary sign", &SignVector{Widths: []int{2}, Values: []int{1, 0}}},
		{"negative width", &SignVector{Widths: []int{-2}, Values: []int{1, 1}}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := Adjudicate(prismDescriptor(tc.sign), caps)
			if got.Outcome != OutcomeUnsupported || got.Reason != ReasonInvalidSignVector {
				t.Fatalf("got %+v, want unsupported/invalid_sign_vector", got)
			}
		})
	}
}

func TestPrismHadamardProvenanceIsPinned(t *testing.T) {
	t.Parallel()
	p, ok := PinnedProvenance(RecipePrismHadamard, "prism-bonsai2-gguf-v1")
	if !ok || p.URI != "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf" {
		t.Fatalf("bad pin: %+v ok=%v", p, ok)
	}
	// A descriptor with a mutated pin is refused, like every other recipe.
	d := prismDescriptor(prismSign())
	d.Provenance.SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
	if got := Validate(d); got.Outcome != OutcomeUnsupported || got.Reason != ReasonMissingProvenance {
		t.Fatalf("got %+v, want unsupported/missing_provenance", got)
	}
}
