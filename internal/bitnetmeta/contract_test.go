package bitnetmeta

import (
	"os"
	"path/filepath"
	"testing"
)

var fixtureCapabilities = Capabilities{
	Schemas:     []string{SchemaV1},
	Formats:     []string{"safetensors@1", "gguf@3"},
	Activations: []string{"integer/8", "bfloat/16", "float/16"},
	Packings:    []string{"bitplane-lsb", "i2_s-pair", "two-bit-codes"},
	Recipes:     []string{"native-bitnet@1", "absmean-ternarize@2", "int2-quantize@1"},
	Runtimes:    []string{"bitnet.cpp@2026.08"},
	Hardware:    []string{"cpu/x86-64-avx2"},
}

func TestGoldenLabelsCannotCollapse(t *testing.T) {
	t.Parallel()
	cases := []struct {
		file     string
		semantic WeightSemantic
		label    string
		origin   ArtifactOrigin
	}{
		{"native-binary.json", WeightNativeBinary, "1-bit", OriginNativeTrained},
		{"native-ternary.json", WeightNativeTernary, "1.58-bit", OriginNativeTrained},
		{"post-training-ternary.json", WeightPostTernary, "ternary", OriginPostTraining},
		{"integer-2bit.json", WeightInteger2Bit, "2-bit", OriginPostTraining},
	}

	seenSemantic := map[WeightSemantic]string{}
	seenLabel := map[string]WeightSemantic{}
	seenID := map[string]string{}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.file, func(t *testing.T) {
			raw := readFixture(t, tc.file)
			got := ParseAndAdjudicate(raw, fixtureCapabilities)
			if got.Outcome != OutcomeAccept || got.Reason != ReasonSupported {
				t.Fatalf("outcome = %s/%s (%s), want accept/supported", got.Outcome, got.Reason, got.Detail)
			}
			if got.Descriptor == nil {
				t.Fatal("accepted result omitted descriptor")
			}
			d := *got.Descriptor
			if d.Weights.Semantic != tc.semantic || d.Weights.Label != tc.label || d.Artifact.Origin != tc.origin {
				t.Fatalf("identity = %q/%q/%q, want %q/%q/%q", d.Weights.Semantic, d.Weights.Label, d.Artifact.Origin, tc.semantic, tc.label, tc.origin)
			}
			if prior, exists := seenSemantic[d.Weights.Semantic]; exists {
				t.Fatalf("semantic %q collapsed fixtures %s and %s", d.Weights.Semantic, prior, tc.file)
			}
			if prior, exists := seenLabel[d.Weights.Label]; exists {
				t.Fatalf("label %q collapsed %s and %s", d.Weights.Label, prior, d.Weights.Semantic)
			}
			id := SemanticID(d)
			if prior, exists := seenID[id]; exists {
				t.Fatalf("semantic id %q collapsed fixtures %s and %s", id, prior, tc.file)
			}
			seenSemantic[d.Weights.Semantic] = tc.file
			seenLabel[d.Weights.Label] = d.Weights.Semantic
			seenID[id] = tc.file
		})
	}
}

func TestMetadataDimensionsRemainIndependent(t *testing.T) {
	t.Parallel()
	native := acceptedFixture(t, "native-ternary.json")
	converted := acceptedFixture(t, "post-training-ternary.json")
	if native.Weights.Levels[1] != converted.Weights.Levels[1] {
		t.Fatal("fixture premise failed: both ternary artifacts should include zero")
	}
	if SemanticID(native) == SemanticID(converted) || native.Recipe.Kind == converted.Recipe.Kind {
		t.Fatal("native training and post-training ternarization collapsed")
	}
	if native.Activation == converted.Activation {
		t.Fatal("activation precision collapsed into weight semantics")
	}
	if native.Packing.Scheme == converted.Packing.Scheme {
		t.Fatal("packing scheme collapsed despite independent fixture values")
	}
	if native.Runtime == nil || converted.Runtime != nil {
		t.Fatal("runtime delegation presence was not preserved independently")
	}
	if !native.Hardware.Measured || converted.Hardware.Measured {
		t.Fatal("measured hardware envelope was not preserved independently")
	}
}

func TestTypedUnknownUnsupportedAndDelegatedResults(t *testing.T) {
	t.Parallel()
	base := acceptedFixture(t, "native-binary.json")
	cases := []struct {
		name   string
		mutate func(*Descriptor)
		caps   Capabilities
		want   Outcome
		reason ReasonCode
	}{
		{"unknown schema abstains", func(d *Descriptor) { d.Schema = "bitnetmeta/v99" }, fixtureCapabilities, OutcomeAbstain, ReasonUnknownSchema},
		{"unknown semantic abstains", func(d *Descriptor) { d.Weights.Semantic = "almost-ternary" }, fixtureCapabilities, OutcomeAbstain, ReasonUnknownWeightSemantic},
		{"unknown artifact version abstains", func(d *Descriptor) { d.Artifact.Version = "99" }, fixtureCapabilities, OutcomeAbstain, ReasonUnknownArtifactFormat},
		{"inconsistent label refuses", func(d *Descriptor) { d.Weights.Label = "2-bit" }, fixtureCapabilities, OutcomeRefuse, ReasonInconsistentArtifact},
		{"unsupported activation refuses", func(d *Descriptor) { d.Activation.Bits = 3 }, fixtureCapabilities, OutcomeRefuse, ReasonUnsupportedActivation},
		{"unsupported packing refuses", func(d *Descriptor) { d.Packing.Scheme = "mystery-pack" }, fixtureCapabilities, OutcomeRefuse, ReasonUnsupportedPacking},
		{"unsupported recipe refuses", func(d *Descriptor) { d.Recipe.Version = "99" }, fixtureCapabilities, OutcomeRefuse, ReasonUnsupportedRecipe},
		{"external runtime delegates", func(d *Descriptor) { d.Runtime = &Runtime{ID: "external", Version: "9"} }, fixtureCapabilities, OutcomeDelegate, ReasonRuntimeDelegation},
		{"unwitnessed hardware abstains", func(d *Descriptor) {
			d.Hardware = HardwareEnvelope{Measured: true, Vendor: "gpu", Architecture: "future", Metric: "latency", Value: 1, Unit: "ms", Witness: "fixture://future"}
		}, fixtureCapabilities, OutcomeAbstain, ReasonHardwareUnverified},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			d := base
			tc.mutate(&d)
			got := Adjudicate(d, tc.caps)
			if got.Outcome != tc.want || got.Reason != tc.reason {
				t.Fatalf("got %s/%s (%s), want %s/%s", got.Outcome, got.Reason, got.Detail, tc.want, tc.reason)
			}
		})
	}

	for _, raw := range [][]byte{[]byte(`{"schema":`), []byte(`{"schema":"bitnetmeta/v1","future":true}`)} {
		malformed := ParseAndAdjudicate(raw, fixtureCapabilities)
		if malformed.Outcome != OutcomeRefuse || malformed.Reason != ReasonMalformed {
			t.Fatalf("malformed = %s/%s, want refuse/malformed", malformed.Outcome, malformed.Reason)
		}
	}
}

func acceptedFixture(t *testing.T, name string) Descriptor {
	t.Helper()
	got := ParseAndAdjudicate(readFixture(t, name), fixtureCapabilities)
	if got.Outcome != OutcomeAccept || got.Descriptor == nil {
		t.Fatalf("fixture %s = %s/%s (%s)", name, got.Outcome, got.Reason, got.Detail)
	}
	return *got.Descriptor
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
