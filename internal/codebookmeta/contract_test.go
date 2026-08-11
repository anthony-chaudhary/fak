package codebookmeta

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

var fixtureCapability = Capability{
	PackingIDs:     []string{"nibble-lsb@1", "bitpack-lsb@1"},
	DecodeFeatures: []string{"per-block-scale", "explicit-codebook"},
	RoutedRuntimes: []string{"lab-runtime@2.4.1"},
}

func readFixture(t *testing.T, name string) Descriptor {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var d Descriptor
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	return d
}

func TestFixtureRoundTripsAndPinsProvenance(t *testing.T) {
	for _, name := range []string{"integer-grid.json", "nf4.json", "learned.json"} {
		t.Run(name, func(t *testing.T) {
			want := readFixture(t, name)
			raw, err := MarshalCanonical(want)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Parse(raw, fixtureCapability)
			if err != nil {
				t.Fatal(err)
			}
			if got.Outcome != OutcomeSupported || got.Reason != ReasonAccepted {
				t.Fatalf("got %#v", got)
			}
			if !reflect.DeepEqual(*got.Descriptor, want) {
				t.Fatalf("round trip changed descriptor\n got: %#v\nwant: %#v", *got.Descriptor, want)
			}
			p := got.Descriptor.Provenance
			if p.ArtifactID == "" || p.ArtifactDigest.Value == "" || p.RecipeID == "" || p.RecipeVersion == "" || p.RecipeDigest.Value == "" || p.RuntimeID == "" || p.RuntimeVersion == "" || p.ModelID == "" || p.ModelRevision == "" {
				t.Fatalf("provenance was not fully pinned: %#v", p)
			}
		})
	}
}

func TestMissingLearnedCodebookPayloadIsTypedRefusal(t *testing.T) {
	d := readFixture(t, "learned.json")
	d.Codebook.Entries = nil
	d.Codebook.Parameters = nil
	d.Codebook.Payload = nil
	d.Codebook.Digest = CodebookDigest(d.Codebook)
	got := Adjudicate(d, fixtureCapability)
	if got.Outcome != OutcomeRefused || got.Reason != ReasonMissingCodebookPayload {
		t.Fatalf("got %#v", got)
	}
}

func TestUnknownAndUnsupportedNeverSilentlyFallBack(t *testing.T) {
	base := readFixture(t, "nf4.json")
	cases := []struct {
		name   string
		mutate func(*Descriptor)
		want   ReasonCode
	}{
		{"schema", func(d *Descriptor) { d.Schema = "fak.codebookmeta/v99" }, ReasonUnknownSchema},
		{"kind", func(d *Descriptor) { d.Codebook.Kind = "mystery" }, ReasonUnknownCodebookKind},
		{"packing", func(d *Descriptor) { d.Packing.ID = "unknown" }, ReasonPackingUnavailable},
		{"digest", func(d *Descriptor) {
			d.Codebook.Digest.Value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		}, ReasonCodebookDigestMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := base
			tc.mutate(&d)
			got := Adjudicate(d, fixtureCapability)
			if got.Outcome != OutcomeRefused || got.Reason != tc.want {
				t.Fatalf("got %s/%s want unsupported/%s", got.Outcome, got.Reason, tc.want)
			}
		})
	}
}

func TestDecodeRequirementDelegatesOnlyToPinnedRuntime(t *testing.T) {
	d := readFixture(t, "learned.json")
	d.DecodeRequirements.RequiredFeatures = append(d.DecodeRequirements.RequiredFeatures, "vendor-fused-decode")
	d.DecodeRequirements.RoutedRuntime = "lab-runtime@2.4.1"
	got := Adjudicate(d, fixtureCapability)
	if got.Outcome != OutcomeDelegate || got.Reason != ReasonRuntimeDelegationRequired || got.Detail != "lab-runtime@2.4.1:vendor-fused-decode" {
		t.Fatalf("got %#v", got)
	}

	d.DecodeRequirements.RoutedRuntime = "unavailable@1"
	got = Adjudicate(d, fixtureCapability)
	if got.Outcome != OutcomeRefused || got.Reason != ReasonDecodeUnavailable {
		t.Fatalf("got %#v", got)
	}
}

func TestModeledAndObservedEvaluationsRemainDistinct(t *testing.T) {
	d := readFixture(t, "learned.json")
	if len(d.Evaluations) != 2 || d.Evaluations[0].Kind != EvidenceObserved || d.Evaluations[1].Kind != EvidenceModeled {
		t.Fatalf("evaluation provenance collapsed: %#v", d.Evaluations)
	}
	if d.Evaluations[0].Value == nil {
		t.Fatal("observed metadata witness must retain its value")
	}
	if d.Evaluations[1].Value != nil {
		t.Fatal("modeled research statement must not masquerade as a measured value")
	}
}

func TestMalformedJSONIsTyped(t *testing.T) {
	got, err := Parse([]byte(`{"schema":`), fixtureCapability)
	if err == nil || got.Outcome != OutcomeRefused || got.Reason != ReasonInvalidJSON {
		t.Fatalf("got %#v err=%v", got, err)
	}
}
