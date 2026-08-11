package mixedprecision

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"
)

const digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

var testSupport = Support{
	Artifacts:  map[string][]string{"safetensors": {"1.0.0"}},
	Recipes:    map[string][]string{"awq": {"0.2.6"}},
	Runtimes:   map[string][]string{"vllm": {"0.10.0"}, "external-engine": {"4.1.0"}},
	Precisions: []string{"fp16", "int8", "int4"},
	Combinations: []Combination{
		{Artifact: "safetensors@1.0.0", Recipe: "awq@0.2.6", Runtime: "vllm@0.10.0", Outcome: OutcomeSupported},
		{Artifact: "safetensors@1.0.0", Recipe: "awq@0.2.6", Runtime: "external-engine@4.1.0", Outcome: OutcomeDelegate},
	},
}

func loadFixture(t *testing.T, name string) Descriptor {
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

func TestGoldenFixturesCoverSupportedUnsupportedAndDelegate(t *testing.T) {
	cases := []struct {
		name    string
		outcome Outcome
		reason  ReasonCode
	}{
		{"supported.json", OutcomeSupported, ReasonSupported},
		{"unsupported.json", OutcomeRefused, ReasonUnknownRuntimeVersion},
		{"delegate.json", OutcomeDelegate, ReasonRuntimeHandoff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Evaluate(loadFixture(t, tc.name), testSupport)
			if got.Outcome != tc.outcome || got.Reason != tc.reason {
				t.Fatalf("got %s/%s: %s", got.Outcome, got.Reason, got.Detail)
			}
		})
	}
}

func TestPropertyCanonicalizationAndCoverageAccounting(t *testing.T) {
	rng := rand.New(rand.NewSource(6254))
	for trial := 0; trial < 500; trial++ {
		n := 1 + rng.Intn(40)
		d := baseDescriptor()
		d.Modules, d.Rules = nil, nil
		var total, matched, weighted uint64
		for i := 0; i < n; i++ {
			params := uint64(1 + rng.Intn(100000))
			name := moduleName(i)
			d.Modules = append(d.Modules, Module{Name: "  " + name + "  ", Parameters: params})
			precision := []string{"int4", "int8", "fp16"}[rng.Intn(3)]
			d.Rules = append(d.Rules, Rule{Pattern: name, Precision: " " + precision + " "})
			total += params
			matched += params
			bits, _ := precisionBits(precision)
			weighted += params * bits
		}
		rng.Shuffle(len(d.Modules), func(i, j int) { d.Modules[i], d.Modules[j] = d.Modules[j], d.Modules[i] })
		got := Evaluate(d, testSupport)
		if got.Outcome != OutcomeSupported {
			t.Fatalf("trial %d: %#v", trial, got)
		}
		if got.Budget.ParametersTotal != total || got.Budget.ParametersMatched != matched || got.Budget.WeightedBits != weighted || got.Budget.Coverage != 1 {
			t.Fatalf("trial %d budget mismatch: %#v", trial, got.Budget)
		}
		if math.Abs(got.Budget.AverageBits-float64(weighted)/float64(total)) > 1e-12 {
			t.Fatalf("trial %d average", trial)
		}
		again := Evaluate(d, testSupport)
		if got.CanonicalID != again.CanonicalID || !reflect.DeepEqual(got.Assignments, again.Assignments) {
			t.Fatalf("trial %d nondeterministic", trial)
		}
	}
}

func TestAmbiguousAndUnmatchedModulesRefuse(t *testing.T) {
	d := baseDescriptor()
	d.Rules = append(d.Rules, Rule{Pattern: "model.layers.*", Precision: "int8"})
	if got := Evaluate(d, testSupport); got.Outcome != OutcomeRefused || got.Reason != ReasonAmbiguousModule {
		t.Fatalf("ambiguous: %#v", got)
	}
	d = baseDescriptor()
	d.Rules = d.Rules[:1]
	if got := Evaluate(d, testSupport); got.Outcome != OutcomeRefused || got.Reason != ReasonUnmatchedModule {
		t.Fatalf("unmatched: %#v", got)
	}
	d.Fallback = Fallback{Mode: FallbackHandoff}
	if got := Evaluate(d, testSupport); got.Outcome != OutcomeDelegate || got.Reason != ReasonUnmatchedModule {
		t.Fatalf("delegate: %#v", got)
	}
}

func TestDeclaredFallbackIsVisibleInBudget(t *testing.T) {
	d := baseDescriptor()
	d.Rules = d.Rules[:1]
	d.Fallback = Fallback{Mode: FallbackAssign, Precision: "fp16"}
	got := Evaluate(d, testSupport)
	if got.Outcome != OutcomeSupported || got.Reason != ReasonAcceptedFallback || got.Budget.ModulesFallback != 1 || got.Budget.ParametersFallback != 200 {
		t.Fatalf("got %#v", got)
	}
}

func TestPinnedVersionsAndEvidenceKindsAreEnforced(t *testing.T) {
	d := baseDescriptor()
	d.Provenance.Recipe.Version = "latest"
	if got := Evaluate(d, testSupport); got.Reason != ReasonUnknownRecipeVersion {
		t.Fatalf("unpinned recipe: %#v", got)
	}
	d = baseDescriptor()
	d.Evidence[0].Kind = EvidenceObserved
	if got := Evaluate(d, testSupport); got.Reason != ReasonInvalidEvidence {
		t.Fatalf("modeled presented as observed: %#v", got)
	}
	d = baseDescriptor()
	d.Evidence = append(d.Evidence, EvaluationEvidence{Kind: EvidenceObserved, Metric: "tokens_per_second", Dataset: pinned("fixed-prompts", "1"), Value: 55, Hardware: &HardwareEnvelope{Accelerator: "lab-gpu", Runtime: "vllm@0.10.0", Driver: "pinned-driver", OS: "linux"}, Samples: 20})
	if got := Evaluate(d, testSupport); got.Outcome != OutcomeSupported || len(got.Evidence) != 2 {
		t.Fatalf("observed envelope: %#v", got)
	}
}

func TestUnknownJSONFieldRefuses(t *testing.T) {
	raw, err := os.ReadFile("testdata/supported.json")
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatal(err)
	}
	object["future_field"] = true
	raw, err = json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseAndEvaluate(raw, testSupport)
	if err == nil || got.Reason != ReasonInvalidDescriptor {
		t.Fatalf("got %#v err=%v", got, err)
	}
}

func baseDescriptor() Descriptor {
	return Descriptor{Schema: SchemaV1, Provenance: Provenance{Artifact: pinned("safetensors", "1.0.0"), Recipe: pinned("awq", "0.2.6"), Runtime: pinned("vllm", "0.10.0")}, Modules: []Module{{Name: "model.layers.0.attn", Parameters: 100}, {Name: "model.layers.0.mlp", Parameters: 200}}, Rules: []Rule{{Pattern: "model.layers.0.attn", Precision: "int8"}, {Pattern: "model.layers.0.mlp", Precision: "int4"}}, Fallback: Fallback{Mode: FallbackRefuse}, Evidence: []EvaluationEvidence{{Kind: EvidenceModeled, Metric: "estimated_weight_bits", Dataset: pinned("module-inventory", "1"), Value: 1600, ModelBasis: "sum(parameters*assigned_bits); excludes scales, zero-points, padding, and runtime workspace"}}}
}
func pinned(id, version string) PinnedRef { return PinnedRef{ID: id, Version: version, SHA256: digest} }
func moduleName(i int) string             { return "model.layers." + fmtInt(i) }
func fmtInt(i int) string {
	if i == 0 {
		return "0"
	}
	b := [20]byte{}
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}
