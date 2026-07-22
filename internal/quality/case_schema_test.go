package quality

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func loadV1Fixture(t *testing.T) QualityCase {
	t.Helper()
	data, err := os.ReadFile("testdata/case_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	c, err := LoadCase(data)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestCanonicalV1FixtureRoundTrip(t *testing.T) {
	want := loadV1Fixture(t)
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadCase(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed case:\n got: %#v\nwant: %#v", got, want)
	}
}

func TestCanonicalV1RejectsMalformedProvenance(t *testing.T) {
	base := loadV1Fixture(t)
	tests := []struct {
		name string
		edit func(*QualityCase)
		want string
	}{
		{"model revision", func(c *QualityCase) { c.Metadata.Model.Revision = "" }, "metadata.model"},
		{"tokenizer revision", func(c *QualityCase) { c.Metadata.Tokenizer.Revision = "" }, "metadata.tokenizer"},
		{"backend", func(c *QualityCase) { c.Metadata.Engine.Backend = "" }, "metadata.engine"},
		{"engine flags", func(c *QualityCase) { c.Metadata.Engine.Flags = nil }, "replay flags"},
		{"determinism", func(c *QualityCase) { c.Metadata.Oracle = OracleEvidence{} }, "non-zero seed or metadata.oracle"},
		{"code revision", func(c *QualityCase) { c.Metadata.Code.Revision = "" }, "metadata.code"},
		{"tolerance provenance", func(c *QualityCase) { c.Metadata.Tolerance.Revision = "" }, "metadata.tolerance"},
		{"baseline provenance", func(c *QualityCase) { c.Metadata.Baseline.Revision = "" }, "metadata.baseline"},
		{"tier", func(c *QualityCase) { c.Metadata.Tier.Name = "weekly" }, "pr, nightly, or release"},
		{"cost", func(c *QualityCase) { c.Metadata.Cost.MemoryMiB = 0 }, "metadata.cost"},
		{"timeout absent", func(c *QualityCase) { c.Metadata.Cost.TimeoutSeconds = 0 }, "positive timeout_seconds"},
		{"timeout below runtime", func(c *QualityCase) { c.Metadata.Cost.RuntimeSeconds = 100 }, "timeout_seconds must be >= runtime_seconds"},
		{"owner", func(c *QualityCase) { c.Metadata.Owner = "" }, "metadata.owner"},
		{"family", func(c *QualityCase) { c.Metadata.Family = "smoke" }, "metadata.family"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := base
			tt.edit(&c)
			data, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			_, err = LoadCase(data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("LoadCase error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestCanonicalV1SeedCanSupplyDeterminism(t *testing.T) {
	c := loadV1Fixture(t)
	c.Params.Seed = 42
	c.Metadata.Oracle = OracleEvidence{}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCase(data); err != nil {
		t.Fatalf("seed-pinned case refused: %v", err)
	}
}

func TestCanonicalFailureReplayIsScrubbedAndIndependentlyPasses(t *testing.T) {
	c := loadV1Fixture(t)
	c.Prompt = "Return canonical answer; api_key=super-secret"
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatal(err)
	}
	defect := ScriptedRunner{Label: "planted-defect", Trace: Trace{Tokens: []string{"wrong", "answer"}, Text: "token=super-secret"}}
	failed, err := RunCase(c, ReferenceRunner{}, defect, oracles)
	if err != nil {
		t.Fatal(err)
	}
	if failed.Pass || failed.FailureBundle == nil {
		t.Fatal("planted defect passed or emitted no replay artifact")
	}
	fb := failed.FailureBundle
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 0 {
		t.Fatalf("first divergence = %#v, want token 0", fb.FirstDivergence)
	}
	artifact, err := json.Marshal(fb)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(artifact), "super-secret") || !fb.Scrubbed {
		t.Fatalf("replay artifact was not scrubbed: %s", artifact)
	}
	clean, err := RunCase(fb.Case, ReferenceRunner{}, ScriptedRunner{Label: "clean-replay", Trace: fb.Reference}, oracles)
	if err != nil {
		t.Fatal(err)
	}
	if !clean.Pass {
		t.Fatalf("independent clean replay failed: %s", Explain(clean))
	}
}

func TestMissingOrInconclusiveEvidenceNeverPasses(t *testing.T) {
	c := loadV1Fixture(t)
	for name, oracles := range map[string][]Oracle{"missing": nil, "inconclusive": {inconclusiveOracle{}}} {
		t.Run(name, func(t *testing.T) {
			result, err := RunCase(c, ReferenceRunner{}, ScriptedRunner{Trace: c.Reference}, oracles)
			if err == nil && result.Pass {
				t.Fatalf("%s evidence passed", name)
			}
		})
	}
}

type inconclusiveOracle struct{}

func (inconclusiveOracle) Name() string { return "inconclusive" }
func (inconclusiveOracle) Kind() string { return "evidence" }
func (inconclusiveOracle) Judge(Trace, Trace, QualityCase) Verdict {
	return Verdict{Oracle: "inconclusive", Kind: "evidence", Detail: "evidence inconclusive"}
}
