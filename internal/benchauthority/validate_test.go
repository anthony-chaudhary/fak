package benchauthority

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
)

// validMeasured returns a fully-admissible MEASURED claim; tests mutate one field
// to exercise a single rule at a time.
func validMeasured() Claim {
	return Claim{
		ID:       "ctx-horizon",
		Title:    "Context horizon",
		Headline: "60.3x TTFT reduction",
		Metric:   "TTFT",
		Value:    "60.3",
		Unit:     "x",
		Status:   Measured,
		Baseline: "no-cache",
		Commit:   "abc123",
		Artifact: "artifacts/ctx.json",
	}
}

// hasProblem reports whether any error in errs contains sub.
func hasProblem(errs []error, sub string) bool {
	for _, e := range errs {
		if strings.Contains(e.Error(), sub) {
			return true
		}
	}
	return false
}

// TestClaimValidate_ValidShapes asserts the admissible claim shapes report zero
// problems — one per number-bearing status plus the honest withheld/tombstone forms.
func TestClaimValidate_ValidShapes(t *testing.T) {
	verified := validMeasured()
	verified.Status = Verified
	verified.Provenance = "WITNESSED"

	theoretical := validMeasured()
	theoretical.Status = Theoretical
	theoretical.Artifact = "" // a projection's derivation may be inline

	gated := Claim{
		ID: "future-num", Title: "Future number", Status: Gated,
		Fences: []string{"blocked on 80GB DGX serving lane (#3059)"},
	}
	pending := Claim{
		ID: "pending-num", Title: "Pending number", Status: Pending,
		Fences: []string{"result packet committed; run not yet executed"},
	}
	retracted := Claim{
		ID: "old-num", Title: "Old claim", Status: Retracted,
		Replacement: "ctx-horizon",
	}

	cases := []struct {
		name string
		c    Claim
	}{
		{"measured", validMeasured()},
		{"verified", verified},
		{"theoretical without artifact", theoretical},
		{"gated withholds a number", gated},
		{"pending withholds a number", pending},
		{"retracted tombstone", retracted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if errs := tc.c.Validate(); len(errs) != 0 {
				t.Fatalf("expected zero problems, got %v", errs)
			}
		})
	}
}

// TestClaimValidate_Rejects asserts each admissibility rule fires with a specific,
// actionable message. Each case mutates exactly one field of a valid claim.
func TestClaimValidate_Rejects(t *testing.T) {
	mut := func(f func(*Claim)) Claim {
		c := validMeasured()
		f(&c)
		return c
	}
	cases := []struct {
		name string
		c    Claim
		want string
	}{
		{"empty ID", mut(func(c *Claim) { c.ID = "" }), "empty ID"},
		{"empty Title", mut(func(c *Claim) { c.Title = " " }), "empty Title"},
		{"unknown status", mut(func(c *Claim) { c.Status = "SORTA" }), "unknown Status"},
		{"non-numeric value (prose)", mut(func(c *Claim) { c.Value = "≈60x" }), "not numeric"},
		{"non-numeric value (adjective)", mut(func(c *Claim) { c.Value = "a lot faster" }), "not numeric"},
		{"empty value", mut(func(c *Claim) { c.Value = "" }), "missing numeric Value"},
		{"NaN value", mut(func(c *Claim) { c.Value = "NaN" }), "not finite"},
		{"Inf value", mut(func(c *Claim) { c.Value = "Inf" }), "not finite"},
		{"missing metric", mut(func(c *Claim) { c.Metric = "" }), "missing Metric"},
		{"missing unit", mut(func(c *Claim) { c.Unit = "" }), "missing Unit"},
		{"missing baseline", mut(func(c *Claim) { c.Baseline = "" }), "needs a Baseline"},
		{"missing artifact", mut(func(c *Claim) { c.Artifact = "" }), "needs an Artifact"},
		{"gated asserts a number", Claim{ID: "g", Title: "g", Status: Gated, Value: "3", Fences: []string{"x"}}, "must not assert a numeric Value"},
		{"gated omits the witness fence", Claim{ID: "g", Title: "g", Status: Gated}, "needs at least one Fence"},
		{"pending asserts a number", Claim{ID: "p", Title: "p", Status: Pending, Value: "3", Fences: []string{"x"}}, "must not assert a numeric Value"},
		{"retracted without replacement", Claim{ID: "r", Title: "r", Status: Retracted}, "needs a Replacement"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.c.Validate()
			if !hasProblem(errs, tc.want) {
				t.Fatalf("expected a problem containing %q, got %v", tc.want, errs)
			}
		})
	}
}

// TestValidateClaims_DuplicateID asserts the cross-claim uniqueness check fires on a
// repeated ID (which would collide the ledger's deep-link anchors).
func TestValidateClaims_DuplicateID(t *testing.T) {
	a := validMeasured()
	b := validMeasured() // same ID "ctx-horizon"
	errs := ValidateClaims([]Claim{a, b})
	if !hasProblem(errs, "duplicate ID") {
		t.Fatalf("expected a duplicate-ID problem, got %v", errs)
	}
}

// TestValidateClaims_AggregatesAllViolations asserts ValidateClaims does not fail
// fast — every bad claim contributes its problem to the aggregate.
func TestValidateClaims_AggregatesAllViolations(t *testing.T) {
	bad1 := validMeasured()
	bad1.ID = "one"
	bad1.Value = "fast" // non-numeric
	bad2 := validMeasured()
	bad2.ID = "two"
	bad2.Baseline = "" // missing baseline
	errs := ValidateClaims([]Claim{bad1, bad2})
	if !hasProblem(errs, "not numeric") || !hasProblem(errs, "needs a Baseline") {
		t.Fatalf("expected both violations aggregated, got %v", errs)
	}
}

// TestValidate_RegistryIsAdmissible asserts the package-level Validate() over the
// shipped registry is clean — the seed registry (empty today) must never carry an
// inadmissible claim, so this test also guards every future transcribed Claim.
func TestValidate_RegistryIsAdmissible(t *testing.T) {
	if errs := Validate(); len(errs) != 0 {
		t.Fatalf("registry is not admissible; fix the offending Claim literal:\n%v", errs)
	}
}

// TestValidateArtifacts asserts the on-disk half: an existing Artifact passes, a
// missing one is reported, and an empty Artifact path is skipped (not every status
// requires one).
func TestValidateArtifacts(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "artifacts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "artifacts", "ctx.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	present := validMeasured() // Artifact artifacts/ctx.json exists under root
	missing := validMeasured()
	missing.ID = "gone"
	missing.Artifact = "artifacts/nope.json"
	empty := validMeasured()
	empty.ID = "noart"
	empty.Artifact = ""

	if errs := ValidateArtifacts(root, []Claim{present, empty}); len(errs) != 0 {
		t.Fatalf("expected present+empty-artifact claims to pass, got %v", errs)
	}
	errs := ValidateArtifacts(root, []Claim{missing})
	if !hasProblem(errs, "not found") {
		t.Fatalf("expected a not-found problem for the missing artifact, got %v", errs)
	}
}

// TestValidateArtifacts_SimulationPromotionBoundary is the benchmark-authority
// lint witness for #9424. The same validator-clean structural-count artifact may
// support a theoretical noncompetitive row, but it cannot be relabelled as a
// measured/verified result or an outward competitive comparison.
func TestValidateArtifacts_SimulationPromotionBoundary(t *testing.T) {
	root := t.TempDir()
	seed := int64(9424)
	ev := benchcli.SimulationEvidence{
		Schema:       benchcli.SimulationEvidenceSchema,
		EvidenceType: benchcli.EvidenceStructuralCount,
		ClaimCeiling: benchcli.ClaimCorrectnessOnly,
		Engine: benchcli.SimulationEngine{
			Name:         "authority-lint-fixture",
			Revision:     "r1",
			ConfigDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		Workload: benchcli.WorkloadProvenance{
			Name:   "fixed-mechanism-count",
			Source: "internal/benchauthority/validate_test.go",
			Digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		},
		ValidityEnvelope: benchcli.ValidityEnvelope{
			Description: "one fixed mechanism-count fixture",
			Dimensions:  map[string]string{"shape": "fixture-v1"},
		},
		ExcludedEffects: []string{"elapsed time", "throughput", "competitor runtime"},
		Replay: benchcli.ReplaySpec{
			Seed:               &seed,
			Stream:             "fixed",
			Repetitions:        1,
			IndependentStreams: 1,
		},
		Cost: benchcli.SimulationCost{HostWallTimeMS: 1, HostCPUTimeMS: 1, Bytes: 256},
	}
	path := writeBenchmarkArtifactFixture(t, root, "simulated.json", benchcli.BenchmarkArtifact{
		RunID:              "authority-lint-simulated",
		SimulationEvidence: &ev,
	})

	theoretical := validMeasured()
	theoretical.ID = "simulated-bound"
	theoretical.Status = Theoretical
	theoretical.Artifact = path
	if errs := ValidateArtifacts(root, []Claim{theoretical}); len(errs) != 0 {
		t.Fatalf("validator-clean noncompetitive theoretical evidence should be admissible, got %v", errs)
	}

	measured := theoretical
	measured.ID = "simulated-measured"
	measured.Status = Measured
	if errs := ValidateArtifacts(root, []Claim{measured}); !hasProblem(errs, "MEASURED row cannot use non-hardware") {
		t.Fatalf("simulated artifact populated a MEASURED row; got %v", errs)
	}

	verified := theoretical
	verified.ID = "simulated-verified"
	verified.Status = Verified
	if errs := ValidateArtifacts(root, []Claim{verified}); !hasProblem(errs, "VERIFIED row cannot use non-hardware") {
		t.Fatalf("simulated artifact populated a VERIFIED row; got %v", errs)
	}

	competitive := theoretical
	competitive.ID = "simulated-competitive"
	competitive.Competitive = true
	if errs := ValidateArtifacts(root, []Claim{competitive}); !hasProblem(errs, "competitive row cannot use non-hardware") {
		t.Fatalf("simulated artifact populated a competitive row; got %v", errs)
	}
}

// TestValidateArtifacts_InvalidSimulationEvidenceFailsClosed proves an invalid
// claim-ceiling promotion is rejected by the shared benchcli validator before
// authority classification, and malformed envelopes cannot evade that validator
// by failing DecodeArtifact.
func TestValidateArtifacts_InvalidSimulationEvidenceFailsClosed(t *testing.T) {
	root := t.TempDir()
	seed := int64(1)
	invalid := benchcli.SimulationEvidence{
		Schema:       benchcli.SimulationEvidenceSchema,
		EvidenceType: benchcli.EvidenceStructuralCount,
		ClaimCeiling: benchcli.ClaimMeasuredAbsolute,
		Engine: benchcli.SimulationEngine{
			Name:         "invalid-promotion",
			Revision:     "r1",
			ConfigDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		Workload: benchcli.WorkloadProvenance{
			Name: "fixture", Source: "validate_test.go",
			Digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		},
		ValidityEnvelope: benchcli.ValidityEnvelope{
			Description: "fixture", Dimensions: map[string]string{"shape": "one"},
		},
		ExcludedEffects: []string{"elapsed time"},
		Replay: benchcli.ReplaySpec{
			Seed: &seed, Stream: "fixed", Repetitions: 1, IndependentStreams: 1,
		},
	}
	invalidPath := writeBenchmarkArtifactFixture(t, root, "invalid.json", benchcli.BenchmarkArtifact{
		RunID: "invalid-simulation-promotion", SimulationEvidence: &invalid,
	})
	c := validMeasured()
	c.Status = Theoretical
	c.Artifact = invalidPath
	if errs := ValidateArtifacts(root, []Claim{c}); !hasProblem(errs, "malformed benchmark_artifact envelope") {
		t.Fatalf("invalid evidence did not fail closed, got %v", errs)
	}

	malformedPath := filepath.ToSlash("artifacts/malformed.json")
	full := filepath.Join(root, filepath.FromSlash(malformedPath))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	// The valid lineage sibling is load-bearing: DecodeArtifact must not fall
	// through from a malformed modern envelope to this legacy compatibility path.
	malformed := []byte(`{"benchmark_artifact":{"run_id":"malformed-simulation","simulation_evidence":"not-an-object"},"lineage":{"lineage_schema":"fak-bench-lineage/1","app_version":"test","utc":"2026-08-27T00:00:00Z","git_commit":"abc123","go_version":"go1.test","node":"fixture"}}`)
	if err := os.WriteFile(full, malformed, 0o644); err != nil {
		t.Fatal(err)
	}
	c.Artifact = malformedPath
	if errs := ValidateArtifacts(root, []Claim{c}); !hasProblem(errs, "malformed benchmark_artifact envelope") {
		t.Fatalf("malformed evidence evaded the lint, got %v", errs)
	}
}

func writeBenchmarkArtifactFixture(t *testing.T, root, name string, art benchcli.BenchmarkArtifact) string {
	t.Helper()
	if art.Schema == "" {
		art.Schema = benchcli.BenchmarkArtifactSchema
	}
	rel := filepath.ToSlash(filepath.Join("artifacts", name))
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]any{"benchmark_artifact": art})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return rel
}
