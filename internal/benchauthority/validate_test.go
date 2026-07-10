package benchauthority

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
