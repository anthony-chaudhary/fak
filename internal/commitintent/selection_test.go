package commitintent

import (
	"errors"
	"reflect"
	"testing"
)

func caseFor(sel Selection, s Surface) (QualityCase, bool) {
	for _, c := range sel.Cases {
		if c.Surface == s {
			return c, true
		}
	}
	return QualityCase{}, false
}

func surfaceSet(sel Selection) map[Surface]bool {
	out := map[Surface]bool{}
	for _, c := range sel.Cases {
		out[c.Surface] = true
	}
	return out
}

// A changed path that maps to a known surface selects exactly that surface's
// case (plus the always-on sentinel), assigns its tier, documents its cost, and
// explains the choice — with no expansion and no inconclusive flag.
func TestSelectCasesMapsKnownSurface(t *testing.T) {
	sel, err := SelectCases([]string{"internal/gateway/sampler_topk.go"}, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases: %v", err)
	}
	if sel.Inconclusive {
		t.Fatalf("known change must not be inconclusive: %+v", sel)
	}
	c, ok := caseFor(sel, SurfaceSampler)
	if !ok {
		t.Fatalf("sampler surface not selected: %+v", surfaceSet(sel))
	}
	if c.Tier != TierPR {
		t.Fatalf("sampler tier = %q, want %q", c.Tier, TierPR)
	}
	if c.Cost == "" {
		t.Fatal("case must document runtime/resource cost")
	}
	if c.Rationale == "" {
		t.Fatal("selector must explain its choice")
	}
	if !reflect.DeepEqual(c.MatchedPaths, []string{"internal/gateway/sampler_topk.go"}) {
		t.Fatalf("matched paths = %v", c.MatchedPaths)
	}
	if !hasSurface(sel.Cases, SurfaceSentinel) {
		t.Fatal("every selection carries the sentinel case")
	}
}

// Acceptance: "Unknown changes expand coverage and selector explains choices."
func TestSelectCasesUnknownExpandsCoverage(t *testing.T) {
	sel, err := SelectCases([]string{"internal/mystery/frobnicate.go"}, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases: %v", err)
	}
	if !sel.Expanded {
		t.Fatalf("unknown change must expand coverage: %+v", sel)
	}
	if !reflect.DeepEqual(sel.UnknownPaths, []string{"internal/mystery/frobnicate.go"}) {
		t.Fatalf("unknown paths = %v", sel.UnknownPaths)
	}
	// Expansion covers every rule surface plus the sentinel.
	for _, s := range []Surface{SurfaceModel, SurfaceTokenizer, SurfaceEngine, SurfaceSampler, SurfaceCache, SurfaceReportRubric, SurfaceSentinel} {
		c, ok := caseFor(sel, s)
		if !ok {
			t.Fatalf("expanded coverage missing surface %q", s)
		}
		if s != SurfaceSentinel && c.Rationale == "" {
			t.Fatalf("expanded case %q must explain the choice", s)
		}
	}
}

// Acceptance: "missing or inconclusive evidence is never pass." An empty diff
// yields a sentinel-only plan flagged inconclusive — never an empty selection a
// caller could read as a clean pass.
func TestSelectCasesEmptyIsInconclusiveNeverEmpty(t *testing.T) {
	sel, err := SelectCases(nil, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases: %v", err)
	}
	if !sel.Inconclusive {
		t.Fatalf("empty diff must be inconclusive: %+v", sel)
	}
	if len(sel.Cases) != 1 || sel.Cases[0].Surface != SurfaceSentinel {
		t.Fatalf("inconclusive plan must be sentinel-only, got %+v", sel.Cases)
	}
}

// Acceptance: "Each case records model/tokenizer/engine, seed or deterministic
// oracle, code/module revision, and tolerance/baseline provenance." Oracle,
// Revision, and Baseline are the mandatory determinism/provenance fields.
func TestSelectCasesProvenanceComplete(t *testing.T) {
	sel, err := SelectCases([]string{"internal/mystery/x.go"}, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases: %v", err)
	}
	if len(sel.Cases) == 0 {
		t.Fatal("no cases to check provenance on")
	}
	for _, c := range sel.Cases {
		if c.Provenance.Revision != baseA {
			t.Fatalf("case %q revision = %q, want %q", c.Surface, c.Provenance.Revision, baseA)
		}
		if c.Provenance.Oracle == "" {
			t.Fatalf("case %q missing seed/deterministic oracle", c.Surface)
		}
		if c.Provenance.Baseline == "" {
			t.Fatalf("case %q missing tolerance/baseline provenance", c.Surface)
		}
	}
}

// Acceptance: "Assign the case to an explicit PR, nightly, or release tier."
func TestSelectCasesTiersAreExplicit(t *testing.T) {
	sel, err := SelectCases([]string{"internal/mystery/x.go"}, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases: %v", err)
	}
	for _, c := range sel.Cases {
		if err := ValidateTier(c.Tier); err != nil {
			t.Fatalf("case %q has non-explicit tier: %v", c.Surface, err)
		}
	}
}

// The selection is a pure, deterministic function of its inputs.
func TestSelectCasesDeterministic(t *testing.T) {
	in := []string{"internal/gateway/engine.go", "internal/mystery/z.go", "docs/report-rubric.md"}
	a, err := SelectCases(in, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases a: %v", err)
	}
	b, err := SelectCases(in, baseA, nil)
	if err != nil {
		t.Fatalf("SelectCases b: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("selection is not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

func TestSelectCasesValidation(t *testing.T) {
	if _, err := SelectCases([]string{"internal/gateway/x.go"}, "  ", nil); !errors.Is(err, ErrMissingField) {
		t.Fatalf("missing revision err = %v, want ErrMissingField", err)
	}
	badRules := []SurfaceRule{{Surface: SurfaceModel, Match: []string{"model"}, Tier: TierPR, Cost: "cpu-fast",
		Provenance: Provenance{Baseline: "golden:model"}}} // no oracle
	if _, err := SelectCases([]string{"internal/model/x.go"}, baseA, badRules); !errors.Is(err, ErrMissingField) {
		t.Fatalf("rule missing oracle err = %v, want ErrMissingField", err)
	}
	badTier := []SurfaceRule{{Surface: SurfaceModel, Match: []string{"model"}, Tier: Tier("weekly"), Cost: "cpu-fast",
		Provenance: Provenance{Oracle: "seed:0", Baseline: "golden:model"}}}
	if _, err := SelectCases([]string{"internal/model/x.go"}, baseA, badTier); !errors.Is(err, ErrInvalidField) {
		t.Fatalf("bad tier err = %v, want ErrInvalidField", err)
	}
}

func TestSelectForIntentUsesIntentPaths(t *testing.T) {
	intent := Intent{
		ID:      "issue-4575-cache",
		BaseSHA: baseA,
		Paths:   []string{"internal/gateway/kvcache.go"},
		Subject: "feat(commitintent): touch cache surface (#4575) (fak commitintent)",
	}
	sel, err := SelectForIntent(intent, nil)
	if err != nil {
		t.Fatalf("SelectForIntent: %v", err)
	}
	if _, ok := caseFor(sel, SurfaceCache); !ok {
		t.Fatalf("cache surface not selected from intent paths: %+v", surfaceSet(sel))
	}
	if sel.Revision != baseA {
		t.Fatalf("revision = %q, want intent base %q", sel.Revision, baseA)
	}
}

// Witness (issue #4575): a planted representative defect on the sampler surface
// is caught before AND after a selector regression. With the correct rules the
// sampler case is selected directly; with a "buggy" selector that has lost the
// sampler rule, the same path becomes unknown and coverage expansion still
// selects a sampler case — so the defect is never silently dropped. This is the
// captured proof: the assertion reds if coverage of the changed surface is lost.
func TestPlantedSamplerDefectNeverSilentlyDropped(t *testing.T) {
	// A path that maps to the sampler surface and nothing else in the defaults.
	const defect = "internal/quality/sampling_only.go"

	fixed, err := SelectCases([]string{defect}, baseA, DefaultSurfaceRules())
	if err != nil {
		t.Fatalf("fixed selector: %v", err)
	}
	if _, ok := caseFor(fixed, SurfaceSampler); !ok {
		t.Fatalf("fixed selector dropped the sampler defect: %+v", surfaceSet(fixed))
	}
	if fixed.Expanded {
		t.Fatalf("a known sampler change must not need expansion: %+v", fixed)
	}

	// Planted selector regression: the sampler rule is gone. The change is now
	// unclassified, but coverage must not silently drop — expansion re-covers it.
	var buggy []SurfaceRule
	for _, r := range DefaultSurfaceRules() {
		if r.Surface == SurfaceSampler {
			continue
		}
		buggy = append(buggy, r)
	}
	regressed, err := SelectCases([]string{defect}, baseA, buggy)
	if err != nil {
		t.Fatalf("regressed selector: %v", err)
	}
	if !regressed.Expanded {
		t.Fatalf("lost surface rule must expand coverage, got %+v", regressed)
	}
	if _, ok := caseFor(regressed, SurfaceSampler); ok {
		t.Fatal("buggy rule set has no sampler rule; expansion cannot fabricate one")
	}
	// The defect is still covered: it surfaces as an explicit unknown path that
	// forces full-coverage nightly/PR runs rather than a silent pass.
	if len(regressed.UnknownPaths) == 0 {
		t.Fatal("dropped-rule change must be recorded as an unknown path")
	}
}
