package commitintent

import (
	"strings"
	"testing"
)

// TestCselFaithfulSelectionMatchesReferenceAndIsProvenanced proves the faithful
// selector routes the reference diff to the expected cases, every case carries
// complete provenance and a valid tier, and the differential oracle passes.
func TestCselFaithfulSelectionMatchesReferenceAndIsProvenanced(t *testing.T) {
	baseline := cselBaseline()
	ref := cselSelect(cselReferenceDiff(), cselFaithfulStrategy())
	if len(ref) == 0 {
		t.Fatal("reference selection is empty — the fixture diff must select cases")
	}

	wantSurfaces := map[cselSurface]bool{
		cselSurfaceModel: true, cselSurfaceSampler: true, cselSurfaceBackend: true,
		cselSurfaceCache: true, cselSurfaceReport: true, cselSurfaceSentinel: true,
	}
	tiersSeen := map[cselTier]bool{}
	for _, c := range ref {
		if !wantSurfaces[c.Surface] {
			t.Fatalf("case %q routed to unexpected surface %q", c.ID, c.Surface)
		}
		delete(wantSurfaces, c.Surface)
		if !cselValidTier(c.Tier) {
			t.Fatalf("case %q has invalid tier %q", c.ID, c.Tier)
		}
		tiersSeen[c.Tier] = true
		if strings.TrimSpace(c.Cost) == "" {
			t.Fatalf("case %q has no runtime/resource cost documented", c.ID)
		}
		if strings.TrimSpace(c.Reason) == "" {
			t.Fatalf("case %q has no reason — the selector must explain choices", c.ID)
		}
		prov := cselProvenanceOf(c, baseline)
		if !prov.complete() {
			t.Fatalf("case %q has incomplete provenance: %+v", c.ID, prov)
		}
	}
	if len(wantSurfaces) != 0 {
		t.Fatalf("faithful selection missed surfaces: %v", wantSurfaces)
	}
	// The parent contract requires an explicit PR / nightly / release split.
	for _, tier := range []cselTier{cselTierPR, cselTierNightly, cselTierRelease} {
		if !tiersSeen[tier] {
			t.Fatalf("reference selection never assigns tier %q — the tier split is not exercised", tier)
		}
	}

	prov := cselProvenanceOf(ref[0], baseline)
	if v := cselJudge(ref, cselSelect(cselReferenceDiff(), cselFaithfulStrategy()), prov); !v.Pass {
		t.Fatalf("faithful selection was not judged pass: %s", v.Detail)
	}
}

// TestCselUnknownChangeExpandsCoverage proves an unmapped changed path expands to
// the sentinel canary and the selector explains the expansion — and that a
// selector without expansion silently drops it (the inverted acceptance).
func TestCselUnknownChangeExpandsCoverage(t *testing.T) {
	unknown := []string{"internal/brand-new-subsystem/thing.go"}

	expanded := cselSelect(unknown, cselFaithfulStrategy())
	var sentinel *cselCase
	for i := range expanded {
		if expanded[i].Surface == cselSurfaceSentinel {
			sentinel = &expanded[i]
		}
	}
	if sentinel == nil {
		t.Fatal("unknown change did not expand to a sentinel case")
	}
	if sentinel.Tier != cselTierRelease {
		t.Fatalf("sentinel expansion tier = %q, want release", sentinel.Tier)
	}
	if !strings.Contains(sentinel.Reason, "expanded:internal/brand-new-subsystem") {
		t.Fatalf("sentinel reason does not explain the expansion: %q", sentinel.Reason)
	}

	noExpand := cselSelect(unknown, cselNoExpandStrategy())
	if len(noExpand) != 0 {
		t.Fatalf("expansion-disabled selector still selected cases for an unknown change: %+v", noExpand)
	}
}

// TestCselPlantedDefectsCaughtAndLocalized is the witness: each planted selection
// defect fails the differential oracle, and the oracle localizes it to the exact
// first case + field with a scrubbed replay artifact.
func TestCselPlantedDefectsCaughtAndLocalized(t *testing.T) {
	baseline := cselBaseline()
	ref := cselSelect(cselReferenceDiff(), cselFaithfulStrategy())

	defects := []struct {
		name      string
		strategy  cselStrategy
		wantCase  string
		wantField string
	}{
		{"dropped-cache-rule", cselDroppedCacheRuleStrategy(), cselCaseID(cselSurfaceCache), "presence"},
		{"no-expand-on-unknown", cselNoExpandStrategy(), cselCaseID(cselSurfaceSentinel), "presence"},
		{"mis-tiered-cache", cselMisTierStrategy(), cselCaseID(cselSurfaceCache), "tier"},
	}

	for _, tc := range defects {
		got := cselSelect(cselReferenceDiff(), tc.strategy)

		// The defect must actually perturb the selection (else it is no witness).
		if d := cselFirstDiff(ref, got); d == nil {
			t.Fatalf("defect %q did not change the selection — not a representative defect", tc.name)
		}

		// The artifact provenance is that of the diverging reference case.
		prov := cselProvenanceOf(cselRefCaseForTest(t, ref, tc.wantCase), baseline)
		v := cselJudge(ref, got, prov)
		if v.Pass {
			t.Fatalf("defect %q was judged pass — a planted defect must fail", tc.name)
		}
		if v.Artifact == nil || v.Artifact.Divergence == nil {
			t.Fatalf("defect %q produced no replay artifact", tc.name)
		}
		if got := v.Artifact.Divergence.CaseID; got != tc.wantCase {
			t.Fatalf("defect %q localized to %q, want %q (detail: %s)", tc.name, got, tc.wantCase, v.Detail)
		}
		if got := v.Artifact.Divergence.Field; got != tc.wantField {
			t.Fatalf("defect %q diverged on field %q, want %q", tc.name, got, tc.wantField)
		}
		// The artifact renders provenance + scrubbed surface routing, never a
		// raw diff or file body (no ".go" file path, no diff hunk marker).
		s := v.Artifact.String()
		if !strings.Contains(s, "surfaces=") || !strings.Contains(s, tc.wantCase) {
			t.Fatalf("defect %q replay artifact is not renderable: %s", tc.name, s)
		}
		if strings.Contains(s, ".go") || strings.Contains(s, "@@") {
			t.Fatalf("defect %q replay artifact leaked a raw path/diff: %s", tc.name, s)
		}
	}
}

// TestCselInconclusiveIsNeverPass proves an empty candidate selection is never a
// pass and still emits a replay artifact.
func TestCselInconclusiveIsNeverPass(t *testing.T) {
	baseline := cselBaseline()
	ref := cselSelect(cselReferenceDiff(), cselFaithfulStrategy())
	prov := cselProvenanceOf(ref[0], baseline)
	v := cselJudge(ref, nil, prov)
	if v.Pass {
		t.Fatal("an empty candidate selection was judged pass — inconclusive evidence must never pass")
	}
	if v.Artifact == nil {
		t.Fatal("inconclusive case produced no replay artifact")
	}
}

// cselRefCaseForTest returns the reference case with the given id, or fails.
func cselRefCaseForTest(t *testing.T, ref []cselCase, id string) cselCase {
	t.Helper()
	for _, c := range ref {
		if c.ID == id {
			return c
		}
	}
	t.Fatalf("reference selection has no case %q", id)
	return cselCase{}
}
