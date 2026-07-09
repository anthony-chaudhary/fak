package bench

import (
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// registryPath is the committed structured benchmark registry the memo
// (docs/generation-benchmark-authority-view.md) classifies. The view reads it live so
// an edit that breaks the memo's stated finding reds this gate, not the prose.
const registryPath = "../../docs/benchmarks/registry.jsonl"

// TestAuthorityView_BindsCommittedRegistry is the #1669 witness: it promotes the
// design memo from described to COMPUTED by folding the LIVE registry through the
// three-axis classification and asserting the exact structural findings the memo's
// first-classification pass records. These numbers are the memo's evidence; binding
// them here makes them re-runnable instead of a hand pass.
func TestAuthorityView_BindsCommittedRegistry(t *testing.T) {
	f, err := os.Open(registryPath)
	if err != nil {
		t.Fatalf("open registry %q: %v", registryPath, err)
	}
	defer f.Close()
	claims, err := LoadAuthorityRegistry(f)
	if err != nil {
		t.Fatalf("LoadAuthorityRegistry: %v", err)
	}
	v := ClassifyAuthorityRegistry(claims)

	// The memo's committed counts (docs/generation-benchmark-authority-view.md,
	// "First classification pass"), asserted against the live registry.
	if v.Total != 52 {
		t.Errorf("Total = %d, want 52 (registry row count the memo classifies)", v.Total)
	}
	// The one confirmed horizon-laundering hit: canonical (top "quote this" tier) x
	// modeled (a geometry witness entitled only to gen/future).
	wantLaundered := []string{"webbench-webvoyager-hero"}
	if !reflect.DeepEqual(v.LaunderedIDs, wantLaundered) {
		t.Errorf("LaunderedIDs = %v, want %v", v.LaunderedIDs, wantLaundered)
	}
	if v.CitableModeled != 6 {
		t.Errorf("CitableModeled = %d, want 6 (modeled witness in a citable status)", v.CitableModeled)
	}
	if v.UnknownProvenance != 6 {
		t.Errorf("UnknownProvenance = %d, want 6", v.UnknownProvenance)
	}
	// Headline current-value vs future-potential split (entitled == now is current value).
	if v.CurrentValue != 30 {
		t.Errorf("CurrentValue = %d, want 30", v.CurrentValue)
	}
	if v.FuturePotential != v.Total-v.CurrentValue {
		t.Errorf("FuturePotential = %d, want %d (Total-CurrentValue)", v.FuturePotential, v.Total-v.CurrentValue)
	}

	// Entitled-horizon histogram over the closed vocabulary.
	wantEntitled := map[string]int{"now": 30, "next": 9, "future": 8, HorizonRetired: 5}
	if !reflect.DeepEqual(v.ByEntitled, wantEntitled) {
		t.Errorf("ByEntitled = %v, want %v", v.ByEntitled, wantEntitled)
	}
	// Promotion-relevance histogram (closed four-token set).
	wantRelevance := map[string]int{RelHolding: 31, RelPromoting: 14, RelGated: 2, RelRetired: 5}
	if !reflect.DeepEqual(v.ByRelevance, wantRelevance) {
		t.Errorf("ByRelevance = %v, want %v", v.ByRelevance, wantRelevance)
	}

	// Orthogonality restated from the single source of truth.
	if v.OrthogonalityNote != metrics.OrthogonalityNote || v.OrthogonalityNote == "" {
		t.Error("OrthogonalityNote must equal metrics.OrthogonalityNote and be non-empty")
	}

	// Closed-vocabulary discipline: every row lands in exactly one entitled bucket
	// (drawn from metrics.RoadmapGenerations + retired) and one relevance token, and
	// the histograms partition the whole registry.
	liveHorizons := map[string]bool{HorizonRetired: true}
	for _, h := range metrics.RoadmapGenerations {
		liveHorizons[h] = true
	}
	relTokens := map[string]bool{RelPromoting: true, RelHolding: true, RelGated: true, RelRetired: true}
	sumE, sumR := 0, 0
	for _, n := range v.ByEntitled {
		sumE += n
	}
	for _, n := range v.ByRelevance {
		sumR += n
	}
	if sumE != v.Total || sumR != v.Total {
		t.Errorf("histograms must partition the registry: sumEntitled=%d sumRelevance=%d total=%d", sumE, sumR, v.Total)
	}
	for _, r := range v.Rows {
		if !liveHorizons[r.EntitledHorizon] {
			t.Errorf("row %s: entitled horizon %q outside the closed set", r.ID, r.EntitledHorizon)
		}
		if !liveHorizons[r.ClaimedHorizon] {
			t.Errorf("row %s: claimed horizon %q outside the closed set", r.ID, r.ClaimedHorizon)
		}
		if !relTokens[r.PromotionRelevance] {
			t.Errorf("row %s: relevance %q outside the closed four-token set", r.ID, r.PromotionRelevance)
		}
		// The invariant the view enforces: laundered iff claimed strictly outranks entitled.
		if r.Laundered != (horizonRank[r.ClaimedHorizon] > horizonRank[r.EntitledHorizon]) {
			t.Errorf("row %s: Laundered=%v inconsistent with claimed=%s entitled=%s", r.ID, r.Laundered, r.ClaimedHorizon, r.EntitledHorizon)
		}
	}
}

// TestAuthorityView_Deterministic pins that the fold is a pure function: rows come
// back in stable ID order and two calls are byte-identical.
func TestAuthorityView_Deterministic(t *testing.T) {
	f, err := os.Open(registryPath)
	if err != nil {
		t.Fatalf("open registry: %v", err)
	}
	defer f.Close()
	claims, err := LoadAuthorityRegistry(f)
	if err != nil {
		t.Fatalf("LoadAuthorityRegistry: %v", err)
	}
	a := ClassifyAuthorityRegistry(claims)
	b := ClassifyAuthorityRegistry(claims)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("ClassifyAuthorityRegistry is not deterministic across two calls")
	}
	for i := 1; i < len(a.Rows); i++ {
		if a.Rows[i-1].ID > a.Rows[i].ID {
			t.Fatalf("rows not sorted by ID at %d: %q > %q", i, a.Rows[i-1].ID, a.Rows[i].ID)
		}
	}
}

// TestAuthorityDerivation pins the witness -> (entitled horizon, promotion relevance,
// laundering) derivation table on synthetic rows, independent of the live registry,
// so the classification logic is witnessed even if the registry churns.
func TestAuthorityDerivation(t *testing.T) {
	issue := 535
	cases := []struct {
		name       string
		claim      AuthorityClaim
		entitled   string
		relevance  string
		laundered  bool
		currentVal bool
	}{
		{"canonical-measured proves now, holds", AuthorityClaim{ID: "a", Status: "canonical", Provenance: "measured"}, "now", RelHolding, false, true},
		{"canonical-modeled is laundering (entitled future)", AuthorityClaim{ID: "b", Status: "canonical", Provenance: "modeled"}, "future", RelHolding, true, false},
		{"live-functional entitled next", AuthorityClaim{ID: "c", Status: "live", Provenance: "functional"}, "next", RelHolding, false, false},
		{"live-measured with blocker promotes", AuthorityClaim{ID: "d", Status: "live", Provenance: "measured", Issue: &issue}, "now", RelPromoting, false, true},
		{"gated withholds a future number", AuthorityClaim{ID: "e", Status: "gated", Provenance: "unknown"}, "future", RelGated, false, false},
		{"pending withholds a future number", AuthorityClaim{ID: "f", Status: "pending", Provenance: "unknown"}, "future", RelGated, false, false},
		{"retracted is a retired tombstone", AuthorityClaim{ID: "g", Status: "retracted", Provenance: "unknown"}, HorizonRetired, RelRetired, false, false},
		{"stale is retired even if it was measured", AuthorityClaim{ID: "h", Status: "stale", Provenance: "measured"}, HorizonRetired, RelRetired, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.claim.EntitledHorizon(); got != tc.entitled {
				t.Errorf("EntitledHorizon = %q, want %q", got, tc.entitled)
			}
			if got := tc.claim.PromotionRelevance(); got != tc.relevance {
				t.Errorf("PromotionRelevance = %q, want %q", got, tc.relevance)
			}
			if got := tc.claim.Laundered(); got != tc.laundered {
				t.Errorf("Laundered = %v, want %v", got, tc.laundered)
			}
			if got := tc.claim.ProvesCurrentValue(); got != tc.currentVal {
				t.Errorf("ProvesCurrentValue = %v, want %v", got, tc.currentVal)
			}
		})
	}
}
