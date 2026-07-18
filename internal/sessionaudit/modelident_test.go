package sessionaudit

import (
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

// TestResolveModelIDCanonicalFleetIDs pins the typed canonical identity layer
// (#4635): every exact fleet model id resolves to ITSELF with "exact"
// provenance, and the raw provider spelling is preserved verbatim on the
// identity so the cost artifact can show raw and canonical side by side.
func TestResolveModelIDCanonicalFleetIDs(t *testing.T) {
	for _, id := range []string{
		"claude-opus-4-8",
		"claude-sonnet-4-6",
		"claude-sonnet-5",
		"claude-haiku-4-5-20251001",
		"claude-fable-5",
	} {
		mi := ResolveModelID(id)
		if !mi.Known() || mi.Canonical != id || mi.Provenance != ProvenanceExact {
			t.Fatalf("ResolveModelID(%q) = %+v, want canonical self with %q provenance", id, mi, ProvenanceExact)
		}
		if mi.Raw != id {
			t.Fatalf("ResolveModelID(%q) lost the raw id: %+v", id, mi)
		}
		r, prov, err := PriceForIdentity(mi)
		if err != nil {
			t.Fatalf("PriceForIdentity(%q) = %v, want a priced row", id, err)
		}
		if r.Input <= 0 || r.Output <= 0 || r.CacheWrite <= 0 || r.CacheRead <= 0 {
			t.Fatalf("%s rate card has a non-positive axis: %+v", id, r)
		}
		if !strings.HasPrefix(prov, "anthropic-published:") {
			t.Fatalf("PriceForIdentity(%q) provenance = %q, want anthropic-published:<tier>", id, prov)
		}
	}
}

// TestResolveModelIDPinnedAliases is the alias fixture #4635 requires: the
// dash/dot respelling and the undated date-suffix family spelling both resolve
// to the SAME canonical id, each carrying the provenance of the rule that
// resolved it — while the raw spelling is preserved.
func TestResolveModelIDPinnedAliases(t *testing.T) {
	cases := []struct {
		raw, canonical, provenance string
	}{
		{"claude-opus-4.8", "claude-opus-4-8", ProvenanceDashDot},
		{"claude-sonnet-4.6", "claude-sonnet-4-6", ProvenanceDashDot},
		{"claude-haiku-4.5-20251001", "claude-haiku-4-5-20251001", ProvenanceDashDot},
		{"Claude-Opus-4-8", "claude-opus-4-8", ProvenanceExact}, // case is not identity
		{"claude-haiku-4-5", "claude-haiku-4-5-20251001", ProvenanceDateSuffix},
		{"claude-haiku-4.5", "claude-haiku-4-5-20251001", ProvenanceDateSuffix},
	}
	for _, c := range cases {
		mi := ResolveModelID(c.raw)
		if mi.Canonical != c.canonical || mi.Provenance != c.provenance {
			t.Fatalf("ResolveModelID(%q) = %+v, want canonical %q via %q", c.raw, mi, c.canonical, c.provenance)
		}
		if mi.Raw != c.raw {
			t.Fatalf("ResolveModelID(%q) lost the raw spelling: %+v", c.raw, mi)
		}
	}
}

// TestResolveModelIDDoesNotOvermatch is the negative half of the fixture: a
// NEIGHBORING spelling — a different date snapshot, a version the fleet does
// not run, a bare tier word — must stay UNKNOWN rather than borrow a priced
// row. This is exactly the overmatch the legacy substring heuristic commits.
func TestResolveModelIDDoesNotOvermatch(t *testing.T) {
	for _, raw := range []string{
		"claude-haiku-4-5-20991231", // different date snapshot: NOT the pinned one
		"claude-sonnet-4-5",         // neighboring minor version, not in the fleet
		"claude-opus-4-80",          // superstring of a canonical id
		"claude-3-5-sonnet-latest",  // external-catalog family spelling, unpinned
		"opus",                      // bare tier word is not a model id
		"claude",                    // bare family word is not a model id
	} {
		mi := ResolveModelID(raw)
		if mi.Known() || mi.Provenance != ProvenanceUnknown {
			t.Fatalf("ResolveModelID(%q) = %+v, want unknown (no overmatch)", raw, mi)
		}
		if _, _, err := PriceForIdentity(mi); !errors.Is(err, ErrUnknownModelPricing) {
			t.Fatalf("PriceForIdentity(%q) err = %v, want ErrUnknownModelPricing", raw, err)
		}
	}
}

// TestStrictModelCostUSDFailsClosed pins the fail-closed pricing contract
// (#4635): an unknown model id returns ErrUnknownModelPricing — it can NEVER
// be reported as a silent $0 — and an unresolved Claude-family spelling
// refuses the neighboring-tier price the legacy substring heuristic would
// assign. Canonical ids, non-billed harness rows, and published non-Claude
// cards keep pricing exactly as before.
func TestStrictModelCostUSDFailsClosed(t *testing.T) {
	const mtok = 1_000_000
	// Canonical fleet ids price from their pinned rows.
	if got, err := StrictModelCostUSD("claude-opus-4-8", 0, 0, 0, mtok); err != nil || got != 75.0 {
		t.Fatalf("strict opus output MTok = (%.2f, %v), want (75, nil)", got, err)
	}
	if got, err := StrictModelCostUSD("claude-haiku-4-5-20251001", mtok, 0, 0, 0); err != nil || math.Abs(got-0.80) > 1e-9 {
		t.Fatalf("strict haiku input MTok = (%.4f, %v), want (0.80, nil)", got, err)
	}
	// The pinned undated alias prices IDENTICALLY to its dated canonical.
	if got, err := StrictModelCostUSD("claude-haiku-4-5", mtok, 0, 0, 0); err != nil || math.Abs(got-0.80) > 1e-9 {
		t.Fatalf("strict undated haiku alias = (%.4f, %v), want (0.80, nil)", got, err)
	}
	// Non-billed harness rows legitimately cost 0 with no error.
	if got, err := StrictModelCostUSD("<synthetic>", mtok, 0, 0, mtok); err != nil || got != 0 {
		t.Fatalf("strict synthetic = (%.2f, %v), want (0, nil)", got, err)
	}
	// A published non-Claude card still prices (the #4823 rows are provenance).
	if got, err := StrictModelCostUSD("deepseek-v4-pro", mtok, 0, 0, 0); err != nil || math.Abs(got-0.435) > 1e-9 {
		t.Fatalf("strict deepseek = (%.4f, %v), want (0.435, nil)", got, err)
	}

	// FAIL CLOSED: an unknown id errors instead of silently reporting $0 —
	// the legacy path is the defect this replaces (CostUSD returns 0.0 here).
	if legacy := CostUSD("mystery-model-x", mtok, 0, 0, 0); legacy != 0 {
		t.Fatalf("legacy CostUSD(mystery) = %.4f, expected the silent-zero defect this test documents", legacy)
	}
	if _, err := StrictModelCostUSD("mystery-model-x", mtok, 0, 0, 0); !errors.Is(err, ErrUnknownModelPricing) {
		t.Fatalf("strict mystery err = %v, want ErrUnknownModelPricing (never a silent $0)", err)
	}

	// FAIL CLOSED on overmatch: the legacy substring heuristic prices an
	// unknown dated Haiku neighbor AS haiku; strict refuses it.
	if legacy := CostUSD("claude-haiku-4-5-20991231", mtok, 0, 0, 0); math.Abs(legacy-0.80) > 1e-9 {
		t.Fatalf("legacy CostUSD(neighbor haiku) = %.4f, expected the 0.80 overmatch this test documents", legacy)
	}
	if _, err := StrictModelCostUSD("claude-haiku-4-5-20991231", mtok, 0, 0, 0); !errors.Is(err, ErrUnknownModelPricing) {
		t.Fatalf("strict neighbor-haiku err = %v, want ErrUnknownModelPricing (no neighboring-model pricing)", err)
	}
}

// TestAggregateCarriesIdentityAndUnknownHold pins the artifact half of #4635:
// the aggregate carries BOTH the raw and canonical id for every billed model
// (with resolution provenance), and a model with no pricing provenance lands
// on the explicit UNKNOWN hold — visible in the aggregate, the compact report
// totals, and a gate-able recommendation — instead of dissolving into a $0.
func TestAggregateCarriesIdentityAndUnknownHold(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("c1", 1_000, 0, 0, withModel("claude-opus-4-8")),
		assistantRecord("u1", 2_000, 0, 0, withModel("mystery-model-x")),
		assistantRecord("n1", 3_000, 0, 0, withModel("claude-haiku-4-5-20991231")),
		assistantRecord("syn", 0, 0, 0, withModel("<synthetic>")),
	}))
	agg := AggregateSessions([]Session{s})

	mi, ok := agg.ModelIdentities["claude-opus-4-8"]
	if !ok || mi.Raw != "claude-opus-4-8" || mi.Canonical != "claude-opus-4-8" || mi.Provenance != ProvenanceExact {
		t.Fatalf("opus identity = %+v (ok=%v), want exact canonical", mi, ok)
	}
	if mi, ok := agg.ModelIdentities["mystery-model-x"]; !ok || mi.Known() || mi.Provenance != ProvenanceUnknown {
		t.Fatalf("mystery identity = %+v (ok=%v), want unknown", mi, ok)
	}
	if _, ok := agg.ModelIdentities["<synthetic>"]; ok {
		t.Fatal("non-billed harness row must not get a model identity")
	}

	if got := agg.UnpricedModels; len(got) != 1 || got[0] != "mystery-model-x" {
		t.Fatalf("UnpricedModels = %v, want [mystery-model-x]", got)
	}
	if got := agg.UnverifiedClaudeIDs; len(got) != 1 || got[0] != "claude-haiku-4-5-20991231" {
		t.Fatalf("UnverifiedClaudeIDs = %v, want [claude-haiku-4-5-20991231]", got)
	}

	rep := BuildCompactReport([]Session{s}, agg, "", nil, false, 0, 1, nil, time.Now())
	if got := rep.Totals.UnpricedModels; len(got) != 1 || got[0] != "mystery-model-x" {
		t.Fatalf("compact totals unpriced = %v, want [mystery-model-x]", got)
	}
	if got := rep.Totals.UnverifiedClaudeIDs; len(got) != 1 || got[0] != "claude-haiku-4-5-20991231" {
		t.Fatalf("compact totals unverified claude = %v, want [claude-haiku-4-5-20991231]", got)
	}
	var hold *CompactRecommendation
	for i := range rep.Recommendations {
		if rep.Recommendations[i].Kind == "unpriced_model_hold" {
			hold = &rep.Recommendations[i]
		}
	}
	if hold == nil {
		t.Fatalf("recommendations = %+v, want an unpriced_model_hold gate", rep.Recommendations)
	}
	for _, want := range []string{"mystery-model-x", "claude-haiku-4-5-20991231"} {
		if !strings.Contains(hold.Evidence, want) {
			t.Fatalf("hold evidence %q missing %q", hold.Evidence, want)
		}
	}

	// A fully-resolved window raises no hold.
	clean := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("c1", 1_000, 0, 0, withModel("claude-opus-4-8")),
	}))
	cleanAgg := AggregateSessions([]Session{clean})
	if len(cleanAgg.UnpricedModels) != 0 || len(cleanAgg.UnverifiedClaudeIDs) != 0 {
		t.Fatalf("clean window hold = %v / %v, want empty", cleanAgg.UnpricedModels, cleanAgg.UnverifiedClaudeIDs)
	}
	cleanRep := BuildCompactReport([]Session{clean}, cleanAgg, "", nil, false, 0, 1, nil, time.Now())
	for _, rec := range cleanRep.Recommendations {
		if rec.Kind == "unpriced_model_hold" {
			t.Fatalf("clean window raised a hold: %+v", rec)
		}
	}
}
