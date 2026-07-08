package modelscore

import (
	"reflect"
	"testing"
)

func TestCatalogProjectsEvidenceOpenRouterShape(t *testing.T) {
	r := NewRegistry()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("add: %v", err)
		}
	}
	// A fully-priced model: measured context, ILLUSTRATIVE price from the same source.
	must(r.Add(ModelEvidence{
		Model:   "beta",
		Cost:    &Cost{In: 3, Out: 15, Provenance: Provenance{Source: "price-page", Confidence: 0.5, Illustrative: true}},
		Context: &ContextWindow{Tokens: 200000, Provenance: Provenance{Source: "price-page", Confidence: 0.9}},
	}))
	// Context only, measured.
	must(r.Add(ModelEvidence{
		Model:   "alpha",
		Context: &ContextWindow{Tokens: 128000, Provenance: Provenance{Source: "docs", Confidence: 0.8}},
	}))
	// Neither cost nor context: id-only, still listed.
	must(r.Add(ModelEvidence{Model: "gamma"}))

	cat := r.Catalog()
	if len(cat) != 3 {
		t.Fatalf("catalog should list every model, got %d", len(cat))
	}
	// Deterministic sort by id: alpha, beta, gamma.
	if cat[0].ID != "alpha" || cat[1].ID != "beta" || cat[2].ID != "gamma" {
		t.Fatalf("catalog not sorted by id: %v", []string{cat[0].ID, cat[1].ID, cat[2].ID})
	}

	// beta: context + pricing; illustrative folded true; one deduped source.
	beta := cat[1]
	if beta.ContextTokens != 200000 || beta.Pricing == nil ||
		beta.Pricing.PromptPerMTok != 3 || beta.Pricing.CompletionPerMTok != 15 {
		t.Fatalf("beta projection wrong: %+v (pricing %+v)", beta, beta.Pricing)
	}
	if !beta.Illustrative {
		t.Fatalf("beta has an illustrative price; the entry must flag illustrative: %+v", beta)
	}
	if !reflect.DeepEqual(beta.Sources, []string{"price-page"}) {
		t.Fatalf("beta sources should dedupe to one, got %v", beta.Sources)
	}

	// alpha: context only, measured (not illustrative), no pricing.
	alpha := cat[0]
	if alpha.ContextTokens != 128000 || alpha.Pricing != nil || alpha.Illustrative {
		t.Fatalf("alpha projection wrong: %+v", alpha)
	}

	// gamma: id-only — every known model appears, even with no figures.
	gamma := cat[2]
	if gamma.ContextTokens != 0 || gamma.Pricing != nil || gamma.Illustrative || len(gamma.Sources) != 0 {
		t.Fatalf("gamma should be id-only: %+v", gamma)
	}
}

func TestCatalogEntryLookup(t *testing.T) {
	r := NewRegistry()
	if err := r.Add(ModelEvidence{
		Model:   "x",
		Context: &ContextWindow{Tokens: 8192, Provenance: Provenance{Source: "s", Confidence: 1}},
	}); err != nil {
		t.Fatalf("add: %v", err)
	}
	e, ok := r.CatalogEntry("x")
	if !ok || e.ContextTokens != 8192 {
		t.Fatalf("CatalogEntry(x) = %+v, %v", e, ok)
	}
	if _, ok := r.CatalogEntry("nope"); ok {
		t.Fatal("CatalogEntry(unknown) should be false")
	}
}
