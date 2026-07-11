package scorecard

import "testing"

// hp is a *float64 literal helper for seeding CacheHealthFacts component healths.
func hp(x float64) *float64 { return &x }

// healthyFacts is a seeded fixture where every cache family clears the pass line, so the fold is
// clean (debt 0, ok) and the headline is the mean of the five family healths.
func healthyFacts() CacheHealthFacts {
	return CacheHealthFacts{
		ManagedCachePosture:        hp(0.90),
		ReuseRatio:                 hp(0.80),
		ShedEffectiveness:          hp(0.70),
		WitnessedObservedAgreement: hp(0.95),
		UpgradeFiredRate:           hp(0.85),
	}
}

// TestCacheHealthNumberMovesWhenAComponentDegrades is the issue #3643 done-condition witness: the
// single 0..1 number moves when a component degrades in a seeded fixture.
func TestCacheHealthNumberMovesWhenAComponentDegrades(t *testing.T) {
	base, basePresent, _ := CacheHealth(healthyFacts())
	if basePresent != 5 {
		t.Fatalf("present = %d, want 5 (all families seeded)", basePresent)
	}
	// mean(0.90,0.80,0.70,0.95,0.85) = 0.84
	if base < 0.839 || base > 0.841 {
		t.Fatalf("baseline cache-health = %.4f, want ~0.84", base)
	}

	degraded := healthyFacts()
	degraded.ReuseRatio = hp(0.20) // one family degrades below the pass line
	got, present, _ := CacheHealth(degraded)
	if present != 5 {
		t.Fatalf("present after degrade = %d, want 5", present)
	}
	if got >= base {
		t.Fatalf("cache-health did not move down: baseline %.4f, degraded %.4f", base, got)
	}
	// mean(0.90,0.20,0.70,0.95,0.85) = 0.72
	if got < 0.719 || got > 0.721 {
		t.Fatalf("degraded cache-health = %.4f, want ~0.72", got)
	}
}

// TestCacheHealthWorklistIsWorstFirst pins the worklist ordering: every scored family, lowest
// health first, canonical order breaking ties.
func TestCacheHealthWorklistIsWorstFirst(t *testing.T) {
	f := healthyFacts()
	f.ShedEffectiveness = hp(0.10) // the worst
	f.ReuseRatio = hp(0.40)        // second worst (also below the 0.5 pass line)
	_, _, worklist := CacheHealth(f)
	if len(worklist) != 5 {
		t.Fatalf("worklist len = %d, want 5 (every scored family)", len(worklist))
	}
	if worklist[0].Component != CacheHealthShed {
		t.Fatalf("worklist[0] = %q, want %q (lowest health first)", worklist[0].Component, CacheHealthShed)
	}
	if worklist[1].Component != CacheHealthReuse {
		t.Fatalf("worklist[1] = %q, want %q", worklist[1].Component, CacheHealthReuse)
	}
	// health must be non-decreasing across the worklist.
	for i := 1; i < len(worklist); i++ {
		if worklist[i].Health < worklist[i-1].Health {
			t.Fatalf("worklist not worst-first at %d: %.3f before %.3f", i, worklist[i-1].Health, worklist[i].Health)
		}
	}
	// only the two below-pass-line families are flagged in debt.
	if !worklist[0].InDebt || !worklist[1].InDebt {
		t.Fatalf("worst two families should be in debt: %+v", worklist[:2])
	}
	for _, r := range worklist[2:] {
		if r.InDebt {
			t.Fatalf("family %q above pass line should not be in debt (health %.3f)", r.Component, r.Health)
		}
	}
}

// TestCacheHealthNilFamilyExcludedNotZero pins the anti-conflation rule: a nil family is EXCLUDED
// from the fold (no evidence), never scored as a measured 0.0 that would tank the headline.
func TestCacheHealthNilFamilyExcludedNotZero(t *testing.T) {
	f := CacheHealthFacts{
		ManagedCachePosture: hp(0.90),
		ReuseRatio:          hp(0.80),
		// the other three families have no evidence this window
	}
	number, present, worklist := CacheHealth(f)
	if present != 2 {
		t.Fatalf("present = %d, want 2 (only the two seeded families)", present)
	}
	if len(worklist) != 2 {
		t.Fatalf("worklist len = %d, want 2", len(worklist))
	}
	// mean(0.90,0.80) = 0.85, NOT dragged toward 0 by the three absent families.
	if number < 0.849 || number > 0.851 {
		t.Fatalf("cache-health = %.4f, want ~0.85 (absent families excluded, not zero)", number)
	}
}

// TestCacheHealthNoEvidenceIsOne pins the empty fixture: no family evidence folds to 1.0 (nothing
// is known-unhealthy), debt 0, ok -- not a spurious F.
func TestCacheHealthNoEvidenceIsOne(t *testing.T) {
	number, present, worklist := CacheHealth(CacheHealthFacts{})
	if present != 0 || number != 1 || len(worklist) != 0 {
		t.Fatalf("empty fixture = (number %.3f, present %d, worklist %d), want (1.0, 0, 0)", number, present, len(worklist))
	}
	p := ComposeCacheHealth(CacheHealthFacts{})
	if !p.OK {
		t.Fatalf("empty fixture payload should be OK (no known-unhealthy family): %s", p.Reason)
	}
	if v, ok := p.Corpus["value"].(float64); !ok || v != 1 {
		t.Fatalf("empty fixture corpus.value = %v, want 1.0", p.Corpus["value"])
	}
}

// TestComposeCacheHealthPayload pins the control-pane payload shape and the debt/ok gate.
func TestComposeCacheHealthPayload(t *testing.T) {
	clean := ComposeCacheHealth(healthyFacts())
	if !clean.OK || clean.Verdict != "OK" {
		t.Fatalf("healthy fixture should be OK/clean, got ok=%v verdict=%s reason=%s", clean.OK, clean.Verdict, clean.Reason)
	}
	if clean.Schema != CacheHealthSchema {
		t.Fatalf("schema = %q, want %q", clean.Schema, CacheHealthSchema)
	}
	if d := clean.Corpus[CacheHealthDebtKey]; d != 0 {
		t.Fatalf("clean debt = %v, want 0", d)
	}
	// corpus.cache_health equals corpus.value by construction (Score == 100*health).
	ch, chOK := clean.Corpus["cache_health"].(float64)
	val, valOK := clean.Corpus["value"].(float64)
	if !chOK || !valOK || ch != val {
		t.Fatalf("cache_health (%v) must equal value (%v)", clean.Corpus["cache_health"], clean.Corpus["value"])
	}
	if cp := clean.Corpus["components_present"]; cp != 5 {
		t.Fatalf("components_present = %v, want 5", cp)
	}

	// degrade two families below the pass line -> debt 2, ACTION, headline drops.
	f := healthyFacts()
	f.ShedEffectiveness = hp(0.10)
	f.UpgradeFiredRate = hp(0.30)
	dirty := ComposeCacheHealth(f)
	if dirty.OK || dirty.Verdict != "ACTION" {
		t.Fatalf("degraded fixture should be ACTION, got ok=%v verdict=%s", dirty.OK, dirty.Verdict)
	}
	if d := dirty.Corpus[CacheHealthDebtKey]; d != 2 {
		t.Fatalf("degraded debt = %v, want 2 (two families below pass line)", d)
	}
	if dch := dirty.Corpus["cache_health"].(float64); dch >= ch {
		t.Fatalf("degraded cache_health %.4f should be below clean %.4f", dch, ch)
	}
}

// TestCacheHealthComponentHelpers pins the family-metric conversions (reused, not re-derived).
func TestCacheHealthComponentHelpers(t *testing.T) {
	// PostureHealth: 0..100 pct -> 0..1; nil passes through.
	if got := PostureHealth(hp(66.7)); got == nil || *got < 0.666 || *got > 0.668 {
		t.Fatalf("PostureHealth(66.7) = %v, want ~0.667", got)
	}
	if PostureHealth(nil) != nil {
		t.Fatal("PostureHealth(nil) must stay nil (no evidence)")
	}
	// ShedEffectivenessHealth: fired/(fired+bailed); no decisions -> nil.
	if got := ShedEffectivenessHealth(3, 1); got == nil || *got != 0.75 {
		t.Fatalf("ShedEffectivenessHealth(3,1) = %v, want 0.75", got)
	}
	if ShedEffectivenessHealth(0, 0) != nil {
		t.Fatal("ShedEffectivenessHealth(0,0) must be nil (no shed decisions)")
	}
	// UpgradeFiredHealth: upgrades/(upgrades+refusals); no heads -> nil.
	if got := UpgradeFiredHealth(3, 1); got == nil || *got != 0.75 {
		t.Fatalf("UpgradeFiredHealth(3,1) = %v, want 0.75", got)
	}
	if UpgradeFiredHealth(0, 0) != nil {
		t.Fatal("UpgradeFiredHealth(0,0) must be nil (no upgrade heads)")
	}
	// WitnessedObservedAgreement: 1 - |gross-net|, reusing GrossNetDivergence.
	if got := WitnessedObservedAgreement(0.30, 0.05); got < 0.749 || got > 0.751 {
		t.Fatalf("WitnessedObservedAgreement(0.30,0.05) = %.4f, want ~0.75", got)
	}
	if got := WitnessedObservedAgreement(0.10, 0.10); got != 1 {
		t.Fatalf("WitnessedObservedAgreement(equal) = %.4f, want 1.0", got)
	}
}
