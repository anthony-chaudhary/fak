package engine

import "testing"

// TestCacheVerdictClosedVocabulary pins the closed verdict vocabulary: exactly the
// six item-31 inventory terms are Valid, and anything else is not. This is the guard
// against a new engine smuggling in a bespoke capability class.
func TestCacheVerdictClosedVocabulary(t *testing.T) {
	want := map[CacheVerdict]bool{
		CacheUnknown:        true,
		CachePassiveObserve: true,
		CacheActiveWarm:     true,
		CacheExactEvict:     true,
		CachePrefixClone:    true,
		CachePagedKV:        true,
	}
	got := CacheVerdicts()
	if len(got) != len(want) {
		t.Fatalf("CacheVerdicts() has %d members, want %d: %v", len(got), len(want), got)
	}
	seen := map[CacheVerdict]bool{}
	for _, v := range got {
		if !want[v] {
			t.Errorf("CacheVerdicts() returned unexpected member %q", v)
		}
		if seen[v] {
			t.Errorf("CacheVerdicts() returned duplicate member %q", v)
		}
		seen[v] = true
		if !v.Valid() {
			t.Errorf("closed-vocabulary member %q reports Valid()==false", v)
		}
	}
	for _, bogus := range []CacheVerdict{"", "warm", "kv", "PASSIVE-OBSERVE", "prefix_clone"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary verdict %q reports Valid()==true", bogus)
		}
	}
	// The zero value is the empty string, which is fail-closed: NOT in the vocabulary
	// and NOT an active behavior, so an unset verdict can never read as a positive.
	var zero CacheVerdict
	if zero.Valid() {
		t.Errorf("zero CacheVerdict %q must not be Valid (fail-closed)", zero)
	}
	if zero.Active() {
		t.Errorf("zero CacheVerdict %q must not be Active (fail-closed)", zero)
	}
}

// TestCacheVerdictActiveClass pins which verdicts describe ACTIVE cache behavior — the
// class that must witness cold-path correctness. Passive/paged/unknown are inert.
func TestCacheVerdictActiveClass(t *testing.T) {
	active := map[CacheVerdict]bool{
		CacheActiveWarm:  true,
		CacheExactEvict:  true,
		CachePrefixClone: true,
	}
	for _, v := range CacheVerdicts() {
		if got := v.Active(); got != active[v] {
			t.Errorf("%q.Active() = %v, want %v", v, got, active[v])
		}
	}
}

// TestCacheProvenanceSeparate pins the four provenance planes as a closed set — the
// #1490 honest-attribution rule keeps them distinct (a provider counter is not a
// kernel witness).
func TestCacheProvenanceSeparate(t *testing.T) {
	want := map[CacheProvenance]bool{
		ProvenanceKernel:   true,
		ProvenanceProvider: true,
		ProvenanceContext:  true,
		ProvenanceForecast: true,
	}
	got := CacheProvenances()
	if len(got) != len(want) {
		t.Fatalf("CacheProvenances() has %d members, want %d: %v", len(got), len(want), got)
	}
	for _, p := range got {
		if !want[p] || !p.Valid() {
			t.Errorf("provenance %q: want in-vocabulary and Valid()", p)
		}
	}
	for _, bogus := range []CacheProvenance{"", "gpu", "PROVIDER"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary provenance %q reports Valid()==true", bogus)
		}
	}
}

// TestCacheCapabilityValid checks well-formedness: a closed-vocabulary verdict and
// provenance is Valid; an out-of-vocabulary member is not.
func TestCacheCapabilityValid(t *testing.T) {
	ok := CacheCapability{Engine: "vllm", Verdict: CachePagedKV, Provenance: ProvenanceProvider}
	if !ok.Valid() {
		t.Errorf("well-formed capability reports Valid()==false: %+v", ok)
	}
	for _, bad := range []CacheCapability{
		{Verdict: "bogus", Provenance: ProvenanceKernel},
		{Verdict: CacheUnknown, Provenance: "bogus"},
	} {
		if bad.Valid() {
			t.Errorf("malformed capability reports Valid()==true: %+v", bad)
		}
	}
}
