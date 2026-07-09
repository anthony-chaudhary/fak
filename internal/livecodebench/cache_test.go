package livecodebench

import "testing"

// TestCacheKeyIdentity is the #2108 cache-key witness: the key must be stable for
// identical inputs and must change when ANY of model, prompt, n, temperature, or
// release changes (the acceptance's cache-key composition), while ignoring fields
// that are not part of the request identity.
func TestCacheKeyIdentity(t *testing.T) {
	base := CacheKeyInput{Model: "m1", Prompt: "solve this", N: 1, Temperature: 0.2, Release: "release_v6"}

	if got := base.CacheKey(); got != base.CacheKey() {
		t.Fatalf("CacheKey not deterministic: %q vs %q", got, base.CacheKey())
	}

	mutate := map[string]CacheKeyInput{
		"model":       {Model: "m2", Prompt: base.Prompt, N: base.N, Temperature: base.Temperature, Release: base.Release},
		"prompt":      {Model: base.Model, Prompt: "solve that", N: base.N, Temperature: base.Temperature, Release: base.Release},
		"n":           {Model: base.Model, Prompt: base.Prompt, N: 5, Temperature: base.Temperature, Release: base.Release},
		"temperature": {Model: base.Model, Prompt: base.Prompt, N: base.N, Temperature: 0.7, Release: base.Release},
		"release":     {Model: base.Model, Prompt: base.Prompt, N: base.N, Temperature: base.Temperature, Release: "release_v5"},
	}
	baseKey := base.CacheKey()
	for field, in := range mutate {
		if in.CacheKey() == baseKey {
			t.Errorf("CacheKey did not change when %s changed; cache would wrongly reuse", field)
		}
	}

	// The field boundary must be forge-proof: (model "ab", prompt "c") and
	// (model "a", prompt "bc") are distinct requests, not a collision.
	a := CacheKeyInput{Model: "ab", Prompt: "c"}
	b := CacheKeyInput{Model: "a", Prompt: "bc"}
	if a.CacheKey() == b.CacheKey() {
		t.Errorf("CacheKey collided across a field boundary: %q", a.CacheKey())
	}
}

// TestPendingWorkResumes is the #2108 --continue-existing witness: a resumed run
// skips completed problems, keeps input order, and never re-emits a duplicate.
func TestPendingWorkResumes(t *testing.T) {
	all := []string{"p1", "p2", "p3", "p2", "p4"} // p2 duplicated in the work list
	done := map[string]bool{"p1": true, "p3": true}

	got := PendingWork(all, done)
	want := []string{"p2", "p4"}
	if len(got) != len(want) {
		t.Fatalf("PendingWork = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("PendingWork = %v, want %v (order/dedup)", got, want)
		}
	}

	// A fully-completed run has nothing pending.
	if p := PendingWork([]string{"p1"}, map[string]bool{"p1": true}); len(p) != 0 {
		t.Errorf("PendingWork over a finished run = %v, want empty", p)
	}
}

// TestCacheStatsHonest is the #2108 cache-hit-accounting witness: the rate is over
// genuine lookups only, and zero lookups reports 0%%, never a vanity 100%%.
func TestCacheStatsHonest(t *testing.T) {
	var s CacheStats
	if s.HitRate() != 0 || s.Lookups() != 0 {
		t.Fatalf("empty stats: HitRate=%v Lookups=%d, want 0/0", s.HitRate(), s.Lookups())
	}
	s.Hit()
	s.Hit()
	s.Hit()
	s.Miss()
	if s.Lookups() != 4 {
		t.Fatalf("Lookups = %d, want 4", s.Lookups())
	}
	if s.HitRate() != 0.75 {
		t.Fatalf("HitRate = %v, want 0.75", s.HitRate())
	}
}
