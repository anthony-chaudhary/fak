package fleetaccounts

import "testing"

// apexRoster is two routable claude workers: the restricted apex seat (Fable 5, tier 0)
// and a frontier seat (Opus, tier 1). The pair is enough to witness that apex is never
// auto-selected and that an apex request degrades to frontier rather than escalating.
func apexRoster() []Account {
	return []Account{
		{
			Dir: "C:/Users/u/.claude-fable-acct", Product: "claude", Account: ".claude-fable-acct",
			Tag: "fable", Kind: KindWorker, Reason: "apex worker",
			ModelTier: intp(TierApex), Model: strp("claude-fable-5"), Available: boolp(true),
		},
		{
			Dir: "C:/Users/u/.claude-opus-acct", Product: "claude", Account: ".claude-opus-acct",
			Tag: "opus", Kind: KindWorker, Reason: "frontier worker",
			ModelTier: intp(TierFrontier), Model: strp("opus"), Available: boolp(true),
		},
	}
}

// TestClassifyTaskApexIsExplicitOnly is the classifier half of the restriction: the apex
// tier (Fable 5) is reachable ONLY through an explicit operator request, and no task-text
// heuristic or other work-kind ever targets tier 0.
func TestClassifyTaskApexIsExplicitOnly(t *testing.T) {
	pol := DefaultPolicy()
	for _, c := range []string{"tier0", "t0", "0", "apex", "fable", "fable5", "fable-5", "APEX", " Apex "} {
		if got := ClassifyTask("", c, pol); got.TargetTier != TierApex {
			t.Errorf("ClassifyTask class=%q target=%d, want apex %d", c, got.TargetTier, TierApex)
		}
	}
	for _, c := range []string{"", "auto", "hard", "default", "tier1", "t1", "1",
		"engineering", "light", "tier2", "tier3", "gardening", "cleanup"} {
		for _, text := range []string{"", "implement a critical production security fix right now",
			"please do the single hardest most important task you can"} {
			if got := ClassifyTask(text, c, pol); got.TargetTier == TierApex {
				t.Errorf("ClassifyTask(text=%q class=%q) reached apex; inference must never target tier 0", text, c)
			}
		}
	}
}

// TestRouteAccountNeverAutoSelectsApex is the account-side half: a roster containing an
// apex (tier 0) seat must not hand it to a default/hard/light task even with tier
// fallback enabled — the restricted seat is reachable only when apex was targeted.
func TestRouteAccountNeverAutoSelectsApex(t *testing.T) {
	pol := DefaultPolicy()
	for _, class := range []string{"", "auto", "hard", "default", "engineering", "light", "tier2"} {
		// allowTierFallback=true is the most permissive ladder; apex still must not appear.
		r := RouteAccount(apexRoster(), "implement the feature", class, true, false, "claude", pol)
		if !r.OK || r.Account == nil {
			t.Fatalf("class %q: expected a routed frontier/light account, got %+v", class, r)
		}
		if r.Account.ModelTier != nil && *r.Account.ModelTier == TierApex {
			t.Fatalf("class %q auto-selected the restricted apex account %q", class, r.Account.Account)
		}
		if r.TargetTier == TierApex {
			t.Fatalf("class %q produced an apex target tier", class)
		}
	}
}

// TestRouteAccountApexRequestSelectsApexSeat proves the positive path: an explicit apex
// request routes to the tier-0 seat when one is offered.
func TestRouteAccountApexRequestSelectsApexSeat(t *testing.T) {
	r := RouteAccount(apexRoster(), "ship the feature", "apex", false, false, "claude", DefaultPolicy())
	if !r.OK || r.Account == nil || r.Account.ModelTier == nil || *r.Account.ModelTier != TierApex {
		t.Fatalf("explicit apex request = %+v, want the tier-0 (Fable 5) account", r)
	}
	if r.TargetTier != TierApex || r.FallbackUsed {
		t.Fatalf("apex hit should be an exact-tier match: target=%d fallback=%v", r.TargetTier, r.FallbackUsed)
	}
}

// TestRouteAccountApexDegradesDownNotUp proves apex degrades DOWN to frontier when no
// apex seat is offered (non-strict), and is apex-or-nothing under strictTier — it never
// falls sideways or up.
func TestRouteAccountApexDegradesDownNotUp(t *testing.T) {
	frontierOnly := apexRoster()[1:] // drop the apex seat, keep the opus frontier seat

	r := RouteAccount(frontierOnly, "x", "apex", false, false, "claude", DefaultPolicy())
	if !r.OK || r.Account == nil || r.Account.ModelTier == nil || *r.Account.ModelTier != TierFrontier || !r.FallbackUsed {
		t.Fatalf("apex request with no apex seat should degrade to frontier: %+v", r)
	}

	r = RouteAccount(frontierOnly, "x", "apex", false, true, "claude", DefaultPolicy())
	if r.OK {
		t.Fatalf("strict apex request with no apex seat must not fall back to frontier: %+v", r)
	}
}

// TestApexNotReachableByNumericProfileOverride pins the sentinel asymmetry: a profile
// with model_tier:0 is treated as UNSET (inferred from the model name), never as apex —
// so apex is reached only by naming a Fable-5 model, not by a casual numeric override.
func TestApexNotReachableByNumericProfileOverride(t *testing.T) {
	// model_tier:0 + a non-fable model → the model's real tier, not apex.
	if p := cleanProfile(ProfileOverride{ModelTier: 0, Model: "opus"}, "test"); p.ModelTier != TierFrontier {
		t.Fatalf("unset tier + opus = %d, want frontier %d (model_tier:0 must not mean apex)", p.ModelTier, TierFrontier)
	}
	// An unset tier with no model name at all → tier 3, not apex.
	if p := cleanProfile(ProfileOverride{}, "test"); p.ModelTier != TierOther {
		t.Fatalf("empty profile = %d, want other %d", p.ModelTier, TierOther)
	}
	// Only a Fable-5 NAME yields apex.
	if p := cleanProfile(ProfileOverride{Model: "claude-fable-5"}, "test"); p.ModelTier != TierApex {
		t.Fatalf("fable-5 profile = %d, want apex %d", p.ModelTier, TierApex)
	}
	// An explicit non-apex tier on a fable model is honored (operator downgrade).
	if p := cleanProfile(ProfileOverride{ModelTier: TierLight, Model: "claude-fable-5"}, "test"); p.ModelTier != TierLight {
		t.Fatalf("explicit tier-2 override on fable = %d, want %d", p.ModelTier, TierLight)
	}
}
