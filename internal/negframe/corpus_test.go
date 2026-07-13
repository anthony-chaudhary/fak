package negframe

import "testing"

// TestGuardRuntimeCorpusEquivalence is the whole-corpus witness (#4421): every enumerated
// guard-runtime surface, once reframed, must clear all three gate properties at once. A new
// guard string added to GuardRuntimeCorpus that ships un-reframed or lossily reframed reds here
// instead of sliding in green. It also asserts the harness exercises a REAL reframe (not an
// all-no-op corpus that would pass vacuously): at least one surface must have an idiom flipped.
func TestGuardRuntimeCorpusEquivalence(t *testing.T) {
	corpus := GuardRuntimeCorpus()
	if len(corpus) == 0 {
		t.Fatal("GuardRuntimeCorpus is empty — nothing to witness")
	}
	verdicts := CorpusEquivalence(corpus)
	if len(verdicts) != len(corpus) {
		t.Fatalf("CorpusEquivalence returned %d verdicts for %d surfaces", len(verdicts), len(corpus))
	}
	for i, v := range verdicts {
		if v.Name != corpus[i].Name {
			t.Errorf("verdict[%d] name = %q, want %q (order not preserved)", i, v.Name, corpus[i].Name)
		}
		if !v.OK {
			t.Errorf("guard surface %q failed equivalence: %s\n  orig: %q\n  reframe: %q",
				v.Name, v.Reason, v.Original, v.Reframed)
		}
	}

	// Prove the harness actually drives the reframe fold: at least one surface must carry a
	// mechanical idiom that gets flipped, so a corpus of pure no-ops cannot pass vacuously.
	totalApplied := 0
	reframedNames := map[string]bool{}
	for _, hp := range corpus {
		if res := ReframePass(hp.Text); res.Applied > 0 {
			totalApplied += res.Applied
			reframedNames[hp.Name] = true
		}
	}
	if totalApplied == 0 {
		t.Fatal("no corpus surface exercised a real reframe — the harness would pass vacuously; add an idiom-bearing fixture")
	}
	if !reframedNames["resume-recovery-prompt"] {
		t.Errorf("expected the resume-recovery-prompt fixture to exercise a reframe of its interpolated fix idiom; reframed = %v", reframedNames)
	}
}

// TestEquivalenceGateRejectsLossyCandidates proves each of the three gate properties independently
// has teeth. Reframe is token-safe by construction, so a lossy candidate cannot be produced by
// Reframe and is synthesized here and fed to the lower-level comparator.
func TestEquivalenceGateRejectsLossyCandidates(t *testing.T) {
	cases := []struct {
		name       string
		orig       string
		candidate  string
		wantOK     bool
		wantReason string
	}{
		{
			name:       "token dropped",
			orig:       "route the write through `fak commit` to clear the `OFF_TRUNK` hold before proceeding",
			candidate:  "route the write through `fak commit` to clear the hold before proceeding", // `OFF_TRUNK` gone
			wantOK:     false,
			wantReason: "dropped a must-keep contract token",
		},
		{
			name:       "polarity flipped more negative",
			orig:       "remember to stamp the commit before pushing",
			candidate:  "remember to stamp the commit before pushing; do not skip it", // injects a negative
			wantOK:     false,
			wantReason: "reframe reads more negative than the original",
		},
		{
			name:       "gross drift below similarity floor",
			orig:       "please route the write through the wrapped commit path before proceeding",
			candidate:  "ok", // wholesale truncation, shares no terms
			wantOK:     false,
			wantReason: "lexical similarity 0.00 below floor 0.45",
		},
		{
			name:      "faithful reframe admitted",
			orig:      "do not forget to stamp the commit",
			candidate: "remember to stamp the commit",
			wantOK:    true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := equivalenceOf(tc.name, tc.orig, tc.candidate)
			if v.OK != tc.wantOK {
				t.Fatalf("OK = %v, want %v (reason: %s)", v.OK, tc.wantOK, v.Reason)
			}
			if tc.wantReason != "" && v.Reason != tc.wantReason {
				t.Fatalf("Reason = %q, want %q", v.Reason, tc.wantReason)
			}
		})
	}
}

// TestEquivalentIsIdempotent: a source already in positive voice (the reframe of any surface)
// gates clean on a second pass — Equivalent(Reframe(x)) is OK with nothing left to flip.
func TestEquivalentIsIdempotent(t *testing.T) {
	for _, hp := range GuardRuntimeCorpus() {
		once := Reframe(hp.Text)
		v := Equivalent(once)
		if !v.OK {
			t.Errorf("second-pass equivalence for %q failed: %s", hp.Name, v.Reason)
		}
		if v.Reframed != once {
			t.Errorf("%q not idempotent: Reframe(Reframe(x)) != Reframe(x)", hp.Name)
		}
	}
}

// TestBroadcastTierWeightOrders pins the tier ordering the paydown weight (#4408) depends on:
// a per-turn surface outweighs a per-session one, which outweighs a cold one.
func TestBroadcastTierWeightOrders(t *testing.T) {
	if !(TierPerTurn.Weight() > TierPerSession.Weight() && TierPerSession.Weight() > TierCold.Weight()) {
		t.Fatalf("tier weights out of order: per-turn=%d per-session=%d cold=%d",
			TierPerTurn.Weight(), TierPerSession.Weight(), TierCold.Weight())
	}
	for _, tc := range []struct {
		tier BroadcastTier
		want string
	}{{TierPerTurn, "per-turn"}, {TierPerSession, "per-session"}, {TierCold, "cold"}} {
		if got := tc.tier.String(); got != tc.want {
			t.Errorf("tier %d String() = %q, want %q", tc.tier, got, tc.want)
		}
	}
}

// TestLexicalCosineBounds sanity-checks the similarity proxy: identical text scores 1, disjoint
// text scores 0, and a small-span reframe lands well above the floor.
func TestLexicalCosineBounds(t *testing.T) {
	if got := lexicalCosine("stamp the commit", "stamp the commit"); got != 1 {
		t.Errorf("identical cosine = %v, want 1", got)
	}
	if got := lexicalCosine("alpha beta", "gamma delta"); got != 0 {
		t.Errorf("disjoint cosine = %v, want 0", got)
	}
	got := lexicalCosine("do not forget to stamp the commit", "remember to stamp the commit")
	if got < SimilarityFloor {
		t.Errorf("faithful reframe cosine = %.3f, below floor %.2f", got, SimilarityFloor)
	}
}
