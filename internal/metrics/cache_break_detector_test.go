package metrics

import (
	"strings"
	"testing"
)

// baseTurn is the established prefix used across these tests: a system prompt, a
// tool schema, and one already-sent turn of history, each with a token length so
// the induced cache_creation is priced from a real span rather than zero.
func baseTurn() TurnPrefix {
	return TurnPrefix{
		System:        "you are fak",
		Tools:         `[{"name":"read"},{"name":"write"}]`,
		HistoryHead:   "u1:hello|a1:hi",
		SystemTokens:  100,
		ToolsTokens:   250,
		HistoryTokens: 400,
	}
}

// The clean pass the contract's witness asks for: an unmutated turn. The first
// turn establishes the prefix and the second matches it byte-for-byte, so nothing
// is witnessed and no cost is booked.
func TestCacheBreakDetectorCleanPassOnUnmutatedTurn(t *testing.T) {
	d := NewCacheBreakDetector(CacheBreakPolicyWarn)

	first := d.Observe(baseTurn())
	if first.Broken {
		t.Fatalf("first turn reported a break: %+v", first)
	}
	if first.ObservedTurn != 1 {
		t.Fatalf("first turn = %d, want 1", first.ObservedTurn)
	}

	second := d.Observe(baseTurn())
	if second.Broken || second.Witness != "" {
		t.Fatalf("unmutated turn reported a break: %+v", second)
	}
	if r := d.Report(); r.Events != 0 || r.CostTokens != 0 {
		t.Fatalf("clean session booked cost: %+v", r)
	}
}

// The false-positive guard the contract names. A normal conversation APPENDS to
// the history head every turn; if growth counted as divergence the detector would
// flag every healthy turn and be worthless. The established head must still be a
// byte-prefix of the grown one, which is exactly what stays intact on an append.
func TestCacheBreakDetectorPureHistoryAppendIsNotABreak(t *testing.T) {
	d := NewCacheBreakDetector(CacheBreakPolicyWarn)
	d.Observe(baseTurn())

	grown := baseTurn()
	grown.HistoryHead += "|u2:more|a2:sure"
	grown.HistoryTokens = 900

	v := d.Observe(grown)
	if v.Broken {
		t.Fatalf("appended turn misreported as a break: %+v", v)
	}
	if !strings.Contains(v.Reason, "extended") {
		t.Fatalf("append reason did not name the extension: %q", v.Reason)
	}

	// The grown head is now the baseline, so appending again is still clean.
	grown2 := grown
	grown2.HistoryHead += "|u3:again"
	if v := d.Observe(grown2); v.Broken {
		t.Fatalf("second append misreported as a break: %+v", v)
	}
}

// Cause attribution is by FIRST divergence in wire order, and the induced
// cache_creation is the new turn's cold span from that component onward — every
// byte after a divergence has to be written to cache too.
func TestCacheBreakDetectorAttributesCauseAndPricesColdSpan(t *testing.T) {
	for _, tc := range []struct {
		name     string
		mutate   func(*TurnPrefix)
		wantCau  CacheBreakCause
		wantCost int64
	}{
		{
			name:     "rebuilt system prompt makes the whole prefix cold",
			mutate:   func(p *TurnPrefix) { p.System = "you are something else" },
			wantCau:  CacheBreakRebuiltPrompt,
			wantCost: 100 + 250 + 400,
		},
		{
			name:     "toolset change leaves the system prompt warm",
			mutate:   func(p *TurnPrefix) { p.Tools = `[{"name":"read"},{"name":"write"},{"name":"bash"}]` },
			wantCau:  CacheBreakToolsetChange,
			wantCost: 250 + 400,
		},
		{
			name:     "reordered tool schema is a real break, not a no-op",
			mutate:   func(p *TurnPrefix) { p.Tools = `[{"name":"write"},{"name":"read"}]` },
			wantCau:  CacheBreakToolsetChange,
			wantCost: 250 + 400,
		},
		{
			name:     "rewritten already-sent turn leaves system+tools warm",
			mutate:   func(p *TurnPrefix) { p.HistoryHead = "u1:goodbye|a1:hi" },
			wantCau:  CacheBreakAlteredTurn,
			wantCost: 400,
		},
		{
			name:     "truncated history head is a rewrite of sent context",
			mutate:   func(p *TurnPrefix) { p.HistoryHead = "u1:h" },
			wantCau:  CacheBreakAlteredTurn,
			wantCost: 400,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := NewCacheBreakDetector(CacheBreakPolicyWarn)
			d.Observe(baseTurn())

			mutated := baseTurn()
			tc.mutate(&mutated)

			v := d.Observe(mutated)
			if !v.Broken {
				t.Fatalf("mutation not detected: %+v", v)
			}
			if v.Cause != tc.wantCau {
				t.Fatalf("cause = %q, want %q", v.Cause, tc.wantCau)
			}
			if v.Event.CostTokens != tc.wantCost {
				t.Fatalf("cost = %d, want %d", v.Event.CostTokens, tc.wantCost)
			}
			// The witness line the contract asks for, verbatim shape.
			if !strings.Contains(v.Witness, "cache broken here, +") {
				t.Fatalf("witness missing the contract phrase: %q", v.Witness)
			}
			if !strings.Contains(v.Witness, string(tc.wantCau)) {
				t.Fatalf("witness did not name the cause: %q", v.Witness)
			}
		})
	}
}

// deny and warn are not two log levels over one behavior — the contract's named
// confusion risk. Only warn advances the established prefix (the mutation reached
// the wire); deny refuses it, so the ORIGINAL prefix is still what the session
// carries and returning to it is clean.
func TestCacheBreakDetectorDenyKeepsBaselineWarnAdvancesIt(t *testing.T) {
	mutated := baseTurn()
	mutated.System = "rebuilt"

	deny := NewCacheBreakDetector(CacheBreakPolicyDeny)
	deny.Observe(baseTurn())
	dv := deny.Observe(mutated)
	if !dv.Broken || !dv.Denied {
		t.Fatalf("deny policy did not refuse the mutation: %+v", dv)
	}
	// Baseline kept: the original prefix is still established, so it is clean.
	if v := deny.Observe(baseTurn()); v.Broken {
		t.Fatalf("deny advanced the baseline; original prefix reported broken: %+v", v)
	}

	warn := NewCacheBreakDetector(CacheBreakPolicyWarn)
	warn.Observe(baseTurn())
	wv := warn.Observe(mutated)
	if !wv.Broken || wv.Denied {
		t.Fatalf("warn policy did not allow the mutation: %+v", wv)
	}
	// Baseline advanced: the mutated prefix is now established, so returning to
	// the original is itself a second break rather than a match.
	if v := warn.Observe(baseTurn()); !v.Broken {
		t.Fatalf("warn did not advance the baseline; revert not reported broken: %+v", v)
	}
}

// Cost the deny policy PREVENTED must never be folded in with cost that was
// actually paid, or a CheckBudget gate would fail a session that spent nothing.
func TestCacheBreakDetectorSeparatesIncurredFromAvoidedCost(t *testing.T) {
	mutated := baseTurn()
	mutated.Tools = "[]"

	deny := NewCacheBreakDetector(CacheBreakPolicyDeny)
	deny.Observe(baseTurn())
	deny.Observe(mutated)

	if r := deny.Report(); r.Events != 0 || r.CostTokens != 0 {
		t.Fatalf("denied break booked incurred cost: %+v", r)
	}
	av := deny.Avoided()
	if av.Events != 1 || av.CostTokens != 250+400 {
		t.Fatalf("avoided report = %+v, want 1 event / 650 tokens", av)
	}
	if len(av.ByCause) != 1 || av.ByCause[0].Cause != CacheBreakToolsetChange {
		t.Fatalf("avoided by-cause = %+v", av.ByCause)
	}

	warn := NewCacheBreakDetector(CacheBreakPolicyWarn)
	warn.Observe(baseTurn())
	warn.Observe(mutated)

	if a := warn.Avoided(); a.Events != 0 {
		t.Fatalf("allowed break booked avoided cost: %+v", a)
	}
	r := warn.Report()
	if r.Events != 1 || r.CostTokens != 250+400 {
		t.Fatalf("incurred report = %+v, want 1 event / 650 tokens", r)
	}
}

// The detector is the live producer #2916's counter was landed waiting for: its
// Report must drop straight into that gate and fail a regressed session.
func TestCacheBreakDetectorReportFeedsTheBudgetGate(t *testing.T) {
	d := NewCacheBreakDetector(CacheBreakPolicyWarn)
	d.Observe(baseTurn())

	rebuilt := baseTurn()
	rebuilt.System = "rebuilt once"
	d.Observe(rebuilt)
	rebuilt.System = "rebuilt twice"
	d.Observe(rebuilt)

	r := d.Report()
	if r.Events != 2 {
		t.Fatalf("events = %d, want 2", r.Events)
	}
	if err := r.CheckBudget(CacheBreakBudget{MaxEvents: 5, MaxCostTokens: 10000}); err != nil {
		t.Fatalf("session within budget failed the gate: %v", err)
	}
	if err := r.CheckBudget(CacheBreakBudget{MaxEvents: 1, MaxCostTokens: 1000}); err == nil {
		t.Fatal("regressed session passed the cache-break budget gate")
	}
}

// A sanctioned prefix change (an acknowledged compaction rewrite or operator
// rebuild) is not a surprise mutation — the invalidating assumption named in the
// file header. Reset is how a caller declares one, and the next turn re-primes.
func TestCacheBreakDetectorResetAbsorbsASanctionedRebuild(t *testing.T) {
	d := NewCacheBreakDetector(CacheBreakPolicyDeny)
	d.Observe(baseTurn())

	compacted := baseTurn()
	compacted.HistoryHead = "summary:earlier turns folded"

	d.Reset()
	if v := d.Observe(compacted); v.Broken {
		t.Fatalf("sanctioned rebuild after Reset was witnessed as a break: %+v", v)
	}
	if r := d.Report(); r.Events != 0 {
		t.Fatalf("Reset session booked a break: %+v", r)
	}
	// The compacted prefix is the new established one; mutating it still breaks.
	compacted.System = "rebuilt after compaction"
	if v := d.Observe(compacted); !v.Broken || v.Cause != CacheBreakRebuiltPrompt {
		t.Fatalf("post-Reset baseline not established: %+v", v)
	}
}

// An unarmed detector must cost an opted-out session nothing: no baseline, no
// witness, no verdict. An unrecognized policy folds here too, so a typo can never
// silently arm a denying gate on live traffic.
func TestCacheBreakDetectorOffIsDisarmedAndUnknownPolicyFoldsToOff(t *testing.T) {
	d := NewCacheBreakDetector(CacheBreakPolicyOff)
	d.Observe(baseTurn())

	mutated := baseTurn()
	mutated.System = "rebuilt"
	v := d.Observe(mutated)
	if v.Broken || v.Witness != "" || v.ObservedTurn != 0 {
		t.Fatalf("disarmed detector produced a verdict: %+v", v)
	}
	if r := d.Report(); r.Events != 0 {
		t.Fatalf("disarmed detector booked events: %+v", r)
	}

	for _, s := range []string{"", "DENY", "loud", "on"} {
		if got := ParseCacheBreakPolicy(s); got != CacheBreakPolicyOff {
			t.Fatalf("ParseCacheBreakPolicy(%q) = %q, want off", s, got)
		}
	}
	if got := ParseCacheBreakPolicy("deny"); got != CacheBreakPolicyDeny {
		t.Fatalf("ParseCacheBreakPolicy(deny) = %q", got)
	}
	if got := ParseCacheBreakPolicy("warn"); got != CacheBreakPolicyWarn {
		t.Fatalf("ParseCacheBreakPolicy(warn) = %q", got)
	}
	if got := NewCacheBreakDetector(CacheBreakPolicy("bogus")).Policy(); got != CacheBreakPolicyOff {
		t.Fatalf("bogus policy armed the detector as %q", got)
	}
}

// The per-turn prefix-hash is component-wise on purpose: a single whole-prefix
// hash proves only that SOMETHING moved, which is the unactionable signal a
// convention already has. It must also be content-free and length-delimited so a
// byte moving across a component boundary cannot forge an unchanged combined
// digest.
func TestCacheBreakDetectorPrefixDigestIsAttributableAndContentFree(t *testing.T) {
	a := DigestTurnPrefix(baseTurn())
	if a != DigestTurnPrefix(baseTurn()) {
		t.Fatal("digest is not deterministic")
	}
	// Pure lowercase hex of fixed width is what makes the retained state
	// structurally incapable of carrying prompt text into a log or journal.
	for _, part := range []string{a.System, a.Tools, a.History, a.Combined} {
		if len(part) != 16 {
			t.Fatalf("digest %q is not the expected 16-hex-char content-free form", part)
		}
		if strings.Trim(part, "0123456789abcdef") != "" {
			t.Fatalf("digest %q is not pure hex; it could carry prompt bytes", part)
		}
	}

	toolsOnly := baseTurn()
	toolsOnly.Tools = "[]"
	b := DigestTurnPrefix(toolsOnly)
	if b.System != a.System || b.History != a.History {
		t.Fatal("a tool-schema change perturbed an unrelated component digest")
	}
	if b.Tools == a.Tools || b.Combined == a.Combined {
		t.Fatal("a tool-schema change did not move its own digest")
	}

	// Same concatenation, different boundary: length delimiting must separate them.
	left := TurnPrefix{System: "ab", Tools: "c"}
	right := TurnPrefix{System: "a", Tools: "bc"}
	if DigestTurnPrefix(left).Combined == DigestTurnPrefix(right).Combined {
		t.Fatal("combined digest is not delimited across component boundaries")
	}
}
