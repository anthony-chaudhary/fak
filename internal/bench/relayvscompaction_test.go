package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRelayVsCompactionBench is the re-runnable witness for issue #1906: it
// produces the comparison report and asserts the acceptance criteria — flat peak
// context for the relay across the duration sweep, plus the cost/accuracy
// advantage over auto-compaction. Regenerate the golden with UPDATE_GOLDEN=1.
func TestRelayVsCompactionBench(t *testing.T) {
	r := BuildRelayVsCompactionReport()

	if r.Schema != "relayvscompaction.v1" {
		t.Fatalf("schema = %q; want relayvscompaction.v1", r.Schema)
	}
	if r.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q; want %q (hermetic model)", r.Provenance.Kind, ProvenanceSimulated)
	}
	if len(r.Sweep) != 7 {
		t.Fatalf("sweep points = %d; want 7 (40,80,100,160,200,300,320)", len(r.Sweep))
	}

	// Done condition (headline): the relay's peak context is FLAT — identical at
	// every swept duration, independent of goal duration (the O(1) invariant).
	if !r.FlatPeakContext {
		t.Fatalf("relay peak context is NOT flat across the sweep: %+v", peaks(r))
	}
	for _, p := range r.Sweep {
		if p.Relay.PeakContext != r.RelayPeakContext {
			t.Fatalf("relay peak drifted at %d turns: %d != %d", p.GoalTurns, p.Relay.PeakContext, r.RelayPeakContext)
		}
	}

	// The relay must peak BELOW the context-rot zone; compaction is pinned at the
	// wall. With the default model that is 60% vs 95% of the window.
	deepest := r.Sweep[len(r.Sweep)-1]
	if deepest.Relay.PeakContextFrac >= deepest.Compaction.PeakContextFrac {
		t.Fatalf("relay peak frac %.2f not below compaction %.2f",
			deepest.Relay.PeakContextFrac, deepest.Compaction.PeakContextFrac)
	}
	if deepest.Compaction.PeakContextFrac < 0.90 {
		t.Errorf("compaction should sit near the wall, got %.2f", deepest.Compaction.PeakContextFrac)
	}

	// Both-lenses acceptance: at EVERY duration the relay is lower-peak, cheaper,
	// busts less cache, and is at least as faithful. Raw cache-hit RATE is reported
	// but NOT an acceptance gate — it flatters compaction's larger prefix; the
	// spine's "compaction busts the cache" claim is the cache-bust COST below.
	for _, p := range r.Sweep {
		if !(p.Relay.PeakContext < p.Compaction.PeakContext) {
			t.Errorf("%d turns: relay peak %d not < compaction %d", p.GoalTurns, p.Relay.PeakContext, p.Compaction.PeakContext)
		}
		if !(p.Relay.BilledTokens < p.Compaction.BilledTokens) {
			t.Errorf("%d turns: relay billed %d not < compaction %d", p.GoalTurns, p.Relay.BilledTokens, p.Compaction.BilledTokens)
		}
		if !(p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens) {
			t.Errorf("%d turns: relay cache-bust %d not < compaction %d", p.GoalTurns, p.Relay.CacheBustTokens, p.Compaction.CacheBustTokens)
		}
		if p.Relay.CacheHitRate <= 0 || p.Compaction.CacheHitRate <= 0 {
			t.Errorf("%d turns: cache-hit rate must be reported for both arms, got relay %.3f / compaction %.3f", p.GoalTurns, p.Relay.CacheHitRate, p.Compaction.CacheHitRate)
		}
		if !(p.Relay.Accuracy >= p.Compaction.Accuracy) {
			t.Errorf("%d turns: relay accuracy %.3f not >= compaction %.3f", p.GoalTurns, p.Relay.Accuracy, p.Compaction.Accuracy)
		}
	}
	if r.Verdict != VerdictRelayWins {
		t.Fatalf("verdict = %q; want %q", r.Verdict, VerdictRelayWins)
	}

	// Accuracy delta: the relay is lossless (fail-closed externalize gate); the
	// deepest-duration compaction has lost load-bearing facts, and that loss GROWS
	// with duration (more compaction events erode more).
	if deepest.Relay.Accuracy != 1.0 {
		t.Errorf("relay accuracy = %.3f; want 1.0 (lossless)", deepest.Relay.Accuracy)
	}
	if deepest.Compaction.Accuracy >= 1.0 {
		t.Errorf("compaction accuracy = %.3f; want < 1.0 (lossy)", deepest.Compaction.Accuracy)
	}
	if !(deepest.Compaction.Accuracy < r.Sweep[0].Compaction.Accuracy) {
		t.Errorf("compaction accuracy should degrade with duration: deepest %.3f not < shortest %.3f",
			deepest.Compaction.Accuracy, r.Sweep[0].Compaction.Accuracy)
	}
	if r.Delta.AccuracyGain <= 0 {
		t.Errorf("accuracy_gain = %.3f; want > 0", r.Delta.AccuracyGain)
	}

	// Net-true hygiene: the report names its invalidating assumptions and its
	// promotion/demotion evidence (the generation frame's closing requirement).
	if len(r.Assumptions) == 0 || r.Promotion == "" || r.DemotionRetirement == "" || r.InvalidatingUnknown == "" {
		t.Errorf("report must name assumptions + promotion + demotion + invalidating unknown")
	}

	// The report marshals to stable JSON (the re-derivable artifact).
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("empty report JSON")
	}

	// Golden: the committed benchmark result artifact (the issue's "checked-in
	// result"). Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "relayvscompaction_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestRelayVsCompactionSavingsGate is the witness for issue #2707: the sweep
// carries an EXPLICIT ≥50% billed-savings gate at each of the operator-requested
// 100/200/300-turn horizons, computed as
// (compaction.BilledTokens - relay.BilledTokens) / compaction.BilledTokens, with a
// per-horizon `savings_meets_50pct_target` verdict that is honest either way — true,
// or a false that names the shortfall. The test does NOT force a pass (the issue
// explicitly permits an honest false-with-shortfall under the analytic model); it
// pins that the gate exists, covers all three horizons, and is internally consistent.
func TestRelayVsCompactionSavingsGate(t *testing.T) {
	r := BuildRelayVsCompactionReport()

	if r.SavingsTarget != RVCSavingsTarget || RVCSavingsTarget != 0.50 {
		t.Fatalf("savings target = %.3f; want %.3f", r.SavingsTarget, 0.50)
	}
	want := RVCSavingsGateHorizons() // {100, 200, 300}
	if len(r.SavingsGates) != len(want) {
		t.Fatalf("savings gates = %d; want %d (100/200/300)", len(r.SavingsGates), len(want))
	}

	// Every requested horizon must be present in the underlying sweep (so the gate
	// is over real arm data, not fabricated).
	inSweep := map[int]RVCPoint{}
	for _, p := range r.Sweep {
		inSweep[p.GoalTurns] = p
	}
	allMeet := true
	for i, g := range r.SavingsGates {
		if g.GoalTurns != want[i] {
			t.Fatalf("gate[%d] horizon = %d; want %d", i, g.GoalTurns, want[i])
		}
		p, ok := inSweep[g.GoalTurns]
		if !ok {
			t.Fatalf("gate horizon %d not present in the sweep", g.GoalTurns)
		}
		if g.CompactionBilledTokens != p.Compaction.BilledTokens || g.RelayBilledTokens != p.Relay.BilledTokens {
			t.Fatalf("gate %d billed tokens desync: gate(c=%d,r=%d) sweep(c=%d,r=%d)",
				g.GoalTurns, g.CompactionBilledTokens, g.RelayBilledTokens, p.Compaction.BilledTokens, p.Relay.BilledTokens)
		}
		// The gate value must equal the issue's exact formula.
		wantFrac := round4(float64(p.Compaction.BilledTokens-p.Relay.BilledTokens) / float64(p.Compaction.BilledTokens))
		if g.BilledSavingsFrac != wantFrac {
			t.Errorf("gate %d savings frac = %.4f; want %.4f", g.GoalTurns, g.BilledSavingsFrac, wantFrac)
		}
		// Verdict must match the computed fraction, and the shortfall must be named
		// exactly when (and only when) the gate is not met — the done condition's
		// "true, or an honest false with the shortfall named".
		if g.SavingsMeets50pctTarget != (g.BilledSavingsFrac >= RVCSavingsTarget) {
			t.Errorf("gate %d verdict %v inconsistent with frac %.4f vs target %.2f",
				g.GoalTurns, g.SavingsMeets50pctTarget, g.BilledSavingsFrac, RVCSavingsTarget)
		}
		if g.SavingsMeets50pctTarget {
			if g.ShortfallFrac != 0 {
				t.Errorf("gate %d met but shortfall = %.4f; want 0", g.GoalTurns, g.ShortfallFrac)
			}
		} else {
			allMeet = false
			if g.ShortfallFrac <= 0 {
				t.Errorf("gate %d NOT met but shortfall = %.4f; want > 0 (name the shortfall)", g.GoalTurns, g.ShortfallFrac)
			}
			if got := round4(RVCSavingsTarget - g.BilledSavingsFrac); g.ShortfallFrac != got {
				t.Errorf("gate %d shortfall = %.4f; want %.4f", g.GoalTurns, g.ShortfallFrac, got)
			}
		}
		t.Logf("horizon %d: relay bills %d vs compaction %d -> %.1f%% savings, meets_50pct=%v (shortfall %.4f)",
			g.GoalTurns, g.RelayBilledTokens, g.CompactionBilledTokens, g.BilledSavingsFrac*100, g.SavingsMeets50pctTarget, g.ShortfallFrac)
	}
	if r.AllHorizonsMeet50pct != allMeet {
		t.Errorf("all_horizons_meet_50pct = %v; want %v (must agree with the per-horizon gates)", r.AllHorizonsMeet50pct, allMeet)
	}
}

// TestRelayVsCompactionFourArmSweep is the witness for issue #2709: the sweep
// widens from two arms (relay, compaction) to FOUR by adding the two additional
// SOTA competitor mechanisms the operator asked to compare against — a
// LangChain-style sliding-window drop (`trim_messages`, the one competitor design
// that ALSO preserves the prompt cache) and Anthropic API context-editing
// (`clear_tool_uses` + memory tool). It pins that all four arms are present at the
// 100/200/300-turn horizons and that the MEASURED numbers reproduce the honest,
// NON-cherry-picked story: the relay is the only arm that is simultaneously
// bounded-peak, cache-preserving, AND lossless — each competitor sacrifices at
// least one axis. No single scalar dominates; the relay's lead over the
// cache-preserving sliding window is specifically FIDELITY.
func TestRelayVsCompactionFourArmSweep(t *testing.T) {
	r := BuildRelayVsCompactionReport()

	// The report advertises all four arms.
	if got := r.Strategies; len(got) != 4 ||
		got[0] != StrategyRelay || got[1] != StrategyCompaction ||
		got[2] != StrategySlidingWindow || got[3] != StrategyContextEditing {
		t.Fatalf("strategies = %v; want [relay compaction sliding_window context_editing]", got)
	}

	// Every requested horizon carries all four arms with the honest relations the
	// model guarantees by construction.
	byDur := map[int]RVCPoint{}
	for _, p := range r.Sweep {
		byDur[p.GoalTurns] = p
	}
	for _, h := range RVCSavingsGateHorizons() { // {100, 200, 300}
		p, ok := byDur[h]
		if !ok {
			t.Fatalf("horizon %d missing from the sweep", h)
		}
		if p.SlidingWindow == nil || p.ContextEditing == nil {
			t.Fatalf("horizon %d missing a new arm: sliding=%v context=%v", h, p.SlidingWindow, p.ContextEditing)
		}
		sw, ce := *p.SlidingWindow, *p.ContextEditing

		// Fidelity: the relay externalizes (lossless); the cache-preserving sliding
		// window drops old turns irrecoverably, so it LOSES load-bearing facts — the
		// honest apples-to-apples separation the issue names. Context-editing recovers
		// cleared content via the memory tool, so it stays lossless like the relay.
		if p.Relay.Accuracy != 1.0 {
			t.Errorf("h%d: relay accuracy = %.3f; want 1.0 (lossless)", h, p.Relay.Accuracy)
		}
		if ce.Accuracy != 1.0 {
			t.Errorf("h%d: context_editing accuracy = %.3f; want 1.0 (recoverable via memory tool)", h, ce.Accuracy)
		}
		if !(sw.Accuracy < p.Relay.Accuracy) {
			t.Errorf("h%d: sliding_window accuracy %.3f not < relay %.3f (a trailing-window drop must lose facts)", h, sw.Accuracy, p.Relay.Accuracy)
		}
		if !(p.Compaction.Accuracy < 1.0) {
			t.Errorf("h%d: compaction accuracy %.3f not < 1.0 (lossy summary)", h, p.Compaction.Accuracy)
		}

		// Peak: the relay, the sliding window, and context-editing all stay off the
		// wall; only compaction is pinned near the ceiling.
		for name, arm := range map[string]RVCArm{"relay": p.Relay, "sliding_window": sw, "context_editing": ce} {
			if !(arm.PeakContext < p.Compaction.PeakContext) {
				t.Errorf("h%d: %s peak %d not < compaction %d (must stay off the wall)", h, name, arm.PeakContext, p.Compaction.PeakContext)
			}
		}

		// Cache: the sliding window is a PURE drop (cache-preserving — zero forced
		// re-writes), while context-editing accepts a cache invalidation on every
		// clear, so it busts strictly more cache than the relay's O(1) baton reset.
		if sw.CacheBustTokens != 0 {
			t.Errorf("h%d: sliding_window cache-bust = %d; want 0 (a drop preserves the cache)", h, sw.CacheBustTokens)
		}
		if !(ce.CacheBustTokens > p.Relay.CacheBustTokens) {
			t.Errorf("h%d: context_editing cache-bust %d not > relay %d (a clear invalidates the cache)", h, ce.CacheBustTokens, p.Relay.CacheBustTokens)
		}
	}

	// The headline the scorecard row cites: at the deepest horizon the relay is the
	// ONLY arm that is lossless (accuracy 1.0) AND off the wall, while the
	// cache-preserving competitor has collapsed on fidelity.
	deep := byDur[300]
	if !(deep.Relay.Accuracy == 1.0 && deep.SlidingWindow.Accuracy < 0.5 && deep.Relay.PeakContext < deep.Compaction.PeakContext) {
		t.Errorf("300-turn headline broke: relay acc %.3f / sliding acc %.3f / relay peak %d vs compaction %d",
			deep.Relay.Accuracy, deep.SlidingWindow.Accuracy, deep.Relay.PeakContext, deep.Compaction.PeakContext)
	}
}

// TestRelayVsCompactionFlatInvariant isolates the O(1) claim: doubling the goal
// duration leaves the relay's peak context UNCHANGED while compaction's total
// work (rotations, billed tokens) scales up. This is the property that makes a
// week-long relay peak no higher than an hour-long one.
func TestRelayVsCompactionFlatInvariant(t *testing.T) {
	m := DefaultRVCModel()
	short := simulateRelay(m, 50)
	long := simulateRelay(m, 400)
	if short.PeakContext != long.PeakContext {
		t.Fatalf("relay peak not invariant to duration: 50-turn %d != 400-turn %d", short.PeakContext, long.PeakContext)
	}
	if !(long.Rotations > short.Rotations) {
		t.Fatalf("longer goal should rotate more legs: %d not > %d", long.Rotations, short.Rotations)
	}
	// Compaction, by contrast, also caps peak (at the wall) but does strictly more
	// summarization work as the goal grows.
	cShort := simulateCompaction(m, 50)
	cLong := simulateCompaction(m, 400)
	if !(cLong.Rotations > cShort.Rotations) {
		t.Fatalf("longer goal should compact more: %d not > %d", cLong.Rotations, cShort.Rotations)
	}
	if !(cLong.PeakContext > long.PeakContext) {
		t.Fatalf("compaction peak %d should exceed relay peak %d", cLong.PeakContext, long.PeakContext)
	}
}

// TestRelayVsCompactionCustomModelSeam checks the observed-run seam: a live leg
// record can feed different constants/durations and still fold into the same
// report shape (the promotion path).
func TestRelayVsCompactionCustomModelSeam(t *testing.T) {
	m := DefaultRVCModel()
	m.WindowCeiling = 100_000
	m.GrowthPerTurn = 4_000
	r := BuildRelayVsCompactionReportFor(m, []int{30, 60})
	if len(r.Sweep) != 2 {
		t.Fatalf("custom sweep points = %d; want 2", len(r.Sweep))
	}
	if r.Model.WindowCeiling != 100_000 {
		t.Fatalf("custom model not threaded through: window = %d", r.Model.WindowCeiling)
	}
	if !r.FlatPeakContext {
		t.Errorf("relay peak should still be flat under the custom model")
	}
}

// TestRelayVsCompactionObservedLegLedger is the promotion witness for issue
// #2659: OBSERVED per-leg records (a replayed capture, provenance != SIMULATED)
// fed through the loader reproduce the SIGN — the relay is lower-peak, cheaper,
// busts less cache, and is at least as faithful at every swept duration — and the
// relay's peak context stays FLAT across the sweep. The checked-in report is the
// issue's "checked-in OBSERVED result artifact". Regenerate with UPDATE_GOLDEN=1.
func TestRelayVsCompactionObservedLegLedger(t *testing.T) {
	ledger := filepath.Join("testdata", "relayvscompaction_observed_legs.json")
	r, err := BuildRelayVsCompactionReportFromLegLedger(DefaultRVCModel(), ledger)
	if err != nil {
		t.Fatalf("BuildRelayVsCompactionReportFromLegLedger: %v", err)
	}

	// Provenance is OBSERVED (record-derived), not the analytic model — the whole
	// point of the promotion seam.
	if r.Provenance.Kind != ProvenanceObserved {
		t.Fatalf("provenance = %q; want %q (record-derived)", r.Provenance.Kind, ProvenanceObserved)
	}
	if r.Schema != "relayvscompaction.v1" {
		t.Fatalf("schema = %q; want relayvscompaction.v1", r.Schema)
	}
	if len(r.Sweep) != 2 {
		t.Fatalf("sweep points = %d; want 2 (durations 6, 12)", len(r.Sweep))
	}

	// The sign the analytic model asserts must survive on OBSERVED records.
	for _, p := range r.Sweep {
		if !(p.Relay.PeakContext < p.Compaction.PeakContext) {
			t.Errorf("%d turns: relay peak %d not < compaction %d", p.GoalTurns, p.Relay.PeakContext, p.Compaction.PeakContext)
		}
		if !(p.Relay.BilledTokens < p.Compaction.BilledTokens) {
			t.Errorf("%d turns: relay billed %d not < compaction %d", p.GoalTurns, p.Relay.BilledTokens, p.Compaction.BilledTokens)
		}
		if !(p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens) {
			t.Errorf("%d turns: relay cache-bust %d not < compaction %d", p.GoalTurns, p.Relay.CacheBustTokens, p.Compaction.CacheBustTokens)
		}
		if !(p.Relay.Accuracy >= p.Compaction.Accuracy) {
			t.Errorf("%d turns: relay accuracy %.3f not >= compaction %.3f", p.GoalTurns, p.Relay.Accuracy, p.Compaction.Accuracy)
		}
	}
	if r.Verdict != VerdictRelayWins {
		t.Fatalf("observed verdict = %q; want %q (sign reproduced)", r.Verdict, VerdictRelayWins)
	}
	// The O(1) invariant holds in the OBSERVED capture too: the relay's peak is
	// identical across the sweep even though the 12-turn run does strictly more
	// rotations than the 6-turn run.
	if !r.FlatPeakContext {
		t.Fatalf("observed relay peak is NOT flat across the sweep: %v", peaks(r))
	}
	if !(r.Sweep[1].Relay.Rotations > r.Sweep[0].Relay.Rotations) {
		t.Fatalf("longer observed goal should rotate more legs: %d not > %d",
			r.Sweep[1].Relay.Rotations, r.Sweep[0].Relay.Rotations)
	}

	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	golden := filepath.Join("testdata", "relayvscompaction_observed_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("observed report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestRelayVsCompactionObservedRefusals pins the fail-closed contract: an OBSERVED
// report is only emitted when the records actually back it. Empty input, a run
// with no turns, an unknown strategy label, a duplicate arm, and a duration
// missing one of the two arms each refuse rather than fabricate an OBSERVED sign.
func TestRelayVsCompactionObservedRefusals(t *testing.T) {
	m := DefaultRVCModel()
	relay6 := RVCRunRecord{Strategy: StrategyRelay, GoalTurns: 6, Turns: []RVCTurnRecord{
		{CachedInputTokens: 6000, UncachedInputTokens: 6000, ResidentTokens: 12000, FactsProduced: 3, FactsRetained: 3},
	}}
	comp6 := RVCRunRecord{Strategy: StrategyCompaction, GoalTurns: 6, Turns: []RVCTurnRecord{
		{CachedInputTokens: 40000, UncachedInputTokens: 20000, ResidentTokens: 60000, FactsProduced: 3, FactsRetained: 3},
	}}

	cases := []struct {
		name    string
		runs    []RVCRunRecord
		wantSub string
	}{
		{"empty", nil, "no observed leg records"},
		{"no-turns", []RVCRunRecord{{Strategy: StrategyRelay, GoalTurns: 6}, comp6}, "has no turn records"},
		{"unknown-strategy", []RVCRunRecord{{Strategy: "guess", GoalTurns: 6, Turns: relay6.Turns}}, "unknown strategy"},
		{"duplicate-arm", []RVCRunRecord{relay6, relay6, comp6}, "duplicate relay run"},
		{"missing-compaction", []RVCRunRecord{relay6}, "missing the compaction arm"},
		{"missing-relay", []RVCRunRecord{comp6}, "missing the relay arm"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildRelayVsCompactionReportFromRecords(m, tc.runs, "fixture")
			if err == nil {
				t.Fatalf("want refusal containing %q, got nil error", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error = %q; want substring %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// TestRelayVsCompactionEmptySweepNoFalseWin pins the fail-closed contract on the
// ANALYTIC seam. BuildRelayVsCompactionReportFor is the exported seam a live
// leg-record run feeds constants into; over an empty duration sweep it has ZERO
// comparisons, so it must not claim the relay wins or that its peak context is flat
// — the same honesty BuildRelayVsCompactionReportFromRecords already enforces on
// the OBSERVED path (TestRelayVsCompactionObservedRefusals). Before the fix the
// analytic seam defaulted the verdict to VerdictRelayWins and flat_peak_context to
// true, fabricating a "relay wins + flat peak" claim over no data points.
func TestRelayVsCompactionEmptySweepNoFalseWin(t *testing.T) {
	for _, durations := range [][]int{nil, {}} {
		r := BuildRelayVsCompactionReportFor(DefaultRVCModel(), durations)
		if len(r.Sweep) != 0 {
			t.Fatalf("empty durations %v should yield an empty sweep, got %d points", durations, len(r.Sweep))
		}
		if r.Verdict != VerdictNoAdvantage {
			t.Errorf("empty sweep verdict = %q; want %q (zero comparisons cannot witness a win)", r.Verdict, VerdictNoAdvantage)
		}
		if r.FlatPeakContext {
			t.Errorf("empty sweep claims flat_peak_context=true with no data points to witness the invariant")
		}
	}
}

func peaks(r RelayVsCompactionReport) []int {
	out := make([]int, 0, len(r.Sweep))
	for _, p := range r.Sweep {
		out = append(out, p.Relay.PeakContext)
	}
	return out
}
