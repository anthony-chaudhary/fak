package ctxplan

import (
	"strings"
	"testing"
)

// turntax_test.go — the witness for issue #1538: pure fak makes a per-turn cache decision by
// default and records WHY. The headline test drives three representative turns (warm prefix,
// cold-but-queryable, no match) through PlanTurnTax and asserts both halves of the target:
// the chosen strategy AND a non-empty reason that names the numbers behind it.

// TestPlanTurnTaxPicksPerTurnStrategy is the three-way witness: the same planner, given
// three representative turns' signals, picks reuse, query, and cold prefill respectively —
// each with a reason. Before this planner nothing chose among the three mechanisms per turn.
func TestPlanTurnTaxPicksPerTurnStrategy(t *testing.T) {
	cases := []struct {
		name        string
		sig         TurnTaxSignals
		want        TurnTaxStrategy
		wantTax     int
		reasonHas   []string
		wantSaved   int
		wantReuse   int
		wantMatched int
	}{
		{
			// A warm turn: the radix index holds 900 of the 1000 prompt tokens and the
			// materialization verdict trusts them, so only the 100-token suffix is computed.
			name: "warm prefix reuses the cached KV",
			sig: TurnTaxSignals{
				PromptTokens:   1000,
				MatchedPrefix:  900,
				PrefixTrusted:  true,
				QueryTokens:    400,
				ResidentBudget: 512,
			},
			want:        TurnTaxReuse,
			wantTax:     100,
			reasonHas:   []string{"trusted prefix covers 900 of 1000", "100", "1000 cold"},
			wantSaved:   900,
			wantReuse:   900,
			wantMatched: 900,
		},
		{
			// A cold-but-queryable turn: no cached prefix at all, the 20k-token history
			// blows past the 512-token resident budget, and the session index can answer it
			// with 300 retrieved tokens. Query is both available and by far the cheapest.
			name: "cold but queryable turn asks the session index",
			sig: TurnTaxSignals{
				PromptTokens:   20000,
				MatchedPrefix:  0,
				PrefixTrusted:  false,
				QueryTokens:    300,
				ResidentBudget: 512,
			},
			want:      TurnTaxQuery,
			wantTax:   300,
			reasonHas: []string{"20000", "512-token resident budget", "300", "20000 to prefill cold"},
			wantSaved: 19700,
		},
		{
			// A first turn: nothing cached, no index to ask. Cold prefill is the fail-open
			// floor and the decision still records why it was taken.
			name: "no match and no index prefills cold",
			sig: TurnTaxSignals{
				PromptTokens:   1000,
				MatchedPrefix:  0,
				QueryTokens:    0,
				ResidentBudget: 512,
			},
			want:      TurnTaxColdPrefill,
			wantTax:   1000,
			reasonHas: []string{"no cached prefix and no queryable session index", "1000"},
			wantSaved: 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := PlanTurnTax(tc.sig)
			if got.Strategy != tc.want {
				t.Errorf("strategy = %s, want %s (reason: %s)", got.Strategy, tc.want, got.Reason)
			}
			if !ValidTurnTaxStrategy(got.Strategy) {
				t.Errorf("strategy %q is outside the closed vocabulary", got.Strategy)
			}
			if got.Reason == "" {
				t.Fatal("decision recorded no reason: 'records why' is the load-bearing half of the target")
			}
			for _, want := range tc.reasonHas {
				if !strings.Contains(got.Reason, want) {
					t.Errorf("reason %q does not name %q", got.Reason, want)
				}
			}
			if got.Tax() != tc.wantTax {
				t.Errorf("tax = %d, want %d", got.Tax(), tc.wantTax)
			}
			if got.Saved() != tc.wantSaved {
				t.Errorf("saved = %d, want %d", got.Saved(), tc.wantSaved)
			}
			if got.ReusableTokens != tc.wantReuse {
				t.Errorf("reusable = %d, want %d", got.ReusableTokens, tc.wantReuse)
			}
			if got.MatchedPrefix != tc.wantMatched {
				t.Errorf("cached prefix = %d, want %d", got.MatchedPrefix, tc.wantMatched)
			}
		})
	}
}

// TestPlanTurnTaxRefusesUntrustedPrefix is the cold-path-correctness witness: a prefix that
// MATCHED the index but whose materialization verdict refused it contributes zero reusable
// tokens, so the turn falls to the cold path — and the refusal is named in the reason rather
// than folded into a plain miss. The planner never trades correctness for tax.
func TestPlanTurnTaxRefusesUntrustedPrefix(t *testing.T) {
	sig := TurnTaxSignals{PromptTokens: 1000, MatchedPrefix: 900, PrefixTrusted: false, ResidentBudget: 4096}
	got := PlanTurnTax(sig)
	if got.Strategy != TurnTaxColdPrefill {
		t.Fatalf("strategy = %s, want %s: an untrusted prefix must never be reused", got.Strategy, TurnTaxColdPrefill)
	}
	if got.ReuseAvailable || got.ReusableTokens != 0 {
		t.Errorf("reuse available=%v reusable=%d, want unavailable/0", got.ReuseAvailable, got.ReusableTokens)
	}
	if got.MatchedPrefix != 900 {
		t.Errorf("cached prefix = %d, want the pre-gate 900 preserved so the refused match stays visible", got.MatchedPrefix)
	}
	if got.Tax() != 1000 {
		t.Errorf("tax = %d, want the full 1000-token cold prefill", got.Tax())
	}
	for _, want := range []string{"900", "verdict refused", "cold"} {
		if !strings.Contains(got.Reason, want) {
			t.Errorf("reason %q does not name %q", got.Reason, want)
		}
	}
}

// TestPlanTurnTaxAlwaysDecides is the "by default" witness: there is no signal state — not
// the zero value, not a garbage over-report, not a negative count — in which the planner
// returns no strategy or no reason. Cold prefill is the always-available floor.
func TestPlanTurnTaxAlwaysDecides(t *testing.T) {
	cases := map[string]TurnTaxSignals{
		"zero value":            {},
		"negative prompt":       {PromptTokens: -50, MatchedPrefix: -10, QueryTokens: -1, ResidentBudget: -1},
		"prefix over-reported":  {PromptTokens: 100, MatchedPrefix: 5000, PrefixTrusted: true},
		"query over-reported":   {PromptTokens: 100, QueryTokens: 5000},
		"budget without prompt": {ResidentBudget: 8000},
	}
	for name, sig := range cases {
		t.Run(name, func(t *testing.T) {
			got := PlanTurnTax(sig)
			if !ValidTurnTaxStrategy(got.Strategy) {
				t.Errorf("strategy %q is outside the closed vocabulary", got.Strategy)
			}
			if got.Reason == "" {
				t.Error("decision recorded no reason")
			}
			if got.Tax() < 0 {
				t.Errorf("tax = %d: a normalized signal set can never mint a negative tax", got.Tax())
			}
			if got.Tax() > got.ColdTax {
				t.Errorf("tax = %d exceeds the %d-token cold floor: the planner chose a strategy worse than cold prefill",
					got.Tax(), got.ColdTax)
			}
			if got.Saved() < 0 {
				t.Errorf("saved = %d, want never negative", got.Saved())
			}
		})
	}
}

// TestPlanTurnTaxQueryNeedsAnOverBudgetPrompt pins the query gate: a query substitutes for
// materializing an OVER-BUDGET context, so a prompt that already fits the sized resident
// budget never routes to the index — even when the query would be nominally cheaper. An
// UNKNOWN (non-positive) budget is not evidence the prompt fits and does not veto the query.
func TestPlanTurnTaxQueryNeedsAnOverBudgetPrompt(t *testing.T) {
	fits := PlanTurnTax(TurnTaxSignals{PromptTokens: 400, QueryTokens: 100, ResidentBudget: 512})
	if fits.Strategy != TurnTaxColdPrefill {
		t.Errorf("strategy = %s, want %s: a prompt inside the resident budget has nothing for a query to displace",
			fits.Strategy, TurnTaxColdPrefill)
	}
	if fits.QueryAvailable {
		t.Error("query reported available for a prompt that fits the resident budget")
	}
	if !strings.Contains(fits.Reason, "already fits") {
		t.Errorf("reason %q does not explain the vetoed query", fits.Reason)
	}

	unknown := PlanTurnTax(TurnTaxSignals{PromptTokens: 400, QueryTokens: 100})
	if unknown.Strategy != TurnTaxQuery {
		t.Errorf("strategy = %s, want %s: an unknown budget must not veto a cheaper query", unknown.Strategy, TurnTaxQuery)
	}
	if !strings.Contains(unknown.Reason, "budget unknown") {
		t.Errorf("reason %q does not record that the budget was unknown", unknown.Reason)
	}
}

// TestPlanTurnTaxTiesStayWarm pins the tie-break: when reuse and query cost the same the
// turn stays on the warm path (reuse), because reuse also leaves the session's KV hot for the
// next turn — a tie-break the cheapest-tax rule alone cannot express.
func TestPlanTurnTaxTiesStayWarm(t *testing.T) {
	// prompt 1000, trusted prefix 700 => reuse tax 300; query tax 300 too.
	got := PlanTurnTax(TurnTaxSignals{
		PromptTokens: 1000, MatchedPrefix: 700, PrefixTrusted: true, QueryTokens: 300, ResidentBudget: 512,
	})
	if got.Strategy != TurnTaxReuse {
		t.Fatalf("strategy = %s, want %s on an equal-tax tie", got.Strategy, TurnTaxReuse)
	}
	if got.ReuseTax != got.QueryTax {
		t.Fatalf("test no longer exercises a tie: reuse=%d query=%d", got.ReuseTax, got.QueryTax)
	}
	// A strictly cheaper query DOES win — the tie-break is a tie-break, not a reuse bias.
	cheaper := PlanTurnTax(TurnTaxSignals{
		PromptTokens: 1000, MatchedPrefix: 700, PrefixTrusted: true, QueryTokens: 299, ResidentBudget: 512,
	})
	if cheaper.Strategy != TurnTaxQuery {
		t.Errorf("strategy = %s, want %s when the query is strictly cheaper (reason: %s)",
			cheaper.Strategy, TurnTaxQuery, cheaper.Reason)
	}
}

// TestTurnTaxLogRecordsAndReplays is the RECORD witness: Append is the only decide-and-record
// call, the log keeps one entry per turn in order, and replaying every entry from its own
// stored signals reproduces the identical strategy — the determinism a recorded decision needs
// to be auditable rather than merely asserted.
func TestTurnTaxLogRecordsAndReplays(t *testing.T) {
	var log TurnTaxLog
	turns := []TurnTaxSignals{
		{PromptTokens: 1000, MatchedPrefix: 900, PrefixTrusted: true, ResidentBudget: 512},
		{PromptTokens: 20000, QueryTokens: 300, ResidentBudget: 512},
		{PromptTokens: 1000, ResidentBudget: 512},
	}
	want := []TurnTaxStrategy{TurnTaxReuse, TurnTaxQuery, TurnTaxColdPrefill}
	for i, sig := range turns {
		if got := log.Append(sig); got.Strategy != want[i] {
			t.Errorf("turn %d strategy = %s, want %s", i, got.Strategy, want[i])
		}
	}
	entries := log.Entries()
	if len(entries) != len(turns) {
		t.Fatalf("log holds %d entries, want one per turn (%d)", len(entries), len(turns))
	}
	for i, e := range entries {
		if e.Decision.Reason == "" {
			t.Errorf("turn %d recorded no reason", i)
		}
	}
	if diverged, allMatch := log.Replay(); !allMatch {
		t.Errorf("replay diverged at %v: the planner is not a pure function of its signals", diverged)
	}

	s := log.Summary()
	if s.Reuse != 1 || s.Query != 1 || s.ColdPrefill != 1 {
		t.Errorf("summary = reuse:%d query:%d cold:%d, want 1/1/1", s.Reuse, s.Query, s.ColdPrefill)
	}
	if wantPrompt := 1000 + 20000 + 1000; s.PromptTokens != wantPrompt {
		t.Errorf("summary prompt tokens = %d, want %d", s.PromptTokens, wantPrompt)
	}
	if wantTax := 100 + 300 + 1000; s.Tax != wantTax {
		t.Errorf("summary tax = %d, want %d", s.Tax, wantTax)
	}
	if s.Saved != s.PromptTokens-s.Tax {
		t.Errorf("summary saved = %d, want prompt-tax = %d", s.Saved, s.PromptTokens-s.Tax)
	}

	report := log.Explain()
	for _, want := range []string{"3 decision(s)", "reuse", "query", "cold_prefill", "reuse=1", "query=1", "cold_prefill=1"} {
		if !strings.Contains(report, want) {
			t.Errorf("Explain() does not name %q:\n%s", want, report)
		}
	}
}

// TestTurnTaxLogIsBounded pins the retained window: like PageFaultLog and ObjectiveLog, a
// long-lived session cannot grow the ledger without limit.
func TestTurnTaxLogIsBounded(t *testing.T) {
	var log TurnTaxLog
	for i := 0; i < defaultMaxLedgerEntries+50; i++ {
		log.Append(TurnTaxSignals{PromptTokens: 10 + i})
	}
	if got := len(log.Entries()); got != defaultMaxLedgerEntries {
		t.Fatalf("log retained %d entries, want the %d-entry window", got, defaultMaxLedgerEntries)
	}
	// The window keeps the MOST RECENT turns, not the oldest.
	if first := log.Entries()[0].Signals.PromptTokens; first != 10+50 {
		t.Errorf("oldest retained turn has prompt %d, want %d (the window trims from the front)", first, 10+50)
	}
}

// TestTurnTaxSignalsForBudgetReadsTheAdaptiveModel pins the tie to adaptive.go: the resident
// budget the query gate uses is the SAME W that RecommendBudget sizes from task difficulty, so
// there is no second budget model to drift. This only READS RecommendBudget — its sizing
// behavior is unchanged.
func TestTurnTaxSignalsForBudgetReadsTheAdaptiveModel(t *testing.T) {
	bounds := BudgetBounds{Floor: 512, Ceil: 8192}
	easy := Difficulty{Intents: 0, Pins: 0, Horizon: 1}
	hard := Difficulty{Intents: 8, Pins: 6, Horizon: 4, FaultRate: 1}

	easySig := TurnTaxSignalsForBudget(4000, 0, false, 300, easy, bounds)
	hardSig := TurnTaxSignalsForBudget(4000, 0, false, 300, hard, bounds)
	if easySig.ResidentBudget != RecommendBudget(easy, bounds).Tokens {
		t.Errorf("easy budget = %d, want RecommendBudget's %d", easySig.ResidentBudget, RecommendBudget(easy, bounds).Tokens)
	}
	if hardSig.ResidentBudget != RecommendBudget(hard, bounds).Tokens {
		t.Errorf("hard budget = %d, want RecommendBudget's %d", hardSig.ResidentBudget, RecommendBudget(hard, bounds).Tokens)
	}
	// The easy turn's tight W leaves the 4000-token prompt over budget, so the index is worth
	// asking; the hard turn's W (the ceiling) holds the whole prompt, so it is not.
	if got := PlanTurnTax(easySig); got.Strategy != TurnTaxQuery {
		t.Errorf("easy turn strategy = %s, want %s (budget %d, reason: %s)",
			got.Strategy, TurnTaxQuery, easySig.ResidentBudget, got.Reason)
	}
	if got := PlanTurnTax(hardSig); got.Strategy != TurnTaxColdPrefill {
		t.Errorf("hard turn strategy = %s, want %s (budget %d, reason: %s)",
			got.Strategy, TurnTaxColdPrefill, hardSig.ResidentBudget, got.Reason)
	}
}

// TestTurnTaxStrategyStringFailsClosed pins the closed vocabulary at the (de)serialization
// boundary: an unset or foreign value renders as a visible tell, never as a plausible strategy.
func TestTurnTaxStrategyStringFailsClosed(t *testing.T) {
	if got := TurnTaxStrategy("").String(); got != "(unset)" {
		t.Errorf("empty strategy renders %q, want %q", got, "(unset)")
	}
	if got := TurnTaxStrategy("warm_start").String(); got != "unknown(warm_start)" {
		t.Errorf("foreign strategy renders %q, want %q", got, "unknown(warm_start)")
	}
	if ValidTurnTaxStrategy("warm_start") {
		t.Error("a foreign strategy passed the membership check")
	}
}
