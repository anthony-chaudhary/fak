package scoreboard

import (
	"math"
	"strings"
	"testing"
)

// servingSLO is a well-formed PR-tier serving SLO. Tests mutate a copy to plant
// exactly one defect, mirroring goodCase in ladder_test.go.
func servingSLO() SLO {
	return SLO{
		Name:          "serving-decode-parity",
		Owner:         "quality@fak",
		Objective:     0.9, // error budget: 10% of the window's cases
		Population:    SLOPopulation{Tier: TierPR},
		WindowSeconds: 3600,
	}
}

// failCase builds a provenance-complete, localized, replayable failing case —
// the only kind of failure the honesty gate lets consume budget as a FAIL.
func failCase(id string) LadderCase {
	c := goodCase(id)
	c.Status = StatusFail
	c.FirstDivergence = &LadderDivergence{Index: 7, Reference: "increased", Engine: "decreased"}
	c.Replay = ".dispatch-runs/quality/" + id + ".bundle.json"
	return c
}

// TestSLOWitnessPlantedDefect is the issue #4582 witness: the same SLO
// evaluation FAILS against a planted serving regression (budget breached, first
// divergence localized, replay artifact attached) and PASSES after the fix, as
// a pure fold that replays identically in this clean test process.
func TestSLOWitnessPlantedDefect(t *testing.T) {
	// --- planted representative defect: 2 of 10 serving cases regressed —
	// a 20% error rate against a 10% budget, so the budget is blown. ---
	cases := []LadderCase{
		goodCase("c1"), goodCase("c2"), goodCase("c3"), goodCase("c4"),
		goodCase("c5"), goodCase("c6"), goodCase("c7"), goodCase("c8"),
		failCase("exec-summary-grounding"), failCase("greedy-decode-drift"),
	}
	before := servingSLO().Evaluate(cases)

	if before.Status != SLOBreached {
		t.Fatalf("planted defect must breach the budget, got %q\n%s", before.Status, before.Summary())
	}
	if before.Green() {
		t.Fatal("a breached SLO must never be green")
	}
	// Burn math witnessed: 2 bad / 10 eligible = 0.2 bad fraction; budget 0.1;
	// burn rate 2.0x.
	if before.Eligible != 10 || before.Bad != 2 {
		t.Fatalf("expected 2 bad / 10 eligible, got %d/%d", before.Bad, before.Eligible)
	}
	if math.Abs(before.BurnRate-2.0) > 1e-9 {
		t.Fatalf("burn rate = %v, want 2.0", before.BurnRate)
	}
	// Failure identifies the first actionable divergence and its replay artifact.
	fa := before.FirstActionable
	if fa == nil || fa.ID != "exec-summary-grounding" {
		t.Fatalf("first actionable must localize the first failing case, got %+v", fa)
	}
	if fa.FirstDivergence == nil || fa.FirstDivergence.Index != 7 || strings.TrimSpace(fa.Replay) == "" {
		t.Fatalf("first actionable must carry the divergence and replay artifact, got %+v", fa)
	}
	summary := before.Summary()
	for _, want := range []string{"burn rate 2.00x", "2 bad / 10 eligible", "replay:", "increased"} {
		if !strings.Contains(summary, want) {
			t.Fatalf("witness summary must contain %q:\n%s", want, summary)
		}
	}

	// --- after the fix: both regressed cases pass again; the SLO is met. ---
	fixed := append([]LadderCase{}, cases[:8]...)
	fixed = append(fixed, goodCase("exec-summary-grounding"), goodCase("greedy-decode-drift"))
	after := servingSLO().Evaluate(fixed)

	if after.Status != SLOMet || !after.Green() {
		t.Fatalf("after the fix the SLO must be met, got %q\n%s", after.Status, after.Summary())
	}
	if after.FirstActionable != nil {
		t.Fatalf("a met SLO must have no actionable divergence, got %+v", after.FirstActionable)
	}

	t.Logf("WITNESS before-fix report:\n%s", before.Summary())
	t.Logf("WITNESS after-fix report:\n%s", after.Summary())
}

// TestSLONoDataNeverGreen proves the first acceptance line of #4582 at every
// no-data shape: an empty case set, a population that matches nothing, evidence
// entirely out of window, everything skipped, and everything excluded must all
// land no-data — never met.
func TestSLONoDataNeverGreen(t *testing.T) {
	aged := goodCase("aged")
	aged.AgeSeconds = 7200 // outside the 3600s window

	skipped := goodCase("skip")
	skipped.Status = StatusSkipped

	nightlySLO := servingSLO()
	nightlySLO.Population = SLOPopulation{Tier: TierNightly}

	excluding := servingSLO()
	excluding.Exclusions = []string{"only"}

	cases := []struct {
		name  string
		slo   SLO
		cases []LadderCase
	}{
		{"no cases at all", servingSLO(), nil},
		{"population matches nothing", nightlySLO, []LadderCase{goodCase("pr-only")}},
		{"all evidence out of window", servingSLO(), []LadderCase{aged}},
		{"all cases skipped", servingSLO(), []LadderCase{skipped}},
		{"all cases excluded", excluding, []LadderCase{goodCase("only")}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.slo.Evaluate(tc.cases)
			if r.Status != SLONoData {
				t.Fatalf("status = %q, want %q\n%s", r.Status, SLONoData, r.Summary())
			}
			if r.Green() {
				t.Fatal("no-data must never be green")
			}
			if !strings.Contains(r.Summary(), "never green") {
				t.Fatalf("no-data summary must explain itself:\n%s", r.Summary())
			}
		})
	}
}

// TestSLOBurnMathWitnessed pins the burn math to hand-computed numbers and
// checks each threshold band: met under the alert, burn-warning at the alert,
// breached at 1.0x. The report must carry every operand, not just the verdict.
func TestSLOBurnMathWitnessed(t *testing.T) {
	build := func(total, bad int) []LadderCase {
		out := make([]LadderCase, 0, total)
		for i := 0; i < total-bad; i++ {
			out = append(out, goodCase("g"+string(rune('a'+i))))
		}
		for i := 0; i < bad; i++ {
			out = append(out, failCase("b"+string(rune('a'+i))))
		}
		return out
	}
	cases := []struct {
		name     string
		bad      int
		wantRate float64
		want     SLOStatus
	}{
		// objective 0.9 => budget 0.1; 20 eligible.
		{"0 bad of 20: met", 0, 0.0, SLOMet},
		{"1 bad of 20 = 0.5x burn: warning at default alert", 1, 0.5, SLOBurnWarning},
		{"2 bad of 20 = 1.0x burn: breached", 2, 1.0, SLOBreached},
		{"4 bad of 20 = 2.0x burn: breached", 4, 2.0, SLOBreached},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := servingSLO().Evaluate(build(20, tc.bad))
			if r.Status != tc.want {
				t.Fatalf("status = %q, want %q\n%s", r.Status, tc.want, r.Summary())
			}
			if math.Abs(r.BurnRate-tc.wantRate) > 1e-9 {
				t.Fatalf("burn rate = %v, want %v", r.BurnRate, tc.wantRate)
			}
			if r.Eligible != 20 || r.Bad != tc.bad {
				t.Fatalf("accounting = %d bad / %d eligible, want %d/20", r.Bad, r.Eligible, tc.bad)
			}
			if math.Abs(r.Budget-0.1) > 1e-9 {
				t.Fatalf("budget = %v, want 0.1", r.Budget)
			}
		})
	}
}

// TestSLOUnbackedEvidenceConsumesBudget re-uses the ladder honesty gate at the
// SLO layer: a declared pass with missing provenance, a stale pass, and a bare
// no-status case all consume budget — missing or inconclusive evidence is never
// good, so it can never prop an objective up.
func TestSLOUnbackedEvidenceConsumesBudget(t *testing.T) {
	unbacked := goodCase("unbacked")
	unbacked.Model = "" // declared pass, no provenance -> inconclusive

	noStatus := goodCase("silent")
	noStatus.Status = "" // never ran -> no-data

	staleCase := goodCase("stale")
	staleCase.AgeSeconds = 500 // within the window but past freshness

	s := servingSLO()
	s.FreshnessSeconds = 100

	r := s.Evaluate([]LadderCase{goodCase("ok"), unbacked, noStatus, staleCase})
	if r.Good != 1 || r.Bad != 3 {
		t.Fatalf("accounting = %d good / %d bad, want 1/3\n%s", r.Good, r.Bad, r.Summary())
	}
	if r.Status != SLOBreached { // 3/4 = 75% error rate vs 10% budget
		t.Fatalf("unbacked evidence must burn the budget, got %q", r.Status)
	}
	if r.Counts[StatusInconclusive] != 1 || r.Counts[StatusNoData] != 1 || r.Counts[StatusStale] != 1 {
		t.Fatalf("counts must witness each demotion: %+v", r.Counts)
	}
}

// TestSLOValidateRefusesMalformedDefinitions: an SLO with no owner, an
// out-of-range objective, or no window is invalid — and an invalid SLO never
// evaluates green.
func TestSLOValidateRefusesMalformedDefinitions(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*SLO)
	}{
		{"no name", func(s *SLO) { s.Name = "" }},
		{"no owner", func(s *SLO) { s.Owner = "" }},
		{"objective 0", func(s *SLO) { s.Objective = 0 }},
		{"objective 1.0 has zero budget", func(s *SLO) { s.Objective = 1.0 }},
		{"objective above 1", func(s *SLO) { s.Objective = 1.5 }},
		{"no window", func(s *SLO) { s.WindowSeconds = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := servingSLO()
			tc.mutate(&s)
			if err := s.Validate(); err == nil {
				t.Fatal("Validate must refuse the malformed SLO")
			}
			r := s.Evaluate([]LadderCase{goodCase("ok")})
			if r.Status != SLOInvalid || r.Green() {
				t.Fatalf("invalid SLO must evaluate invalid and never green, got %q", r.Status)
			}
			if strings.TrimSpace(r.Reason) == "" {
				t.Fatal("invalid report must carry the reason")
			}
		})
	}
}

// TestSLOExclusionsAreWitnessed: a contract exclusion that actually removes a
// case is listed in the report (and in the summary), and an exclusion that
// matches nothing does not appear — an exclusion can never silently widen.
func TestSLOExclusionsAreWitnessed(t *testing.T) {
	s := servingSLO()
	s.Exclusions = []string{"quarantined", "never-existed"}

	r := s.Evaluate([]LadderCase{goodCase("ok"), failCase("quarantined")})
	if r.Eligible != 1 || r.Bad != 0 {
		t.Fatalf("excluded case must leave the denominator: %d eligible / %d bad", r.Eligible, r.Bad)
	}
	if strings.Join(r.Excluded, ",") != "quarantined" {
		t.Fatalf("report must witness exactly the exclusions that hit, got %v", r.Excluded)
	}
	if !strings.Contains(r.Summary(), "excluded: quarantined") {
		t.Fatalf("summary must show the hit exclusion:\n%s", r.Summary())
	}
	if r.Status != SLOMet {
		t.Fatalf("with the quarantined failure excluded the SLO is met, got %q", r.Status)
	}
}

// TestSLOPopulationSlices: the population selector governs which cases count —
// a nightly SLO ignores PR-tier evidence and vice versa.
func TestSLOPopulationSlices(t *testing.T) {
	nightly := goodCase("n1")
	nightly.Tier = TierNightly

	s := servingSLO() // population tier=pr
	r := s.Evaluate([]LadderCase{goodCase("p1"), failCase("p2"), nightly})
	if r.Eligible != 2 {
		t.Fatalf("pr SLO must see only pr cases, got %d eligible", r.Eligible)
	}

	s.Population = SLOPopulation{Tier: TierNightly}
	r = s.Evaluate([]LadderCase{goodCase("p1"), failCase("p2"), nightly})
	if r.Eligible != 1 || r.Bad != 0 {
		t.Fatalf("nightly SLO must see only the nightly case, got %d eligible / %d bad", r.Eligible, r.Bad)
	}
}
