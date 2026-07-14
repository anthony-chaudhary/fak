package scoreboard

import (
	"fmt"
	"strings"
)

// Quality SLOs and error budgets for serving (issue #4582, epic #4509): the
// contract layer ABOVE the per-case quality ladder. ladder.go decides whether a
// single case's evidence can be trusted (Effective); this file decides whether a
// POPULATION of trusted evidence is meeting an objective, and how fast it is
// burning its error budget when it is not.
//
// Like the ladder it is a pure fold — an SLO plus a []LadderCase in, a witnessed
// report out — so the same evaluation replays identically in CI, nightly, and on
// a host. The burn math is carried IN the report (counts, fractions, rate,
// thresholds), never just the verdict, so a reader can re-derive every number.

// SLOStatus is the closed set of outcomes an SLO evaluation renders. Only
// SLOMet is green; every other state — including "we had no data" and "the SLO
// itself is malformed" — demands attention rather than quietly counting green.
type SLOStatus string

const (
	SLOMet         SLOStatus = "met"          // objective held; burn under the alert threshold
	SLOBurnWarning SLOStatus = "burn-warning" // objective still held, but budget burning past the alert threshold
	SLOBreached    SLOStatus = "breached"     // error budget for the window is exhausted
	SLONoData      SLOStatus = "no-data"      // no eligible evidence in the window — NEVER green
	SLOInvalid     SLOStatus = "invalid"      // the SLO definition itself is malformed — never green
)

// SLOPopulation selects which ladder cases an SLO governs. Empty fields match
// everything on that dimension, so {} is "all serving cases" and {Tier: pr} is
// "the cheap always-on slice".
type SLOPopulation struct {
	Tier   LadderTier `json:"tier,omitempty"`
	Engine string     `json:"engine,omitempty"`
}

func (p SLOPopulation) matches(c LadderCase) bool {
	if p.Tier != "" && c.Tier != p.Tier {
		return false
	}
	if p.Engine != "" && c.Engine != p.Engine {
		return false
	}
	return true
}

// String renders the population selector for the witnessed report.
func (p SLOPopulation) String() string {
	parts := []string{}
	if p.Tier != "" {
		parts = append(parts, "tier="+string(p.Tier))
	}
	if p.Engine != "" {
		parts = append(parts, "engine="+p.Engine)
	}
	if len(parts) == 0 {
		return "all"
	}
	return strings.Join(parts, " ")
}

// SLO is one serving-quality objective: who owns it, which cases count, over
// what window, what fraction must be good, and when budget burn should alert.
// This is the full vocabulary the issue #4582 scope names — objective,
// population, window, exclusions, freshness, owner, burn alert.
type SLO struct {
	Name  string `json:"name"`
	Owner string `json:"owner"` // explicit owner; an unowned SLO is invalid

	// Objective is the target good fraction, in (0,1) exclusive. The error
	// budget is its complement (1 - Objective). A 1.0 objective is refused:
	// zero budget makes every burn rate infinite and the alert math undefined.
	Objective float64 `json:"objective"`

	Population SLOPopulation `json:"population"`

	// WindowSeconds bounds the evidence the SLO evaluates: a case whose
	// evidence is older than the window is out of population (reported, not
	// silently dropped). Must be > 0 — an unbounded window has no burn story.
	WindowSeconds int64 `json:"window_seconds"`

	// FreshnessSeconds is the max age at which a pass still counts good; an
	// older pass demotes to stale via Effective and CONSUMES budget. 0 means
	// the window itself is the only freshness bound.
	FreshnessSeconds int64 `json:"freshness_seconds,omitempty"`

	// Exclusions are case IDs excluded by contract (e.g. a quarantined case
	// with a tracked issue). Exclusions that actually hit are witnessed in the
	// report so an exclusion can never silently widen.
	Exclusions []string `json:"exclusions,omitempty"`

	// BurnAlertAt is the burn-rate threshold (fraction of the window's error
	// budget consumed) that trips SLOBurnWarning. 0 defaults to 0.5 — alert
	// when half the budget is gone, breach at 1.0.
	BurnAlertAt float64 `json:"burn_alert_at,omitempty"`
}

// Budget is the error budget: the bad fraction the window is allowed.
func (s SLO) Budget() float64 { return 1 - s.Objective }

// burnAlertAt applies the documented default.
func (s SLO) burnAlertAt() float64 {
	if s.BurnAlertAt <= 0 {
		return 0.5
	}
	return s.BurnAlertAt
}

// Validate refuses a malformed SLO definition. An invalid SLO must never
// evaluate green — Evaluate short-circuits to SLOInvalid with the reason.
func (s SLO) Validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return fmt.Errorf("slo has no name")
	}
	if strings.TrimSpace(s.Owner) == "" {
		return fmt.Errorf("slo %q has no owner — every objective needs an explicit owner", s.Name)
	}
	if s.Objective <= 0 || s.Objective >= 1 {
		return fmt.Errorf("slo %q objective %v must be in (0,1): a 100%% objective has zero error budget and no burn math", s.Name, s.Objective)
	}
	if s.WindowSeconds <= 0 {
		return fmt.Errorf("slo %q has no evaluation window", s.Name)
	}
	return nil
}

// SLOReport is the witnessed evaluation of one SLO: every number the verdict
// was derived from rides along, so the burn math is checkable by a reader and
// a test — never a bare verdict.
type SLOReport struct {
	Name   string    `json:"name"`
	Owner  string    `json:"owner"`
	Status SLOStatus `json:"status"`
	Reason string    `json:"reason,omitempty"` // set when Status == SLOInvalid

	Objective float64 `json:"objective"`
	Budget    float64 `json:"budget"`

	// Population accounting: what was considered, what was set aside, and why.
	Eligible    int      `json:"eligible"`      // denominator: good + bad
	Good        int      `json:"good"`          // effective pass
	Bad         int      `json:"bad"`           // effective fail / stale / inconclusive / no-data — all consume budget
	Skipped     int      `json:"skipped"`       // deliberately not run; outside the denominator, still witnessed
	OutOfWindow int      `json:"out_of_window"` // evidence older than the window
	Excluded    []string `json:"excluded,omitempty"` // contract exclusions that actually hit

	// Burn math (witnessed): BadFraction = Bad/Eligible; BurnRate =
	// BadFraction/Budget. Alert at BurnAlertAt, breach at 1.0.
	BadFraction float64 `json:"bad_fraction"`
	BurnRate    float64 `json:"burn_rate"`
	BurnAlertAt float64 `json:"burn_alert_at"`

	Counts map[LadderStatus]int `json:"counts"` // effective statuses over in-window, non-excluded cases

	// FirstActionable is the first failing case's localized divergence and
	// replay artifact — the SLO-level restatement of acceptance criterion 3.
	FirstActionable *LadderCaseView `json:"first_actionable,omitempty"`
}

// Green reports whether the SLO evaluation may render green. Only a met
// objective is green: no-data, an invalid definition, a burn warning, and a
// breach all demand attention.
func (r SLOReport) Green() bool { return r.Status == SLOMet }

// Evaluate folds the cases through the SLO contract. Each case's status is
// re-derived by Effective (the ladder honesty gate), so an unbacked declared
// pass consumes budget here exactly as it refuses to render green there.
func (s SLO) Evaluate(cases []LadderCase) SLOReport {
	r := SLOReport{
		Name:        s.Name,
		Owner:       s.Owner,
		Objective:   s.Objective,
		Budget:      s.Budget(),
		BurnAlertAt: s.burnAlertAt(),
		Counts:      map[LadderStatus]int{},
	}
	if err := s.Validate(); err != nil {
		r.Status = SLOInvalid
		r.Reason = err.Error()
		return r
	}
	for _, st := range allStatuses {
		r.Counts[st] = 0
	}
	excluded := map[string]bool{}
	for _, id := range s.Exclusions {
		excluded[id] = true
	}
	hit := map[string]bool{}
	for _, c := range cases {
		if !s.Population.matches(c) {
			continue
		}
		if excluded[c.ID] {
			hit[c.ID] = true
			continue
		}
		if c.AgeSeconds > s.WindowSeconds {
			r.OutOfWindow++
			continue
		}
		eff := c.Effective(s.FreshnessSeconds)
		r.Counts[eff]++
		switch eff {
		case StatusSkipped:
			r.Skipped++
			continue
		case StatusPass:
			r.Good++
		default: // fail, stale, inconclusive, no-data — every one consumes budget
			r.Bad++
			if r.FirstActionable == nil && eff == StatusFail {
				r.FirstActionable = &LadderCaseView{
					ID:              c.ID,
					Status:          eff,
					Declared:        c.Status,
					Tier:            c.Tier,
					Cost:            c.Cost,
					Revision:        c.Revision,
					FirstDivergence: c.FirstDivergence,
					Replay:          c.Replay,
				}
			}
		}
	}
	r.Eligible = r.Good + r.Bad
	r.Excluded = sortedKeys(hit)
	if r.Eligible == 0 {
		// No eligible evidence in the window. This is the acceptance
		// criterion's first line: no-data NEVER appears green.
		r.Status = SLONoData
		return r
	}
	r.BadFraction = float64(r.Bad) / float64(r.Eligible)
	r.BurnRate = r.BadFraction / r.Budget
	switch {
	case r.BurnRate >= 1:
		r.Status = SLOBreached
	case r.BurnRate >= r.BurnAlertAt:
		r.Status = SLOBurnWarning
	default:
		r.Status = SLOMet
	}
	return r
}

// Summary renders the report as the captured, human-checkable witness: the
// verdict, the owner, the full population accounting, and the burn math with
// every operand printed.
func (r SLOReport) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "quality SLO %s — %s (owner: %s)\n", r.Name, r.Status, r.Owner)
	if r.Status == SLOInvalid {
		fmt.Fprintf(&b, "invalid: %s\n", r.Reason)
		return b.String()
	}
	fmt.Fprintf(&b, "objective: %.2f%%  error budget: %.2f%%\n", r.Objective*100, r.Budget*100)
	fmt.Fprintf(&b, "population: eligible %d (good %d, bad %d) · skipped %d · out-of-window %d",
		r.Eligible, r.Good, r.Bad, r.Skipped, r.OutOfWindow)
	if len(r.Excluded) > 0 {
		fmt.Fprintf(&b, " · excluded: %s", strings.Join(r.Excluded, ","))
	}
	b.WriteByte('\n')
	if r.Status == SLONoData {
		b.WriteString("no eligible evidence in window — no-data is never green\n")
		return b.String()
	}
	fmt.Fprintf(&b, "burn math: %d bad / %d eligible = %.2f%% error rate; budget %.2f%%; burn rate %.2fx (alert ≥ %.2fx, breach ≥ 1.00x)\n",
		r.Bad, r.Eligible, r.BadFraction*100, r.Budget*100, r.BurnRate, r.BurnAlertAt)
	if r.FirstActionable != nil {
		fmt.Fprintf(&b, "first actionable: case %s", r.FirstActionable.ID)
		if r.FirstActionable.FirstDivergence != nil {
			fmt.Fprintf(&b, " — %s", r.FirstActionable.FirstDivergence.String())
		}
		if r.FirstActionable.Replay != "" {
			fmt.Fprintf(&b, " (replay: %s)", r.FirstActionable.Replay)
		}
		b.WriteByte('\n')
	}
	return b.String()
}
