package quality

import (
	"fmt"
	"sort"
	"strings"
)

// This file is the suite splitter (#4574, under epic #4509): the layer that takes
// a whole quality corpus and partitions it into the PR / nightly / release suites
// top stacks keep separate, splitting BY EVIDENCE COST so a cheap per-PR check is
// never trapped behind a nightly hardware run. It is additive — it registers no
// oracles and edits no runner core, consuming only the QualityCase envelope
// (case.go) each case already carries.
//
// The split is fail-closed the same way the release gate (release_gate.go) is: a
// case that does not carry a complete routing header (tier, timeout, resources,
// owner, evidence family) is REJECTED, never silently dropped into a suite, and a
// case whose declared tier cannot afford its evidence cost is rejected with the
// budget it broke — "expensive evidence in a cheap tier" is never a pass.

// SuitePlanSchema is the versioned tag on a split plan. Consumers pin the major so
// a schema bump is a conscious migration (the #4519 house rule), not field drift.
const SuitePlanSchema = "fak-quality-suite-plan/1"

// EvidenceFamily is the class of evidence a case produces. #4574 scope names the
// five families top stacks separate; the splitter records the family so an
// operator can see WHAT a suite is qualifying, not just how long it costs.
type EvidenceFamily string

const (
	// FamilyDeterministic is an exact-oracle / greedy differential check: cheap,
	// reproducible from a fixed oracle, the natural per-PR gate.
	FamilyDeterministic EvidenceFamily = "deterministic"
	// FamilyGPUParity compares device backends and needs an accelerator, so it can
	// never ride the CPU-only PR lane.
	FamilyGPUParity EvidenceFamily = "gpu_parity"
	// FamilyStatistics is a distribution/sampling check that needs many samples —
	// too slow for a PR, the natural nightly tenant.
	FamilyStatistics EvidenceFamily = "statistics"
	// FamilyCorpora replays a task corpus: broad, slow, nightly-or-release.
	FamilyCorpora EvidenceFamily = "corpora"
	// FamilyReview is a rubric/judge review of report quality: release-cadence
	// qualification, not a per-PR blocker.
	FamilyReview EvidenceFamily = "review"
)

func validFamily(s string) bool {
	switch EvidenceFamily(s) {
	case FamilyDeterministic, FamilyGPUParity, FamilyStatistics, FamilyCorpora, FamilyReview:
		return true
	}
	return false
}

// TierBudget is the evidence-cost ceiling one tier admits. A tier is a cost class:
// the PR tier buys a fast, CPU-only signal on every push; nightly buys a longer
// CPU budget for sampling and corpora; release buys unbounded time and the
// accelerators GPU-parity and hardware qualification need. A case whose declared
// cost exceeds its tier's budget is refused — that refusal IS the "split by
// evidence cost" contract.
type TierBudget struct {
	Tier            Tier
	MaxTimeout      int64 // seconds; 0 = unbounded
	MaxAccelerators int   // peak accelerators the tier admits
}

// DefaultBudgets is the built-in cost split. PR is fast and CPU-only so it can gate
// every push; nightly extends the wall for statistics and corpora but stays on CPU;
// release is unbounded and admits accelerators for GPU parity and hardware review.
func DefaultBudgets() map[Tier]TierBudget {
	return map[Tier]TierBudget{
		TierPR:      {Tier: TierPR, MaxTimeout: 120, MaxAccelerators: 0},
		TierNightly: {Tier: TierNightly, MaxTimeout: 3600, MaxAccelerators: 0},
		TierRelease: {Tier: TierRelease, MaxTimeout: 0, MaxAccelerators: 64},
	}
}

// SuiteCase is one placed case's routing header: enough for an operator to read who
// owns it, what it proves, which tier runs it, and what it costs — without opening
// the full case.
type SuiteCase struct {
	CaseID string         `json:"case_id"`
	Family EvidenceFamily `json:"family"`
	Owner  string         `json:"owner"`
	Tier   Tier           `json:"tier"`
	Cost   CostSpec       `json:"cost"`
}

// Suite is one tier's ordered case list plus its summed cost. Cases are ordered
// cheapest-evidence-first so the suite fails fast: the quickest check that can
// localize a defect runs before the expensive ones.
type Suite struct {
	Tier            Tier        `json:"tier"`
	Cases           []SuiteCase `json:"cases"`
	TotalRuntimeSec int64       `json:"total_runtime_seconds"`
	TotalTimeoutSec int64       `json:"total_timeout_seconds"`
	MaxMemoryMiB    int64       `json:"max_memory_mib"`
	MaxCPU          int         `json:"max_cpu"`
	MaxAccelerators int         `json:"max_accelerators"`
}

// SuiteReject is one case the split refused, with the reason it could not be
// routed. A rejected case is never placed in a suite — missing routing evidence or
// evidence too expensive for its tier is never a pass.
type SuiteReject struct {
	CaseID string `json:"case_id"`
	Tier   Tier   `json:"tier,omitempty"`
	Reason string `json:"reason"`
}

// SuitePlan is the machine-readable output of a split: the three ordered suites and
// every rejected case. It is a pure function of (cases, budgets) — same corpus,
// same plan — so a split replays.
type SuitePlan struct {
	Schema   string        `json:"schema"`
	Suites   []Suite       `json:"suites"`
	Rejected []SuiteReject `json:"rejected,omitempty"`
}

// SplitSuites partitions a corpus into PR / nightly / release suites under the given
// budgets (nil = DefaultBudgets). A case is rejected — never placed — when it fails
// the canonical routing contract (bad/absent tier, timeout, resources, owner, or
// family) or when its declared cost exceeds its tier's budget. Placed cases are
// ordered cheapest-evidence-first within each suite; the suites themselves are
// returned in fixed PR, nightly, release order so the plan is deterministic.
func SplitSuites(cases []QualityCase, budgets map[Tier]TierBudget) SuitePlan {
	if budgets == nil {
		budgets = DefaultBudgets()
	}
	plan := SuitePlan{Schema: SuitePlanSchema}
	buckets := map[Tier][]SuiteCase{}

	for _, c := range cases {
		if err := c.ValidateCanonical(); err != nil {
			plan.Rejected = append(plan.Rejected, SuiteReject{
				CaseID: c.ID, Tier: Tier(c.Metadata.Tier.Name),
				Reason: "incomplete routing header: " + err.Error(),
			})
			continue
		}
		tier := Tier(c.Metadata.Tier.Name)
		budget, ok := budgets[tier]
		if !ok {
			plan.Rejected = append(plan.Rejected, SuiteReject{
				CaseID: c.ID, Tier: tier, Reason: "no budget defined for tier " + string(tier),
			})
			continue
		}
		if reason, ok := affords(budget, c.Metadata.Cost); !ok {
			plan.Rejected = append(plan.Rejected, SuiteReject{CaseID: c.ID, Tier: tier, Reason: reason})
			continue
		}
		buckets[tier] = append(buckets[tier], SuiteCase{
			CaseID: c.ID, Family: EvidenceFamily(c.Metadata.Family),
			Owner: c.Metadata.Owner, Tier: tier, Cost: c.Metadata.Cost,
		})
	}

	for _, tier := range []Tier{TierPR, TierNightly, TierRelease} {
		plan.Suites = append(plan.Suites, buildSuite(tier, buckets[tier]))
	}
	sort.SliceStable(plan.Rejected, func(i, j int) bool {
		return plan.Rejected[i].CaseID < plan.Rejected[j].CaseID
	})
	return plan
}

// affords reports whether a tier's budget admits a case's declared cost, returning
// the first broken ceiling so the rejection is actionable rather than a bare "too
// expensive". A zero MaxTimeout means the tier is unbounded on time.
func affords(b TierBudget, cost CostSpec) (string, bool) {
	if b.MaxTimeout > 0 && cost.TimeoutSeconds > b.MaxTimeout {
		return fmt.Sprintf("timeout %ds exceeds tier %s budget of %ds — route to a slower tier",
			cost.TimeoutSeconds, b.Tier, b.MaxTimeout), false
	}
	if cost.Accelerators > b.MaxAccelerators {
		return fmt.Sprintf("needs %d accelerator(s) but tier %s admits %d — route to release",
			cost.Accelerators, b.Tier, b.MaxAccelerators), false
	}
	return "", true
}

// buildSuite orders a tier's cases cheapest-evidence-first and sums the suite's cost
// envelope. Ordering is (timeout, runtime, memory, id) so the cheapest, fastest
// check that can localize a defect runs first and the order is total (id breaks
// ties, keeping the plan replayable).
func buildSuite(tier Tier, cases []SuiteCase) Suite {
	sort.SliceStable(cases, func(i, j int) bool {
		a, b := cases[i].Cost, cases[j].Cost
		switch {
		case a.TimeoutSeconds != b.TimeoutSeconds:
			return a.TimeoutSeconds < b.TimeoutSeconds
		case a.RuntimeSeconds != b.RuntimeSeconds:
			return a.RuntimeSeconds < b.RuntimeSeconds
		case a.MemoryMiB != b.MemoryMiB:
			return a.MemoryMiB < b.MemoryMiB
		default:
			return cases[i].CaseID < cases[j].CaseID
		}
	})
	s := Suite{Tier: tier, Cases: cases}
	for _, sc := range cases {
		s.TotalRuntimeSec += sc.Cost.RuntimeSeconds
		s.TotalTimeoutSec += sc.Cost.TimeoutSeconds
		if sc.Cost.MemoryMiB > s.MaxMemoryMiB {
			s.MaxMemoryMiB = sc.Cost.MemoryMiB
		}
		if sc.Cost.CPU > s.MaxCPU {
			s.MaxCPU = sc.Cost.CPU
		}
		if sc.Cost.Accelerators > s.MaxAccelerators {
			s.MaxAccelerators = sc.Cost.Accelerators
		}
	}
	return s
}

// ExplainPlan renders a SuitePlan as an operator readout: each suite with its case
// count and cost envelope, its ordered cases, then every rejected case with the
// budget or header it broke. It mirrors Explain / ExplainRelease — the bridge from a
// machine plan to "here is what each suite costs and why a case was left out".
func ExplainPlan(p SuitePlan) string {
	var b strings.Builder
	for _, s := range p.Suites {
		fmt.Fprintf(&b, "SUITE %-8s %d case(s)  runtime=%ds timeout=%ds cpu<=%d mem<=%dMiB accel<=%d\n",
			s.Tier, len(s.Cases), s.TotalRuntimeSec, s.TotalTimeoutSec, s.MaxCPU, s.MaxMemoryMiB, s.MaxAccelerators)
		for _, sc := range s.Cases {
			fmt.Fprintf(&b, "  %-14s %-13s owner=%s timeout=%ds accel=%d\n",
				sc.CaseID, sc.Family, sc.Owner, sc.Cost.TimeoutSeconds, sc.Cost.Accelerators)
		}
	}
	if len(p.Rejected) > 0 {
		fmt.Fprintf(&b, "REJECTED %d case(s) — not placed in any suite:\n", len(p.Rejected))
		for _, r := range p.Rejected {
			fmt.Fprintf(&b, "  %-14s tier=%-8s %s\n", r.CaseID, r.Tier, r.Reason)
		}
	}
	return b.String()
}
