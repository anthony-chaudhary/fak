package quality

import (
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
)

// This file is the corpus half of the #4574 suite split. suite_split.go partitions
// an in-memory corpus by evidence cost; SplitCorpus reads that corpus from the
// canonical case files as committed, so a split replays from BYTES in a clean
// checkout instead of from structs a caller assembled in memory. The difference
// matters: "the splitter routes correctly when handed good input" is a property of
// the function, while "this corpus, as it sits on disk, routes to these suites" is
// evidence about the suites an operator actually runs.
//
// It is fail-closed at every step a corpus can go wrong. A file that will not load
// is REJECTED rather than skipped — a silently skipped case is a case that stops
// being checked with nobody noticing, which is exactly the "missing evidence is
// never a pass" rule this ladder exists to hold. Two files claiming one case id are
// refused as ambiguous rather than racing on directory order. And Green refuses a
// plan that placed nothing at all, so an empty or wholly unreadable corpus can
// never read as a green split.

// SplitCorpus loads every *.json case under dir in fsys and splits it into the
// PR / nightly / release suites (see SplitSuites; nil budgets = DefaultBudgets).
//
// A file that fails the canonical case contract becomes a SuiteReject naming the
// file rather than an error: one broken case must not stop the rest of the corpus
// from being routed, and it must not disappear either. An error is returned only
// when the corpus DIRECTORY itself cannot be read — an infrastructure failure,
// which is never reported as a quality verdict.
func SplitCorpus(fsys fs.FS, dir string, budgets map[Tier]TierBudget) (SuitePlan, error) {
	entries, err := fs.ReadDir(fsys, dir)
	if err != nil {
		return SuitePlan{}, fmt.Errorf("read quality corpus %s: %w", dir, err)
	}

	var (
		cases   []QualityCase
		refused []SuiteReject
		byID    = map[string]string{}
	)
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".json") {
			continue
		}
		data, err := fs.ReadFile(fsys, path.Join(dir, name))
		if err != nil {
			refused = append(refused, SuiteReject{
				CaseID: name, Reason: "unreadable corpus file: " + err.Error(),
			})
			continue
		}
		c, err := LoadCase(data)
		if err != nil {
			refused = append(refused, SuiteReject{
				CaseID: name, Reason: "unloadable corpus file: " + err.Error(),
			})
			continue
		}
		if first, dup := byID[c.ID]; dup {
			refused = append(refused, SuiteReject{
				CaseID: c.ID, Tier: Tier(c.Metadata.Tier.Name),
				Reason: fmt.Sprintf("duplicate case id: %s also declares it — one id cannot route two ways", first),
			})
			continue
		}
		byID[c.ID] = name
		cases = append(cases, c)
	}

	plan := SplitSuites(cases, budgets)
	plan.Rejected = append(plan.Rejected, refused...)
	sort.SliceStable(plan.Rejected, func(i, j int) bool {
		return plan.Rejected[i].CaseID < plan.Rejected[j].CaseID
	})
	return plan, nil
}

// Green is the fail-closed verdict on a split plan, for a caller that gates on it
// (a CI step, an operator readout) rather than reading the whole plan. It is true
// only when the split rejected nothing AND actually placed evidence: a plan that
// refused a case is not a pass — refusing it is the point — and a plan whose suites
// are all empty is not a pass either, because a corpus that qualifies nothing has
// produced no evidence. The returned string names the first blocking reason, in the
// plan's deterministic case order, so a caller can print it and act on it.
func (p SuitePlan) Green() (bool, string) {
	if len(p.Rejected) > 0 {
		r := p.Rejected[0]
		why := fmt.Sprintf("case %s not placed: %s", r.CaseID, r.Reason)
		if more := len(p.Rejected) - 1; more > 0 {
			why += fmt.Sprintf(" (+%d more rejected)", more)
		}
		return false, why
	}
	placed := 0
	for _, s := range p.Suites {
		placed += len(s.Cases)
	}
	if placed == 0 {
		return false, "no case placed in any suite — a split that qualifies nothing is not evidence"
	}
	return true, ""
}
