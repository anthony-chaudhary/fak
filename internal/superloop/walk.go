package superloop

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// Walk evaluates an intent across the statuses of its members and returns the
// ranked work-list and satisfaction verdict.
func Walk(s Super, statuses []MemberStatus, opts ...WalkOpt) WalkReport {
	var cfg walkConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	rep := WalkReport{
		Schema:                WalkSchema,
		Name:                  s.Name,
		Title:                 s.Title,
		Floor:                 s.Floor,
		IssueTarget:           s.IssueTarget,
		IssueProgressMeasured: cfg.issueProgressMeasured,
		IssueProgressed:       cfg.issueProgressed,
		DeclaredMembers:       len(s.Members),
		Members:               len(s.Members),
		Statuses:              statuses,
	}

	// Preserve declared order as the stable tiebreaker.
	order := map[string]int{}
	for i, m := range s.Members {
		order[memberKey(m)] = i
	}

	for _, st := range statuses {
		if st.Container {
			rep.Containers++
			continue // a descend-pointer: not weighed, not counted (see MemberStatus).
		}
		if st.Measured {
			rep.Walked++
			rep.TotalDebt += st.Debt
		} else {
			rep.Unmeasured++
		}
		if st.Dark {
			rep.Dark++
		}
		if st.Progress == ProgressSpinning {
			rep.Spinning++
		}
		if st.FollowOn == FollowonOrphaned {
			rep.Orphaned++
		}
	}

	// When template members expand (e.g. KindLoopFleet:all or KindTrajectory:open),
	// the evaluated direct candidate denominator is Walked + Unmeasured + Containers.
	if rep.Walked+rep.Unmeasured > rep.Members {
		rep.Members = rep.Walked + rep.Unmeasured + rep.Containers
	}

	populateWalkRollup(&rep, s, statuses)
	populateWalkWorklist(&rep, s, statuses, order)

	// Fold the declared issue-target headline against the measured live progress: an
	// unmet target with progress in hand is a shortfall that gates satisfaction (the
	// number is a promise, not a decoration). Only bites when the intent DECLARES a
	// target and the shell actually MEASURED progress — a surface-only walk never gates.
	if s.IssueTarget > 0 && cfg.issueProgressMeasured && cfg.issueProgressed < s.IssueTarget {
		rep.IssueShortfall = s.IssueTarget - cfg.issueProgressed
	}

	directSatisfied := rep.Unmeasured == 0 && rep.Dark == 0 && rep.Spinning == 0 && rep.Orphaned == 0 && rep.TotalDebt <= s.Floor && rep.IssueShortfall == 0
	rep.Satisfied = directSatisfied && rep.Rollup.Satisfied
	rep.Verdict, rep.Finding, rep.Reason, rep.NextAction = walkVerdict(s, rep)
	return rep
}

func populateWalkRollup(rep *WalkReport, s Super, statuses []MemberStatus) {
	descendedIntents := map[string]bool{s.Name: true}
	leafMap := make(map[string]MemberStatus)
	var rollupContainers int
	var subwalkShortfall int
	var subwalkTarget int
	var subwalkProgressed int

	for _, st := range statuses {
		if st.Container {
			rollupContainers++
			continue
		}
		if st.Subwalk != nil {
			descendedIntents[st.Subwalk.Intent] = true
			for _, in := range st.Subwalk.DescendedIntents {
				descendedIntents[in] = true
			}
			subwalkShortfall += st.Subwalk.IssueShortfall
			subwalkTarget += st.Subwalk.IssueTarget
			subwalkProgressed += st.Subwalk.IssueProgressed
			rollupContainers += st.Subwalk.Rollup.Containers

			if len(st.Subwalk.LeafStatuses) > 0 {
				for _, ls := range st.Subwalk.LeafStatuses {
					k := memberKey(ls.Member)
					if _, exists := leafMap[k]; !exists {
						leafMap[k] = ls
					}
				}
			} else {
				k := memberKey(st.Member)
				if _, exists := leafMap[k]; !exists {
					synth := st
					if st.Subwalk.Unmeasured > 0 {
						synth.Measured = false
					}
					if st.Subwalk.Dark > 0 {
						synth.Dark = true
					}
					leafMap[k] = synth
				}
			}
		} else {
			k := memberKey(st.Member)
			if _, exists := leafMap[k]; !exists {
				leafMap[k] = st
			}
		}
	}

	rep.Rollup.Intents = len(descendedIntents)
	rep.Rollup.Floor = s.Floor
	rep.Rollup.Containers = rollupContainers

	descendedList := make([]string, 0, len(descendedIntents))
	for in := range descendedIntents {
		descendedList = append(descendedList, in)
	}
	sort.Strings(descendedList)
	rep.Rollup.DescendedIntents = descendedList

	keys := make([]string, 0, len(leafMap))
	for k := range leafMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		rep.LeafStatuses = append(rep.LeafStatuses, leafMap[k])
	}

	if rep.Rollup.Intents == 1 {
		rep.Rollup.LeafMembers = rep.Walked + rep.Unmeasured
		rep.Rollup.Walked = rep.Walked
		rep.Rollup.Unmeasured = rep.Unmeasured
		rep.Rollup.Dark = rep.Dark
		rep.Rollup.Spinning = rep.Spinning
		rep.Rollup.Orphaned = rep.Orphaned
		rep.Rollup.TotalDebt = rep.TotalDebt
		rep.Rollup.IssueTarget = rep.IssueTarget
		rep.Rollup.IssueProgressed = rep.IssueProgressed
		rep.Rollup.IssueShortfall = rep.IssueShortfall
		rep.Rollup.Satisfied = rep.Unmeasured == 0 && rep.Dark == 0 && rep.Spinning == 0 && rep.Orphaned == 0 && rep.TotalDebt <= s.Floor && rep.IssueShortfall == 0
	} else {
		for _, ls := range rep.LeafStatuses {
			if ls.Measured {
				rep.Rollup.Walked++
				rep.Rollup.TotalDebt += ls.Debt
			} else {
				rep.Rollup.Unmeasured++
			}
			if ls.Dark {
				rep.Rollup.Dark++
			}
			if ls.Progress == ProgressSpinning {
				rep.Rollup.Spinning++
			}
			if ls.FollowOn == FollowonOrphaned {
				rep.Rollup.Orphaned++
			}
		}
		rep.Rollup.LeafMembers = rep.Rollup.Walked + rep.Rollup.Unmeasured
		rep.Rollup.IssueTarget = rep.IssueTarget + subwalkTarget
		rep.Rollup.IssueProgressed = rep.IssueProgressed + subwalkProgressed
		rep.Rollup.IssueShortfall = rep.IssueShortfall + subwalkShortfall
		rep.Rollup.TotalDebt += rep.Rollup.IssueShortfall
		rep.Rollup.Satisfied = rep.Rollup.Unmeasured == 0 && rep.Rollup.Dark == 0 && rep.Rollup.Spinning == 0 && rep.Rollup.Orphaned == 0 && rep.Rollup.TotalDebt <= s.Floor && rep.Rollup.IssueShortfall == 0
	}
}

func populateWalkWorklist(rep *WalkReport, s Super, statuses []MemberStatus, order map[string]int) {
	targetPct := resolveThroughputTarget(s.ThroughputTargetPct)
	var gardening, throughput, neutral int
	for _, st := range statuses {
		if !workEligible(st) {
			continue
		}
		switch classifyWork(st.Member) {
		case WorkGardening:
			gardening++
		case WorkThroughput:
			throughput++
		default:
			neutral++
		}
	}
	favor := favoredClass(gardening, throughput, targetPct)
	rep.Mix = WorkMix{
		Gardening:           gardening,
		Throughput:          throughput,
		Neutral:             neutral,
		TargetThroughputPct: targetPct,
		Favor:               favor,
	}

	ranked := append([]MemberStatus(nil), statuses...)
	sort.SliceStable(ranked, func(i, j int) bool {
		ti, tj := tier(ranked[i]), tier(ranked[j])
		if ti != tj {
			return ti < tj
		}
		if ranked[i].Debt != ranked[j].Debt {
			return ranked[i].Debt > ranked[j].Debt
		}
		if favor != "" {
			fi := classifyWork(ranked[i].Member) == favor
			fj := classifyWork(ranked[j].Member) == favor
			if fi != fj {
				return fi
			}
		}
		return order[memberKey(ranked[i].Member)] < order[memberKey(ranked[j].Member)]
	})

	for _, st := range ranked {
		if !workEligible(st) {
			continue
		}
		rep.Worklist = append(rep.Worklist, WorkItem{
			Member:         st.Member,
			Debt:           st.Debt,
			Dark:           st.Dark,
			Container:      st.Container,
			Progress:       st.Progress,
			ProgressReason: st.ProgressReason,
			FollowOn:       st.FollowOn,
			FollowOnReason: st.FollowOnReason,
			Action:         actionFor(st),
			Detail:         workDetail(st),
		})
	}
	for i := range rep.Worklist {
		rep.Worklist[i].Rank = i + 1
	}

	rows, alloc := divideBudget(s.Budget, len(rep.Worklist))
	rep.Budget = rows
	for i := range rep.Worklist {
		rep.Worklist[i].Allocation = alloc
	}
}

// divideBudget folds a declared budget into the walk's per-dimension rows and the
// per-member allocation.
func divideBudget(b GenerationBudget, n int) ([]BudgetRow, Allocation) {
	rows := make([]BudgetRow, 0, len(budgetDims))
	var alloc Allocation
	for _, d := range budgetDims {
		total := b.cap(d.Name)
		row := BudgetRow{
			Dimension: d.Name,
			Unit:      d.Unit,
			Stream:    b.Stream,
			Budgeted:  total > 0,
			Total:     total,
			Members:   n,
		}
		if total <= 0 {
			row.Hold = "unbudgeted — held for later-horizon work; declare a " + d.Name + " cap to reserve capacity"
			alloc.Held = append(alloc.Held, d.Name)
			rows = append(rows, row)
			continue
		}
		if n > 0 {
			row.PerMember = total / n
		}
		switch d.Name {
		case BudgetTime:
			alloc.MaxMinutes = row.PerMember
		case BudgetTokens:
			alloc.TokenCeiling = row.PerMember
		case BudgetWorkers:
			alloc.MaxWorkers = row.PerMember
		case BudgetReview:
			alloc.ReviewSlots = row.PerMember
		}
		rows = append(rows, row)
	}
	return rows, alloc
}

// SubwalkStatus folds a completed sub-walk into the member status the parent walk
// weighs.
func SubwalkStatus(m Member, rep WalkReport) MemberStatus {
	debt := rep.TotalDebt + rep.IssueShortfall
	if !rep.Satisfied && debt <= 0 {
		debt = 1
	}
	dark := rep.Dark > 0 || rep.Rollup.Dark > 0
	var prog MemberProgress
	var progReason string
	if rep.Spinning > 0 || rep.Rollup.Spinning > 0 {
		prog = ProgressSpinning
		progReason = relay.ReasonNoProgress
	}
	var followOn MemberFollowon
	var followOnReason string
	if rep.Orphaned > 0 || rep.Rollup.Orphaned > 0 {
		followOn = FollowonOrphaned
		followOnReason = relay.ReasonOrphanedFollowon
	}

	leaves := rep.Rollup.LeafMembers
	if leaves == 0 {
		leaves = rep.Members
	}
	unm := rep.Unmeasured
	if rep.Rollup.Unmeasured > unm {
		unm = rep.Rollup.Unmeasured
	}
	darkCount := rep.Dark
	if rep.Rollup.Dark > darkCount {
		darkCount = rep.Rollup.Dark
	}

	detail := fmt.Sprintf("descended: %s (%s) — debt %d, shortfall %d, unmeasured %d, dark %d across %d member(s)",
		rep.Verdict, rep.Finding, rep.TotalDebt, rep.IssueShortfall, unm, darkCount, leaves)

	descended := make([]string, 0, len(rep.Rollup.DescendedIntents)+1)
	descended = append(descended, rep.Name)
	for _, in := range rep.Rollup.DescendedIntents {
		if in != rep.Name {
			descended = append(descended, in)
		}
	}
	sort.Strings(descended)

	sub := &SubwalkSummary{
		Intent:           rep.Name,
		Title:            rep.Title,
		Verdict:          rep.Verdict,
		Finding:          rep.Finding,
		Satisfied:        rep.Satisfied,
		TotalDebt:        debt,
		Floor:            rep.Floor,
		Members:          rep.Members,
		Walked:           rep.Walked,
		Unmeasured:       rep.Unmeasured,
		Dark:             rep.Dark,
		Spinning:         rep.Spinning,
		Orphaned:         rep.Orphaned,
		IssueTarget:      rep.IssueTarget,
		IssueProgressed:  rep.IssueProgressed,
		IssueShortfall:   rep.IssueShortfall,
		Rollup:           rep.Rollup,
		LeafStatuses:     rep.LeafStatuses,
		DescendedIntents: descended,
	}

	return MemberStatus{
		Member:         m,
		Measured:       true,
		Debt:           debt,
		Dark:           dark,
		Progress:       prog,
		ProgressReason: progReason,
		FollowOn:       followOn,
		FollowOnReason: followOnReason,
		Detail:         detail,
		Subwalk:        sub,
	}
}
