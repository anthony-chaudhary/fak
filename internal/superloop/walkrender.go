package superloop

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// tier ranks a member status into a worst-first band (lower = enter sooner):
//
//	0  a dark / unmeasured LEAF — its status is bad or unknown; most urgent
//	1  a measured leaf carrying debt, SPINNING (#4956: ticking, zero advanced
//	   verified progress), or ORPHANED (#4957: emitting work nobody advances) —
//	   so a spinning or orphaned member out-ranks every clean live leaf even
//	   before its shell-side debt term lands
//	2  a container (garden / super loop) — descend to learn its status
//	3  a measured, clean, live leaf — nothing to do
//
// workEligible reports whether a status belongs on the worklist — the exact inverse of
// the clean-and-measured drop condition. A container (always surfaced for descent), an
// unmeasured or dark member, any debt-bearing leaf, a SPINNING member, or an ORPHANED
// member (#4957: emitting work nobody advances) is work to enter; a measured, clean,
// live leaf is not. Shared by the worklist filter and the mix pre-count so the two can
// never disagree on what counts as "work to enter".
func workEligible(st MemberStatus) bool {
	return st.Container || !st.Measured || st.Dark || st.Debt > 0 ||
		st.Progress == ProgressSpinning || st.FollowOn == FollowonOrphaned
}

func tier(st MemberStatus) int {
	if st.Container {
		return 2
	}
	if st.Dark || !st.Measured {
		return 0
	}
	if st.Debt > 0 || st.Progress == ProgressSpinning || st.FollowOn == FollowonOrphaned {
		return 1
	}
	return 3
}

// spinningAction is the revive/redirect entry for a SPINNING member. Unlike a dark
// loop (restart it), a spinning one IS running — the action is to redirect it at
// work that advances its verified ledger, or stop paying for its ticks.
func spinningAction(st MemberStatus) string {
	if e := strings.TrimSpace(st.Member.Enter); e != "" {
		return fmt.Sprintf("revive/redirect via `%s` — %s is SPINNING (%s: ticking, zero advanced verified progress)",
			e, st.Member.Ref, relay.ReasonNoProgress)
	}
	return fmt.Sprintf("revive/redirect the %s loop — SPINNING (%s: ticking, zero advanced verified progress)",
		st.Member.Ref, relay.ReasonNoProgress)
}

// orphanedAction is the chase/redirect entry for an ORPHANED member (#4957). Unlike a
// dark loop (restart it) or a spinning one (make it produce), an orphaned loop
// produces fine — the action is to route its EMITTED work to an owner who advances
// it, or stop emitting into the void. It names the closed relay token so the verdict
// stays checkable against dos_check_reason, never free text.
func orphanedAction(st MemberStatus) string {
	if e := strings.TrimSpace(st.Member.Enter); e != "" {
		return fmt.Sprintf("chase/redirect via `%s` — %s is ORPHANED (%s: emits follow-on work nobody advances; route its output to an owner)",
			e, st.Member.Ref, relay.ReasonOrphanedFollowon)
	}
	return fmt.Sprintf("chase/redirect the %s loop — ORPHANED (%s: emits follow-on work nobody advances; route its output to an owner)",
		st.Member.Ref, relay.ReasonOrphanedFollowon)
}

// loopDriveAction is the shared next-action ladder for a driveable loop member (both
// KindLoop and KindLoopFleet run it): a spinning loop or one emitting orphaned follow-on
// work reports that first, then an explicit enter-command wins, and a loop that has gone
// dark asks to be revived rather than driven. `darkSubject` names the member in the
// revive-via-command wording — the only phrasing the two kinds spell differently.
func loopDriveAction(st MemberStatus, darkSubject string) string {
	if st.Progress == ProgressSpinning {
		return spinningAction(st)
	}
	if st.FollowOn == FollowonOrphaned {
		return orphanedAction(st)
	}
	if e := strings.TrimSpace(st.Member.Enter); e != "" {
		if st.Dark {
			return fmt.Sprintf("revive via `%s` — %s has gone dark", e, darkSubject)
		}
		return fmt.Sprintf("drive via `%s`", e)
	}
	if st.Dark {
		return fmt.Sprintf("revive the %s loop — it has gone dark", st.Member.Ref)
	}
	return fmt.Sprintf("drive the %s loop", st.Member.Ref)
}

func actionFor(st MemberStatus) string {
	switch st.Member.Kind {
	case KindScorecard:
		if !st.Measured {
			return fmt.Sprintf("run `fak scorecard` / the %s scorecard to measure it", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to retire %s debt", e, st.Member.Ref)
		}
		return fmt.Sprintf("enter the %s scorecard's reduce loop (its skill) to retire debt", st.Member.Ref)
	case KindLoop:
		return loopDriveAction(st, st.Member.Ref)
	case KindGarden:
		return "run `fak garden` then `fak garden tick` to tend the bundle"
	case KindSuperloop:
		return fmt.Sprintf("descend: `fak superloop walk %s`", st.Member.Ref)
	case KindSurface:
		return fmt.Sprintf("enter `%s`", st.Member.Ref)
	case KindUtilization:
		if !st.Measured {
			return fmt.Sprintf("read the %s pool's live utilization to measure its unused headroom", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to spend the idle %s capacity", e, st.Member.Ref)
		}
		return fmt.Sprintf("put the idle %s capacity to work", st.Member.Ref)
	case KindLoopFleet:
		if !st.Measured {
			return fmt.Sprintf("read `fak superloop roster` — %q has no foldable ledger here (known gap)", st.Member.Ref)
		}
		return loopDriveAction(st, "fleet loop "+st.Member.Ref)
	case KindTrajectory:
		if !st.Measured {
			return fmt.Sprintf("read the trajctl ledger to fold objective %q's curve", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to steer objective %s back on-course", e, st.Member.Ref)
		}
		return fmt.Sprintf("steer trajectory objective %q worst-first (`fak trajctl curve --objective %s`)", st.Member.Ref, st.Member.Ref)
	default:
		return "enter the member's loop"
	}
}

func workDetail(st MemberStatus) string {
	if st.Container {
		return "DESCEND — " + firstNonEmpty(st.Detail, st.Member.Why)
	}
	if !st.Measured {
		if strings.TrimSpace(st.Detail) != "" {
			return "UNMEASURED — " + st.Detail
		}
		return "UNMEASURED — status could not be read"
	}
	if st.Dark {
		return "DARK — " + firstNonEmpty(st.Detail, "loop has gone quiet past its cadence")
	}
	if st.Progress == ProgressSpinning {
		return "SPINNING — " + firstNonEmpty(st.Detail,
			"ticking on cadence with zero advanced verified progress ("+relay.ReasonNoProgress+")")
	}
	if st.FollowOn == FollowonOrphaned {
		return "ORPHANED-FOLLOWON — " + firstNonEmpty(st.Detail,
			"emitted follow-on work open with no advance past its cadence window ("+relay.ReasonOrphanedFollowon+")")
	}
	return firstNonEmpty(st.Detail, st.Member.Why)
}

func walkVerdict(s Super, rep WalkReport) (verdict, finding, reason, next string) {
	if rep.Unmeasured > 0 {
		return "ACTION", "superloop_unmeasured",
			fmt.Sprintf("walking %q: %d/%d member(s) could not be read, so the intent is not proven tended (debt %d across %d measured)",
				s.Name, rep.Unmeasured, rep.Members, rep.TotalDebt, rep.Walked),
			"repair/read the unmeasured member(s) first: " + worklistHead(rep)
	}
	if rep.Dark > 0 {
		return "ACTION", "superloop_dark",
			fmt.Sprintf("walking %q: %d member loop(s) have gone DARK; revive them before chasing debt (debt %d)", s.Name, rep.Dark, rep.TotalDebt),
			"worst-first: " + worklistHead(rep)
	}
	if rep.Spinning > 0 {
		return "ACTION", "superloop_spinning",
			fmt.Sprintf("walking %q: %d member loop(s) are SPINNING (%s) — ticking on cadence with zero advanced verified progress; revive or redirect them before chasing debt (debt %d)",
				s.Name, rep.Spinning, relay.ReasonNoProgress, rep.TotalDebt),
			"worst-first: " + worklistHead(rep)
	}
	if rep.Orphaned > 0 {
		return "ACTION", "superloop_orphaned",
			fmt.Sprintf("walking %q: %d member loop(s) are ORPHANED (%s) — emitting follow-on work nobody advances; chase or redirect their output before chasing debt (debt %d)",
				s.Name, rep.Orphaned, relay.ReasonOrphanedFollowon, rep.TotalDebt),
			"worst-first: " + worklistHead(rep)
	}
	if rep.TotalDebt > s.Floor {
		return "ACTION", "superloop_debt",
			fmt.Sprintf("walking %q: aggregate debt %d > floor %d across %d member(s); enter the worst first", s.Name, rep.TotalDebt, s.Floor, rep.Members),
			"worst-first: " + worklistHead(rep)
	}
	if rep.IssueShortfall > 0 {
		return "ACTION", "superloop_issue_shortfall",
			fmt.Sprintf("walking %q: debt at-or-below floor, but %d/%d headline issue(s) still owed (progressed %d) — the target is a gate, not a decoration",
				s.Name, rep.IssueShortfall, rep.IssueTarget, rep.IssueProgressed),
			"progress the remaining issues: " + worklistHead(rep)
	}
	if rep.Rollup.Intents > 1 {
		if rep.Rollup.Unmeasured > 0 {
			return "ACTION", "superloop_unmeasured",
				fmt.Sprintf("walking %q: %d/%d rolled-up leaf member(s) across %d intent(s) could not be read, so the intent is not proven tended (rollup debt %d)",
					s.Name, rep.Rollup.Unmeasured, rep.Rollup.LeafMembers, rep.Rollup.Intents, rep.Rollup.TotalDebt),
				"repair/read the unmeasured rolled-up member(s) first: " + worklistHead(rep)
		}
		if rep.Rollup.Dark > 0 {
			return "ACTION", "superloop_dark",
				fmt.Sprintf("walking %q: %d rolled-up member loop(s) across %d intent(s) have gone DARK; revive them before chasing debt (rollup debt %d)",
					s.Name, rep.Rollup.Dark, rep.Rollup.Intents, rep.Rollup.TotalDebt),
				"worst-first: " + worklistHead(rep)
		}
		if rep.Rollup.Spinning > 0 {
			return "ACTION", "superloop_spinning",
				fmt.Sprintf("walking %q: %d rolled-up member loop(s) across %d intent(s) are SPINNING (%s); revive or redirect them",
					s.Name, rep.Rollup.Spinning, rep.Rollup.Intents, relay.ReasonNoProgress),
				"worst-first: " + worklistHead(rep)
		}
		if rep.Rollup.Orphaned > 0 {
			return "ACTION", "superloop_orphaned",
				fmt.Sprintf("walking %q: %d rolled-up member loop(s) across %d intent(s) are ORPHANED (%s); chase or redirect their output",
					s.Name, rep.Rollup.Orphaned, rep.Rollup.Intents, relay.ReasonOrphanedFollowon),
				"worst-first: " + worklistHead(rep)
		}
		if rep.Rollup.TotalDebt > s.Floor {
			return "ACTION", "superloop_debt",
				fmt.Sprintf("walking %q: rollup aggregate debt %d > floor %d across %d leaf member(s) in %d intent(s); enter the worst first",
					s.Name, rep.Rollup.TotalDebt, s.Floor, rep.Rollup.LeafMembers, rep.Rollup.Intents),
				"worst-first: " + worklistHead(rep)
		}
		if rep.Rollup.IssueShortfall > 0 {
			return "ACTION", "superloop_issue_shortfall",
				fmt.Sprintf("walking %q: debt at-or-below floor, but %d headline issue(s) still owed in rollup — the target is a gate, not a decoration",
					s.Name, rep.Rollup.IssueShortfall),
				"progress the remaining issues: " + worklistHead(rep)
		}
	}
	target := ""
	if rep.IssueProgressMeasured && rep.IssueTarget > 0 {
		target = fmt.Sprintf("; headline %d/%d issue(s) progressed", rep.IssueProgressed, rep.IssueTarget)
	}
	return "OK", "superloop_satisfied",
		fmt.Sprintf("walking %q: aggregate debt %d at-or-below floor %d; every member measured and live%s", s.Name, rep.TotalDebt, s.Floor, target),
		"hold the line; the member loops keep it tended"
}

func worklistHead(rep WalkReport) string {
	if len(rep.Worklist) == 0 {
		return "(nothing to enter)"
	}
	w := rep.Worklist[0]
	return fmt.Sprintf("%s %q — %s", w.Member.Kind, w.Member.Ref, w.Action)
}

func memberKey(m Member) string { return string(m.Kind) + ":" + m.Ref }

func firstNonEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}
