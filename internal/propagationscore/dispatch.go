package propagationscore

// dispatch.go is the fan-out half: it turns each HARD propagation gap (a proven scorecard
// convention a sibling card has not adopted) into a deduped, dispatchable GitHub issue by
// mapping onto the EXISTING internal/dogfoodissues backlog bridge -- the same Decide ->
// ToActionItem -> BuildPlan -> Sync path internal/guardroute established, so there is no new
// gh-issue code here. The stable Key is content-addressed (convention + member, no timestamp),
// so a re-run UPDATES the same issue in place instead of opening a duplicate. This is the
// "10 auto-created tickets beat the operator remembering" mechanism: one tracked, scoped issue
// per laggard, routed to the lane that owns that card's files.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// Gap is one HARD un-propagated convention: a Member that has not adopted a Convention which the
// family has otherwise proven (a declared standard, or one past the quorum). Adopters/Total carry
// the adoption ratio at detection time for the issue body.
type Gap struct {
	Member     Member
	Convention Convention
	Adopters   int
	Total      int
}

// Gaps re-probes the tree and returns every HARD propagation gap, sorted deterministically
// (convention work-list order, then member verb) so the dispatcher's cap and dedup are stable.
func Gaps(root string) []Gap {
	probes := ProbeMembers(root, Family)
	order := map[string]int{}
	for i, c := range Conventions {
		order[c.Key] = i
	}
	var gaps []Gap
	for _, c := range Conventions {
		_, g := kpiForConvention(c, probes)
		gaps = append(gaps, g...)
	}
	sort.SliceStable(gaps, func(i, j int) bool {
		if order[gaps[i].Convention.Key] != order[gaps[j].Convention.Key] {
			return order[gaps[i].Convention.Key] < order[gaps[j].Convention.Key]
		}
		return gaps[i].Member.Verb < gaps[j].Member.Verb
	})
	return gaps
}

// Key is the CONTENT-STABLE dedup identity (no run-id / timestamp): convention + member. A
// re-run folds onto the same issue, and a recurrence after a "fixed" close re-files the same key.
func (g Gap) Key() string {
	return "propagation-debt/" + g.Convention.Key + "/" + slug(g.Member.Verb)
}

// Title is the one-line issue subject, e.g. "propagate the shared scorecard kernel to fak dogfood-score".
func (g Gap) Title() string {
	return "propagate " + g.Convention.Short + " to fak " + g.Member.Verb
}

// witnessHint returns the per-convention command that proves the extension landed.
func (g Gap) witnessHint() string {
	switch g.Convention.Key {
	case "kernel":
		return "go test ./" + g.Member.PkgDir + " && go build ./..."
	case "compare":
		return "fak " + g.Member.Verb + " --json > /tmp/base.json && fak " + g.Member.Verb + " --compare /tmp/base.json"
	case "markdown":
		return "fak " + g.Member.Verb + " --markdown"
	case "json":
		return "fak " + g.Member.Verb + " --json"
	case "test":
		return "go test ./" + g.Member.PkgDir
	case "controlpane":
		return "python tools/scorecard_control_pane.py --json"
	default:
		return "go build ./... && go test ./" + g.Member.PkgDir
	}
}

// inScopeHint returns the per-convention shape of the extension work.
func (g Gap) inScopeHint() string {
	switch g.Convention.Key {
	case "kernel":
		return "Port `" + g.Member.PkgDir + "` onto pkg/scorecard.Fold (select GradeStd/GradeStrict from grade.go) so its grade table can no longer drift, mirroring an adopter such as internal/conflationscore."
	case "compare":
		return "Add a `--compare BASELINE.json` flag to `" + g.Member.CmdFile + "` that prints the debt delta via pkg/scorecard.Compare, mirroring cmd/fak/conflationscore.go (note: the shared scorecardCmdSetup helper does NOT provide --compare)."
	case "markdown":
		return "Add a `--markdown` snapshot flag to `" + g.Member.CmdFile + "` via pkg/scorecard.Markdown, mirroring an adopter."
	case "json":
		return "Add a `--json` control-pane payload flag to `" + g.Member.CmdFile + "`, mirroring an adopter."
	case "test":
		return "Add a `_test.go` to `" + g.Member.PkgDir + "` with a per-KPI fixture and a live-floor smoke, mirroring an adopter."
	case "controlpane":
		return "Register `" + g.Member.Verb + "` in tools/scorecard_control_pane.py SCORECARDS (and re-pin the baseline) so its debt folds into the portfolio ratchet."
	default:
		return "Extend `" + g.Member.Verb + "` to adopt the convention, mirroring an adopter."
	}
}

// ToActionItem maps a gap onto the EXISTING dogfoodissues.ActionItem so the GitHub-issue half is
// pure reuse (IssueBody/BuildPlanWithOptions/Sync). evidencePath is the scorecard JSON the issue
// cites. Routing rides Paths (the member's package + shell), so dispatch lands it in the lane
// that owns the card's code -- the worker that fixes it is the one that owns the file.
func (g Gap) ToActionItem(evidencePath string) dogfoodissues.ActionItem {
	declared := ""
	if g.Convention.Declared {
		declared = fmt.Sprintf(" This is a declared standard (%s), so the laggard is debt regardless of count.", g.Convention.Source)
	}
	return dogfoodissues.ActionItem{
		Key:          "propagation-debt/" + g.Convention.Key + "/" + slug(g.Member.Verb),
		Title:        g.Title(),
		SourceProbe:  "propagation-scorecard",
		ScoreName:    "adoption",
		Score:        fmt.Sprintf("%d/%d", g.Adopters, g.Total),
		Grade:        adoptionGrade(g.Adopters, g.Total),
		DebtName:     DebtKey,
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   "Extend " + g.Convention.Short + " to `fak " + g.Member.Verb + "`.",
		Finding:      g.Key(),
		ParentRef:    "fak propagation-scorecard",
		CurrentState: fmt.Sprintf("The propagation scorecard found that %d/%d scorecard-family cards have adopted %s, but `fak %s` (%s) does not -- the improvement has not fanned out to this sibling.",
			g.Adopters, g.Total, g.Convention.Short, g.Member.Verb, g.Member.PkgDir),
		WhyNow:         "A scoring concept proven across the family has stalled at this sibling; extending it is mechanical, removes one un-propagated gap, and spares the operator from remembering to do it by hand." + declared,
		WorkingSpine:   "Extend the existing, proven convention to `fak " + g.Member.Verb + "` -- copy how the adopters already do it; do not reinvent it.",
		WorkUnit:       "leaf",
		ExpectedSteps:  4,
		Assumptions:    []string{"The convention is already adopted by enough sibling scorecards to copy rather than redesign."},
		ConfusionRisks: []string{"Do not batch multiple laggard members into one worker issue."},
		Coordination:   []string{"One generated issue owns one scorecard member and convention key."},
		Trigger:        "Propagation scorecard reports missing convention `" + g.Convention.Key + "` for member `" + g.Member.Verb + "`.",
		BatchPolicy:    "One issue per propagation-debt convention/member key; reruns update by stable marker.",
		InScope:        g.inScopeHint(),
		OutOfScope:     "Do not change this card's KPIs, scores, or grade thresholds, retune the kernel, or touch other family members in the same change.",
		DoneCondition:  "A re-run of `fak propagation-scorecard --json` no longer reports `fak " + g.Member.Verb + "` as missing " + g.Convention.Short + " (the `" + g.Key() + "` gap is gone).",
		Witness:        g.witnessHint(),
		AcceptanceGate: "go test ./internal/propagationscore ./" + g.Member.PkgDir + " && go build ./...",
		Lane:           "",
		Paths:          []string{g.Member.PkgDir + "/**", g.Member.CmdFile},
		Labels:         []string{"propagation-debt", "scorecard"},
		BoundaryNotes:  []string{"Public scorecard-family convention only; no private or lab-local evidence."},
		ClosureBinding: "Resolving commit cites `#N` in the subject and carries a `(fak <leaf>)` trailer for the member's lane.",
	}
}

// ActionItems maps a set of gaps onto dogfoodissues.ActionItems, the input to the backlog bridge.
func ActionItems(gaps []Gap, evidencePath string) []dogfoodissues.ActionItem {
	items := make([]dogfoodissues.ActionItem, 0, len(gaps))
	for _, g := range gaps {
		items = append(items, g.ToActionItem(evidencePath))
	}
	return items
}

func adoptionGrade(adopters, total int) string {
	if total == 0 {
		return "F"
	}
	return scoreGrade(100.0 * float64(adopters) / float64(total))
}

func scoreGrade(s float64) string {
	switch {
	case s >= 90:
		return "A"
	case s >= 80:
		return "B"
	case s >= 70:
		return "C"
	case s >= 60:
		return "D"
	default:
		return "F"
	}
}

func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "member"
	}
	return out
}
