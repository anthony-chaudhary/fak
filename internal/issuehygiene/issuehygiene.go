// Package issuehygiene is the pure KPI core behind `fak score issue-hygiene` --
// the deterministic scorecard that grades how well the DEFAULT GitHub issue
// surface is CREATED and TAGGED, folded into one issue_hygiene_debt integer.
//
// WHY THIS CARD EXISTS. Issue creation in this repo is a fan of independent
// producers (fak issue create, dogfood, complain, maturity route, idea-scout,
// the signal bots) that all converge on `gh issue create`. Creation is
// deliberately fast and structurally UNGATED: the issue contract
// (internal/issuecontract), the near-dup index (internal/issuededup) and the
// class/priority tagging (tools/issue_lane_router.py, tools/issue_triage.py)
// are all POST-HOC audit surfaces, not wired into the create path. The result
// is a backlog where an unknown fraction of issues are not pickup-ready: no
// class:* label (invisible to the dev-leaves / infra / front-door views), no
// priority/P? label (ranks at the default weight so it can't be ordered for
// pickup), a body missing the worker-ready sections a fresh agent needs, or a
// silent duplicate of work already filed. Each of those is wasted pickup
// attention. This card MEASURES that waste as a debt count so a
// continuous-improvement loop can drive it down, and so "creation and tagging"
// stops being an unscored surface.
//
// THE MEASUREMENT. Score grades the set of OPEN issues against the same axes
// the downstream gates already own -- it is the fold of those scattered signals
// into the scorecard family's control-pane envelope, nothing the gates don't
// already assert. The HARD debt axes (each defect is one concrete issue to fix):
//
//   - class_coverage:        a dispatchable issue with no class:* label.
//   - priority_coverage:     a dispatchable issue with no priority/P? label.
//   - contract_completeness: a dispatchable issue whose body is missing the
//     worker-ready sections (mirrors issuepolicy.missingRequiredIssueSections).
//   - dedupe_integrity:      a dispatchable issue that is a near-duplicate of a
//     lower-numbered open issue (via internal/issuededup).
//   - leaf_shape:            an epic with no linked children (nothing to
//     dispatch under it).
//
// The SOFT axes (advisory, never counted, never gate -- the family anti-gaming
// rule) surface kind/area coverage, the triage backlog depth, and staleness.
//
// DETERMINISM. Score is a pure function of (issues, nowUnix): two callers with
// the same backlog snapshot and reference clock produce byte-identical output.
// The impure part -- fetching the live backlog via `gh` -- lives in the cmd
// shell, exactly like internal/maturity's issue bridge. "dispatchable" mirrors
// the ready-leaves view exclusions (not an epic, not blocked-by-human, not a
// triage/provenance label) because those are the issues whose creation+tagging
// quality actually decides whether the next pickup is one obvious choice.
package issuehygiene

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuededup"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
	"strconv"
)

// Parse decodes cached `gh issue list --json
// number,title,body,labels,assignees,state,createdAt,updatedAt` output into the
// Issue slice Score grades. It tolerates a UTF-8 BOM (PowerShell redirection
// stamps one), keeping the whole path offline-safe: a cached snapshot in, a
// verdict out, no gh anywhere in the core.
func Parse(b []byte) ([]Issue, error) {
	trimmed := stripBOM(string(b))
	var issues []Issue
	if err := json.Unmarshal([]byte(trimmed), &issues); err != nil {
		return nil, fmt.Errorf("not a gh issue list --json array: %w", err)
	}
	return issues, nil
}

// Schema is the control-pane envelope id for this card.
const Schema = "fak-issue-hygiene-scorecard/1"

// DebtKey is the headline integer the control-pane folds. Keep it in sync with
// any control-pane roster row that lands this card (tools/scorecard_control_pane.py).
const DebtKey = "issue_hygiene_debt"

// staleDays mirrors tools/issue_triage.py STALE_DAYS (60): a dispatchable,
// unassigned issue untouched this long is advisory-stale.
const staleDays = 60

// dupThreshold is the near-dup cosine floor. It matches issuededup.DefaultThreshold
// (0.80) via issuededup.Check's threshold<=0 default -- we pass 0 to inherit it,
// so the scorecard and the write-time gate never drift on what counts as a dup.
const dupThreshold = 0

// Label is one gh label ({name}). Assignee is one gh assignee ({login}).
type Label struct {
	Name string `json:"name"`
}

// Assignee is one gh assignee login.
type Assignee struct {
	Login string `json:"login"`
}

// Issue is one row of `gh issue list --json
// number,title,body,labels,assignees,createdAt,updatedAt,state` -- the only
// shape the core reads, so the whole path stays offline-safe (a cached snapshot
// in, a verdict out). Timestamps are RFC3339 strings kept verbatim; staleness is
// computed from UpdatedAt against the injected reference clock so the core needs
// no clock of its own.
type Issue struct {
	Number    int        `json:"number"`
	Title     string     `json:"title"`
	Body      string     `json:"body"`
	Labels    []Label    `json:"labels"`
	Assignees []Assignee `json:"assignees"`
	State     string     `json:"state"`
	CreatedAt string     `json:"createdAt"`
	UpdatedAt string     `json:"updatedAt"`
}

// labelSet lowercases the issue's labels into a membership set.
func (is Issue) labelSet() map[string]bool {
	out := make(map[string]bool, len(is.Labels))
	for _, l := range is.Labels {
		out[strings.ToLower(strings.TrimSpace(l.Name))] = true
	}
	return out
}

// hasPrefix reports whether any label starts with prefix (lowercased).
func hasLabelPrefix(labels map[string]bool, prefix string) bool {
	for name := range labels {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func hasAnyLabel(labels map[string]bool, names ...string) bool {
	for _, n := range names {
		if labels[n] {
			return true
		}
	}
	return false
}

// triageProvenanceLabels are the labels that HIDE an issue from every dispatch
// view (the ready-leaves exclusion set + the dispatch router TriageOnlyLabels).
// An issue carrying any of these is un-triaged by design, so its creation/tagging
// quality is not yet the pickup surface -- it is excluded from the dispatchable
// set and folded into the soft triage_backlog signal instead.
var triageProvenanceLabels = []string{
	"idea-scout", "needs-triage", "triage-only", "triage_only",
	"guard-complaint", "research", "needs-scope",
}

// kindLabels / areaLabels mirror tools/issue_triage.py KIND / AREA -- the soft
// coverage axes (their absence is a nudge, not debt: dispatch does not require them).
var kindLabels = []string{
	"bug", "enhancement", "documentation", "question", "performance", "build", "research",
}
var areaLabels = []string{
	"agentic-serving", "trust-floor", "model-arch", "compute", "gpu", "model",
	"substrate", "loader", "security", "dispatch", "rsi", "licensing",
}

func (is Issue) isEpic() bool {
	if is.labelSet()["epic"] {
		return true
	}
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(is.Title)), "epic(")
}

func (is Issue) isTriageProvenance() bool {
	return hasAnyLabel(is.labelSet(), triageProvenanceLabels...)
}

func (is Issue) isBlockedByHuman() bool {
	return is.labelSet()["blocked-by-human"]
}

// isDispatchable mirrors the ready-leaves view: an open issue that is not an
// epic, not blocked on a human, and not carrying a triage/provenance label. This
// is the pickup-candidate set the HARD axes grade.
func (is Issue) isDispatchable() bool {
	if is.State != "" && !strings.EqualFold(is.State, "open") {
		return false
	}
	return !is.isEpic() && !is.isTriageProvenance() && !is.isBlockedByHuman()
}

func (is Issue) assigned() bool { return len(is.Assignees) > 0 }

// requiredSections mirrors issuepolicy.missingRequiredIssueSections's heading
// half: the worker-ready body a fresh-context agent needs. We can only see the
// rendered body (not the contract's parsed scope fields), so we grade heading
// PRESENCE -- the structural signal the create path leaves ungated.
var requiredSections = []struct {
	field   string
	anyOf   []string
	require int // number of the anyOf headings that must be present (1, or 2 for scope)
}{
	{"current_state", []string{"Current state"}, 1},
	{"scope", []string{"Scope"}, 1}, // OR (In scope AND Out of scope), handled specially
	{"done_condition", []string{"Done condition", "Done condition / witness"}, 1},
	{"witness", []string{"Witness", "Done condition / witness"}, 1},
	{"likely_files", []string{"Likely files", "Path hints", "Paths", "Files"}, 1},
}

// headingRE matches a markdown/bolded heading line carrying phrase (case-insensitive):
// optional leading '#'s or '*'s, the phrase, then end / ':' / closing '*'.
func headingRE(phrase string) *regexp.Regexp {
	return regexp.MustCompile(`(?im)^\s{0,3}[#*]{0,6}\s*` + regexp.QuoteMeta(phrase) + `\b`)
}

var headingCache = map[string]*regexp.Regexp{}

func hasHeading(body, phrase string) bool {
	re, ok := headingCache[phrase]
	if !ok {
		re = headingRE(phrase)
		headingCache[phrase] = re
	}
	return re.MatchString(body)
}

// missingSections returns the required worker-ready sections absent from body.
func missingSections(body string) []string {
	var missing []string
	for _, s := range requiredSections {
		if s.field == "scope" {
			if hasHeading(body, "Scope") || ((hasHeading(body, "Core through-line") || hasHeading(body, "In scope")) && (hasHeading(body, "Gold-plating boundary") || hasHeading(body, "Out of scope"))) {
				continue
			}
			missing = append(missing, "scope")
			continue
		}
		present := false
		for _, h := range s.anyOf {
			if hasHeading(body, h) {
				present = true
				break
			}
		}
		if !present {
			missing = append(missing, s.field)
		}
	}
	return missing
}

// childRefRE matches a task-list child (- [ ] #123) or an explicit child/blocks
// reference an epic uses to enumerate the leaves it decomposes into.
var childRefRE = regexp.MustCompile(`(?im)(^\s*[-*]\s*\[[ xX]\]\s*#\d+|(?:child|children|blocks|depends on|subtask)[^\n]*#\d+)`)

func hasChildRefs(body string) bool { return childRefRE.MatchString(body) }

// Reference is the injected clock: NowUnix is epoch seconds used only for the
// soft staleness axis, so the core stays a pure function of its inputs.
type Reference struct {
	NowUnix int64
}

// pct returns 100*ok/total, or 100 when total==0 (an empty axis is clean).
func pct(ok, total int) float64 {
	if total == 0 {
		return 100
	}
	return 100 * float64(ok) / float64(total)
}

// Score grades the open backlog into the control-pane Payload. It is pure:
// identical (issues, ref) in -> identical Payload out.
func Score(issues []Issue, ref Reference) scorecard.Payload {
	// The dispatchable pickup-candidate set: the issues whose creation+tagging
	// quality decides whether the next pickup is one obvious, ready choice.
	var dispatchable []Issue
	openCount, triageBacklog := 0, 0
	for _, is := range issues {
		if is.State != "" && !strings.EqualFold(is.State, "open") {
			continue
		}
		openCount++
		if is.isTriageProvenance() {
			triageBacklog++
		}
		if is.isDispatchable() {
			dispatchable = append(dispatchable, is)
		}
	}

	// HARD axis 1+2+3+6+8: per-dispatchable tagging + body checks in one pass.
	classOK, prioOK, contractOK, kindAreaOK, pickupReady := 0, 0, 0, 0, 0
	var classDefects, prioDefects, contractDefects []string
	var kindAreaSoft, staleSoft []string
	for _, is := range dispatchable {
		labels := is.labelSet()
		hasClass := hasLabelPrefix(labels, "class:")
		hasPrio := hasLabelPrefix(labels, "priority/")
		miss := missingSections(is.Body)
		hasKind := hasAnyLabel(labels, kindLabels...)
		hasArea := hasAnyLabel(labels, areaLabels...)

		if hasClass {
			classOK++
		} else {
			classDefects = append(classDefects, ref0(is, "no class:* label (invisible to dev-leaves / infra / front-door views)"))
		}
		if hasPrio {
			prioOK++
		} else {
			prioDefects = append(prioDefects, ref0(is, "no priority/P? label (ranks at default weight; cannot be ordered for pickup)"))
		}
		if len(miss) == 0 {
			contractOK++
		} else {
			contractDefects = append(contractDefects, ref0(is, "body missing worker-ready section(s): "+strings.Join(miss, ", ")))
		}
		if hasKind && hasArea {
			kindAreaOK++
		} else {
			var gaps []string
			if !hasKind {
				gaps = append(gaps, "kind")
			}
			if !hasArea {
				gaps = append(gaps, "area")
			}
			kindAreaSoft = append(kindAreaSoft, ref0(is, "no "+strings.Join(gaps, "/")+" label"))
		}
		if hasClass && hasPrio && len(miss) == 0 {
			pickupReady++
		}
		if !is.assigned() && staleDaysSince(is.UpdatedAt, ref.NowUnix) > staleDays {
			staleSoft = append(staleSoft, ref0(is, "open+unassigned, untouched > "+strconv.Itoa(staleDays)+"d"))
		}
	}

	// HARD axis 4: near-duplicate dispatchable issues (a higher-numbered issue
	// that near-matches a lower-numbered OPEN issue = the redundant one to fix).
	dupDefects := dedupeDefects(issues, dispatchable)
	// pickupReady already counted class+priority+contract; a dup is not
	// pickup-ready either, but we keep pickup_ready to the per-issue axes above
	// so the two signals stay independent.

	// HARD axis 5: epics that decompose into nothing.
	var leafDefects []string
	epicCount := 0
	for _, is := range issues {
		if is.State != "" && !strings.EqualFold(is.State, "open") {
			continue
		}
		if !is.isEpic() {
			continue
		}
		epicCount++
		if !hasChildRefs(is.Body) {
			leafDefects = append(leafDefects, ref0(is, "epic has no linked children (nothing to dispatch under it)"))
		}
	}

	nDisp := len(dispatchable)
	kpis := []scorecard.KPI{
		{
			Key: "class_coverage", Group: "tagging",
			Score:   pct(classOK, nDisp),
			Detail:  plural(len(classDefects), "dispatchable issue") + " missing a class:* label",
			Defects: classDefects,
		},
		{
			Key: "priority_coverage", Group: "tagging",
			Score:   pct(prioOK, nDisp),
			Detail:  plural(len(prioDefects), "dispatchable issue") + " missing a priority/P? label",
			Defects: prioDefects,
		},
		{
			Key: "contract_completeness", Group: "creation",
			Score:   pct(contractOK, nDisp),
			Detail:  plural(len(contractDefects), "dispatchable issue") + " missing worker-ready body sections",
			Defects: contractDefects,
		},
		{
			Key: "dedupe_integrity", Group: "waste",
			Score:   pct(nDisp-len(dupDefects), nDisp),
			Detail:  plural(len(dupDefects), "dispatchable issue") + " is a near-duplicate of an earlier open issue",
			Defects: dupDefects,
		},
		{
			Key: "leaf_shape", Group: "creation",
			Score:   pct(epicCount-len(leafDefects), epicCount),
			Detail:  plural(len(leafDefects), "epic") + " with no linked children",
			Defects: leafDefects,
		},
		{
			Key: "kind_area_coverage", Group: "tagging",
			Score:  pct(kindAreaOK, nDisp),
			Detail: plural(len(kindAreaSoft), "dispatchable issue") + " missing a kind/area label (advisory)",
			Soft:   kindAreaSoft,
		},
		{
			Key: "triage_backlog", Group: "waste",
			Score:  triageBacklogScore(triageBacklog),
			Detail: strconv.Itoa(triageBacklog) + " open issue(s) waiting in the triage inbox (advisory)",
			Soft:   triageBacklogSoft(triageBacklog),
		},
		{
			Key: "staleness", Group: "waste",
			Score:  pct(nDisp-len(staleSoft), nDisp),
			Detail: plural(len(staleSoft), "dispatchable issue") + " stale (unassigned, untouched > " + strconv.Itoa(staleDays) + "d) (advisory)",
			Soft:   staleSoft,
		},
	}

	debt := len(classDefects) + len(prioDefects) + len(contractDefects) + len(dupDefects) + len(leafDefects)
	pickupPct := pct(pickupReady, nDisp)

	findingClean := "every dispatchable issue is class+priority tagged, contract-complete, de-duplicated -- the backlog is pickup-ready"
	nextClean := "hold -- re-run `fak score issue-hygiene --live` after the next batch of issues lands"
	findingDebt := plural(debt, "issue-hygiene defect") + ": " + strconv.Itoa(nDisp-pickupReady) + " of " + strconv.Itoa(nDisp) + " dispatchable issues are not pickup-ready (untagged, contract-incomplete, or duplicated)"
	nextDebt := "tag class+priority, complete the worker-ready body, or close the duplicate on the offending issues; run `python tools/issue_lane_router.py --apply-labels --apply-labels-write` to backfill class:*"

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         findingDebt,
		FindingClean:    findingClean,
		NextAction:      nextDebt,
		NextActionClean: nextClean,
		ExtraCorpus: map[string]any{
			"open_issues":      openCount,
			"dispatchable":     nDisp,
			"pickup_ready":     pickupReady,
			"pickup_ready_pct": scorecard.Round1(pickupPct),
			"triage_backlog":   triageBacklog,
			"epics":            epicCount,
		},
	})
	return p
}

// dedupeDefects flags each dispatchable issue that near-duplicates a
// LOWER-numbered open issue (the earlier issue is the canonical one; the later
// is the redundant defect). It builds one issuededup.Index over the whole open
// backlog and checks each dispatchable issue against it, keeping only matches to
// a smaller issue number so a pair is counted once, on the newer twin.
func dedupeDefects(all []Issue, dispatchable []Issue) []string {
	backlog := make([]issuededup.BacklogIssue, 0, len(all))
	for _, is := range all {
		if is.State != "" && !strings.EqualFold(is.State, "open") {
			continue
		}
		backlog = append(backlog, issuededup.BacklogIssue{Number: is.Number, Title: is.Title, Body: is.Body})
	}
	ix := issuededup.NewIndex(backlog)
	if ix.Len() == 0 {
		return nil
	}
	var defects []string
	for _, is := range dispatchable {
		verdicts := ix.Check(issuededup.Candidate{Number: is.Number, Title: is.Title, Body: is.Body}, 0, dupThreshold)
		for _, v := range verdicts {
			if v.IssueNumber < is.Number {
				defects = append(defects, ref0(is, "near-duplicate of #"+strconv.Itoa(v.IssueNumber)+" ("+sim(v.Similarity)+" "+v.MatchedOn+"); close or merge"))
				break // one defect per redundant issue
			}
		}
	}
	sort.Strings(defects)
	return defects
}

// ref0 prefixes a defect with the issue's #number and a trimmed title.
func ref0(is Issue, msg string) string {
	title := strings.TrimSpace(is.Title)
	if len(title) > 60 {
		title = title[:57] + "..."
	}
	return "#" + strconv.Itoa(is.Number) + " \"" + title + "\" -- " + msg
}
