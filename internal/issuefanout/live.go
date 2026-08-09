// Live filing (#2531): the decision layer behind `fak issue fanout --live`.
// It follows the gh-touching creator pattern (dogfoodissues, learningdebt):
// plan offline, shell to gh only behind the explicit flag, and dedupe by
// marker key FIRST — a candidate whose `fanout-<leaf>-<slug>` key already
// appears in any existing issue body (open or closed) is skipped, so a rerun
// files zero and spams nothing. That batch policy is what makes automated
// filing safe.
//
// This file keeps the leaf's purity contract: it composes gh argv and decides
// file-vs-skip, but never executes a subprocess — the caller (the verb in
// cmd/fak) injects the Runner that actually runs gh, and tests inject fakes.

package issuefanout

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

// LiveSchema identifies the machine-readable live-filing result.
const LiveSchema = "fak.issue-fanout-live.v1"

// DefaultDedupeCap bounds the existing-issue scan a --live run dedupes
// against. A bounded scan is required: an uncapped tracker walk is the other
// half of the spam failure mode the marker-key contract exists to prevent.
const DefaultDedupeCap = 300

var liveIssueURLRE = regexp.MustCompile(`https?://\S+/issues/([0-9]+)`)

// Issue is the subset of a `gh issue list --json number,body` row the
// marker-key dedupe scan reads.
type Issue struct {
	Number int    `json:"number"`
	Body   string `json:"body"`
}

// Runner executes one gh invocation and reports its stdout, stderr, and an ok
// flag (true when the process exited 0). The leaf never supplies a default:
// the effectful runner lives with the caller.
type Runner func(args []string) (stdout, stderr string, ok bool)

// LiveOptions tunes one live filing run.
type LiveOptions struct {
	Repo      string // owner/repo for gh; "" = current repo
	DedupeCap int    // bounded existing-issue scan size (<=0 = DefaultDedupeCap)
	Runner    Runner // required: the gh executor
}

// FileRow is one per-candidate outcome of a live run.
type FileRow struct {
	Key    string `json:"key"`
	Title  string `json:"title"`
	Action string `json:"action"` // "filed" | "skipped" | "failed"
	Number *int   `json:"number,omitempty"`
	URL    string `json:"url,omitempty"`
	SeenIn *int   `json:"seen_in,omitempty"` // issue whose body already carries the marker key
	Reason string `json:"reason,omitempty"`
}

// LiveResult is the machine-readable fold of one live filing run.
type LiveResult struct {
	Schema    string    `json:"schema"`
	Input     Input     `json:"input"`
	DedupeCap int       `json:"dedupe_cap"`
	Scanned   int       `json:"scanned"`
	Filed     int       `json:"filed"`
	Skipped   int       `json:"skipped"`
	Failed    int       `json:"failed"`
	Rows      []FileRow `json:"rows"`
}

// MilestoneForGeneration maps a candidate's gen/* stream to the roadmap
// milestone issues are filed under, mirroring docs/generation.md (the same
// table internal/devindex serves). Unknown streams map to "" (no milestone).
func MilestoneForGeneration(generation string) string {
	switch strings.TrimPrefix(strings.ToLower(strings.TrimSpace(generation)), "gen/") {
	case "now":
		return "Generation G0 - Now / Immediate"
	case "next":
		return "Generation G1 - Next Gen"
	case "second-next":
		return "Generation G2 - Second Next Gen"
	case "future":
		return "Generation G3 - Future"
	}
	return ""
}

// LiveLabels is the label set filed with a candidate: its planned labels
// (fanout + area) plus its generation and priority streams, deduplicated in
// order.
func LiveLabels(c issuecontract.Candidate) []string {
	seen := map[string]bool{}
	var out []string
	for _, label := range append(append([]string{}, c.Labels...), c.Generation, c.Priority) {
		label = strings.TrimSpace(label)
		if label == "" || seen[label] {
			continue
		}
		seen[label] = true
		out = append(out, label)
	}
	return out
}

// LiveBody renders the marker-stamped issue body for one candidate. It leads
// with the dedupe marker (a member of issuecontract's fak-*-key grammar whose
// value is the candidate's `fanout-<leaf>-<slug>` key), then the standard
// contract sections, so the filed issue re-parses as a dispatchable candidate
// and a rerun's substring scan finds the key.
func LiveBody(c issuecontract.Candidate) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-issuefanout-key: %s -->\n\n", c.Key)
	section := func(title, body string) {
		if body = strings.TrimSpace(body); body != "" {
			fmt.Fprintf(&b, "## %s\n\n%s\n\n", title, body)
		}
	}
	list := func(title string, items []string) {
		if len(items) == 0 {
			return
		}
		fmt.Fprintf(&b, "## %s\n\n", title)
		for _, item := range items {
			fmt.Fprintf(&b, "- %s\n", item)
		}
		b.WriteString("\n")
	}
	section("Generation stream", c.Generation)
	section("Parent context", c.ParentRef)
	section("Current state", c.CurrentState)
	section("Why this is next", c.WhyNow)
	section("Working spine", c.WorkingSpine)
	section("In scope", c.InScope)
	section("Out of scope", c.OutOfScope)
	section("Root point", c.RootPoint)
	section("Origin signal", c.OriginSignal)
	section("Prevents recurrence", c.PreventsRecurrence)
	section("Done condition", c.DoneCondition)
	section("Witness", c.Witness)
	section("Acceptance gate", c.AcceptanceGate)
	section("Work estimate", c.WorkEstimate)
	section("Overall completion contribution", c.ScopeContribution)
	section("Completion standard", c.CompletionStandard)
	section("Target operating envelope", c.TargetEnvelope)
	section("Witnessed operating envelope", c.WitnessedEnvelope)
	section("Closure binding", c.ClosureBinding)
	section("Lane", c.Lane)
	section("Work unit", c.WorkUnit)
	if c.ExpectedSteps > 0 {
		section("Expected steps", strconv.Itoa(c.ExpectedSteps))
	}
	list("Assumptions", c.Assumptions)
	list("Confusion risks", c.ConfusionRisks)
	list("Coordination", c.Coordination)
	section("Trigger", c.Trigger)
	section("Batch policy", c.BatchPolicy)
	if len(c.Paths) > 0 {
		b.WriteString("## Likely files\n\n")
		for _, p := range c.Paths {
			fmt.Fprintf(&b, "- `%s`\n", p)
		}
		b.WriteString("\n")
	}
	b.WriteString("Filed by `fak issue fanout --live`; the marker key above is the rerun dedupe contract.\n")
	return b.String()
}

// ListExistingArgs composes the gh argv for the bounded dedupe scan: every
// issue, open or closed, capped at the dedupe cap.
func ListExistingArgs(repo string, dedupeCap int) []string {
	if dedupeCap <= 0 {
		dedupeCap = DefaultDedupeCap
	}
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(dedupeCap), "--json", "number,body"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	return args
}

// FileLive files each planned candidate whose marker key is unseen among the
// existing issue bodies, and skips the rest. It refuses to run without an
// injected Runner — the leaf owns no subprocess.
func FileLive(plan Plan, existing []Issue, opt LiveOptions) (LiveResult, error) {
	if opt.Runner == nil {
		return LiveResult{}, refusef("issuefanout: live filing needs a gh Runner — set LiveOptions.Runner to the gh executor (the verb wires the real gh; tests wire a fake)")
	}
	dedupeCap := opt.DedupeCap
	if dedupeCap <= 0 {
		dedupeCap = DefaultDedupeCap
	}
	res := LiveResult{Schema: LiveSchema, Input: plan.Input, DedupeCap: dedupeCap, Scanned: len(existing)}
	for _, c := range plan.Candidates {
		if plan.Input.ParentIssue <= 0 || plan.Input.ParentBaseline <= 0 {
			return res, refusef("issuefanout: live filing requires --parent-issue and --parent-baseline-points — set both flags to the parent issue number and its declared scope baseline")
		}
		row := FileRow{Key: c.Key, Title: c.Title}
		if n, seen := seenIn(existing, c.Key); seen {
			row.Action = "skipped"
			row.SeenIn = &n
			row.Reason = fmt.Sprintf("marker key already in issue #%d", n)
			res.Skipped++
			res.Rows = append(res.Rows, row)
			continue
		}
		// Build already reviewed the plan offline; a live run re-reviews under the
		// armed contract (agent-context + noise-control fields required) so a
		// candidate the strict gate would refuse is reported, never filed.
		if r := issuecontract.ReviewCandidate(c, issuecontract.Options{Live: true, DedupeChecked: true, DedupeCap: dedupeCap}); !r.OK || r.Dispatchability != issuecontract.Dispatchable {
			row.Action = "failed"
			row.Reason = "issue contract (live-armed): " + strings.Join(r.Reasons, ", ")
			res.Failed++
			res.Rows = append(res.Rows, row)
			continue
		}
		body := LiveBody(c)
		args := []string{"issue", "create", "--title", c.Title, "--body", body}
		for _, label := range LiveLabels(c) {
			args = append(args, "--label", label)
		}
		if m := MilestoneForGeneration(c.Generation); m != "" {
			args = append(args, "--milestone", m)
		}
		if opt.Repo != "" {
			args = append(args, "--repo", opt.Repo)
		}
		stdout, stderr, ok := opt.Runner(args)
		if !ok {
			row.Action = "failed"
			row.Reason = strings.TrimSpace(stderr)
			res.Failed++
			res.Rows = append(res.Rows, row)
			continue
		}
		n, url, parsed := createdIssue(stdout)
		if !parsed {
			row.Action = "failed"
			row.Reason = "gh issue create exited 0 but printed no issue URL"
			res.Failed++
			res.Rows = append(res.Rows, row)
			continue
		}
		row.Action = "filed"
		row.Number = &n
		row.URL = url
		res.Filed++
		res.Rows = append(res.Rows, row)
		// The fresh body carries the key, so a same-batch duplicate also dedupes.
		existing = append(existing, Issue{Number: n, Body: body})
	}
	return res, nil
}

// seenIn returns the first existing issue whose body carries the marker key.
// The scan is a plain substring match so hand-filed issues that mention the
// key in prose dedupe the same as marker-stamped ones.
func seenIn(existing []Issue, key string) (int, bool) {
	for _, issue := range existing {
		if strings.Contains(issue.Body, key) {
			return issue.Number, true
		}
	}
	return 0, false
}

// createdIssue parses the URL gh prints after `gh issue create`.
func createdIssue(stdout string) (number int, url string, ok bool) {
	m := liveIssueURLRE.FindStringSubmatch(strings.TrimSpace(stdout))
	if m == nil {
		return 0, "", false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, "", false
	}
	return n, m[0], true
}

// RenderLive prints the live result for a human: the filed/skipped fold the
// witness contract asks for, one line per candidate.
func RenderLive(r LiveResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fanout --live: filed %d, skipped %d, failed %d (scanned %d existing, dedupe cap %d)\n",
		r.Filed, r.Skipped, r.Failed, r.Scanned, r.DedupeCap)
	for _, row := range r.Rows {
		switch row.Action {
		case "filed":
			fmt.Fprintf(&b, "  [filed  ] #%d %s\n", *row.Number, row.Key)
		case "skipped":
			fmt.Fprintf(&b, "  [skipped] %s (%s)\n", row.Key, row.Reason)
		default:
			fmt.Fprintf(&b, "  [failed ] %s (%s)\n", row.Key, row.Reason)
		}
	}
	if r.Filed == 0 && r.Failed == 0 && r.Skipped > 0 {
		b.WriteString("rerun clean: every candidate's marker key is already on the tracker\n")
	}
	return b.String()
}
