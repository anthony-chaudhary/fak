package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontractrepair"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// issue-repair dispositions: the closed set of what the consumer will do with a
// repairable row. Only auto-apply ever writes, and only under --live.
const (
	repairDispAutoApply   = "auto-apply"      // template rows with a provably-safe merge
	repairDispProposeOnly = "propose-only"    // fix is computed/scaffolded but a human/agent applies it
	repairDispRefuse      = "refuse"          // no canonical fix (boundary/content-loss)
	repairDispDefer       = "defer-to-phase3" // needs decomposition, not a field edit
	repairSchema          = "fak.issue-repair.v1"
)

// repairKindDisposition maps every non-template manifest kind to its fixed
// disposition. template is handled dynamically (its safety is per-body), so it
// is deliberately absent here. Kinds come only from issuecontractrepair's
// review-reason vocabulary (split/scope/route/noise/private/template/other) —
// the router's live-state kinds (duplicate/human/decide) never reach this pass.
var repairKindDisposition = map[string]string{
	"scope":   repairDispProposeOnly,
	"noise":   repairDispProposeOnly,
	"route":   repairDispProposeOnly,
	"other":   repairDispProposeOnly,
	"private": repairDispRefuse,
	"split":   repairDispDefer,
}

type issueRepairRow struct {
	Number       int                               `json:"number"`
	Title        string                            `json:"title"`
	Kind         string                            `json:"kind"`
	Kinds        []string                          `json:"kinds,omitempty"`
	Disposition  string                            `json:"disposition"`
	Reasons      []string                          `json:"reasons,omitempty"`
	NextAction   string                            `json:"next_action,omitempty"`
	MissingField []issuecontractrepair.FieldPrompt `json:"missing_fields,omitempty"`
	ProposedLane *string                           `json:"proposed_lane,omitempty"`
	ProposedBody string                            `json:"proposed_body,omitempty"` // template auto-apply rows
	ProposedArgs []string                          `json:"proposed_args,omitempty"` // gh argv a live apply would run
	Unsafe       string                            `json:"unsafe,omitempty"`        // template merge fail-closed reason
	Applied      bool                              `json:"applied"`                 // written under --live
	URL          string                            `json:"url,omitempty"`
	Error        string                            `json:"error,omitempty"`
}

type issueRepairCounts struct {
	Examined    int            `json:"examined"`
	NeedsRepair int            `json:"needs_repair"`
	AutoApply   int            `json:"auto_apply"`
	Applied     int            `json:"applied"`
	ProposeOnly int            `json:"propose_only"`
	Refuse      int            `json:"refuse"`
	Defer       int            `json:"defer"`
	Unsafe      int            `json:"unsafe"`
	Failed      int            `json:"failed"`
	ByKind      map[string]int `json:"by_kind"`
}

type issueRepairResult struct {
	Schema    string            `json:"schema"`
	Live      bool              `json:"live"`
	Workspace string            `json:"workspace"`
	Limit     int               `json:"limit"`
	MaxApply  int               `json:"max_apply"`
	Counts    issueRepairCounts `json:"counts"`
	Rows      []issueRepairRow  `json:"rows"`
}

func runIssueRepair(stdout, stderr io.Writer, argv []string) int {
	return runIssueRepairWith(stdout, stderr, argv, nil, nil)
}

// runIssueRepairWith consumes the read-only issue-contract-repair manifest and
// turns it into dispositions. It defaults to a DRY-RUN plan (read-only, never
// touches GitHub); --live arms writes but writes ONLY template rows whose
// ApplyTemplateRepair merge is provably lossless, bounded by --max-apply. Every
// other kind is propose-only / refuse / defer and is never written here.
//
// The issues param injects a fixed draft set (tests, or a pre-fetched cache);
// nil means fetch open issues via gh. The runner param injects the gh executor
// (tests), nil means the real runTaskHandoffGH — the same seam as issue edit.
func runIssueRepairWith(stdout, stderr io.Writer, argv []string, issues []issuepolicy.IssueDraft, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue repair", flag.ContinueOnError)
	fs.SetOutput(stderr)
	live := fs.Bool("live", false, "arm writes: apply safe template repairs (default: dry-run plan only)")
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON file from `gh issue list --json number,title,body,labels,url` (offline)")
	kindFilter := fs.String("kind", "", "comma-separated kinds to include (default: all)")
	issueSel := fs.String("issue", "", "restrict to a comma-separated set of issue numbers")
	limit := fs.Int("limit", 50, "cap the number of repairable rows examined")
	maxApply := fs.Int("max-apply", 10, "blast-radius fuse: refuse a --live run that would write more than N issues")
	workspace := fs.String("workspace", "", "repo root (default: resolved from cwd)")
	repo := fs.String("repo", "", "owner/name override passed through to gh")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	asMarkdown := fs.Bool("markdown", false, "emit a Markdown report")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue repair: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *asJSON && *asMarkdown {
		fmt.Fprintln(stderr, "fak-dev issue repair: pass at most one of --json or --markdown")
		return 2
	}

	root := resolveRoot(*workspace)

	// Source the drafts: injected > --from-issues file > live gh fetch.
	if issues == nil {
		if strings.TrimSpace(*fromIssues) != "" {
			raw, err := os.ReadFile(*fromIssues)
			if err != nil {
				fmt.Fprintf(stderr, "fak-dev issue repair: read --from-issues: %v\n", err)
				return 2
			}
			if err := json.Unmarshal(raw, &issues); err != nil {
				fmt.Fprintf(stderr, "fak-dev issue repair: parse --from-issues: %v\n", err)
				return 2
			}
		} else {
			fetched, err := issuecontractrepair.FetchOpenIssues(root, issuecontractrepair.DefaultCap)
			if err != nil {
				fmt.Fprintf(stderr, "fak-dev issue repair: %v\n", err)
				return 2
			}
			issues = fetched
		}
	}

	// Optional issue-number restriction, applied before the manifest so --limit
	// counts only the rows the operator asked about.
	if want := parseIssueNumberSet(*issueSel); want != nil {
		filtered := issues[:0:0]
		for _, d := range issues {
			if want[d.Number] {
				filtered = append(filtered, d)
			}
		}
		issues = filtered
	}
	draftByNumber := make(map[int]issuepolicy.IssueDraft, len(issues))
	for _, d := range issues {
		draftByNumber[d.Number] = d
	}

	manifest := issuecontractrepair.BuildManifest(root, issues, issuecontractrepair.Options{Limit: *limit})

	kinds := kindFilterSet(*kindFilter)

	result := issueRepairResult{
		Schema: repairSchema, Live: *live, Workspace: root,
		Limit: *limit, MaxApply: *maxApply,
		Counts: issueRepairCounts{
			Examined: manifest.Counts.CandidatesExamined,
			ByKind:   map[string]int{},
		},
	}

	// First pass: classify every row into a disposition (no writes yet).
	//
	// The manifest's row.Kind is the PRIMARY (highest-priority) repair kind, and
	// template is deliberately the LOWEST-ranked kind (issuecontractrepair
	// kindRank) — a corrupt-header issue is almost always ALSO scope-incomplete,
	// so its primary kind is "scope", not "template". We therefore key the
	// auto-apply on row.Kinds CONTAINING template: the generated-header merge is
	// an orthogonal, provably-lossless mechanical fix worth applying even when
	// other reasons remain. Those residual reasons stay visible in Reasons/Kinds
	// and the field scaffold below, so "auto-apply" describes only what --live
	// writes (the header), never a claim that the issue is now complete.
	for _, row := range manifest.Issues {
		if kinds != nil && !rowMatchesKindFilter(row, kinds) {
			continue
		}
		rr := issueRepairRow{
			Number: row.Number, Title: row.Title, Kind: row.Kind, Kinds: row.Kinds,
			Reasons: row.Reasons, NextAction: row.NextAction,
			ProposedLane: row.ProposedLane,
		}
		if scaffoldKind(row.Kind) {
			rr.MissingField = row.MissingFields
		}
		if repairHasKind(row.Kinds, "template") {
			apply, ok := issuepolicy.ApplyTemplateRepair(draftByNumber[row.Number])
			switch {
			case ok && apply.Safe:
				rr.Disposition = repairDispAutoApply
				rr.ProposedBody = apply.NewBody
				rr.ProposedArgs = []string{"issue", "edit", strconv.Itoa(row.Number), "--body-file", "<repaired-body>"}
			case ok:
				rr.Disposition = repairDispProposeOnly
				rr.Unsafe = apply.Unsafe
			default:
				rr.Disposition = repairDispProposeOnly
				rr.Unsafe = issuepolicy.TemplateUnsafeNoHeaderBlock
			}
		} else {
			rr.Disposition = repairKindDisposition[row.Kind]
			if rr.Disposition == "" {
				rr.Disposition = repairDispProposeOnly
			}
		}
		result.Rows = append(result.Rows, rr)
	}

	// Tally dispositions.
	autoApplyRows := make([]int, 0)
	for i := range result.Rows {
		rr := &result.Rows[i]
		result.Counts.NeedsRepair++
		result.Counts.ByKind[rr.Kind]++
		switch rr.Disposition {
		case repairDispAutoApply:
			result.Counts.AutoApply++
			autoApplyRows = append(autoApplyRows, i)
		case repairDispRefuse:
			result.Counts.Refuse++
		case repairDispDefer:
			result.Counts.Defer++
		default:
			result.Counts.ProposeOnly++
		}
		if rr.Unsafe != "" {
			result.Counts.Unsafe++
		}
	}

	// Live apply: only auto-apply (safe template) rows, fused by --max-apply.
	exit := 0
	if *live && len(autoApplyRows) > 0 {
		if len(autoApplyRows) > *maxApply {
			fmt.Fprintf(stderr, "fak-dev issue repair: --live would write %d issues but --max-apply is %d; re-run with --max-apply %d to proceed\n",
				len(autoApplyRows), *maxApply, len(autoApplyRows))
			return 2
		}
		run := runner
		if run == nil {
			run = runTaskHandoffGH
		}
		for _, idx := range autoApplyRows {
			rr := &result.Rows[idx]
			if applyOneTemplateRepair(rr, *repo, run, stderr) {
				result.Counts.Applied++
			} else {
				result.Counts.Failed++
				exit = 1
			}
		}
	}

	switch {
	case *asJSON:
		return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue repair")
	case *asMarkdown:
		fmt.Fprint(stdout, renderIssueRepairMarkdown(result))
	default:
		fmt.Fprint(stdout, renderIssueRepairText(result))
	}
	return exit
}

// applyOneTemplateRepair writes the repaired body via a temp --body-file and the
// injected gh runner. It returns true on success and records the outcome on rr.
func applyOneTemplateRepair(rr *issueRepairRow, repo string, run issueCreateRunner, stderr io.Writer) bool {
	f, err := os.CreateTemp("", "fak-issue-repair-*.md")
	if err != nil {
		rr.Error = fmt.Sprintf("temp file: %v", err)
		return false
	}
	path := f.Name()
	defer os.Remove(path)
	if _, err := f.WriteString(rr.ProposedBody); err != nil {
		f.Close()
		rr.Error = fmt.Sprintf("write temp body: %v", err)
		return false
	}
	if err := f.Close(); err != nil {
		rr.Error = fmt.Sprintf("close temp body: %v", err)
		return false
	}
	args := []string{"issue", "edit", strconv.Itoa(rr.Number), "--body-file", path}
	if strings.TrimSpace(repo) != "" {
		args = append(args, "--repo", repo)
	}
	out, errOut, ok := run(args)
	if !ok {
		rr.Error = strings.TrimSpace(errOut)
		return false
	}
	rr.Applied = true
	rr.URL = strings.TrimSpace(out)
	return true
}

// parseIssueNumberSet parses "12,34" into a set; returns nil for empty input
// (meaning "no restriction").
func parseIssueNumberSet(csv string) map[int]bool {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil
	}
	set := map[int]bool{}
	for _, tok := range strings.Split(csv, ",") {
		tok = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(tok), "#"))
		if tok == "" {
			continue
		}
		if n, err := strconv.Atoi(tok); err == nil && n > 0 {
			set[n] = true
		}
	}
	if len(set) == 0 {
		return nil
	}
	return set
}

// repairHasKind reports whether kinds contains want.
func repairHasKind(kinds []string, want string) bool {
	for _, k := range kinds {
		if k == want {
			return true
		}
	}
	return false
}

// rowMatchesKindFilter reports whether ANY of a row's kinds (primary or
// secondary) is selected — so `--kind template` surfaces a scope-primary row
// that also carries a template problem, matching the auto-apply keying.
func rowMatchesKindFilter(row issuecontractrepair.RepairRow, sel map[string]bool) bool {
	if sel[row.Kind] {
		return true
	}
	return repairHasKindSet(row.Kinds, sel)
}

func repairHasKindSet(kinds []string, sel map[string]bool) bool {
	for _, k := range kinds {
		if sel[k] {
			return true
		}
	}
	return false
}

// kindFilterSet parses a comma-separated --kind value; nil means "all kinds".
func kindFilterSet(csv string) map[string]bool {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil
	}
	set := map[string]bool{}
	for _, tok := range strings.Split(csv, ",") {
		if tok = strings.TrimSpace(tok); tok != "" {
			set[strings.ToLower(tok)] = true
		}
	}
	if len(set) == 0 {
		return nil
	}
	return set
}

func scaffoldKind(kind string) bool {
	switch kind {
	case "scope", "noise", "split", "other":
		return true
	default:
		return false
	}
}

func renderIssueRepairText(r issueRepairResult) string {
	var b strings.Builder
	mode := "dry-run plan"
	if r.Live {
		mode = "LIVE"
	}
	fmt.Fprintf(&b, "fak-dev issue repair (%s) — examined %d, needs repair %d\n",
		mode, r.Counts.Examined, r.Counts.NeedsRepair)
	fmt.Fprintf(&b, "  auto-apply %d · applied %d · propose-only %d · refuse %d · defer %d · unsafe %d · failed %d\n",
		r.Counts.AutoApply, r.Counts.Applied, r.Counts.ProposeOnly, r.Counts.Refuse, r.Counts.Defer, r.Counts.Unsafe, r.Counts.Failed)
	if !r.Live && r.Counts.AutoApply > 0 {
		fmt.Fprintf(&b, "  (%d template row(s) would be written by --live; re-run with --live --max-apply %d)\n",
			r.Counts.AutoApply, r.Counts.AutoApply)
	}
	for _, row := range r.Rows {
		line := fmt.Sprintf("  #%d [%s] %s", row.Number, row.Kind, row.Disposition)
		if row.Unsafe != "" {
			line += " (" + row.Unsafe + ")"
		}
		if row.Applied {
			line += " ✓ applied"
		}
		if row.Error != "" {
			line += " ✗ " + row.Error
		}
		b.WriteString(line + "\n")
	}
	return b.String()
}

func renderIssueRepairMarkdown(r issueRepairResult) string {
	var b strings.Builder
	mode := "dry-run plan"
	if r.Live {
		mode = "LIVE"
	}
	fmt.Fprintf(&b, "# Issue repair — %s\n\n", mode)
	fmt.Fprintf(&b, "**examined:** %d · **needs repair:** %d · **auto-apply:** %d · **applied:** %d · **propose-only:** %d · **refuse:** %d · **defer:** %d · **unsafe:** %d · **failed:** %d\n\n",
		r.Counts.Examined, r.Counts.NeedsRepair, r.Counts.AutoApply, r.Counts.Applied,
		r.Counts.ProposeOnly, r.Counts.Refuse, r.Counts.Defer, r.Counts.Unsafe, r.Counts.Failed)
	b.WriteString("> Dry-run is read-only. `--live` writes only template rows with a provably-lossless header merge, bounded by `--max-apply`. Every other kind is proposed for a human/agent, never written here.\n\n")
	kinds := make([]string, 0, len(r.Counts.ByKind))
	for k := range r.Counts.ByKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	b.WriteString("| # | kind | disposition | note |\n|---|---|---|---|\n")
	for _, row := range r.Rows {
		note := row.Unsafe
		if row.Applied {
			note = "applied"
		} else if row.Error != "" {
			note = "error: " + row.Error
		}
		fmt.Fprintf(&b, "| #%d | %s | %s | %s |\n", row.Number, row.Kind, row.Disposition, note)
	}
	return b.String()
}
