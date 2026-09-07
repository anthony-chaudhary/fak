package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const (
	DiscoverabilitySchema = "fak.issue-discoverability-audit/1"

	PlacementParallelWave   = "parallel_wave"
	PlacementSerialWave     = "serial_wave"
	PlacementSubdivideQueue = "subdivide_queue"
	PlacementTriageQueue    = "triage_queue"
)

// IssueDiscoverabilityAuditResult is the JSON output schema for discoverability audits.
type IssueDiscoverabilityAuditResult struct {
	Schema       string                    `json:"schema"`
	OK           bool                      `json:"ok"`
	Total        int                       `json:"total"`
	Dispatchable int                       `json:"dispatchable"`
	Subdivide    int                       `json:"subdivide"`
	Triage       int                       `json:"triage"`
	Serial       int                       `json:"serial"`
	Parallel     int                       `json:"parallel"`
	Issues       []IssueDiscoverabilityRow `json:"issues"`
}

// IssueDiscoverabilityRow captures the evaluation and predicted placement for one issue.
type IssueDiscoverabilityRow struct {
	Number           int      `json:"number,omitempty"`
	Title            string   `json:"title"`
	Key              string   `json:"key,omitempty"`
	URL              string   `json:"url,omitempty"`
	Dispatchable     bool     `json:"dispatchable"`
	Placement        string   `json:"placement"`
	Dispatchability  string   `json:"dispatchability"`
	Lane             string   `json:"lane,omitempty"`
	Paths            []string `json:"paths,omitempty"`
	InternalPackages []string `json:"internal_packages,omitempty"`
	ExpectedSteps    int      `json:"expected_steps"`
	Centrality       string   `json:"centrality,omitempty"`
	IsSubdivide      bool     `json:"is_subdivide"`
	IsTriage         bool     `json:"is_triage"`
	IsSerial         bool     `json:"is_serial"`
	Reasons          []string `json:"reasons,omitempty"`
	MissingSections  []string `json:"missing_sections,omitempty"`
	MissingFields    []string `json:"missing_fields,omitempty"`
	RepairActions    []string `json:"repair_actions,omitempty"`
}

func runIssueDiscoverability(stdout, stderr io.Writer, argv []string) int {
	return runIssueDiscoverabilityWith(stdout, stderr, argv, nil)
}

func runIssueDiscoverabilityWith(stdout, stderr io.Writer, argv []string, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue discoverability", flag.ContinueOnError)
	fs.SetOutput(stderr)

	title := fs.String("title", "", "draft issue title")
	body := fs.String("body", "", "draft markdown body text")
	bodyFile := fs.String("body-file", "", "path to markdown draft file")
	file := fs.String("file", "", "alias for --body-file or JSON file")
	fromIssues := fs.String("from-issues", "", "path to issues JSON file or '-' for stdin (batch audit)")
	issueNum := fs.Int("issue", 0, "GitHub issue number to fetch via gh and audit")
	repo := fs.String("repo", "", "owner/repo override (default: inferred from cwd)")
	asJSON := fs.Bool("json", false, "emit typed JSON")
	strictBornRouted := fs.Bool("strict-born-routed", false, "strict routing check (requires lane, class label, priority label)")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue discoverability: unexpected positional arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	hasIssue := *issueNum > 0
	hasFromIssues := strings.TrimSpace(*fromIssues) != ""
	hasFile := strings.TrimSpace(*file) != ""
	hasBodyFile := strings.TrimSpace(*bodyFile) != ""
	hasBody := strings.TrimSpace(*body) != ""
	hasTitle := strings.TrimSpace(*title) != ""

	if hasIssue && hasFromIssues {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: pass at most one of --issue or --from-issues")
		return 2
	}
	if hasBody && hasBodyFile {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: pass at most one of --body or --body-file")
		return 2
	}
	if hasFile && hasBodyFile {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: pass at most one of --file or --body-file")
		return 2
	}
	if hasFile && hasBody {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: pass at most one of --file or --body")
		return 2
	}
	if (hasIssue || hasFromIssues) && (hasBody || hasBodyFile) {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: cannot combine --issue/--from-issues with --body/--body-file")
		return 2
	}
	if !hasIssue && !hasFromIssues && !hasFile && !hasBodyFile && !hasBody && !hasTitle {
		fmt.Fprintln(stderr, "fak-dev issue discoverability: provide an issue to audit via --title/--body, --file, --from-issues, or --issue")
		return 2
	}

	var drafts []issuepolicy.IssueDraft

	switch {
	case hasIssue:
		if *issueNum <= 0 {
			fmt.Fprintln(stderr, "fak-dev issue discoverability: --issue must be a positive integer")
			return 2
		}
		run := runner
		if run == nil {
			run = runTaskHandoffGH
		}
		args := []string{"issue", "view", strconv.Itoa(*issueNum), "--json", "number,title,body,labels,url"}
		if strings.TrimSpace(*repo) != "" {
			args = append(args, "--repo", strings.TrimSpace(*repo))
		}
		out, errOut, ok := run(args)
		if !ok {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: fetch issue #%d: %s\n", *issueNum, strings.TrimSpace(errOut))
			return 1
		}
		var issueRow struct {
			Number int    `json:"number"`
			Title  string `json:"title"`
			Body   string `json:"body"`
			URL    string `json:"url"`
			Labels []struct {
				Name string `json:"name"`
			} `json:"labels"`
		}
		if err := json.Unmarshal([]byte(out), &issueRow); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: decode gh issue: %v\n", err)
			return 1
		}
		draft := issuepolicy.IssueDraft{
			Number: issueRow.Number,
			Title:  issueRow.Title,
			Body:   issueRow.Body,
			URL:    issueRow.URL,
		}
		for _, l := range issueRow.Labels {
			draft.Labels = append(draft.Labels, issuepolicy.IssueLabel{Name: l.Name})
		}
		drafts = []issuepolicy.IssueDraft{draft}

	case hasFromIssues:
		var raw []byte
		var err error
		if *fromIssues == "-" {
			raw, err = io.ReadAll(os.Stdin)
		} else {
			abs, absErr := filepath.Abs(*fromIssues)
			if absErr != nil {
				fmt.Fprintf(stderr, "fak-dev issue discoverability: resolve --from-issues path: %v\n", absErr)
				return 2
			}
			raw, err = os.ReadFile(abs)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: read --from-issues: %v\n", err)
			return 2
		}
		drafts, err = decodeDiscoverabilityIssues(raw)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: parse --from-issues: %v\n", err)
			return 2
		}

	case hasFile:
		abs, absErr := filepath.Abs(*file)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: resolve --file path: %v\n", absErr)
			return 2
		}
		raw, err := os.ReadFile(abs)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: read --file: %v\n", err)
			return 2
		}
		trimmed := strings.TrimSpace(string(raw))
		if strings.HasSuffix(strings.ToLower(*file), ".json") || strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
			if decoded, decErr := decodeDiscoverabilityIssues(raw); decErr == nil && len(decoded) > 0 {
				drafts = decoded
			}
		}
		if len(drafts) == 0 {
			draftTitle := strings.TrimSpace(*title)
			if draftTitle == "" {
				draftTitle = extractTitleFromMarkdown(trimmed)
			}
			drafts = []issuepolicy.IssueDraft{{
				Title: draftTitle,
				Body:  trimmed,
			}}
		}

	case hasBodyFile:
		abs, absErr := filepath.Abs(*bodyFile)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: resolve --body-file path: %v\n", absErr)
			return 2
		}
		raw, err := os.ReadFile(abs)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: read --body-file: %v\n", err)
			return 2
		}
		trimmed := strings.TrimSpace(string(raw))
		draftTitle := strings.TrimSpace(*title)
		if draftTitle == "" {
			draftTitle = extractTitleFromMarkdown(trimmed)
		}
		drafts = []issuepolicy.IssueDraft{{
			Title: draftTitle,
			Body:  trimmed,
		}}

	default:
		draftTitle := strings.TrimSpace(*title)
		if draftTitle == "" {
			draftTitle = extractTitleFromMarkdown(*body)
		}
		drafts = []issuepolicy.IssueDraft{{
			Title: draftTitle,
			Body:  *body,
		}}
	}

	opts := issuepolicy.Options{
		StrictBornRouted: *strictBornRouted,
	}

	result := IssueDiscoverabilityAuditResult{
		Schema: DiscoverabilitySchema,
		OK:     true,
		Total:  len(drafts),
		Issues: make([]IssueDiscoverabilityRow, 0, len(drafts)),
	}

	for _, d := range drafts {
		row := auditIssueDraftDiscoverability(d, opts)
		if !row.Dispatchable {
			result.OK = false
		}
		switch row.Placement {
		case PlacementSubdivideQueue:
			result.Subdivide++
		case PlacementTriageQueue:
			result.Triage++
		case PlacementSerialWave:
			result.Dispatchable++
			result.Serial++
		case PlacementParallelWave:
			result.Dispatchable++
			result.Parallel++
		}
		result.Issues = append(result.Issues, row)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue discoverability: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderIssueDiscoverabilityText(result))
	}

	if !result.OK {
		return 3
	}
	return 0
}

func auditIssueDraftDiscoverability(draft issuepolicy.IssueDraft, opts issuepolicy.Options) IssueDiscoverabilityRow {
	candidate := issuepolicy.CandidateFromIssueDraft(draft)
	review := issuepolicy.ReviewIssueDraft(draft, opts)
	frame := issuepolicy.AssessProblemFrame(draft)

	internalPkgs := extractInternalPackages(review.Paths)
	isSub := isSubdivideTarget(review.ExpectedSteps, internalPkgs)
	isTri := isTriageTarget(draft.Title, review.Dispatchability)
	isSer := isSerialSingleton(review.Lane, review.Paths)

	var placement string
	switch {
	case isSub:
		placement = PlacementSubdivideQueue
	case isTri:
		placement = PlacementTriageQueue
	case isSer:
		placement = PlacementSerialWave
	default:
		placement = PlacementParallelWave
	}

	dispatchable := (placement == PlacementParallelWave || placement == PlacementSerialWave)

	var reasons []string
	var repairActions []string

	if review.ExpectedSteps > 15 {
		reasons = append(reasons, fmt.Sprintf("expected steps (%d) exceeds limit of 15", review.ExpectedSteps))
		repairActions = append(repairActions, fmt.Sprintf("decompose oversized issue into child tasks with <= %d expected steps", issuepolicy.MaxDispatchExpectedSteps))
	}
	if len(internalPkgs) >= 3 {
		reasons = append(reasons, fmt.Sprintf("touches %d distinct internal packages (%s)", len(internalPkgs), strings.Join(internalPkgs, ", ")))
		repairActions = append(repairActions, fmt.Sprintf("partition changes across internal packages (%s) into separate scoped issues", strings.Join(internalPkgs, ", ")))
	}

	if strings.TrimSpace(draft.Title) == "" {
		reasons = append(reasons, "issue title is empty")
		repairActions = append(repairActions, "add a concise, descriptive title using conventional commit prefix (e.g. feat:, fix:)")
	}

	for _, r := range review.Reasons {
		reasons = append(reasons, r)
		switch r {
		case issuepolicy.ReasonScopeIncomplete:
			if len(review.MissingSections) > 0 {
				repairActions = append(repairActions, fmt.Sprintf("add missing required sections: %s", strings.Join(review.MissingSections, ", ")))
			} else {
				repairActions = append(repairActions, "add missing contract sections (## Working spine, ## Core through-line, ## Done condition / witness)")
			}
		case issuepolicy.ReasonUnrouted:
			repairActions = append(repairActions, "add a ## Lane or ## Path hints section so the issue routes to an execution lane")
		case issuepolicy.ReasonProblemFrameIncomplete:
			if len(frame.RepairActions) > 0 {
				repairActions = append(repairActions, frame.RepairActions...)
			} else {
				repairActions = append(repairActions, "complete problem frame with concrete evidence for Centrality and P1-P4")
			}
		case issuepolicy.ReasonUnexpandedTemplate:
			repairActions = append(repairActions, "expand or remove unexpanded template tokens ($(@{...})) in the issue body")
		case issuepolicy.ReasonPrivateBoundary:
			repairActions = append(repairActions, "remove private references or move the issue to the private companion repository")
		case issuepolicy.ReasonLiveUnarmored:
			repairActions = append(repairActions, "arm live dedupe check and scan cap before syncing")
		case issuepolicy.ReasonNoiseIncomplete:
			repairActions = append(repairActions, "add ## Creation trigger and ## Noise control (batch policy) sections")
		case issuepolicy.ReasonAgentIncomplete:
			repairActions = append(repairActions, "add ## Assumptions, ## Known confusion, and ## Handoff notes sections")
		case issuepolicy.ReasonModelTierIncomplete:
			repairActions = append(repairActions, "declare required model tier (tier/T1-required .. tier/T4-required)")
		case issuepolicy.ReasonNotBornRouted:
			repairActions = append(repairActions, "ensure issue has lane, class label (class:*), and priority label (priority/P*)")
		case issuepolicy.ReasonNotDispatchLeaf:
			repairActions = append(repairActions, "decompose non-leaf issue into dispatchable child tasks")
		case issuepolicy.ReasonOversizedSteps:
			if review.ExpectedSteps <= 15 {
				repairActions = append(repairActions, fmt.Sprintf("reduce expected steps (%d) to <= %d", review.ExpectedSteps, issuepolicy.MaxDispatchExpectedSteps))
			}
		case issuepolicy.ReasonScaleUndeclared:
			repairActions = append(repairActions, "declare work scale (S0..S4 or leaf/feature/epic)")
		case issuepolicy.ReasonWitnessScaleMismatch:
			repairActions = append(repairActions, "upgrade witness to match work scale")
		case issuepolicy.ReasonTargetEnvelopeMissing:
			repairActions = append(repairActions, "declare target operating envelope")
		case issuepolicy.ReasonEnvelopeInvalid:
			repairActions = append(repairActions, "fix invalid operating envelope entries")
		case issuepolicy.ReasonEnvelopeUnderTarget:
			repairActions = append(repairActions, "ensure witnessed envelope meets target envelope")
		case issuepolicy.ReasonScaleEvidenceInvalid:
			repairActions = append(repairActions, "fix invalid scale evidence records")
		case issuepolicy.ReasonScaleStageMissing:
			repairActions = append(repairActions, "add missing scale evidence stages")
		case issuepolicy.ReasonWitnessForgeable:
			repairActions = append(repairActions, "provide a verifiable, non-forgeable witness command")
		case issuepolicy.ReasonProjectWorkMissing:
			repairActions = append(repairActions, "declare project-work estimate and contribution")
		case issuepolicy.ReasonProjectWorkInvalid:
			repairActions = append(repairActions, "fix invalid project-work estimate/contribution values")
		case issuepolicy.ReasonClosureWitnessMissing:
			repairActions = append(repairActions, "add closure witness standard")
		case issuepolicy.ReasonClosureWitnessMismatch:
			repairActions = append(repairActions, "reconcile closure claim with witnessed standard")
		case issuepolicy.ReasonClosureProductionGap:
			repairActions = append(repairActions, "bridge production gap for closure")
		default:
			repairActions = append(repairActions, "resolve contract refusal: "+r)
		}
	}

	if frame.Enforced && !frame.Ready {
		for _, pfr := range frame.Reasons {
			reasons = append(reasons, pfr)
		}
		for _, pfa := range frame.RepairActions {
			repairActions = append(repairActions, pfa)
		}
	}

	if !dispatchable && len(repairActions) == 0 {
		repairActions = append(repairActions, "review issue contract requirements and ensure all required sections and lanes are specified")
	}

	reasons = dedupeStrings(reasons)
	repairActions = dedupeStrings(repairActions)

	key := candidate.Key
	if key == "" && draft.Number > 0 {
		key = fmt.Sprintf("issue-%d", draft.Number)
	}

	centrality := string(candidate.ProblemFrame.Centrality)
	if centrality == "" {
		centrality = string(frame.Centrality)
	}

	return IssueDiscoverabilityRow{
		Number:           draft.Number,
		Title:            draft.Title,
		Key:              key,
		URL:              draft.URL,
		Dispatchable:     dispatchable,
		Placement:        placement,
		Dispatchability:  review.Dispatchability,
		Lane:             review.Lane,
		Paths:            review.Paths,
		InternalPackages: internalPkgs,
		ExpectedSteps:    review.ExpectedSteps,
		Centrality:       centrality,
		IsSubdivide:      isSub,
		IsTriage:         isTri,
		IsSerial:         isSer,
		Reasons:          reasons,
		MissingSections:  review.MissingSections,
		MissingFields:    review.MissingFields,
		RepairActions:    repairActions,
	}
}

func extractInternalPackages(paths []string) []string {
	pkgs := make(map[string]bool)
	for _, p := range paths {
		norm := filepath.ToSlash(filepath.Clean(p))
		if strings.HasPrefix(norm, "internal/") {
			parts := strings.Split(norm, "/")
			if len(parts) >= 2 && parts[1] != "" {
				pkgs[parts[1]] = true
			}
		}
	}
	out := make([]string, 0, len(pkgs))
	for pkg := range pkgs {
		out = append(out, pkg)
	}
	sort.Strings(out)
	return out
}

func isSubdivideTarget(steps int, internalPkgs []string) bool {
	if steps > 15 {
		return true
	}
	return len(internalPkgs) >= 3
}

func isTriageTarget(title, dispatchability string) bool {
	if strings.TrimSpace(title) == "" {
		return true
	}
	if dispatchability != "" && dispatchability != issuepolicy.Dispatchable {
		return true
	}
	return false
}

func isSerialSingleton(lane string, paths []string) bool {
	l := strings.ToLower(strings.TrimSpace(lane))
	switch l {
	case "abi", "kernel", "adjudicator", "policy", "gateway", "vdso", "shipgate", "architest":
		return true
	}
	for _, p := range paths {
		norm := filepath.ToSlash(filepath.Clean(p))
		if norm == "go.mod" || norm == "go.sum" || norm == "dos.toml" {
			return true
		}
		for _, core := range []string{"internal/abi", "internal/kernel", "internal/adjudicator", "internal/policy", "internal/gateway", "internal/vdso"} {
			if norm == core || strings.HasPrefix(norm, core+"/") {
				return true
			}
		}
	}
	return false
}

func extractTitleFromMarkdown(body string) string {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		if strings.HasPrefix(l, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(l, "# "))
		}
		break
	}
	return ""
}

func dedupeStrings(in []string) []string {
	seen := make(map[string]bool, len(in))
	var out []string
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

func decodeDiscoverabilityIssues(raw []byte) ([]issuepolicy.IssueDraft, error) {
	var drafts []issuepolicy.IssueDraft
	if err := json.Unmarshal(raw, &drafts); err == nil && len(drafts) > 0 {
		return drafts, nil
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		for _, key := range []string{"issues", "items"} {
			if sub, ok := obj[key]; ok {
				var subDrafts []issuepolicy.IssueDraft
				if err := json.Unmarshal(sub, &subDrafts); err == nil && len(subDrafts) > 0 {
					return subDrafts, nil
				}
			}
		}
		if sub, ok := obj["candidates"]; ok {
			var candidates []issuepolicy.Candidate
			if err := json.Unmarshal(sub, &candidates); err == nil && len(candidates) > 0 {
				return candidatesToDrafts(candidates), nil
			}
		}
	}

	var candidates []issuepolicy.Candidate
	if err := json.Unmarshal(raw, &candidates); err == nil && len(candidates) > 0 {
		return candidatesToDrafts(candidates), nil
	}

	var single issuepolicy.IssueDraft
	if err := json.Unmarshal(raw, &single); err == nil && (single.Title != "" || single.Body != "") {
		return []issuepolicy.IssueDraft{single}, nil
	}

	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	var jsonlDrafts []issuepolicy.IssueDraft
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var d issuepolicy.IssueDraft
		if err := json.Unmarshal([]byte(line), &d); err == nil && (d.Title != "" || d.Body != "") {
			jsonlDrafts = append(jsonlDrafts, d)
		}
	}
	if len(jsonlDrafts) > 0 {
		return jsonlDrafts, nil
	}

	return nil, fmt.Errorf("unable to parse issue drafts from input payload")
}

func candidatesToDrafts(candidates []issuepolicy.Candidate) []issuepolicy.IssueDraft {
	drafts := make([]issuepolicy.IssueDraft, 0, len(candidates))
	for _, c := range candidates {
		drafts = append(drafts, candidateToDraft(c))
	}
	return drafts
}

func candidateToDraft(c issuepolicy.Candidate) issuepolicy.IssueDraft {
	var b strings.Builder
	if c.ParentRef != "" {
		b.WriteString("## Parent context\n\n" + c.ParentRef + "\n\n")
	}
	if c.InScope != "" {
		b.WriteString("## Core through-line\n\n" + c.InScope + "\n\n")
	}
	if c.OutOfScope != "" {
		b.WriteString("## Gold-plating boundary\n\n" + c.OutOfScope + "\n\n")
	}
	if c.DoneCondition != "" || c.Witness != "" {
		b.WriteString("## Done condition / witness\n\n")
		if c.DoneCondition != "" {
			b.WriteString(c.DoneCondition + "\n")
		}
		if c.Witness != "" {
			b.WriteString("Witness: " + c.Witness + "\n")
		}
		b.WriteString("\n")
	}
	if c.Lane != "" {
		b.WriteString("## Lane\n\n" + c.Lane + "\n\n")
	}
	if len(c.Paths) > 0 {
		b.WriteString("## Likely files\n\n")
		for _, p := range c.Paths {
			b.WriteString("- " + p + "\n")
		}
		b.WriteString("\n")
	}
	if c.ExpectedSteps > 0 {
		fmt.Fprintf(&b, "## Expected steps\n\n%d\n\n", c.ExpectedSteps)
	}
	var labels []issuepolicy.IssueLabel
	for _, l := range c.Labels {
		labels = append(labels, issuepolicy.IssueLabel{Name: l})
	}
	return issuepolicy.IssueDraft{
		Number: c.IssueNumber,
		Title:  c.Title,
		Body:   b.String(),
		Labels: labels,
	}
}

func renderIssueDiscoverabilityText(result IssueDiscoverabilityAuditResult) string {
	var b strings.Builder
	if len(result.Issues) == 1 {
		iss := result.Issues[0]
		b.WriteString("=== Issue Discoverability Audit ===\n")
		title := iss.Title
		if strings.TrimSpace(title) == "" {
			title = "(untitled)"
		}
		if iss.Number > 0 {
			fmt.Fprintf(&b, "Issue: #%d: %s\n", iss.Number, title)
		} else {
			fmt.Fprintf(&b, "Issue: %s\n", title)
		}
		if iss.Key != "" {
			fmt.Fprintf(&b, "Key: %s\n", iss.Key)
		}
		dispStr := "DISPATCHABLE"
		if !iss.Dispatchable {
			dispStr = "NOT DISPATCHABLE"
		}
		fmt.Fprintf(&b, "Status: %s\n", dispStr)
		fmt.Fprintf(&b, "Placement: %s\n", iss.Placement)
		if iss.Lane != "" {
			fmt.Fprintf(&b, "Lane: %s\n", iss.Lane)
		}
		fmt.Fprintf(&b, "Expected Steps: %d\n", iss.ExpectedSteps)
		if len(iss.Paths) > 0 {
			fmt.Fprintf(&b, "Paths: %s\n", strings.Join(iss.Paths, ", "))
		}
		if len(iss.InternalPackages) > 0 {
			fmt.Fprintf(&b, "Internal Packages: %s\n", strings.Join(iss.InternalPackages, ", "))
		}
		if len(iss.MissingSections) > 0 {
			fmt.Fprintf(&b, "Missing Sections: %s\n", strings.Join(iss.MissingSections, ", "))
		}
		if len(iss.Reasons) > 0 {
			b.WriteString("\nReasons:\n")
			for _, r := range iss.Reasons {
				fmt.Fprintf(&b, "  - %s\n", r)
			}
		}
		if len(iss.RepairActions) > 0 {
			b.WriteString("\nRepair Actions:\n")
			for i, a := range iss.RepairActions {
				fmt.Fprintf(&b, "  %d. %s\n", i+1, a)
			}
		}
		return b.String()
	}

	b.WriteString("=== Issue Discoverability Audit ===\n")
	fmt.Fprintf(&b, "Total: %d | Dispatchable: %d (Parallel: %d, Serial: %d) | Subdivide: %d | Triage: %d\n\n",
		result.Total, result.Dispatchable, result.Parallel, result.Serial, result.Subdivide, result.Triage)

	for i, iss := range result.Issues {
		title := iss.Title
		if strings.TrimSpace(title) == "" {
			title = "(untitled)"
		}
		numPrefix := ""
		if iss.Number > 0 {
			numPrefix = fmt.Sprintf("#%d: ", iss.Number)
		}
		dispStr := "DISPATCHABLE"
		if !iss.Dispatchable {
			dispStr = "NOT DISPATCHABLE"
		}
		fmt.Fprintf(&b, "[%d] %s%s\n", i+1, numPrefix, title)
		fmt.Fprintf(&b, "    Placement: %s (%s)\n", iss.Placement, dispStr)
		if iss.Lane != "" || iss.ExpectedSteps > 0 {
			fmt.Fprintf(&b, "    Lane: %s | Steps: %d\n", firstNonEmpty(iss.Lane, "(unrouted)"), iss.ExpectedSteps)
		}
		if len(iss.MissingSections) > 0 {
			fmt.Fprintf(&b, "    Missing Sections: %s\n", strings.Join(iss.MissingSections, ", "))
		}
		if len(iss.Reasons) > 0 {
			b.WriteString("    Reasons:\n")
			for _, r := range iss.Reasons {
				fmt.Fprintf(&b, "      - %s\n", r)
			}
		}
		if len(iss.RepairActions) > 0 {
			b.WriteString("    Repair Actions:\n")
			for j, a := range iss.RepairActions {
				fmt.Fprintf(&b, "      %d. %s\n", j+1, a)
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}
