package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// issue finding is the live adapter over modelroute's pure cross-audit finding
// planner (#3857). It reads a batch of verified audit receipts plus the findings
// already filed, asks the planner what to do (CREATE/UPDATE/REOPEN/COMMENT/
// NOOP/ESCALATE), renders each CREATE as an issue-contract-DISPATCHABLE
// candidate, and — only when explicitly armed — applies the bounded GitHub
// mutations through the same governed gh atoms `fak-dev issue create` uses. It
// DEFAULTS to a dry-run plan that never touches GitHub, and it refuses to emit a
// candidate the strict issue contract would not admit, so a broken generator is
// caught before it can file spam.

// findingKeyMarkerRE and findingReceiptMarkerRE recover a filed finding's dedupe
// key and last-recorded receipt digest from its issue body. They mirror the
// markers renderFindingIssueBody writes, and the key marker is a member of
// issuecontract's fak-*-key marker grammar so the contract's own dedupe scan
// sees it too.
var findingKeyMarkerRE = regexp.MustCompile(`<!--\s*fak-crossaudit-finding-key:\s*([^>\s]+)\s*-->`)
var findingReceiptMarkerRE = regexp.MustCompile(`<!--\s*fak-crossaudit-finding-receipt:\s*([^>\s]+)\s*-->`)

// issueFindingDeps injects the receipt/finding sources and the gh runner so
// tests exercise the whole adapter without a ledger file or a live gh. When a
// field is left at its zero value the adapter falls back to the flag-driven file
// sources and the real governed gh runner.
type issueFindingDeps struct {
	receipts    []modelroute.IssueAuditReceipt
	receiptsSet bool
	existing    []modelroute.ExistingFinding
	existingSet bool
	gh          issueCreateRunner
}

type issueFindingResult struct {
	Schema      string                    `json:"schema"`
	Live        bool                      `json:"live"`
	DryRun      bool                      `json:"dry_run"`
	Lane        string                    `json:"lane"`
	DedupeCap   int                       `json:"dedupe_cap"`
	MaxApply    int                       `json:"max_apply"`
	OK          bool                      `json:"ok"`
	Counts      map[string]int            `json:"counts"`
	Mutations   int                       `json:"mutations"`
	Items       []issueFindingItem        `json:"items"`
	Candidates  []issuecontract.Candidate `json:"candidates,omitempty"`
	Escalations []issueFindingEscalation  `json:"escalations,omitempty"`
	Applied     []issueFindingApplied     `json:"applied,omitempty"`
	Refusal     string                    `json:"refusal,omitempty"`
}

type issueFindingItem struct {
	Action        string `json:"action"`
	Reason        string `json:"reason"`
	Key           string `json:"key"`
	AuditedIssue  int    `json:"audited_issue"`
	TargetIssue   int    `json:"target_issue,omitempty"`
	Verdict       string `json:"verdict"`
	Severity      string `json:"severity,omitempty"`
	Dispatchable  string `json:"dispatchable,omitempty"`
	ContractOK    *bool  `json:"contract_ok,omitempty"`
	ReceiptDigest string `json:"receipt_digest,omitempty"`
}

type issueFindingEscalation struct {
	Key          string `json:"key"`
	AuditedIssue int    `json:"audited_issue"`
	Verdict      string `json:"verdict"`
	Reason       string `json:"reason"`
	Detail       string `json:"detail,omitempty"`
}

type issueFindingApplied struct {
	Action      string   `json:"action"`
	Key         string   `json:"key"`
	TargetIssue int      `json:"target_issue,omitempty"`
	Args        []string `json:"args"`
	URL         string   `json:"url,omitempty"`
	OK          bool     `json:"ok"`
	Error       string   `json:"error,omitempty"`
}

func runIssueFinding(stdout, stderr io.Writer, argv []string) int {
	return runIssueFindingWith(stdout, stderr, argv, issueFindingDeps{})
}

func runIssueFindingWith(stdout, stderr io.Writer, argv []string, deps issueFindingDeps) int {
	fs := flag.NewFlagSet("issue finding", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "audit receipt ledger JSONL to read receipts from (verified chain)")
	receipts := fs.String("receipts", "", "JSON array of audit receipts (offline/test source)")
	fromIssues := fs.String("from-issues", "", "existing findings as gh issue list --json number,body,state JSON")
	lane := fs.String("lane", "crossaudit", "routing lane hint stamped on generated finding candidates")
	repo := fs.String("repo", "", "owner/name override for live gh mutations")
	live := fs.Bool("live", false, "arm bounded live GitHub mutations (default: dry-run plan only)")
	dedupeCap := fs.Int("dedupe-cap", 0, "bounded issue-scan cap proven before live sync (required with --live)")
	parentBaseline := fs.Float64("parent-baseline-points", 0, "audited issue production-scope baseline points (required for REFUTE create)")
	completionStandard := fs.String("completion-standard", "production", "finding maturity (default production)")
	targetEnvelope := fs.String("target-envelope", "", "production target operating envelope")
	witnessedEnvelope := fs.String("witnessed-envelope", "", "currently witnessed operating envelope")
	maxApply := fs.Int("max-apply", 10, "refuse the run if planned mutations exceed this cap")
	asJSON := fs.Bool("json", false, "emit the machine-readable plan/result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue finding: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// Resolve the receipt source: injected deps win, then exactly one of the file
	// flags. --ledger is the verified path; --receipts is an offline convenience.
	var receiptRows []modelroute.IssueAuditReceipt
	switch {
	case deps.receiptsSet:
		receiptRows = deps.receipts
	default:
		selected := 0
		for _, v := range []string{*ledger, *receipts} {
			if strings.TrimSpace(v) != "" {
				selected++
			}
		}
		if selected != 1 {
			fmt.Fprintln(stderr, "fak-dev issue finding: pass exactly one of --ledger LEDGER.jsonl or --receipts RECEIPTS.json")
			return 2
		}
		var err error
		if strings.TrimSpace(*ledger) != "" {
			receiptRows, err = loadFindingLedgerReceipts(*ledger)
		} else {
			receiptRows, err = loadFindingReceiptsFile(*receipts)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue finding: %v\n", err)
			return 2
		}
	}

	// Resolve existing findings: injected deps win, else parse --from-issues (an
	// absent file means "no findings filed yet").
	var existing []modelroute.ExistingFinding
	if deps.existingSet {
		existing = deps.existing
	} else if strings.TrimSpace(*fromIssues) != "" {
		var err error
		existing, err = loadFindingExistingFile(*fromIssues)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue finding: %v\n", err)
			return 2
		}
	}

	plan := modelroute.PlanCrossAuditFindings(receiptRows, existing)

	result := issueFindingResult{
		Schema:    "fak.issue-finding-result.v1",
		Live:      *live,
		DryRun:    !*live,
		Lane:      strings.TrimSpace(*lane),
		DedupeCap: *dedupeCap,
		MaxApply:  *maxApply,
		OK:        true,
		Counts:    map[string]int{},
	}

	// Every generated CREATE candidate is held to the strict, armed issue
	// contract: a candidate the contract would not admit is a generator bug, not
	// something to file. dedupeCap defaults to a positive proof value for the
	// dry-run review so the contract's live-armed gate is what we validate against.
	reviewCap := *dedupeCap
	if reviewCap <= 0 {
		reviewCap = 1
	}
	contractOpts := issuecontract.Options{
		Live:              true,
		DedupeChecked:     true,
		DedupeCap:         reviewCap,
		StrictModelTier:   true,
		StrictScale:       true,
		StrictBornRouted:  true,
		StrictProjectWork: true,
	}

	authoring := findingProjectWork{Baseline: *parentBaseline, Standard: *completionStandard, TargetEnvelope: *targetEnvelope, WitnessedEnvelope: *witnessedEnvelope}
	createIndex := map[string]int{} // finding key -> index into result.Candidates
	for _, item := range plan.Items {
		result.Counts[string(item.Action)]++
		row := issueFindingItem{
			Action:        string(item.Action),
			Reason:        item.Reason,
			Key:           item.Key,
			AuditedIssue:  item.AuditedIssue,
			TargetIssue:   item.TargetIssue,
			Verdict:       string(item.Verdict),
			Severity:      string(item.Severity),
			ReceiptDigest: item.ReceiptDigest,
		}
		switch item.Action {
		case modelroute.FindingCreate:
			candidate, authorErr := buildFindingCandidateWithProjectWork(item, result.Lane, reviewCap, authoring)
			if authorErr != nil {
				v := false
				row.ContractOK = &v
				result.OK = false
				result.Items = append(result.Items, row)
				continue
			}
			review := issuecontract.ReviewCandidate(candidate, contractOpts)
			ok := review.OK && review.Dispatchability == issuecontract.Dispatchable
			row.Dispatchable = review.Dispatchability
			row.ContractOK = &ok
			if !ok {
				result.OK = false
			}
			createIndex[item.Key] = len(result.Candidates)
			result.Candidates = append(result.Candidates, candidate)
		case modelroute.FindingEscalate:
			result.Escalations = append(result.Escalations, issueFindingEscalation{
				Key:          item.Key,
				AuditedIssue: item.AuditedIssue,
				Verdict:      string(item.Verdict),
				Reason:       item.Reason,
				Detail:       item.Detail,
			})
		}
		result.Items = append(result.Items, row)
		if item.Action.Mutating() {
			result.Mutations++
		}
	}

	// Blast-radius guard: a run that would exceed the declared mutation cap is
	// refused wholesale (dry-run and live alike surface it) so a runaway batch
	// cannot file dozens of issues.
	if result.Mutations > *maxApply {
		result.OK = false
		result.Refusal = fmt.Sprintf("planned mutations %d exceed --max-apply %d", result.Mutations, *maxApply)
	}

	if *live {
		if code := applyFindingLive(stdout, stderr, plan, &result, *repo, deps.gh, authoring); code != 0 {
			// applyFindingLive has already populated result / printed context.
			result.OK = false
			emitFindingResult(stdout, stderr, result, *asJSON)
			return code
		}
	}

	emitFindingResult(stdout, stderr, result, *asJSON)
	if !result.OK {
		if result.Refusal != "" {
			return 2
		}
		return 3
	}
	return 0
}

// applyFindingLive performs the bounded, armed mutations. It requires a proven
// dedupe cap and a clean plan (all candidates dispatchable, under the blast-radius
// cap) before it touches GitHub. Returns 0 on success, or a non-zero exit code.
func applyFindingLive(stdout, stderr io.Writer, plan modelroute.FindingPlan, result *issueFindingResult, repo string, runner issueCreateRunner, authoring findingProjectWork) int {
	if result.DedupeCap <= 0 {
		fmt.Fprintln(stderr, "fak-dev issue finding: --live requires --dedupe-cap N (a bounded issue-scan cap)")
		result.Refusal = "live sync not armed: missing dedupe cap"
		return 2
	}
	if !result.OK {
		// A contract-failing candidate or a blast-radius refusal blocks live sync.
		if result.Refusal == "" {
			result.Refusal = "live sync blocked: a generated candidate is not dispatchable"
		}
		fmt.Fprintf(stderr, "fak-dev issue finding: %s\n", result.Refusal)
		return 2
	}
	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	for _, item := range plan.Items {
		if !item.Action.Mutating() {
			continue
		}
		args, ok := findingMutationArgs(item, repo, authoring)
		if !ok {
			// A non-create mutation targeting a finding created earlier in this same
			// batch has no issue number yet; it is folded into the create, so skip.
			continue
		}
		out, errOut, ghOK := run(args)
		applied := issueFindingApplied{
			Action:      string(item.Action),
			Key:         item.Key,
			TargetIssue: item.TargetIssue,
			Args:        args,
			OK:          ghOK,
			URL:         strings.TrimSpace(out),
		}
		if !ghOK {
			applied.Error = strings.TrimSpace(errOut)
			result.Applied = append(result.Applied, applied)
			result.Refusal = fmt.Sprintf("gh %s failed for %s", item.Action, item.Key)
			fmt.Fprintf(stderr, "fak-dev issue finding: gh %s failed: %s\n", item.Action, applied.Error)
			return 1
		}
		result.Applied = append(result.Applied, applied)
	}
	return 0
}

// findingMutationArgs renders the gh argv for one mutating item. CREATE always
// resolves; the edit/comment/reopen family needs a real target issue number
// (ok=false when the target is a finding opened earlier in this batch).
func findingMutationArgs(item modelroute.FindingPlanItem, repo string, authoring findingProjectWork) ([]string, bool) {
	withRepo := func(args []string) []string {
		if strings.TrimSpace(repo) != "" {
			args = append(args, "--repo", repo)
		}
		return args
	}
	switch item.Action {
	case modelroute.FindingCreate:
		title := findingIssueTitle(item.AuditedIssue)
		body, err := renderFindingIssueBodyWithProjectWork(item, authoring)
		if err != nil {
			return nil, false
		}
		args := []string{"issue", "create", "--title", title, "--body", body}
		for _, label := range findingIssueLabels() {
			args = append(args, "--label", label)
		}
		return withRepo(args), true
	case modelroute.FindingReopen:
		if item.TargetIssue <= 0 {
			return nil, false
		}
		return withRepo([]string{"issue", "reopen", itoaFinding(item.TargetIssue), "--comment", findingComment(item)}), true
	case modelroute.FindingUpdate, modelroute.FindingComment:
		if item.TargetIssue <= 0 {
			return nil, false
		}
		return withRepo([]string{"issue", "comment", itoaFinding(item.TargetIssue), "--body", findingComment(item)}), true
	default:
		return nil, false
	}
}

func emitFindingResult(stdout, stderr io.Writer, result issueFindingResult, asJSON bool) {
	if asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue finding: encode json: %v\n", err)
		}
		return
	}
	fmt.Fprintln(stdout, renderFindingResult(result))
}

func renderFindingResult(r issueFindingResult) string {
	mode := "dry-run"
	if r.Live {
		mode = "live"
	}
	lines := []string{
		fmt.Sprintf("issue-finding: %s  ok=%t  items=%d  mutations=%d  lane=%s", mode, r.OK, len(r.Items), r.Mutations, r.Lane),
	}
	order := []modelroute.FindingAction{
		modelroute.FindingCreate, modelroute.FindingUpdate, modelroute.FindingReopen,
		modelroute.FindingComment, modelroute.FindingEscalate, modelroute.FindingNoop,
	}
	parts := make([]string, 0, len(order))
	for _, a := range order {
		parts = append(parts, fmt.Sprintf("%s=%d", a, r.Counts[string(a)]))
	}
	lines = append(lines, "  counts: "+strings.Join(parts, " "))
	if r.Refusal != "" {
		lines = append(lines, "  refusal: "+r.Refusal)
	}
	for _, item := range r.Items {
		line := fmt.Sprintf("  [%s] %s audited=#%d verdict=%s reason=%s", item.Action, item.Key, item.AuditedIssue, item.Verdict, item.Reason)
		if item.TargetIssue > 0 {
			line += fmt.Sprintf(" target=#%d", item.TargetIssue)
		}
		if item.ContractOK != nil {
			line += fmt.Sprintf(" contract=%s", item.Dispatchable)
		}
		lines = append(lines, line)
	}
	for _, esc := range r.Escalations {
		lines = append(lines, fmt.Sprintf("  escalate: %s audited=#%d verdict=%s (%s)", esc.Key, esc.AuditedIssue, esc.Verdict, esc.Reason))
	}
	for _, applied := range r.Applied {
		status := "ok"
		if !applied.OK {
			status = "FAILED: " + applied.Error
		}
		lines = append(lines, fmt.Sprintf("  applied %s %s -> %s", applied.Action, applied.Key, status))
	}
	return strings.Join(lines, "\n")
}

type findingProjectWork struct {
	Baseline                                    float64
	Standard, TargetEnvelope, WitnessedEnvelope string
}

func buildFindingCandidateWithProjectWork(item modelroute.FindingPlanItem, lane string, cap int, a findingProjectWork) (issuecontract.Candidate, error) {
	c := buildFindingCandidate(item, lane, cap)
	if item.AuditedIssue > 0 {
		c.ParentRef = fmt.Sprintf("#%d", item.AuditedIssue)
	}
	points := float64(c.ExpectedSteps)
	c.WorkEstimate = fmt.Sprintf("Estimate: %g points", points)
	c.ScopeContribution = fmt.Sprintf("Contribution: %g/%g points", points, a.Baseline)
	c.CompletionStandard = strings.TrimSpace(a.Standard)
	if c.CompletionStandard == "" {
		c.CompletionStandard = "production"
	}
	if c.CompletionStandard == "production" {
		c.TargetEnvelope = strings.TrimSpace(a.TargetEnvelope)
		c.WitnessedEnvelope = strings.TrimSpace(a.WitnessedEnvelope)
	}
	if a.Baseline <= 0 {
		return c, fmt.Errorf("finding create requires --parent-baseline-points")
	}
	return c, nil
}

// buildFindingCandidate renders one REFUTE finding as a fully-scoped,
// issue-contract-dispatchable candidate. Every field the strict armed contract
// checks — spine, scope, witness, routing, agent context, noise control, model
// tier, scale, and born-routed labels — is populated so the generated fixture is
// admitted rather than filed as an unscoped worker prompt. The done-condition
// binds closure to a witnessed fix PLUS an independent re-audit PASS.
func buildFindingCandidate(item modelroute.FindingPlanItem, lane string, dedupeCap int) issuecontract.Candidate {
	n := item.AuditedIssue
	sev := strings.ToUpper(strings.TrimSpace(string(item.Severity)))
	if sev == "" {
		sev = string(modelroute.AuditSeverityUnknown)
	}
	ref := strings.TrimSpace(item.Subject.IssueURL)
	if ref == "" {
		ref = fmt.Sprintf("#%d", n)
	}
	detail := strings.TrimSpace(item.Detail)
	if detail == "" {
		detail = "the claimed resolution is not supported by the audited evidence bundle"
	}
	if lane == "" {
		lane = "crossaudit"
	}
	return issuecontract.Candidate{
		Schema:         issuecontract.Schema,
		Key:            item.Key,
		Title:          findingIssueTitle(n),
		ParentRef:      ref,
		CurrentState:   fmt.Sprintf("Closed issue #%d was re-audited by an independent cross-model auditor, which returned REFUTE (severity %s): %s.", n, sev, detail),
		WhyNow:         fmt.Sprintf("A resolved issue is failing an independent re-audit right now: #%d's closing change does not hold up, so the working path it claimed is still red and is the current weak point.", n),
		WorkingSpine:   fmt.Sprintf("Restore #%d's resolution to true — land a corrected closing change and prove it with a fresh independent cross-audit PASS, closing the gap the re-audit found.", n),
		InScope:        fmt.Sprintf("Reproduce the re-audit REFUTE for #%d, land a corrected closing change, and re-run the cross-audit to a PASS.", n),
		OutOfScope:     "Unrelated refactors or polish, and closing this finding on the auditor's word alone without a witnessed fix and an independent re-audit.",
		DoneCondition:  fmt.Sprintf("A new closing commit for #%d lands AND an independent cross-model re-audit of that commit returns PASS.", n),
		Witness:        fmt.Sprintf("`fak-dev issue audit --issue %d` returns PASS on the new closing commit (a fresh crossaudit PASS receipt) with the fix commit green under `go test ./...`.", n),
		AcceptanceGate: "The cross-audit re-run on the corrected commit returns PASS and its receipt is appended to the audit ledger.",
		ClosureBinding: fmt.Sprintf("Close only when witnessed by a new closing commit for #%d AND an independent re-audit PASS — never on this finding report alone.", n),
		Lane:           lane,
		Paths:          findingCandidatePaths(item.EvidenceRefs),
		WorkUnit:       "leaf",
		ExpectedSteps:  3,
		Scale:          "S1",
		Assumptions: []string{
			fmt.Sprintf("The audited commit %s is the change that closed #%d.", findingShortSHA(item.Subject.CommitSHA), n),
		},
		ConfusionRisks: []string{
			"A REFUTE means the resolution is unproven against the evidence bundle, not that the code is malicious or that the auditor overrides a human reviewer.",
		},
		Coordination: []string{
			"One finding issue per audit key; if this key already has an open finding, update it in place instead of filing a duplicate.",
		},
		Trigger:     "Filed when an independent cross-model audit receipt returns REFUTE for a closed issue.",
		BatchPolicy: fmt.Sprintf("One issue per audit key; dedupe against the existing finding marker and update in place, with a bounded issue scan capped at %d per run.", dedupeCap),
		Labels:      findingIssueLabels(),
		Priority:    "P2",
		// Body fallback tiers (equal, so no contradiction): the finding remediation
		// is default-capability work.
		RequiredModelTier: "T2",
		OptimalModelTier:  "T2",
	}
}

func findingIssueTitle(n int) string {
	return fmt.Sprintf("crossaudit finding: independent re-audit REFUTED resolution of #%d", n)
}

func findingIssueLabels() []string {
	// class:* and priority/p* satisfy the born-routed contract; crossaudit-finding
	// is the marker label the dedupe scan filters on.
	return []string{"crossaudit-finding", "class:bug", "priority/p2"}
}

// findingCandidatePaths surfaces path-like evidence refs as routing hints. It is
// deliberately conservative: only refs that look like repo-relative paths (a
// slash, no whitespace, not a URL/sha/marker) are kept, capped at a few.
func findingCandidatePaths(refs []modelroute.EvidenceRef) []string {
	var out []string
	seen := map[string]bool{}
	for _, ref := range refs {
		p := strings.TrimSpace(ref.Ref)
		if p == "" || strings.ContainsAny(p, " \t\r\n") {
			continue
		}
		if !strings.Contains(p, "/") {
			continue
		}
		lower := strings.ToLower(p)
		if strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://") ||
			strings.HasPrefix(lower, "sha256:") || strings.HasPrefix(lower, "audit:") || strings.HasPrefix(p, "#") {
			continue
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
		if len(out) >= 5 {
			break
		}
	}
	return out
}

// renderFindingIssueBody renders the markdown body filed for a CREATE. It leads
// with the dedupe markers (finding key + last-recorded receipt digest) so a
// later run recovers the finding's state, then lays out the standard issue
// sections so the filed issue re-parses to the same dispatchable candidate under
// `fak-dev issue contract --from-issues`.
func renderFindingIssueBody(item modelroute.FindingPlanItem) string {
	body, _ := renderFindingIssueBodyWithProjectWork(item, findingProjectWork{Baseline: float64(buildFindingCandidate(item, "crossaudit", 1).ExpectedSteps), Standard: "demo"})
	return body
}
func renderFindingIssueBodyWithProjectWork(item modelroute.FindingPlanItem, a findingProjectWork) (string, error) {
	c, err := buildFindingCandidateWithProjectWork(item, "crossaudit", 1, a)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-crossaudit-finding-key: %s -->\n", item.Key)
	if item.ReceiptDigest != "" {
		fmt.Fprintf(&b, "<!-- fak-crossaudit-finding-receipt: %s -->\n", item.ReceiptDigest)
	}
	if item.AuditKey != "" {
		fmt.Fprintf(&b, "<!-- fak-crossaudit-finding-audit-key: %s -->\n", item.AuditKey)
	}
	fmt.Fprintf(&b, "\n# %s\n\n", c.Title)
	fmt.Fprintf(&b, "## Parent context\n%s\n\n", c.ParentRef)
	fmt.Fprintf(&b, "## Current state\n%s\n\n", c.CurrentState)
	fmt.Fprintf(&b, "## Why this is next\n%s\n\n", c.WhyNow)
	fmt.Fprintf(&b, "## Working spine\n%s\n\n", c.WorkingSpine)
	fmt.Fprintf(&b, "## In scope\n%s\n\n", c.InScope)
	fmt.Fprintf(&b, "## Out of scope\n%s\n\n", c.OutOfScope)
	fmt.Fprintf(&b, "## Done condition / witness\nDone condition: %s\nWitness: %s\n\n", c.DoneCondition, c.Witness)
	fmt.Fprintf(&b, "## Acceptance gate\n%s\n\n", c.AcceptanceGate)
	fmt.Fprintf(&b, "## Work estimate\n%s\n\n", c.WorkEstimate)
	fmt.Fprintf(&b, "## Overall completion contribution\n%s\n\n", c.ScopeContribution)
	fmt.Fprintf(&b, "## Completion standard\n%s\n\n", c.CompletionStandard)
	if c.TargetEnvelope != "" {
		fmt.Fprintf(&b, "## Target operating envelope\n%s\n\n", c.TargetEnvelope)
	}
	if c.WitnessedEnvelope != "" {
		fmt.Fprintf(&b, "## Witnessed operating envelope\n%s\n\n", c.WitnessedEnvelope)
	}
	fmt.Fprintf(&b, "## Closure binding\n%s\n\n", c.ClosureBinding)
	fmt.Fprintf(&b, "## Lane\n%s\n\n", c.Lane)
	b.WriteString("## Likely files\n")
	if len(c.Paths) == 0 {
		b.WriteString("- (none recorded on the audit receipt)\n")
	}
	for _, p := range c.Paths {
		fmt.Fprintf(&b, "- `%s`\n", p)
	}
	fmt.Fprintf(&b, "\n## Work unit\n%s\n\n", c.WorkUnit)
	fmt.Fprintf(&b, "## Expected steps\n%d\n\n", c.ExpectedSteps)
	fmt.Fprintf(&b, "## Work scale\n%s\n\n", c.Scale)
	b.WriteString("## Assumptions\n")
	for _, a := range c.Assumptions {
		fmt.Fprintf(&b, "- %s\n", a)
	}
	b.WriteString("\n## Confusion risks\n")
	for _, cr := range c.ConfusionRisks {
		fmt.Fprintf(&b, "- %s\n", cr)
	}
	b.WriteString("\n## Coordination\n")
	for _, co := range c.Coordination {
		fmt.Fprintf(&b, "- %s\n", co)
	}
	fmt.Fprintf(&b, "\n## Trigger\n%s\n\n", c.Trigger)
	fmt.Fprintf(&b, "## Batch policy\n%s\n\n", c.BatchPolicy)
	fmt.Fprintf(&b, "Required model tier: %s\nOptimal model tier: %s\n", c.RequiredModelTier, c.OptimalModelTier)
	return b.String(), nil
}

func findingComment(item modelroute.FindingPlanItem) string {
	sev := strings.ToUpper(strings.TrimSpace(string(item.Severity)))
	switch item.Action {
	case modelroute.FindingReopen:
		return fmt.Sprintf("Reopening: a fresh independent cross-audit REFUTE recurred after this finding was closed (receipt `%s`, severity %s). The prior resolution did not hold. %s", item.ReceiptDigest, sev, item.Detail)
	case modelroute.FindingComment:
		return fmt.Sprintf("Independent cross-audit re-audit returned PASS (receipt `%s`). Closure still requires a witnessed fix commit before this finding is closed.", item.ReceiptDigest)
	default: // UPDATE
		return fmt.Sprintf("Independent cross-audit REFUTE recurred with new evidence (receipt `%s`, severity %s). %s", item.ReceiptDigest, sev, item.Detail)
	}
}

func findingShortSHA(sha string) string {
	sha = strings.TrimSpace(sha)
	if len(sha) > 12 {
		return sha[:12]
	}
	if sha == "" {
		return "(unknown)"
	}
	return sha
}

func itoaFinding(n int) string { return fmt.Sprintf("%d", n) }

// --- source decoders --------------------------------------------------------

func loadFindingLedgerReceipts(path string) ([]modelroute.IssueAuditReceipt, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(abs)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rows, err := modelroute.ParseAuditReceiptLedger(f)
	if err != nil {
		return nil, fmt.Errorf("read audit ledger: %w", err)
	}
	out := make([]modelroute.IssueAuditReceipt, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Receipt)
	}
	return out, nil
}

func loadFindingReceiptsFile(path string) ([]modelroute.IssueAuditReceipt, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	return decodeFindingReceipts(b)
}

func decodeFindingReceipts(b []byte) ([]modelroute.IssueAuditReceipt, error) {
	var arr []modelroute.IssueAuditReceipt
	if err := json.Unmarshal(b, &arr); err == nil {
		return arr, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse receipts JSON: %w", err)
	}
	for _, key := range []string{"receipts", "items"} {
		if raw, ok := obj[key]; ok {
			var rows []modelroute.IssueAuditReceipt
			if err := json.Unmarshal(raw, &rows); err != nil {
				return nil, fmt.Errorf("%s must be a receipt array: %w", key, err)
			}
			return rows, nil
		}
	}
	return nil, fmt.Errorf("receipts JSON must be an array or an object with a receipts/items array")
}

type findingIssueRow struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url"`
}

func loadFindingExistingFile(path string) ([]modelroute.ExistingFinding, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	rows, err := decodeFindingIssueRows(b)
	if err != nil {
		return nil, err
	}
	return existingFindingsFromIssues(rows), nil
}

func decodeFindingIssueRows(b []byte) ([]findingIssueRow, error) {
	var arr []findingIssueRow
	if err := json.Unmarshal(b, &arr); err == nil {
		return arr, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, fmt.Errorf("parse issues JSON: %w", err)
	}
	for _, key := range []string{"issues", "items"} {
		if raw, ok := obj[key]; ok {
			var rows []findingIssueRow
			if err := json.Unmarshal(raw, &rows); err != nil {
				return nil, fmt.Errorf("%s must be a GitHub issue array: %w", key, err)
			}
			return rows, nil
		}
	}
	return nil, fmt.Errorf("issues JSON must be an array or an object with an issues/items array")
}

// existingFindingsFromIssues keeps only rows carrying a finding-key marker (a
// filed cross-audit finding), pairing each with its recorded state and last
// receipt digest so the planner can dedupe/reopen against them.
func existingFindingsFromIssues(rows []findingIssueRow) []modelroute.ExistingFinding {
	var out []modelroute.ExistingFinding
	for _, row := range rows {
		km := findingKeyMarkerRE.FindStringSubmatch(row.Body)
		if km == nil {
			continue
		}
		state := modelroute.FindingStateOpen
		if strings.EqualFold(strings.TrimSpace(row.State), string(modelroute.FindingStateClosed)) {
			state = modelroute.FindingStateClosed
		}
		digest := ""
		if rm := findingReceiptMarkerRE.FindStringSubmatch(row.Body); rm != nil {
			digest = strings.TrimSpace(rm[1])
		}
		out = append(out, modelroute.ExistingFinding{
			Key:           strings.TrimSpace(km[1]),
			IssueNumber:   row.Number,
			State:         state,
			ReceiptDigest: digest,
		})
	}
	return out
}
