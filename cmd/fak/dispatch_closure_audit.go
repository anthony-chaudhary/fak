package main

// dispatch_closure_audit.go — `fak dispatch closure-audit`, the native port of
// the read/grade half of tools/issue_closure_audit.py (#1406). It binds commits
// to issue numbers from commit text and grades each issue into one witness
// bucket, re-reading the per-SHA `dos commit-audit` verdict for every resolving
// commit — never trusting a worker claim. All classification lives in the pure
// internal/closureaudit fold; this file is the I/O shell (gh issue list, git log,
// dos commit-audit) and the human/JSON/Markdown render.
//
//	# the human closure card
//	fak dispatch closure-audit
//	# machine-readable payload (schema fleet-issue-closure-audit/1)
//	fak dispatch closure-audit --json
//	# the same card as an operator Markdown block
//	fak dispatch closure-audit --markdown
//
// The gh/git coverage-truncation block of the Python auditor is ported here: when
// the --issue-limit fetch or the --max-commits git-log window is narrower than the
// backlog, the report carries a COVERAGE_INCOMPLETE / AUDIT_WINDOW_TRUNCATED
// coverage block so a narrowed audit can never present as complete coverage. This
// command owns the native binding + witness-gated grade + coverage honesty.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/closureaudit"
)

// I/O seams, overridable in tests so the shell is exercised hermetically without
// gh/git/dos on PATH.
var (
	closureAuditFetchIssues  = closureAuditFetchIssuesGH
	closureAuditReadCommits  = closureAuditReadCommitsGit
	closureAuditCommitAudit  = closureAuditCommitAuditDOS
	closureAuditCommitAudits = closureAuditCommitAuditsBounded
	closureAuditTotalCommits = closureAuditTotalCommitsGit
)

const closureAuditWorkers = 8

func runDispatchClosureAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch closure-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	maxCommits := fs.Int("max-commits", 2000, "git log window to scan for issue refs")
	issueLimit := fs.Int("issue-limit", 1000, "max issues to fetch from gh")
	asJSON := fs.Bool("json", false, "emit the fleet-issue-closure-audit/1 JSON payload")
	asMarkdown := fs.Bool("markdown", false, "render the operator closure card as Markdown")
	resolveAcceptance := fs.Bool("resolve-acceptance", false,
		"also resolve each still-open issue by ACCEPTANCE-SYMBOL presence on --ref, not by commit-subject grep (#5435)")
	ref := fs.String("ref", closureaudit.DefaultRef, "git ref the acceptance-symbol probes resolve against")
	acceptanceLimit := fs.Int("acceptance-limit", 25, "cap how many still-open issues are acceptance-resolved (each costs gh + git probes)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch closure-audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *asJSON && *asMarkdown {
		fmt.Fprintln(stderr, "fak dispatch closure-audit: choose at most one of --json or --markdown")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	rep := collectDispatchClosureAudit(root, *maxCommits, *issueLimit)
	// The subject-grep binding above is blind to a landing whose commit subject
	// named a different issue or named none, and to an acceptance item that landed
	// inside a peer's unrelated commit. Resolve the still-open issues by what their
	// acceptance NAMES instead, so this view stops reporting shipped work as open
	// and stops presenting a caller-less primitive as done (#5435).
	if *resolveAcceptance {
		resolveAcceptanceForReport(root, *ref, &rep, *acceptanceLimit)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch closure-audit: encode json: %v\n", err)
			return 1
		}
	} else if *asMarkdown {
		fmt.Fprint(stdout, renderClosureAuditMarkdown(rep))
	} else {
		fmt.Fprint(stdout, renderClosureAudit(rep))
	}
	if rep.OK {
		return 0
	}
	return 1
}

// collectDispatchClosureAudit is the I/O half: read issues + commits + per-SHA
// audits, then hand the pre-read facts to the pure grader. It audits only the
// resolving commits of issues actually fetched (deduped by SHA), mirroring the
// Python collect().
func collectDispatchClosureAudit(root string, maxCommits, issueLimit int) closureaudit.Report {
	issues, issuesErr := closureAuditFetchIssues(root, issueLimit)
	// Freeze the history denominator before the slow DOS witness pass. Shared
	// trunk can advance while thousands of commits are audited; counting HEAD at
	// the end would compare two different snapshots and fabricate truncation.
	totalCommits := closureAuditTotalCommits(root)
	commits, _ := closureAuditReadCommits(root, maxCommits)
	refs := closureaudit.RefsFromCommits(commits)

	fetched := map[int]bool{}
	for _, is := range issues {
		fetched[is.Number] = true
	}
	toAudit := map[string]bool{}
	for num, rs := range refs {
		if !fetched[num] {
			continue
		}
		for _, r := range rs {
			if r.Kind == closureaudit.Resolving {
				toAudit[r.SHA] = true
			}
		}
	}
	shas := make([]string, 0, len(toAudit))
	for sha := range toAudit {
		shas = append(shas, sha)
	}
	sort.Strings(shas)
	audits := closureAuditCommitAudits(root, shas)

	auditError := ""
	if issuesErr != nil {
		auditError = "gh issue read-back failed: " + issuesErr.Error()
	} else if len(issues) == 0 {
		auditError = "gh returned no issues (auth/network?) — cannot grade closure"
	}
	rep := closureaudit.Build(root, issues, refs, audits, auditError)
	// Attach the coverage-truncation block: the pure fold has no window facts, so
	// the shell measures them (issues fetched vs cap, commits scanned vs the repo
	// total from `git rev-list --count`) and surfaces AUDIT_WINDOW_TRUNCATED when
	// the audit saw only a slice of the backlog.
	cov := closureaudit.ComputeCoverage(len(issues), issueLimit, len(commits), maxCommits,
		totalCommits)
	rep.Coverage = &cov
	return rep
}

// closureAuditTotalCommitsGit returns the count of commits reachable from HEAD,
// or nil if git can't answer — the denominator for detecting a git-log window
// narrower than the repo's history.
func closureAuditTotalCommitsGit(root string) *int {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-list", "--count", "HEAD")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return nil
	}
	return &n
}

func closureAuditFetchIssuesGH(root string, limit int) ([]closureaudit.Issue, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "all",
		"--limit", fmt.Sprintf("%d", limit), "--json", "number,state,stateReason,title")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("gh issue list: %w", err)
	}
	var rows []struct {
		Number      int    `json:"number"`
		State       string `json:"state"`
		StateReason string `json:"stateReason"`
		Title       string `json:"title"`
	}
	if err := json.Unmarshal(out, &rows); err != nil {
		return nil, fmt.Errorf("gh issue list: parse json: %w", err)
	}
	issues := make([]closureaudit.Issue, 0, len(rows))
	for _, r := range rows {
		issues = append(issues, closureaudit.Issue{
			Number: r.Number, State: r.State, StateReason: r.StateReason, Title: r.Title,
		})
	}
	return issues, nil
}

func closureAuditReadCommitsGit(root string, maxCommits int) ([]closureaudit.Commit, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pretty := "--pretty=format:%H" + commitLinkFieldSep + "%s" + commitLinkFieldSep + "%b" + commitLinkRecordSep
	cmd := exec.CommandContext(ctx, "git", "log", fmt.Sprintf("-%d", maxCommits), pretty)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, errOut, err := runBufferedCommand(cmd)
	if err != nil {
		return nil, fmt.Errorf("git log: %w: %s", err, strings.TrimSpace(errOut))
	}
	return parseClosureAuditLog(out), nil
}

// parseClosureAuditLog splits the %x1f/%x1e-delimited `git log` text into Commit
// facts. Deterministic and process-free, so it is tested directly.
func parseClosureAuditLog(raw string) []closureaudit.Commit {
	var commits []closureaudit.Commit
	for _, rec := range strings.Split(raw, commitLinkRecordSep) {
		rec = strings.Trim(rec, "\n")
		if rec == "" {
			continue
		}
		parts := strings.SplitN(rec, commitLinkFieldSep, 3)
		if len(parts) < 2 {
			continue
		}
		body := ""
		if len(parts) == 3 {
			body = parts[2]
		}
		commits = append(commits, closureaudit.Commit{
			SHA: strings.TrimSpace(parts[0]), Subject: parts[1], Body: body,
		})
	}
	return commits
}

func closureAuditCommitAuditDOS(root, sha string) closureaudit.Audit {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// The ref is POSITIONAL (`dos commit-audit <ref>`); `dos commit-audit --json`
	// emits a JSON ARRAY (one record per audited commit), so take the first row.
	cmd := exec.CommandContext(ctx, "dos", "commit-audit", sha, "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return closureaudit.Audit{}
	}
	return parseCommitAuditRecord(out)
}

// closureAuditCommitAuditsBounded keeps complete-window audits practical while
// preserving the existing one-SHA DOS witness boundary. A full repository scan
// can contain thousands of resolving commits; serial process startup turned the
// coverage-honesty path into an hour-scale operation.
func closureAuditCommitAuditsBounded(root string, shas []string) map[string]closureaudit.Audit {
	audits := make(map[string]closureaudit.Audit, len(shas))
	if len(shas) == 0 {
		return audits
	}

	workers := min(closureAuditWorkers, len(shas))
	jobs := make(chan string)
	var mu sync.Mutex
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sha := range jobs {
				audit := closureAuditCommitAudit(root, sha)
				mu.Lock()
				audits[sha] = audit
				mu.Unlock()
			}
		}()
	}
	for _, sha := range shas {
		jobs <- sha
	}
	close(jobs)
	wg.Wait()
	return audits
}

func parseCommitAuditRecord(out []byte) closureaudit.Audit {
	type rec struct {
		Verdict   string `json:"verdict"`
		Witness   string `json:"witness"`
		ClaimKind string `json:"claim_kind"`
	}
	var arr []rec
	if err := json.Unmarshal(out, &arr); err == nil && len(arr) > 0 {
		return closureaudit.Audit{Verdict: arr[0].Verdict, Witness: arr[0].Witness, ClaimKind: arr[0].ClaimKind}
	}
	var one rec
	if err := json.Unmarshal(out, &one); err == nil {
		return closureaudit.Audit{Verdict: one.Verdict, Witness: one.Witness, ClaimKind: one.ClaimKind}
	}
	return closureaudit.Audit{}
}

func closureRateField(r *float64) string {
	if r == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.4g", *r)
}

func renderClosureAudit(rep closureaudit.Report) string {
	c := rep.Counts
	var b strings.Builder
	fmt.Fprintf(&b, "issue-closure audit: %s (%s)\n", rep.Verdict, rep.Finding)
	fmt.Fprintf(&b, "closure_rate=%s honest_close_rate=%s  next: %s\n",
		closureRateField(rep.ClosureRate), closureRateField(rep.HonestCloseRate), rep.NextAction)
	fmt.Fprintf(&b, "buckets: true=%d data=%d claimed=%d open_witnessed=%d open=%d not_planned=%d\n",
		c[closureaudit.TrueResolved], c[closureaudit.DataResolved], c[closureaudit.ClaimedClosed],
		c[closureaudit.OpenWitnessed], c[closureaudit.Open], c[closureaudit.ClosedNotPlanned])
	if cov := rep.Coverage; cov != nil && cov.Warning != "" {
		fmt.Fprintf(&b, "  WARN %s — %s\n", cov.Warning, strings.Join(cov.Notes, "; "))
		fmt.Fprintf(&b, "    re-run for full coverage: %s\n", cov.Recommended.Command)
	}
	b.WriteString(closureAuditAcceptanceBlock(rep))
	var actionable []closureaudit.Graded
	for _, g := range rep.Issues {
		if g.Bucket == closureaudit.ClaimedClosed || g.Bucket == closureaudit.OpenWitnessed {
			actionable = append(actionable, g)
		}
	}
	if len(actionable) > 0 {
		b.WriteString("  attention:\n")
		for i, g := range actionable {
			if i >= 15 {
				break
			}
			commits := "-"
			if wc := firstNonEmptyCommits(g); wc != "" {
				commits = wc
			}
			fmt.Fprintf(&b, "    #%-5d %-14s [%s] %s\n", g.Number, g.Bucket, commits, g.Title)
		}
	}
	return b.String()
}

func renderClosureAuditMarkdown(rep closureaudit.Report) string {
	c := rep.Counts
	var b strings.Builder
	fmt.Fprintf(&b, "### issue-closure audit — %s (%s)\n\n", rep.Verdict, rep.Finding)
	fmt.Fprintf(&b, "- **closure_rate**: `%s` (strict diff-witness)\n", closureRateField(rep.ClosureRate))
	fmt.Fprintf(&b, "- **honest_close_rate**: `%s` (credits the data rung)\n", closureRateField(rep.HonestCloseRate))
	fmt.Fprintf(&b, "- **next**: %s\n\n", rep.NextAction)
	b.WriteString("| bucket | count |\n|---|---|\n")
	for _, bucket := range []string{
		closureaudit.TrueResolved, closureaudit.DataResolved, closureaudit.ClaimedClosed,
		closureaudit.OpenWitnessed, closureaudit.Open, closureaudit.ClosedNotPlanned,
	} {
		fmt.Fprintf(&b, "| %s | %d |\n", bucket, c[bucket])
	}
	if cov := rep.Coverage; cov != nil && cov.Warning != "" {
		fmt.Fprintf(&b, "\n> **%s** — coverage incomplete. %s\n", cov.Warning, strings.Join(cov.Notes, " "))
		fmt.Fprintf(&b, ">\n> re-run for full coverage: `%s`\n", cov.Recommended.Command)
	}
	if block := closureAuditAcceptanceBlock(rep); block != "" {
		b.WriteString("\n```\n" + block + "```\n")
	}
	return b.String()
}

func firstNonEmptyCommits(g closureaudit.Graded) string {
	if len(g.WitnessedCommits) > 0 {
		return strings.Join(g.WitnessedCommits, ",")
	}
	if len(g.ResolvingCommits) > 0 {
		return strings.Join(g.ResolvingCommits, ",")
	}
	return ""
}
