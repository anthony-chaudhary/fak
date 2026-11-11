package devcmd

// stale-fixed closes the loop on open issues that trunk already fixed: it scans
// a trunk ref for GitHub closing-keyword commits (Fixes/Closes/Resolves #N),
// joins them against the open issue set, and reports each candidate with its
// SHA evidence. Dry-run by default; --apply posts an evidence comment and then
// closes each listed issue through the governed gh atom (runTaskHandoffGH), the
// same seam every other issue verb uses instead of raw gh calls.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuecontractrepair"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const staleFixedSchema = "fak.issue-stale-fixed/1"

const staleFixedGitTimeout = 120 * time.Second

// staleFixedCloseRE matches GitHub closing-keyword references in commit
// messages: fix/fixes/fixed/close/closes/closed/resolve/resolves/resolved,
// optionally followed by ':', an optional owner/repo qualifier, and #N. The
// word boundary keeps prose like "prefixes #123" or "closeout #123" out.
var staleFixedCloseRE = regexp.MustCompile(`(?i)\b(fixed|fixes|fix|closed|closes|close|resolved|resolves|resolve)\s*:?\s*(?:[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)?#([0-9]+)`)

type staleFixedCommit struct {
	SHA     string
	Message string
}

// staleFixedScanner resolves the trunk ref to a SHA and returns its history
// newest-first as commit records. Tests inject fixtures; nil runs real git.
type staleFixedScanner func(root, ref string, depth int) (trunkSHA string, commits []staleFixedCommit, err error)

type staleFixedRow struct {
	Number    int      `json:"number"`
	Title     string   `json:"title"`
	URL       string   `json:"url,omitempty"`
	Shas      []string `json:"shas"`
	Applied   bool     `json:"applied"`
	CommentOK bool     `json:"comment_ok,omitempty"`
	CloseOK   bool     `json:"close_ok,omitempty"`
	Error     string   `json:"error,omitempty"`
}

type staleFixedResult struct {
	Schema         string          `json:"schema"`
	Repo           string          `json:"repo,omitempty"`
	TrunkRef       string          `json:"trunk_ref"`
	TrunkSHA       string          `json:"trunk_sha"`
	Depth          int             `json:"depth,omitempty"`
	ScannedCommits int             `json:"scanned_commits"`
	OpenIssues     int             `json:"open_issues"`
	Apply          bool            `json:"apply"`
	Rows           []staleFixedRow `json:"rows"`
	AppliedCount   int             `json:"applied_count"`
	FailedCount    int             `json:"failed_count"`
}

func runIssueStaleFixed(stdout, stderr io.Writer, argv []string) int {
	return runIssueStaleFixedWith(stdout, stderr, argv, nil, nil)
}

func runIssueStaleFixedWith(stdout, stderr io.Writer, argv []string, runner issueCreateRunner, scanner staleFixedScanner) int {
	fs := flag.NewFlagSet("issue stale-fixed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "owner/name passed to gh (default: gh infers from cwd)")
	root := fs.String("root", ".", "local repository root holding trunk history")
	trunkRef := fs.String("trunk-ref", "", "trunk ref to scan (default: configured development branch)")
	limit := fs.Int("limit", 0, "cap on open issues fetched live (0 = default cap)")
	depth := fs.Int("depth", 0, "scan at most this many commits (0 = full history)")
	fromIssues := fs.String("from-issues", "", "cached open-issue JSON (gh issue list --json number,title,body,labels); '-' reads stdin")
	apply := fs.Bool("apply", false, "post the evidence comment and close each listed issue")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue stale-fixed: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	resolvedRef, err := issueInventoryTrunkRef(*root, *trunkRef)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue stale-fixed: branch roles: %v\n", err)
		return 1
	}
	var issues []issuepolicy.IssueDraft
	if strings.TrimSpace(*fromIssues) != "" {
		raw, rerr := readIssueDedupInput(*fromIssues)
		if rerr != nil {
			fmt.Fprintf(stderr, "fak-dev issue stale-fixed: read --from-issues: %v\n", rerr)
			return 2
		}
		issues, err = decodeIssueContractIssues(raw)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue stale-fixed: decode --from-issues: %v\n", err)
			return 2
		}
	} else {
		issues, err = issuecontractrepair.FetchOpenIssues(resolveRoot(*root), *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue stale-fixed: fetch open issues (pass --from-issues to work offline): %v\n", err)
			return 1
		}
	}
	scan := scanner
	if scan == nil {
		scan = staleFixedScanGit
	}
	trunkSHA, commits, err := scan(*root, resolvedRef, *depth)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue stale-fixed: %v\n", err)
		return 1
	}
	result := staleFixedResult{
		Schema:         staleFixedSchema,
		Repo:           *repo,
		TrunkRef:       resolvedRef,
		TrunkSHA:       trunkSHA,
		Depth:          *depth,
		ScannedCommits: len(commits),
		OpenIssues:     len(issues),
		Apply:          *apply,
		Rows:           []staleFixedRow{},
	}
	buildStaleFixedRows(&result, issues, commits)
	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	if *apply {
		applyStaleFixed(&result, *repo, run)
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, result, "fak-dev issue stale-fixed"); code != 0 {
			return code
		}
	} else {
		renderStaleFixed(stdout, result)
	}
	if result.FailedCount > 0 {
		return 1
	}
	return 0
}

// buildStaleFixedRows joins the open issue set against closing-keyword commits:
// an open issue becomes a row only when at least one scanned commit carries a
// closing keyword referencing it. SHAs keep git's newest-first order within a
// row, so the first entry is the most recent closing claim.
func buildStaleFixedRows(result *staleFixedResult, issues []issuepolicy.IssueDraft, commits []staleFixedCommit) {
	byNumber := make(map[int]issuepolicy.IssueDraft, len(issues))
	for _, is := range issues {
		if is.Number <= 0 {
			continue
		}
		if _, dup := byNumber[is.Number]; !dup {
			byNumber[is.Number] = is
		}
	}
	shasByIssue := map[int][]string{}
	for _, c := range commits {
		seenInCommit := map[int]bool{}
		for _, m := range staleFixedCloseRE.FindAllStringSubmatch(c.Message, -1) {
			n, err := strconv.Atoi(m[2])
			if err != nil || n <= 0 || seenInCommit[n] {
				continue
			}
			seenInCommit[n] = true
			shasByIssue[n] = append(shasByIssue[n], c.SHA)
		}
	}
	nums := make([]int, 0, len(shasByIssue))
	for n := range shasByIssue {
		if _, open := byNumber[n]; open {
			nums = append(nums, n)
		}
	}
	sort.Ints(nums)
	for _, n := range nums {
		is := byNumber[n]
		result.Rows = append(result.Rows, staleFixedRow{Number: n, Title: is.Title, URL: is.URL, Shas: shasByIssue[n]})
	}
}

// applyStaleFixed posts the evidence comment and closes each row through the
// governed gh atom. A failed comment skips that row's close; failures never
// stop the remaining rows (mirrors decompose's per-row application).
func applyStaleFixed(result *staleFixedResult, repo string, run issueCreateRunner) {
	for i := range result.Rows {
		row := &result.Rows[i]
		body := renderStaleFixedEvidence(row, result.TrunkRef)
		commentArgs := []string{"issue", "comment", strconv.Itoa(row.Number)}
		if repo != "" {
			commentArgs = append(commentArgs, "--repo", repo)
		}
		commentArgs = append(commentArgs, "--body", body)
		if _, errOut, ok := run(commentArgs); !ok {
			row.Error = "gh issue comment failed: " + strings.TrimSpace(errOut)
			result.FailedCount++
			continue
		}
		row.CommentOK = true
		closeArgs := []string{"issue", "close", strconv.Itoa(row.Number)}
		if repo != "" {
			closeArgs = append(closeArgs, "--repo", repo)
		}
		if _, errOut, ok := run(closeArgs); !ok {
			row.Error = "gh issue close failed: " + strings.TrimSpace(errOut)
			result.FailedCount++
			continue
		}
		row.CloseOK = true
		row.Applied = true
		result.AppliedCount++
	}
}

func renderStaleFixedEvidence(row *staleFixedRow, trunkRef string) string {
	var b strings.Builder
	b.WriteString("Closing-keyword commits found on trunk `" + trunkRef + "` referencing this issue:\n\n")
	for _, sha := range row.Shas {
		fmt.Fprintf(&b, "- %s\n", sha)
	}
	b.WriteString("\nAuto-closed by `fak-dev issue stale-fixed` (fak-private#3392). Verify the cited commits before relying on this closure.\n")
	return b.String()
}

func renderStaleFixed(w io.Writer, result staleFixedResult) {
	fmt.Fprintf(w, "stale-fixed: trunk=%s@%s scanned=%d open_issues=%d candidates=%d\n",
		result.TrunkRef, shortSHA(result.TrunkSHA), result.ScannedCommits, result.OpenIssues, len(result.Rows))
	for _, row := range result.Rows {
		fmt.Fprintf(w, "#%d %s\n", row.Number, row.Title)
		fmt.Fprintf(w, "  closed-by: %s\n", strings.Join(row.Shas, ", "))
		if row.Applied {
			fmt.Fprintln(w, "  applied: commented + closed")
		}
		if row.Error != "" {
			fmt.Fprintf(w, "  ERROR: %s\n", row.Error)
		}
	}
	if len(result.Rows) == 0 {
		fmt.Fprintln(w, "no open issue has a closing-keyword commit on trunk")
	}
	if !result.Apply {
		fmt.Fprintln(w, "dry-run: nothing was closed; re-run with --apply to post evidence comments and close")
	}
}

// --- git scanning -----------------------------------------------------------

// staleFixedScanGit resolves the trunk ref and streams its history with
// NUL-separated records: tformat prepends a newline per record, %H%x00 marks the
// SHA, %B carries subject+body, and %x1e terminates each record so multi-line
// messages cannot bleed across boundaries. The %xNN hex escapes keep the argv
// free of raw control bytes, which Windows CreateProcess rejects.
func staleFixedScanGit(root, ref string, depth int) (string, []staleFixedCommit, error) {
	ctx, cancel := context.WithTimeout(context.Background(), staleFixedGitTimeout)
	defer cancel()
	sha, err := staleFixedGitOut(ctx, root, "rev-parse", "--verify", "--end-of-options", ref+"^{commit}")
	if err != nil {
		return "", nil, fmt.Errorf("resolve trunk ref %q: %w", ref, err)
	}
	sha = strings.TrimSpace(sha)
	args := []string{"--no-lazy-fetch", "--no-replace-objects", "--no-optional-locks", "log"}
	if depth > 0 {
		args = append(args, fmt.Sprintf("--max-count=%d", depth))
	}
	args = append(args, ref, "--pretty=tformat:%H%x00%B%x1e")
	out, err := staleFixedGitOut(ctx, root, args...)
	if err != nil {
		return sha, nil, err
	}
	return sha, parseStaleFixedLog(out), nil
}

func staleFixedGitOut(ctx context.Context, root string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", root}, args...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	verb := args[0]
	for _, a := range args {
		if !strings.HasPrefix(a, "-") {
			verb = a
			break
		}
	}
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("git %s timed out after %s", verb, staleFixedGitTimeout)
	}
	if err != nil {
		return "", fmt.Errorf("git %s: %v (%s)", verb, err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

// parseStaleFixedLog splits tformat records ("\n<sha>\x00<message>\x1e" chunks)
// into commit records in git's newest-first order. Malformed chunks are skipped
// rather than failing the scan; scanned_commits reports how much survived.
func parseStaleFixedLog(raw string) []staleFixedCommit {
	var commits []staleFixedCommit
	for _, chunk := range strings.Split(raw, "\x1e") {
		chunk = strings.TrimPrefix(chunk, "\n")
		i := strings.IndexByte(chunk, '\x00')
		if i <= 0 {
			continue
		}
		sha := chunk[:i]
		if len(sha) != 40 || !staleFixedIsHex(sha) {
			continue
		}
		commits = append(commits, staleFixedCommit{SHA: sha, Message: chunk[i+1:]})
	}
	return commits
}

func staleFixedIsHex(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
