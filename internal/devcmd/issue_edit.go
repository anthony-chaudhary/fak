package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// issueEditResult is the --json shape for `fak-dev issue edit`: the rendered gh argv
// is always included (even on a dry run) so a caller sees exactly what would run
// or did run — the same contract as issueCreateResult (issue_create.go:18).
type issueEditResult struct {
	OK           bool     `json:"ok"`
	DryRun       bool     `json:"dry_run"`
	Issue        int      `json:"issue"`
	Title        string   `json:"title,omitempty"`
	Repo         string   `json:"repo,omitempty"`
	AddLabels    []string `json:"add_labels,omitempty"`
	RemoveLabels []string `json:"remove_labels,omitempty"`
	// DroppedLabels are agent-proposed --add-label tokens that named no real repo
	// label and were clamped out before dispatch (#4047). Surfaced (not silently
	// dropped) so the caller — human or agent — sees exactly which hallucinated
	// tokens never reached `gh issue edit`.
	DroppedLabels []string `json:"dropped_labels,omitempty"`
	Args          []string `json:"args"`
	URL           string   `json:"url,omitempty"`
	Error         string   `json:"error,omitempty"`
}

func runIssueEdit(stdout, stderr io.Writer, argv []string) int {
	return runIssueEditWith(stdout, stderr, argv, nil)
}

// runIssueEditWith is the governed mutation twin of runIssueCreateWith
// (issue_create.go:42): it shells to `gh issue edit N ...` from the trusted fak
// binary rather than the model proposing a raw `gh issue edit` via Bash. It is
// the single apply atom every repair path routes through — `fak-dev issue repair`
// and (later) the decompose consumer build gh argv and hand it to this same
// injected runner, so live edits have one auditable seam and one place tests
// pin. Unlike create it defaults LIVE (a repair loop wants writes), with
// --dry-run as the opt-out; the runner param exists purely so tests inject a
// fake and assert on the built argv without ever invoking real gh.
func runIssueEditWith(stdout, stderr io.Writer, argv []string, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue edit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issue := fs.Int("issue", 0, "issue number to edit (required)")
	title := fs.String("title", "", "new issue title")
	body := fs.String("body", "", "new issue body text (exactly one of --body/--body-file)")
	bodyFile := fs.String("body-file", "", "path to a file containing the new issue body")
	addLabel := fs.String("add-label", "", "comma-separated labels to add")
	removeLabel := fs.String("remove-label", "", "comma-separated labels to remove")
	repo := fs.String("repo", "", "owner/name override (default: gh infers from cwd)")
	dryRun := fs.Bool("dry-run", false, "render the gh argv without calling gh")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue edit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *issue <= 0 {
		fmt.Fprintln(stderr, "fak-dev issue edit: --issue N (a positive issue number) is required")
		return 2
	}
	if strings.TrimSpace(*body) != "" && strings.TrimSpace(*bodyFile) != "" {
		fmt.Fprintln(stderr, "fak-dev issue edit: pass at most one of --body or --body-file")
		return 2
	}
	resolvedBody := *body
	haveBody := strings.TrimSpace(*body) != ""
	if strings.TrimSpace(*bodyFile) != "" {
		b, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue edit: read --body-file: %v\n", err)
			return 2
		}
		resolvedBody = string(b)
		haveBody = true
	}

	addLabels := issueFanoutSplit(*addLabel)
	removeLabels := issueFanoutSplit(*removeLabel)
	haveTitle := strings.TrimSpace(*title) != ""
	if !haveTitle && !haveBody && len(addLabels) == 0 && len(removeLabels) == 0 {
		fmt.Fprintln(stderr, "fak-dev issue edit: nothing to change — pass at least one of --title/--body/--body-file/--add-label/--remove-label")
		return 2
	}

	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}

	// Closed-vocabulary label clamp at the actuator (#4047): filter agent-proposed
	// --add-label tokens against the repo's REAL label set (`gh label list`) so a
	// hallucinated label never reaches the `gh issue edit` side effect. LOUD, not
	// silent — every dropped token is recorded in issueEditResult.DroppedLabels and
	// echoed to stderr, giving the agent a signal it can self-correct on (fak's
	// refuse-don't-drop norm, improving on the source's silent drop). The clamp runs
	// on the dry-run path too so the rendered argv is truthful. Fail-OPEN on a
	// label-list outage (keep the proposed set, warn) because a read failure must
	// not wedge a repair loop, and an unknown label already errors at gh itself.
	var droppedLabels []string
	if len(addLabels) > 0 {
		if canonical, ok := repoCanonicalLabels(run, *repo); ok {
			addLabels, droppedLabels = clampLabelsToCanonical(addLabels, canonical)
			if len(droppedLabels) > 0 {
				fmt.Fprintf(stderr, "fak-dev issue edit: dropped %d label(s) not in the repo label set: %s\n",
					len(droppedLabels), strings.Join(droppedLabels, ", "))
			}
		} else {
			fmt.Fprintln(stderr, "fak-dev issue edit: warning: could not fetch repo label set; skipping label clamp")
		}
	}

	args := []string{"issue", "edit", strconv.Itoa(*issue)}
	if haveTitle {
		args = append(args, "--title", *title)
	}
	if haveBody {
		args = append(args, "--body", resolvedBody)
	}
	for _, l := range addLabels {
		args = append(args, "--add-label", l)
	}
	for _, l := range removeLabels {
		args = append(args, "--remove-label", l)
	}
	if strings.TrimSpace(*repo) != "" {
		args = append(args, "--repo", *repo)
	}

	result := issueEditResult{
		DryRun: *dryRun, Issue: *issue, Title: *title, Repo: *repo,
		AddLabels: addLabels, RemoveLabels: removeLabels, DroppedLabels: droppedLabels, Args: args,
	}

	if *dryRun {
		result.OK = true
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue edit")
		}
		fmt.Fprintf(stdout, "fak-dev issue edit --dry-run: would run `gh %s`\n", strings.Join(args, " "))
		return 0
	}

	out, errOut, ok := run(args)
	result.OK = ok
	result.URL = strings.TrimSpace(out)
	if !ok {
		result.Error = strings.TrimSpace(errOut)
		if *asJSON {
			encodeJSONOrFail(stdout, stderr, result, "fak-dev issue edit")
		} else {
			fmt.Fprintf(stderr, "fak-dev issue edit: gh failed: %s\n", result.Error)
		}
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue edit")
	}
	fmt.Fprintln(stdout, result.URL)
	return 0
}

// repoCanonicalLabels fetches the repo's real label set via `gh label list`,
// reusing the SAME injected runner the edit dispatch uses so a test can stub both
// the list and the edit through one fake. Returns the label names and whether the
// fetch succeeded; a failed or unparseable fetch returns ok=false so the caller can
// fail-open loudly rather than clamp against an empty set (which would drop every
// proposed label). --limit 500 matches the source shim's ceiling.
func repoCanonicalLabels(run issueCreateRunner, repo string) ([]string, bool) {
	args := []string{"label", "list", "--json", "name", "--limit", "500"}
	if strings.TrimSpace(repo) != "" {
		args = append(args, "--repo", repo)
	}
	out, _, ok := run(args)
	if !ok {
		return nil, false
	}
	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &rows); err != nil {
		return nil, false
	}
	names := make([]string, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Name) != "" {
			names = append(names, r.Name)
		}
	}
	return names, true
}

// clampLabelsToCanonical keeps only proposed labels that name a real repo label
// (matched case-insensitively, emitted in the canonical spelling); every other
// proposed token is DROPPED and returned separately so the caller can surface it
// loudly. This is the closed-vocabulary clamp at the actuator (#4047): a
// hallucinated label never reaches `gh issue edit --add-label`. Order is preserved
// and duplicates in the proposed list are collapsed to a single canonical entry.
func clampLabelsToCanonical(proposed, canonical []string) (kept, dropped []string) {
	byLower := make(map[string]string, len(canonical))
	for _, c := range canonical {
		byLower[strings.ToLower(strings.TrimSpace(c))] = c
	}
	seen := make(map[string]bool, len(proposed))
	for _, p := range proposed {
		key := strings.ToLower(strings.TrimSpace(p))
		if canon, ok := byLower[key]; ok {
			if !seen[key] {
				seen[key] = true
				kept = append(kept, canon)
			}
			continue
		}
		dropped = append(dropped, p)
	}
	return kept, dropped
}
