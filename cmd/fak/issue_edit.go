package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// issueEditResult is the --json shape for `fak issue edit`: the rendered gh argv
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
	Args         []string `json:"args"`
	URL          string   `json:"url,omitempty"`
	Error        string   `json:"error,omitempty"`
}

func runIssueEdit(stdout, stderr io.Writer, argv []string) int {
	return runIssueEditWith(stdout, stderr, argv, nil)
}

// runIssueEditWith is the governed mutation twin of runIssueCreateWith
// (issue_create.go:42): it shells to `gh issue edit N ...` from the trusted fak
// binary rather than the model proposing a raw `gh issue edit` via Bash. It is
// the single apply atom every repair path routes through — `fak issue repair`
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
		fmt.Fprintf(stderr, "fak issue edit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *issue <= 0 {
		fmt.Fprintln(stderr, "fak issue edit: --issue N (a positive issue number) is required")
		return 2
	}
	if strings.TrimSpace(*body) != "" && strings.TrimSpace(*bodyFile) != "" {
		fmt.Fprintln(stderr, "fak issue edit: pass at most one of --body or --body-file")
		return 2
	}
	resolvedBody := *body
	haveBody := strings.TrimSpace(*body) != ""
	if strings.TrimSpace(*bodyFile) != "" {
		b, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue edit: read --body-file: %v\n", err)
			return 2
		}
		resolvedBody = string(b)
		haveBody = true
	}

	addLabels := issueFanoutSplit(*addLabel)
	removeLabels := issueFanoutSplit(*removeLabel)
	haveTitle := strings.TrimSpace(*title) != ""
	if !haveTitle && !haveBody && len(addLabels) == 0 && len(removeLabels) == 0 {
		fmt.Fprintln(stderr, "fak issue edit: nothing to change — pass at least one of --title/--body/--body-file/--add-label/--remove-label")
		return 2
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
		AddLabels: addLabels, RemoveLabels: removeLabels, Args: args,
	}

	if *dryRun {
		result.OK = true
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak issue edit")
		}
		fmt.Fprintf(stdout, "fak issue edit --dry-run: would run `gh %s`\n", strings.Join(args, " "))
		return 0
	}

	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	out, errOut, ok := run(args)
	result.OK = ok
	result.URL = strings.TrimSpace(out)
	if !ok {
		result.Error = strings.TrimSpace(errOut)
		if *asJSON {
			encodeJSONOrFail(stdout, stderr, result, "fak issue edit")
		} else {
			fmt.Fprintf(stderr, "fak issue edit: gh failed: %s\n", result.Error)
		}
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak issue edit")
	}
	fmt.Fprintln(stdout, result.URL)
	return 0
}
