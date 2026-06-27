package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// cmdBlockers posts a BLOCKER to the central Slack #blockers channel — the one place
// the fleet records an ongoing impediment and surfaces a human-needed one.
//
//	fak blockers post --title "GPU-gated, waiting on DGX hours"            # background status
//	fak blockers post --severity operator --title "DA33 host unreachable" \
//	      --detail "CPU node not responding" --action "restart the serve"  # pages <!here>
//	fak blockers post --severity operator --owner "<@U0OPS>" --title "..."  # pages one person
//	fak blockers feed --issues issues.json --label blocked                  # CI roll-up of the backlog
//
// It targets the FAK_BLOCKERS_* surface (a public channel in the scoreboard Slack
// workspace, separate from the lab/DGX control bridge); the token falls back to the
// scoreboard bot token, the channel to the built-in #blockers default. --dry-run renders
// the card and prints it without posting, matching the scoreboard/bench "safe by
// default" idiom.
func cmdBlockers(argv []string) {
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "fak blockers: missing subcommand (post | feed)")
		os.Exit(2)
	}
	switch argv[0] {
	case "post":
		os.Exit(runBlockersPost(os.Stdout, os.Stderr, argv[1:]))
	case "feed":
		os.Exit(runBlockersFeed(os.Stdout, os.Stderr, argv[1:]))
	default:
		fmt.Fprintf(os.Stderr, "fak blockers: unknown subcommand %q (want: post | feed)\n", argv[0])
		os.Exit(2)
	}
}

// runBlockersPost handles `fak blockers post` — one hand-built blocker.
func runBlockersPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak blockers post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	severity := fs.String("severity", "status", "status (background, no page) | operator (surfaced, pages a human) | clear (all-clear heartbeat)")
	title := fs.String("title", "", "short headline, e.g. \"DA33 host unreachable\"")
	detail := fs.String("detail", "", "one-line what is blocked / why")
	owner := fs.String("owner", "", "operator: who to page — a Slack mention like \"<@U123>\" or \"<!here>\" (default: <!here>)")
	action := fs.String("action", "", "operator: a \"do this next\" label, e.g. \"restart the DA33 serve\"")
	actionURL := fs.String("action-url", "", "operator: a link for the do-this-next button (runbook / issue / docs)")
	ref := fs.String("ref", "", "optional stable key shown in context, e.g. \"#921\" or a hostname")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BLOCKERS_CHANNEL / .env.slack.local / #blockers)")
	token := fs.String("token", "", "override bot token (default: $FAK_BLOCKERS_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	sev, ok := blockerpost.ParseSeverity(*severity)
	if !ok {
		fmt.Fprintf(stderr, "fak blockers post: unknown --severity %q (want: status | operator | clear)\n", *severity)
		return 2
	}
	if *title == "" && *detail == "" {
		fmt.Fprintln(stderr, "fak blockers post: nothing to post: pass at least --title (and usually --detail)")
		return 2
	}

	b := blockerpost.Blocker{
		Severity:  sev,
		Title:     *title,
		Detail:    *detail,
		Owner:     *owner,
		Action:    *action,
		ActionURL: *actionURL,
		Ref:       *ref,
		Source:    resolveBlockerSource(*source),
	}
	return emitBlocker(stdout, stderr, b, *channel, *token, *dryRun)
}

// runBlockersFeed handles `fak blockers feed` — the CI cadence roll-up. It folds a
// `gh issue list --json number,title,url,assignees,labels` payload (the open backlog
// for a blocker label) into ONE roll-up blocker: clear when empty, operator when any
// issue is unowned, background status when all are assigned.
func runBlockersFeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak blockers feed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesPath := fs.String("issues", "", "fold a `gh issue list --json number,title,url,assignees,labels` payload from this file (- for stdin)")
	label := fs.String("label", "blocked", "the issue label the backlog was filtered by (for prose + the triage link)")
	repoURL := fs.String("repo-url", "", "repo base URL for the operator triage link, e.g. https://github.com/owner/repo")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BLOCKERS_CHANNEL / .env.slack.local / #blockers)")
	token := fs.String("token", "", "override bot token (default: $FAK_BLOCKERS_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	issues, err := loadFeedIssues(*issuesPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak blockers feed: %v\n", err)
		return 2
	}
	b := blockerpost.FoldIssues(issues, *label, *repoURL)
	b.Source = resolveBlockerSource(*source)
	return emitBlocker(stdout, stderr, b, *channel, *token, *dryRun)
}

// loadFeedIssues reads the gh issue-list payload from a file (or stdin for "-"). An
// empty path yields no issues, which folds to the all-clear card — so a feeder whose
// `gh` step produced nothing still posts an honest "no standing blockers".
func loadFeedIssues(path string) ([]blockerpost.Issue, error) {
	if path == "" {
		return nil, nil
	}
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, err
	}
	raw = trimSpaceBytes(raw)
	if len(raw) == 0 {
		return nil, nil
	}
	var issues []blockerpost.Issue
	if err := json.Unmarshal(raw, &issues); err != nil {
		return nil, fmt.Errorf("parse --issues payload: %w", err)
	}
	return issues, nil
}

// emitBlocker is the shared dry-run / post tail: render to stdout under --dry-run, else
// resolve channel+token and post via the scoreboard transport (the same chat.postMessage
// client every feeder reuses).
func emitBlocker(stdout, stderr io.Writer, b blockerpost.Blocker, channel, token string, dryRun bool) int {
	if dryRun {
		fmt.Fprintln(stdout, b.Text())
		return 0
	}
	ch := channel
	if ch == "" {
		ch = blockerpost.ResolveChannel()
	}
	if ch == "" {
		fmt.Fprintln(stderr, "fak blockers: no channel: pass --channel, set FAK_BLOCKERS_CHANNEL, or add it to .env.slack.local")
		return 2
	}
	tok := token
	if tok == "" {
		tok = blockerpost.ResolveToken()
	}
	client, err := scoreboard.NewClient(tok)
	if err != nil {
		fmt.Fprintf(stderr, "fak blockers: %v\n", err)
		return 2
	}
	ts, err := client.Post(ctx(), ch, b.Text(), b.Blocks())
	if err != nil {
		fmt.Fprintf(stderr, "fak blockers: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}

// resolveBlockerSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveBlockerSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}

// trimSpaceBytes drops leading/trailing ASCII whitespace so an empty-but-whitespace gh
// payload (e.g. a lone newline) folds to the all-clear card rather than a parse error.
func trimSpaceBytes(b []byte) []byte {
	start, end := 0, len(b)
	for start < end && asciiSpace(b[start]) {
		start++
	}
	for end > start && asciiSpace(b[end-1]) {
		end--
	}
	return b[start:end]
}

func asciiSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
