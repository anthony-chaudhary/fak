package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/ghexec"
)

// cmdBlockers posts a BLOCKER to the central Slack #blockers channel — the one place
// the fleet records an ongoing impediment and surfaces a human-needed one.
//
//	fak blockers post --title "GPU-gated, waiting on GPU-server hours"      # background status
//	fak blockers post --severity operator --title "CPU host unreachable" \
//	      --detail "CPU node not responding" --action "restart the serve"  # pages <!here>
//	fak blockers post --severity operator --owner "<@U0OPS>" --title "..."  # pages one person
//	fak blockers source --repo owner/repo --label blocked \                 # CI acquisition
//	      --issues-out issues.json --status-out blockers-source.json
//	fak blockers feed --issues issues.json --label blocked                  # CI roll-up of the backlog
//
// It targets the FAK_BLOCKERS_* surface (a public channel in the scoreboard Slack
// workspace, separate from the lab/DGX control bridge); the token falls back to the
// scoreboard bot token, the channel to the built-in #blockers default. --dry-run renders
// the card and prints it without posting, matching the scoreboard/bench "safe by
// default" idiom.
func cmdBlockers(argv []string) {
	dispatchSubcommands("blockers", "post | source | feed | selfcheck", argv,
		subcommand{"post", runBlockersPost},
		subcommand{"source", runBlockersSource},
		subcommand{"feed", runBlockersFeed},
		subcommand{"selfcheck", runBlockersSelfcheck},
	)
}

// runBlockersPost handles `fak blockers post` — one hand-built blocker.
func runBlockersPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak blockers post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	severity := fs.String("severity", "status", "status (background, no page) | operator (surfaced, pages a human) | clear (all-clear heartbeat)")
	title := fs.String("title", "", "short headline, e.g. \"CPU host unreachable\"")
	detail := fs.String("detail", "", "one-line what is blocked / why")
	owner := fs.String("owner", "", "operator: who to page — a Slack mention like \"<@U123>\" or \"<!here>\" (default: <!here>)")
	action := fs.String("action", "", "operator: a \"do this next\" label, e.g. \"restart the CPU-host serve\"")
	actionURL := fs.String("action-url", "", "operator: a link for the do-this-next button (runbook / issue / docs)")
	ref := fs.String("ref", "", "optional stable key shown in context, e.g. \"#921\" or a hostname")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BLOCKERS_CHANNEL / .env.slack.local / #blockers)")
	token := fs.String("token", "", "override bot token (default: $FAK_BLOCKERS_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
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

type blockerFeedGHRunner func(args ...string) ([]byte, error)

func runBlockerFeedGH(args ...string) ([]byte, error) {
	cmd, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, args...)
	defer cancel()
	out, err := cmd.Output()
	if err == nil {
		return out, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return nil, fmt.Errorf("gh %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(exitErr.Stderr)))
	}
	return nil, fmt.Errorf("gh %s: %w", strings.Join(args, " "), err)
}

// runBlockersSource acquires the scheduled feed's GitHub payload and writes its
// fail-closed source marker. The exact configured label is witnessed immediately
// before and after the issue query, closing the missing-label race around a successful
// zero-result response. The marker starts UNKNOWN and becomes OK only after the label
// probes, query, payload validation, and output write all succeed.
func runBlockersSource(stdout, stderr io.Writer, argv []string) int {
	return runBlockersSourceWithGH(stdout, stderr, argv, runBlockerFeedGH)
}

func runBlockersSourceWithGH(stdout, stderr io.Writer, argv []string, gh blockerFeedGHRunner) int {
	fs := flag.NewFlagSet("fak blockers source", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "GitHub repository in owner/name form")
	label := fs.String("label", "blocked", "exact issue label to query")
	issuesOut := fs.String("issues-out", "", "write the validated issue-list JSON array here")
	statusOut := fs.String("status-out", "", "write the source status JSON here (UNKNOWN until every acquisition check succeeds)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*issuesOut) == "" || strings.TrimSpace(*statusOut) == "" {
		fmt.Fprintln(stderr, "fak blockers source: --issues-out and --status-out are required")
		return 2
	}

	status := blockerFeedSourceStatus{
		Status: "UNKNOWN",
		Reason: "source acquisition did not complete",
	}
	if err := writeBlockerFeedSourceStatus(*statusOut, status); err != nil {
		fmt.Fprintf(stderr, "fak blockers source: initialize UNKNOWN marker: %v\n", err)
		return 1
	}
	fail := func(reason string) int {
		status.Reason = reason
		if err := writeBlockerFeedSourceStatus(*statusOut, status); err != nil {
			fmt.Fprintf(stderr, "fak blockers source: blocker source UNKNOWN: %s (also failed to update marker: %v)\n", reason, err)
			return 1
		}
		fmt.Fprintf(stderr, "fak blockers source: blocker source UNKNOWN: %s\n", reason)
		return 1
	}

	repoName := strings.TrimSpace(*repo)
	if repoName == "" {
		return fail("GitHub repository is empty")
	}
	labelName := strings.TrimSpace(*label)
	if labelName == "" {
		return fail("configured label is empty")
	}
	labelEndpoint := fmt.Sprintf("repos/%s/labels/%s", repoName, url.PathEscape(labelName))
	probeLabel := func(when string) error {
		if _, err := gh("api", "--silent", labelEndpoint); err != nil {
			return fmt.Errorf("configured label %q could not be resolved %s the issue query: %w", labelName, when, err)
		}
		return nil
	}

	if err := probeLabel("before"); err != nil {
		return fail(err.Error())
	}
	payload, err := gh(
		"issue", "list",
		"--repo", repoName,
		"--state", "open",
		"--label", labelName,
		"--limit", "200",
		"--json", "number,title,url,assignees,labels",
	)
	if err != nil {
		return fail(fmt.Sprintf("gh issue list failed: %v", err))
	}
	issues, err := decodeFeedIssues(payload)
	if err != nil {
		return fail(fmt.Sprintf("gh issue list returned an unusable payload: %v", err))
	}
	if err := probeLabel("after"); err != nil {
		return fail(err.Error())
	}
	canonical, err := json.Marshal(issues)
	if err != nil {
		return fail(fmt.Sprintf("encode validated issue payload: %v", err))
	}
	canonical = append(canonical, '\n')
	if err := os.WriteFile(*issuesOut, canonical, 0o644); err != nil {
		return fail(fmt.Sprintf("write --issues-out: %v", err))
	}

	status.Status = "OK"
	status.Reason = "GitHub issue query completed with the configured label present before and after acquisition"
	if err := writeBlockerFeedSourceStatus(*statusOut, status); err != nil {
		fmt.Fprintf(stderr, "fak blockers source: finalize OK marker: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "blocker source OK: %d open %q issue(s)\n", len(issues), labelName)
	return 0
}

// runBlockersFeed handles `fak blockers feed` — the CI cadence roll-up. It folds a
// `gh issue list --json number,title,url,assignees,labels` payload (the open backlog
// for a blocker label) into ONE roll-up blocker: clear when empty, operator when any
// issue is unowned, background status when all are assigned.
func runBlockersFeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak blockers feed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesPath := fs.String("issues", "", "fold a `gh issue list --json number,title,url,assignees,labels` payload from this file (- for stdin)")
	sourceStatusPath := fs.String("source-status", "", "optional source-acquisition status JSON; when set, status must be OK before rendering or posting")
	label := fs.String("label", "blocked", "the issue label the backlog was filtered by (for prose + the triage link)")
	repoURL := fs.String("repo-url", "", "repo base URL for the operator triage link, e.g. https://github.com/owner/repo")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BLOCKERS_CHANNEL / .env.slack.local / #blockers)")
	token := fs.String("token", "", "override bot token (default: $FAK_BLOCKERS_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}

	if err := requireFeedSourceOK(*sourceStatusPath); err != nil {
		fmt.Fprintf(stderr, "fak blockers feed: %v\n", err)
		return 1
	}
	issues, err := loadFeedIssues(*issuesPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak blockers feed: %v\n", err)
		return 2
	}
	// Decenter the human at the feed seam: under FAK_BLOCKERS_TRIAGE_GATE=enforce,
	// an unowned issue only pages when it names authority a person holds; a
	// fleet-routable backlog is recorded as background status. Default ("", "warn")
	// keeps the ownership-only paging so the change can soak.
	var b blockerpost.Blocker
	if blockerpost.TriageEnforced(os.Getenv("FAK_BLOCKERS_TRIAGE_GATE")) {
		b = blockerpost.FoldIssuesTriaged(issues, *label, *repoURL)
	} else {
		b = blockerpost.FoldIssues(issues, *label, *repoURL)
	}
	b.Source = resolveBlockerSource(*source)
	return emitBlocker(stdout, stderr, b, *channel, *token, *dryRun)
}

// runBlockersSelfcheck runs the deterministic decenter-the-human proof for the
// blocker feed — no key, no network, no fixtures.
func runBlockersSelfcheck(stdout, stderr io.Writer, argv []string) int {
	return runReportSelfcheck(stdout, stderr, argv, "blockers", blockerpost.TriageSelfcheck,
		"SELFCHECK OK -- decenter-the-human at the blocker feed: an unowned issue that "+
			"names authority still pages; a fleet-routable one routes to the fleet and stops paging.")
}

type blockerFeedSourceStatus struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

func writeBlockerFeedSourceStatus(path string, status blockerFeedSourceStatus) error {
	raw, err := json.Marshal(status)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(path, raw, 0o644)
}

// requireFeedSourceOK checks the workflow's source-acquisition marker before any issue
// payload is read. The marker is optional for direct/manual folds, but when supplied it
// must carry the exact OK status. A missing, malformed, or UNKNOWN marker is an
// operational failure, never evidence of zero blockers.
func requireFeedSourceOK(path string) error {
	if path == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("blocker source UNKNOWN: read --source-status: %w; refusing to render or post", err)
	}
	var status blockerFeedSourceStatus
	if err := json.Unmarshal(trimSpaceBytes(raw), &status); err != nil {
		return fmt.Errorf("blocker source UNKNOWN: parse --source-status: %w; refusing to render or post", err)
	}
	if strings.TrimSpace(status.Status) != "OK" {
		reason := strings.TrimSpace(status.Reason)
		if reason == "" {
			reason = "source acquisition did not prove a usable issue query"
		}
		return fmt.Errorf("blocker source UNKNOWN: %s; refusing to render or post", reason)
	}
	return nil
}

// loadFeedIssues reads the gh issue-list payload from a file (or stdin for "-").
// The input must be an explicit, non-empty JSON array. Only a successful query that
// produced [] may fold to the all-clear card; absent, blank, null, or malformed input
// is UNKNOWN and fails closed.
func loadFeedIssues(path string) ([]blockerpost.Issue, error) {
	if path == "" {
		return nil, fmt.Errorf("--issues is required; an absent source is UNKNOWN, not zero issues")
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
	return decodeFeedIssues(raw)
}

func decodeFeedIssues(raw []byte) ([]blockerpost.Issue, error) {
	raw = trimSpaceBytes(raw)
	if len(raw) == 0 {
		return nil, fmt.Errorf("--issues payload is empty; source is UNKNOWN, not zero issues")
	}
	var issues []blockerpost.Issue
	if err := json.Unmarshal(raw, &issues); err != nil {
		return nil, fmt.Errorf("parse --issues payload: %w", err)
	}
	if issues == nil {
		return nil, fmt.Errorf("parse --issues payload: want a JSON array; null is UNKNOWN, not zero issues")
	}
	for idx, issue := range issues {
		switch {
		case issue.Number <= 0:
			return nil, fmt.Errorf("parse --issues payload: issue[%d] has no positive number", idx)
		case strings.TrimSpace(issue.Title) == "":
			return nil, fmt.Errorf("parse --issues payload: issue[%d] has no title", idx)
		case strings.TrimSpace(issue.URL) == "":
			return nil, fmt.Errorf("parse --issues payload: issue[%d] has no URL", idx)
		case issue.Assignees == nil:
			return nil, fmt.Errorf("parse --issues payload: issue[%d] has no assignees array", idx)
		case issue.Labels == nil:
			return nil, fmt.Errorf("parse --issues payload: issue[%d] has no labels array", idx)
		}
	}
	return issues, nil
}

// emitBlocker is the shared dry-run / post tail: render to stdout under --dry-run, else
// resolve channel+token and post via the scoreboard transport (the same chat.postMessage
// client every feeder reuses).
func emitBlocker(stdout, stderr io.Writer, b blockerpost.Blocker, channel, token string, dryRun bool) int {
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           b,
		channel:        channel,
		token:          token,
		dryRun:         dryRun,
		label:          "fak blockers",
		chanEnv:        "FAK_BLOCKERS_CHANNEL",
		resolveChannel: blockerpost.ResolveChannel,
		resolveToken:   blockerpost.ResolveToken,
	})
}

// resolveBlockerSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveBlockerSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}

// trimSpaceBytes drops leading/trailing ASCII whitespace so a whitespace-only source
// is recognized as unusable rather than misread as a successful zero-result query.
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
