package main

// dispatch_skipped.go — `fak dispatch skipped`, the surface that pushes the issues the
// dispatch router HELD BACK FOR A HUMAN into a dedicated Slack channel. The router already
// classifies every skipped issue (`fak dispatch route`'s skipped_human_blocked set); the
// subset with skip reason BLOCKED_BY_HUMAN is exactly the work no worker may take until a
// person acts. This verb folds that subset into ONE compact card so an operator watching one
// channel sees the fleet's human-blocked backlog without reading a router dump.
//
//	fak dispatch skipped --dry-run                 # render the card, post nothing (default-safe)
//	fak dispatch skipped --channel C0ABC123        # post to a specific channel
//	fak dispatch skipped --repo-url https://github.com/owner/repo   # link the issue rows
//	fak dispatch skipped                           # post to $FAK_SKIPPED_CHANNEL
//
// It invents no transport: it reuses the same router seam `fak dispatch route` uses and posts
// through the internal/scoreboard chat.postMessage client every Slack surface reuses, resolving
// the token the shared env-then-.env.slack.local way. Slack is internal-only for now, so there
// is NO public channel default: an unresolved channel is a skip (or a usage error on a live
// run), never a misroute to a baked-in id.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// skippedChannelEnv is the dedicated channel key for the human-blocked surface. It has no
// public default (Slack is internal-only): resolve it from --channel, this env var, or a
// FAK_SKIPPED_CHANNEL= line in .env.slack.local, or the post is skipped.
const skippedChannelEnv = "FAK_SKIPPED_CHANNEL"

// skippedMaxRows caps how many issue rows the card lists so a large human-blocked backlog stays
// one readable post; the overflow is summarised as a "… and N more" tail.
const skippedMaxRows = 20

// reasonBlockedByHuman is the router skip-reason wire value for an issue held back from dispatch
// because a human must unblock it. It is matched by value here rather than imported from
// dispatchtick so this surface stays decoupled from the router package's internal reason
// constants — the router has emitted this exact string for the blocked-by-human skip since the
// classifier was written.
const reasonBlockedByHuman = "BLOCKED_BY_HUMAN"

// runDispatchSkipped is the testable core of `fak dispatch skipped`. It returns the process exit
// code (0 ok/dry-run, 1 a runtime/post error, 2 a usage/config error).
func runDispatchSkipped(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dispatch skipped", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	repoURL := fs.String("repo-url", "", "repo base URL for issue links, e.g. https://github.com/owner/repo")
	channel := fs.String("channel", "", "target channel id (default: $"+skippedChannelEnv+" / .env.slack.local)")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	root := strings.TrimSpace(*workspace)
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch skipped: getwd: %v\n", err)
			return 1
		}
		root = wd
	}

	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch skipped: %v\n", err)
		return 1
	}
	blocked := humanBlockedSkipped(router)
	card := renderSkippedHumanBlockedCard(blocked, *repoURL)

	if *dryRun {
		fmt.Fprintln(stdout, card)
		fmt.Fprintf(stdout, "\n(dry-run: %d human-blocked issue(s); would post to %s)\n",
			len(blocked), orUnset(resolveSkippedChannel(*channel)))
		return 0
	}

	target := resolveSkippedChannel(*channel)
	if target == "" {
		fmt.Fprintf(stderr, "fak dispatch skipped: no target channel — set --channel or %s (env / %s)\n",
			skippedChannelEnv, slackenv.EnvFileName)
		return 2
	}
	tok := strings.TrimSpace(*token)
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}
	if tok == "" {
		fmt.Fprintf(stderr, "fak dispatch skipped: no bot token — set --token, FAK_SCOREBOARD_TOKEN, or add it to %s\n",
			slackenv.EnvFileName)
		return 2
	}

	var opts []scoreboard.Option
	if *apiBase != "" {
		opts = append(opts, scoreboard.WithAPIBase(*apiBase))
	}
	c, err := scoreboard.NewClient(tok, opts...)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch skipped: %v\n", err)
		return 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ts, err := c.Post(ctx, target, card, nil)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch skipped: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak dispatch skipped: posted %d human-blocked issue(s) to %s (ts=%s)\n", len(blocked), target, ts)
	return 0
}

// humanBlockedSkipped selects the router's skipped issues whose reason is BLOCKED_BY_HUMAN — the
// subset a human must unblock — out of the full skipped_human_blocked set (which also holds
// epics, triage-only, duplicate-risk, and scope-incomplete rows that are NOT a human's to
// clear). The router already sorts the skipped slice highest-number-first, so order is preserved.
func humanBlockedSkipped(router dispatchtick.RouterPayload) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0, len(router.SkippedHumanBlocked))
	for _, s := range router.SkippedHumanBlocked {
		if s.Reason == reasonBlockedByHuman {
			out = append(out, s)
		}
	}
	return out
}

// renderSkippedHumanBlockedCard folds the human-blocked skipped issues into one Slack card
// (mrkdwn text). Empty → a quiet all-clear line so a scheduled run posts an honest "nothing
// held for a human"; non-empty → a headline count plus one row per issue, capped at
// skippedMaxRows with an overflow tail. It is pure so it is unit-tested without gh, the network,
// or a Slack token.
func renderSkippedHumanBlockedCard(issues []dispatchtick.SkippedIssue, repoURL string) string {
	if len(issues) == 0 {
		return ":white_check_mark: *no human-blocked issues* — the dispatch router is holding nothing for a human"
	}
	var b strings.Builder
	fmt.Fprintf(&b, ":no_entry: *%d issue(s) blocked by a human* — skipped from dispatch, waiting on you", len(issues))
	shown := issues
	if len(shown) > skippedMaxRows {
		shown = shown[:skippedMaxRows]
	}
	for _, s := range shown {
		fmt.Fprintf(&b, "\n• %s", skippedIssueRow(s, repoURL))
	}
	if len(issues) > skippedMaxRows {
		fmt.Fprintf(&b, "\n… and %d more", len(issues)-skippedMaxRows)
	}
	return b.String()
}

// skippedIssueRow renders one issue line: a Slack #-link when a repo URL is set (else a bare
// #N), the title, and the router's "next action" hint (what a human must do to unblock it).
func skippedIssueRow(s dispatchtick.SkippedIssue, repoURL string) string {
	ref := fmt.Sprintf("#%d", s.Number)
	if base := strings.TrimRight(strings.TrimSpace(repoURL), "/"); base != "" {
		ref = fmt.Sprintf("<%s/issues/%d|#%d>", base, s.Number, s.Number)
	}
	row := ref
	if t := strings.TrimSpace(s.Title); t != "" {
		row += " " + t
	}
	if n := strings.TrimSpace(s.NextAction); n != "" {
		row += " — " + n
	}
	return row
}

// resolveSkippedChannel picks the target channel: the flag, else FAK_SKIPPED_CHANNEL from the
// env or a .env.slack.local line (the shared resolver every Slack surface reads). There is no
// public built-in default, so an unset channel returns "" and the caller skips or errors —
// never a misroute to a baked-in id.
func resolveSkippedChannel(flagVal string) string {
	if v := strings.TrimSpace(flagVal); v != "" {
		return v
	}
	return slackenv.Lookup(skippedChannelEnv).Value
}
