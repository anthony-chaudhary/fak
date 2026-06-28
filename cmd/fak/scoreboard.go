package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// cmdScoreboard posts fak status to the scoreboard Slack channel.
//
//	fak scoreboard post --from card.json --debt-key conflation_debt
//	fak scoreboard post --kpi code-debt --value 10 --grade A --verdict OK
//	fak scoreboard post --kpi gate --detail "make ci green" --dry-run
//
// It targets the FAK_SCOREBOARD_* workspace (a separate Slack workspace from the
// lab/DGX control bridge), so a CI job or a local agent publishes a number the
// moment it changes without touching the lab plumbing.
func cmdScoreboard(argv []string) {
	dispatchSubcommands("scoreboard", "post", argv,
		subcommand{"post", runScoreboardPost},
	)
}

func runScoreboardPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scoreboard post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	from := fs.String("from", "", "read a pkg/scorecard control-pane JSON payload from this file (- for stdin)")
	debtKey := fs.String("debt-key", "", "with --from: which corpus integer is the headline debt (e.g. conflation_debt)")
	title := fs.String("title", "", "post title (default: derived from --from schema or --kpi)")
	kpi := fs.String("kpi", "", "ad-hoc post: the metric name (e.g. code-debt)")
	value := fs.String("value", "", "ad-hoc post: the metric value")
	grade := fs.String("grade", "", "ad-hoc post: A-F grade")
	verdict := fs.String("verdict", "", "ad-hoc post: OK | ACTION")
	detail := fs.String("detail", "", "ad-hoc post: one-line finding")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_SCOREBOARD_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	up, err := buildUpdate(stderr, *from, *debtKey, *title, *kpi, *value, *grade, *verdict, *detail, *source)
	if err != nil {
		fmt.Fprintf(stderr, "fak scoreboard post: %v\n", err)
		return 2
	}

	return scoreboardPostFlow(stdout, stderr, up, scoreboardPostOpts{
		prefix:        "fak scoreboard post",
		channelFlag:   *channel,
		tokenFlag:     *token,
		resolveChan:   scoreboard.ResolveChannel,
		noChannelHint: "fak scoreboard post: no channel: pass --channel, set FAK_SCOREBOARD_CHANNEL, or add it to .env.slack.local",
		dryRun:        *dryRun,
		dedupe:        true,
	})
}

// buildUpdate assembles the scoreboard Update from either a --from payload or the
// ad-hoc --kpi flags. Exactly one path must produce content.
func buildUpdate(stderr io.Writer, from, debtKey, title, kpi, value, grade, verdict, detail, source string) (scoreboard.Update, error) {
	src := source
	if src == "" {
		src = defaultSource()
	}

	if from != "" {
		return scoreboardPayloadUpdate(from, debtKey, title, src, "scorecard")
	}

	if kpi == "" {
		return scoreboard.Update{}, fmt.Errorf("nothing to post: pass --from <payload.json> or --kpi <name>")
	}
	return scoreboardKPIUpdate(title, kpi, value, grade, verdict, detail, src), nil
}

// scoreboardPayloadUpdate reads a pkg/scorecard control-pane JSON payload from
// `from`, folds it into a scoreboard.Update via FromPayload (tagged with src),
// and titles it from --title, else the payload schema, else fallbackTitle. It
// is the shared --from branch used by the scoreboard/nodeusage/product feeds.
func scoreboardPayloadUpdate(from, debtKey, title, src, fallbackTitle string) (scoreboard.Update, error) {
	raw, err := readFromFile(from)
	if err != nil {
		return scoreboard.Update{}, err
	}
	var p scorecard.Payload
	if err := json.Unmarshal(raw, &p); err != nil {
		return scoreboard.Update{}, fmt.Errorf("parse --from payload: %w", err)
	}
	return scoreboardPayloadUpdateFrom(p, debtKey, title, src, fallbackTitle), nil
}

// scoreboardPayloadUpdateFrom folds an already-parsed scorecard payload into an
// Update, applying the shared title-resolution (--title > schema > fallback).
func scoreboardPayloadUpdateFrom(p scorecard.Payload, debtKey, title, src, fallbackTitle string) scoreboard.Update {
	t := title
	if t == "" {
		t = p.Schema
	}
	if t == "" {
		t = fallbackTitle
	}
	up := scoreboard.FromPayload(t, p, debtKey)
	up.Source = src
	return up
}

// scoreboardKPIUpdate builds the ad-hoc --kpi Update shared by the scoreboard
// and nodeusage feeds: title defaults to the kpi name.
func scoreboardKPIUpdate(title, kpi, value, grade, verdict, detail, src string) scoreboard.Update {
	t := title
	if t == "" {
		t = kpi
	}
	return scoreboard.Update{
		Title:   t,
		Score:   value,
		Grade:   grade,
		Verdict: verdict,
		Detail:  detail,
		Source:  src,
	}
}

func readFromFile(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}

func defaultSource() string {
	if v := os.Getenv("FAK_SCOREBOARD_SOURCE"); v != "" {
		return v
	}
	if h, err := os.Hostname(); err == nil {
		return h
	}
	return ""
}
