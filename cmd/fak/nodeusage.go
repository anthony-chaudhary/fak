package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/nodeusagepost"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// cmdNodeUsage posts compute-node-usage status to the #node-usage Slack channel.
//
//	fak nodeusage post --fleet snap.json            # fold a `fak lab status --json` snapshot
//	fak lab status --json | fak nodeusage post --fleet -
//	fak nodeusage post --kpi active-workers --value 3 --grade A --verdict OK
//	fak nodeusage post --from card.json --debt-key <key>
//
// It targets the SAME workspace as `fak scoreboard` (team FAK_SCOREBOARD_TEAM) but a
// DIFFERENT channel: #node-usage carries the latest compute-node usage (fleet/node
// readiness, worker count, inbound load). The bot token is shared
// (FAK_SCOREBOARD_TOKEN, via the node-usage fallback); only the channel differs
// (FAK_NODE_USAGE_CHANNEL). A node-usage post NEVER falls back to #scoreboard — that
// would put node-usage status in the number feed.
func cmdNodeUsage(argv []string) {
	dispatchSubcommands("nodeusage", "post", argv,
		subcommand{"post", runNodeUsagePost},
	)
}

func runNodeUsagePost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak nodeusage post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fleetSrc := fs.String("fleet", "", "fold a `fak lab status --json` fleet snapshot into the card (- for stdin)")
	from := fs.String("from", "", "read a pkg/scorecard control-pane JSON payload from this file (- for stdin)")
	debtKey := fs.String("debt-key", "", "with --from: which corpus integer is the headline debt")
	title := fs.String("title", "", "post title (default: derived from --fleet/--from/--kpi)")
	kpi := fs.String("kpi", "", "ad-hoc post: the metric name (e.g. active-workers, open-issues)")
	value := fs.String("value", "", "ad-hoc post: the metric value")
	grade := fs.String("grade", "", "ad-hoc post: A-F grade")
	verdict := fs.String("verdict", "", "ad-hoc post: OK | ACTION")
	detail := fs.String("detail", "", "ad-hoc post: one-line finding")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_NODE_USAGE_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_NODE_USAGE_TOKEN / $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	up, err := buildNodeUsageUpdate(*fleetSrc, *from, *debtKey, *title, *kpi, *value, *grade, *verdict, *detail, *source)
	if err != nil {
		fmt.Fprintf(stderr, "fak nodeusage post: %v\n", err)
		return 2
	}

	return scoreboardPostFlow(stdout, stderr, up, scoreboardPostOpts{
		prefix:        "fak nodeusage post",
		channelFlag:   *channel,
		tokenFlag:     *token,
		resolveChan:   nodeusagepost.ResolveChannel,
		resolveToken:  nodeusagepost.ResolveToken,
		noChannelHint: "fak nodeusage post: no channel: pass --channel, set FAK_NODE_USAGE_CHANNEL, or add it to .env.slack.local",
		dryRun:        *dryRun,
		dedupe:        false,
	})
}

// buildNodeUsageUpdate assembles the Update from exactly one content source: a fleet
// snapshot (--fleet, the headline node-usage signal), a scorecard payload (--from), or
// an ad-hoc KPI (--kpi, the worker-count / inbound-load signals).
func buildNodeUsageUpdate(fleetSrc, from, debtKey, title, kpi, value, grade, verdict, detail, source string) (scoreboard.Update, error) {
	src := source
	if src == "" {
		src = defaultSource()
	}

	if !exactlyOneNodeUsageSource(fleetSrc, from, kpi) {
		return scoreboard.Update{}, fmt.Errorf("pass exactly one of --fleet <snapshot.json> / --from <payload.json> / --kpi <name>")
	}

	switch {
	case fleetSrc != "":
		raw, err := readFromFile(fleetSrc)
		if err != nil {
			return scoreboard.Update{}, err
		}
		snap, err := nodeusagepost.ParseSnapshot(raw)
		if err != nil {
			return scoreboard.Update{}, err
		}
		up := nodeusagepost.FromSnapshot(snap, src)
		if title != "" {
			up.Title = title
		}
		return up, nil

	case from != "":
		return scoreboardPayloadUpdate(from, debtKey, title, src, "node usage")

	default: // kpi != ""
		return scoreboardKPIUpdate(title, kpi, value, grade, verdict, detail, src), nil
	}
}

// exactlyOneNodeUsageSource reports whether exactly one content source was selected, so
// the build refuses an ambiguous --fleet+--kpi combo up front.
func exactlyOneNodeUsageSource(fleetSrc, from, kpi string) bool {
	n := 0
	for _, s := range []string{fleetSrc, from, kpi} {
		if s != "" {
			n++
		}
	}
	return n == 1
}
