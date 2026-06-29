package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluepost"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// cmdCachevalue posts the cache-effectiveness P&L roll-up — fak's WITNESSED kernel
// cache-value trend (Track 1 of epic #1301) — to the central Slack #cache-value channel
// (C0BDSB81XDZ), the one durable place the fleet reads "is fak's cache method paying off,
// and is it trending up or down?".
//
//	fak cachevalue feed                                  # fold the dogfooded ledger → card
//	fak cachevalue feed --dry-run                        # render the exact card; do not post
//	fak cachevalue feed --ledger docs/nightrun/cache-value.jsonl
//	fak cachevalue post --report-json report.json        # post a pre-rolled report (- for stdin)
//
// It targets the FAK_CACHEVALUE_* surface (a public channel in the scoreboard Slack
// workspace, separate from the lab/DGX control bridge); the token falls back to the
// scoreboard bot token, the channel to the built-in #cache-value default. --dry-run
// renders the card and prints it without posting, matching the scoreboard/bench/blockers
// "safe by default" idiom.
func cmdCachevalue(argv []string) {
	dispatchSubcommands("cachevalue", "post | feed", argv,
		subcommand{"post", runCachevaluePost},
		subcommand{"feed", runCachevalueFeed},
	)
}

// runCachevalueFeed handles `fak cachevalue feed` — the cadence roll-up. It reads the
// durable kernel cache-value ledger (docs/nightrun/cache-value.jsonl), folds it into the
// weekly Track-1 trend report (internal/cachevaluereport), and posts ONE card. A missing
// or empty ledger folds to the honest INSUFFICIENT card rather than failing.
func runCachevalueFeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue feed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "the durable cache-value ledger to fold (docs/nightrun/cache-value.jsonl)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #cache-value)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	rows := cachevalueledger.ReadLedgerFile(*ledger)
	report := cachevaluereport.Fold(rows, time.Now())
	card := cachevaluepost.Fold(report)
	card.Source = resolveCachevalueSource(*source)
	return emitCachevalue(stdout, stderr, card, *channel, *token, *dryRun)
}

// runCachevaluePost handles `fak cachevalue post` — post a PRE-ROLLED report. It folds a
// `fak cachevalue report --json` style payload (a cachevaluereport.Report) from a file or
// stdin into the card, the path for posting a specific window an upstream rung produced.
func runCachevaluePost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportJSON := fs.String("report-json", "", "fold a pre-rolled cachevaluereport.Report JSON from this file (- for stdin)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #cache-value)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	report, err := loadCachevalueReport(*reportJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachevalue post: %v\n", err)
		return 2
	}
	card := cachevaluepost.Fold(report)
	card.Source = resolveCachevalueSource(*source)
	return emitCachevalue(stdout, stderr, card, *channel, *token, *dryRun)
}

// loadCachevalueReport reads a pre-rolled report payload from a file (or stdin for "-").
// An empty path is an error: `post` has no ledger to fall back to, so the caller must say
// what to post (use `feed` to fold the ledger).
func loadCachevalueReport(path string) (cachevaluereport.Report, error) {
	var report cachevaluereport.Report
	if path == "" {
		return report, fmt.Errorf("nothing to post: pass --report-json <file> (or use `fak cachevalue feed` to fold the ledger)")
	}
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return report, err
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, fmt.Errorf("parse --report-json payload: %w", err)
	}
	return report, nil
}

// emitCachevalue is the shared dry-run / post tail: render to stdout under --dry-run, else
// resolve channel+token and post via the scoreboard transport (the same chat.postMessage
// client every feeder reuses).
func emitCachevalue(stdout, stderr io.Writer, card cachevaluepost.Card, channel, token string, dryRun bool) int {
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           card,
		channel:        channel,
		token:          token,
		dryRun:         dryRun,
		label:          "fak cachevalue post",
		chanEnv:        "FAK_CACHEVALUE_CHANNEL",
		resolveChannel: cachevaluepost.ResolveChannel,
		resolveToken:   cachevaluepost.ResolveToken,
	})
}

// resolveCachevalueSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveCachevalueSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}
