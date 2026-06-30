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
//	fak cachevalue feed                                  # fold both cache-value ledgers → Slack card
//	fak cachevalue feed --dry-run                        # render the exact card; do not post
//	fak cachevalue feed --ledger docs/nightrun/cache-value.jsonl --savings-ledger docs/nightrun/cache-savings.jsonl
//	fak cachevalue post --report-json report.json        # post a pre-rolled report (- for stdin)
//	fak cachevalue report --since 2026-06-22             # the two-track P&L (WITNESSED + OBSERVED $) + NET (#1304)
//	fak cachevalue review --since 2026-06-22 --json      # inspect cache-frontier review row
//	fak cachevalue review --date 2026-06-29 --append-ledger docs/cache-frontier/review-ledger.jsonl --markdown-out docs/cache-frontier/reviews/2026-06-29.md
//
// It targets the FAK_CACHEVALUE_* surface (a public channel in the scoreboard Slack
// workspace, separate from the lab/DGX control bridge); the token falls back to the
// scoreboard bot token, the channel to the built-in #cache-value default. --dry-run
// renders the card and prints it without posting, matching the scoreboard/bench/blockers
// "safe by default" idiom.
func cmdCachevalue(argv []string) {
	dispatchSubcommands("cachevalue", "report | review | post | feed", argv,
		subcommand{"report", runCachevalueReport},
		subcommand{"review", runCachevalueReview},
		subcommand{"post", runCachevaluePost},
		subcommand{"feed", runCachevalueFeed},
	)
}

// foldAndEmitCachevalue folds a report into the post card, stamps the resolved source, and
// emits it — the shared tail of the feed/post subcommands.
func foldAndEmitCachevalue(stdout, stderr io.Writer, report cachevaluereport.Report, source, channel, token string, dryRun bool) int {
	card := cachevaluepost.Fold(report)
	card.Source = resolveCachevalueSource(source)
	return emitCachevalue(stdout, stderr, card, channel, token, dryRun)
}

// runCachevalueFeed handles `fak cachevalue feed` — the cadence roll-up. It reads the
// durable kernel cache-value ledger (Track 1) and the OBSERVED-$ savings ledger (Track 2),
// folds them into the two-track P&L report, and posts ONE card. Missing or empty ledgers
// fold to the honest INSUFFICIENT / missing-track card rather than failing.
func runCachevalueFeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue feed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "the durable cache-value ledger to fold (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "the Track-2 OBSERVED-$ ledger to fold (docs/nightrun/cache-savings.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #cache-value)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue feed: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)
	report := cachevaluereport.FoldTwoTrack(track1, track2, time.Now())
	card := cachevaluepost.FoldTwoTrack(report)
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
	return foldAndEmitCachevalue(stdout, stderr, report, *source, *channel, *token, *dryRun)
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
