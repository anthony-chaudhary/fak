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
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// cmdCachevalue posts the cache-effectiveness P&L roll-up — fak's WITNESSED kernel
// cache-value trend (Track 1 of epic #1301) — to the central Slack #scoreboard channel
// (C0BEF8B8KMW), the fleet's top-level status channel and the one durable place the fleet
// reads "is fak's cache method paying off, and is it trending up or down?".
//
//	fak cachevalue feed                                  # fold both cache-value ledgers → Slack card
//	fak cachevalue feed --dry-run                        # render the exact card; do not post
//	fak cachevalue feed --ledger docs/nightrun/cache-value.jsonl --savings-ledger .fak/nightrun/cache-savings.jsonl
//	fak cachevalue weekly --dry-run                      # weekly fleet cache-HEALTH digest (posture adoption + reuse trend + shed + refused upgrades, #3646)
//	fak cachevalue census                                # LIVE fleet managed-cache posture census: %ACTIVE and %upgrade-fired among ACTIVE (#3650)
//	fak cachevalue census --json                         # the same census fold as JSON, for a periodic poster
//	fak cachevalue post --report-json report.json        # post a pre-rolled report (- for stdin)
//	fak cachevalue report --since 2026-06-22             # the two-track P&L (WITNESSED + OBSERVED $) + NET (#1304)
//	fak cachevalue metrics                               # the same two-track fold + ablation arms as a Prometheus exposition (Grafana surface)
//	fak cachevalue metrics --serve --addr 127.0.0.1:9097 # serve /metrics, re-folding the ledgers on each scrape
//	fak cachevalue shapes --since 2026-06-22             # the WITNESSED ledger folded by session shape (length × realized-reuse outcome)
//	fak cachevalue shapes --json                         # emit the ShapeReport for downstream posting
//	fak cachevalue compaction --since 2026-07-09         # compaction shed/fire/bail SEGMENTED by budget regime × session-length band (keeps 48k/96k regimes apart)
//	fak cachevalue compaction --by week                  # same segmentation with a TIME axis, so a within-regime trend ("did shed% decline recently?") is legible
//	fak cachevalue compaction --json                     # emit the CompactionReport for downstream posting
//	fak cachevalue status --json                         # cache-plane health, owner, dependency, fidelity, and next action
//	fak cachevalue status --session transcript.jsonl --vcache-score-report score.json
//	fak cachevalue status --artifact-dir diagnostics/cache
//	fak cachevalue status --vcache-observe-report observe.json
//	fak cachevalue status --vcache-actions-report actions.json
//	fak cachevalue status --vcache-context-join-report context-join.json
//	fak cachevalue status --vcache-context-witness-report context-witness.json
//	fak cachevalue status --ablation-report ablate.json --headroom-bench-report headroom.json
//	fak cachevalue review --since 2026-06-22 --json      # inspect cache-frontier review row
//	fak cachevalue review --date 2026-06-29 --append-ledger docs/cache-frontier/review-ledger.jsonl --markdown-out docs/cache-frontier/reviews/2026-06-29.md
//
// It targets the FAK_CACHEVALUE_* surface (a public channel in the scoreboard Slack
// workspace, separate from the lab/DGX control bridge); the token falls back to the
// scoreboard bot token, the channel to the built-in #scoreboard default. --dry-run
// renders the card and prints it without posting, matching the scoreboard/bench/blockers
// "safe by default" idiom.

//fak:ctxplan verb=cachevalue enters="nothing live — an offline fold over the durable cache-value, cache-savings, and gateway-usage JSONL ledgers on disk" pages="nothing into a model window — it renders a cache-effectiveness P&L card and posts it to the #scoreboard Slack channel (or prints it under --dry-run)" warms="nothing — it REPORTS on whether the kernel prompt-cache method is paying off; it warms no prompt cache or KV itself"
func cmdCachevalue(argv []string) {
	dispatchSubcommands("cachevalue", "report | shapes | compaction | status | review | post | feed | weekly | census | metrics | regress", argv,
		subcommand{"report", runCachevalueReport},
		subcommand{"shapes", runCachevalueShapes},
		subcommand{"compaction", runCachevalueCompaction},
		subcommand{"status", runCachevalueStatus},
		subcommand{"review", runCachevalueReview},
		subcommand{"post", runCachevaluePost},
		subcommand{"feed", runCachevalueFeed},
		subcommand{"weekly", runCachevalueWeekly},
		subcommand{"census", runCachevalueCensus},
		subcommand{"metrics", runCachevalueMetrics},
		subcommand{"regress", runCachevalueRegress},
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
// durable kernel cache-value ledger (Track 1), the OBSERVED-$ savings ledger (Track 2), and
// the gateway-usage ledger (cumulative fleet usage/session-extension counters), folds them
// into the two-track P&L report, and posts ONE card. Missing or empty ledgers fold to the
// honest INSUFFICIENT / missing-track card rather than failing.
func runCachevalueFeed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue feed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "the durable cache-value ledger to fold (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "the Track-2 OBSERVED-$ ledger to fold (.fak/nightrun/cache-savings.jsonl)")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger for cumulative fleet usage/session-extension counters (.fak/nightrun/gateway-usage.jsonl)")
	fabricLedger := fs.String("microcontext-ledger", "", "controlled micro-context prefix A/B ledger to map into Track 1")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	contextBudget := fs.Uint64("context-budget-tokens", 0, "optional session context budget denominator; normalizes witnessed shed tokens into window-equivalent extension")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #scoreboard)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue feed: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	if *fabricLedger != "" {
		row, err := cachevaluereport.FabricTrack1Row(*fabricLedger)
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue feed: micro-context provenance: %v\n", err)
			return 1
		}
		track1 = append(track1, row)
	}
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)
	usage := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(*usageLedger), *since)
	report := cachevaluereport.FoldTwoTrackWithUsage(track1, track2, usage, time.Now(), cachevaluereport.FleetBenefitOptions{
		ContextBudgetTokens: *contextBudget,
	})
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
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #scoreboard)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
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
