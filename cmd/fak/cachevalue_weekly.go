package main

import (
	"flag"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluepost"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// runCachevalueWeekly handles `fak cachevalue weekly` — the weekly OPERATIONAL
// cache-health digest (#3646), a separate cadence/card from the daily $ feed: not
// "what did the cache save" but "is the cache machinery actually working across the
// fleet this week". It folds the durable gateway-usage ledger (posture adoption,
// shed effectiveness, refused-upgrade rate) and the Track-1 cache-value ledger
// (realized-reuse trend) into one week-over-week card and delivers it through the
// shared durable Slack outbox tail. Missing or empty ledgers fold to the honest
// INSUFFICIENT card rather than failing.
func runCachevalueWeekly(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue weekly", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "the durable cache-value ledger to fold (docs/nightrun/cache-value.jsonl)")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger for posture/shed/refusal counters (.fak/nightrun/gateway-usage.jsonl)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_CACHEVALUE_CHANNEL / .env.slack.local / #scoreboard)")
	token := fs.String("token", "", "override bot token (default: $FAK_CACHEVALUE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}

	digest := cachevaluereport.FoldWeeklyDigest(
		cachevalueledger.ReadLedgerFile(*ledger),
		gatewayusageledger.ReadLedgerFile(*usageLedger),
		time.Now(),
	)
	card := cachevaluepost.FoldWeekly(digest)
	card.Source = resolveCachevalueSource(*source)
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           card,
		channel:        *channel,
		token:          *token,
		dryRun:         *dryRun,
		label:          "fak cachevalue weekly",
		chanEnv:        "FAK_CACHEVALUE_CHANNEL",
		resolveChannel: cachevaluepost.ResolveChannel,
		resolveToken:   cachevaluepost.ResolveToken,
	})
}
