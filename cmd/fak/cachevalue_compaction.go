package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// runCachevalueCompaction handles `fak cachevalue compaction` — the compaction-health
// view of the gateway-usage ledger, SEGMENTED by budget regime × session-length band.
//
// Where `fak cachevalue report` folds by week and `shapes` folds Track-1 reuse by
// session shape, this folds the WITNESSED compaction attempt counters (fired / bailed /
// shed) so the one question the blended numbers hide becomes legible: "is compaction
// still shedding what it should on long sessions, PER budget regime?" It exists because
// the interactive 48k and headless 96k budgets are only comparable within a regime —
// blend them and a deliberate 48k→96k switch reads as a compaction regression (the
// headless band correctly bails `under_budget` where the 48k band shed heavily). See
// gatewayusageledger.FoldCompaction for the regime/quarantine rationale.
//
// It defaults to the LIVE DefaultLedgerRel (.fak/nightrun/gateway-usage.jsonl), NOT the
// committed docs mirror, so a stale published copy cannot masquerade as a recent cliff.
// --since floors the fold to rows on/after the date; --json emits the CompactionReport.
//
//fak:ctxplan verb="cachevalue compaction" enters="nothing live — an offline fold over the gateway-usage JSONL ledger on disk" pages="nothing into a model window — it prints a compaction-by-regime table (or --json report) to stdout" warms="nothing — it REPORTS compaction shed/fire/bail health; it warms no prompt cache or KV itself"
func runCachevalueCompaction(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue compaction", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", gatewayusageledger.DefaultLedgerRel, "gateway usage ledger to fold (defaults to the LIVE .fak/nightrun copy, not the committed docs mirror)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit the CompactionReport as JSON instead of the table")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue compaction: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	rows := filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(*ledger), *since)
	report := gatewayusageledger.FoldCompaction(rows, *since)

	if *asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue compaction: marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprint(stdout, gatewayusageledger.RenderCompaction(report))
	return 0
}
