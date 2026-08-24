package main

import (
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
// --ledger is REPEATABLE: pass it more than once to fold several ledgers in one view (e.g.
// the live copy AND the docs mirror), which is how a regime comparison spanning both
// windows becomes a single command — rows that appear in more than one ledger are deduped
// by (pid, unix_millis, generated_at) so an overlapping session is never double-counted.
// --since floors the fold to rows on/after the date; --json emits the CompactionReport.
// --by day|week adds a TIME axis within each regime×band so the trend the single-window
// fold hides — "is shed% still declining WITHIN the headless regime this week?" — becomes
// legible, the temporal question a point-in-time table structurally cannot answer.
//
//fak:ctxplan verb="cachevalue compaction" enters="nothing live — an offline fold over the gateway-usage JSONL ledger(s) on disk" pages="nothing into a model window — it prints a compaction-by-regime table (or --json report) to stdout" warms="nothing — it REPORTS compaction shed/fire/bail health; it warms no prompt cache or KV itself"
func runCachevalueCompaction(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue compaction", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var ledgers ablationPathList
	fs.Var(&ledgers, "ledger", "gateway usage ledger to fold (repeatable; defaults to the LIVE .fak/nightrun copy, not the committed docs mirror — pass twice to merge live + docs mirror)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	by := fs.String("by", "", "bucket the fold by time within each regime×band to see a trend: day | week (default: one window over the whole --since span)")
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
	switch *by {
	case "", "none", "day", "week":
	default:
		fmt.Fprintf(stderr, "fak cachevalue compaction: --by must be day or week (got %q)\n", *by)
		return 2
	}
	if len(ledgers) == 0 {
		ledgers = ablationPathList{gatewayusageledger.DefaultLedgerRel}
	}

	rows := filterGatewayUsageSince(readGatewayLedgersDedup(ledgers), *since)
	report := gatewayusageledger.FoldCompactionByPeriod(rows, *since, *by)

	if *asJSON {
		if rc := encodeJSONOrFailPrefixed(stdout, stderr, report, "fak cachevalue compaction: marshal"); rc != 0 {
			return rc
		}
		return 0
	}
	fmt.Fprint(stdout, gatewayusageledger.RenderCompaction(report))
	return 0
}

// readGatewayLedgersDedup reads every ledger path in order and concatenates their rows,
// dropping any row already seen under an identical (pid, unix_millis, generated_at) key.
// That triple identifies one process's one emitted snapshot, so a session exit present in
// BOTH a live ledger and its published docs mirror (the overlapping window) folds exactly
// once — the merge that lets a single invocation span two ledger windows without inflating
// the corpus. First occurrence wins; a single-path call is unchanged (nothing to dedup).
func readGatewayLedgersDedup(paths []string) []gatewayusageledger.Row {
	seen := map[string]bool{}
	var out []gatewayusageledger.Row
	for _, p := range paths {
		for _, r := range gatewayusageledger.ReadLedgerFile(p) {
			key := fmt.Sprintf("%d|%d|%s", r.PID, r.UnixMillis, r.GeneratedAt)
			if seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, r)
		}
	}
	return out
}
