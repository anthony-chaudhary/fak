package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// runCachevalueReport handles `fak cachevalue report` — the #1304 two-track P&L.
// It folds BOTH durable ledgers side by side, never blended:
//
//	Track 1 (WITNESSED kernel)  — docs/nightrun/cache-value.jsonl   (realized KV reuse)
//	Track 2 (OBSERVED $)        — docs/nightrun/cache-savings.jsonl (provider rebate +
//	                              compaction token-shed − write premium − API spend)
//
// and prints both tracks plus a single NET line per period with the running total
// crossing break-even shown explicitly. --since floors both ledgers to rows on or
// after the date; --json emits the cachevaluereport.TwoTrackReport for downstream
// posting. A missing Track-2 ledger folds to the honest "rung B not appending yet"
// report rather than failing — Track 2's live append is epic #1301 rung B (#1303).
func runCachevalueReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	savingsLedger := fs.String("savings-ledger", cachevaluereport.DefaultSavingsLedgerRel, "Track-2 OBSERVED-$ ledger (docs/nightrun/cache-savings.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit the two-track report as JSON instead of the table")
	markdown := fs.Bool("markdown", false, "emit the two-track report as markdown (mermaid xychart trends + sparklines + a provenance-labelled KPI table) instead of the terminal table")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue report: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	track1 := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	track2 := filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(*savingsLedger), *since)

	report := cachevaluereport.FoldTwoTrack(track1, track2, time.Now().UTC())
	report.Since = *since

	if *asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue report: marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	if *markdown {
		fmt.Fprint(stdout, cachevaluereport.RenderTwoTrackMarkdown(report))
		return 0
	}
	fmt.Fprint(stdout, cachevaluereport.RenderTwoTrack(report))
	return 0
}

// filterTrack1Since drops WITNESSED rows dated before `since` (empty since = keep
// all). The date compare is lexical on the YYYY-MM-DD string, which orders
// chronologically.
func filterTrack1Since(rows []cachevalueledger.Row, since string) []cachevalueledger.Row {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}

// filterTrack2Since drops OBSERVED-$ rows dated before `since` (empty since = keep all).
func filterTrack2Since(rows []cachevaluereport.SavingsRow, since string) []cachevaluereport.SavingsRow {
	if since == "" {
		return rows
	}
	out := rows[:0:0]
	for _, r := range rows {
		if r.Date >= since {
			out = append(out, r)
		}
	}
	return out
}
