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

// runCachevalueShapes handles `fak cachevalue shapes` — the session-SHAPE view of the
// WITNESSED kernel ledger (Track 1). Where `fak cachevalue report` folds by week ×
// session_type (a trend over time), this folds by the SHAPE of each session — its
// length band (single / short / long) crossed with its realized-reuse outcome band
// (n/a / cold / partial / warm) — so a reader can see WHICH KINDS of sessions earn
// reuse, a fact the time trend hides. It shares the #1066 honesty fence: outcome bands
// are cut on the WITNESSED realized reuse ratio, never the vs-naive re-prefill multiple.
//
// --since floors the ledger to rows on or after the date; --json emits the
// cachevaluereport.ShapeReport for downstream posting.
func runCachevalueShapes(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue shapes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit the shape report as JSON instead of the table")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak cachevalue shapes: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}

	rows := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	report := cachevaluereport.FoldShapes(rows, time.Now().UTC())
	report.Since = *since

	if *asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue shapes: marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprint(stdout, cachevaluereport.RenderShapes(report))
	return 0
}
