package main

import (
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
// cachevaluereport.ShapeReport for downstream posting. --trend swaps the static
// all-corpus snapshot for the LONGITUDINAL cachevaluereport.ShapeTrendReport — each
// shape's within-week share of reused tokens trended week over week (drift), so a reader
// can see a warm×long share shrinking, the early-regression signal the snapshot hides.
func runCachevalueShapes(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue shapes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger (docs/nightrun/cache-value.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit the shape report as JSON instead of the table")
	asTrend := fs.Bool("trend", false, "render each shape's week-over-week reuse-share drift instead of the static snapshot")
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

	if *asTrend {
		report := cachevaluereport.FoldShapeTrend(rows, time.Now().UTC())
		report.Since = *since
		if *asJSON {
			if rc := encodeJSONOrFailPrefixed(stdout, stderr, report, "fak cachevalue shapes: marshal"); rc != 0 {
				return rc
			}
			return 0
		}
		fmt.Fprint(stdout, cachevaluereport.RenderShapeTrend(report))
		return 0
	}

	report := cachevaluereport.FoldShapes(rows, time.Now().UTC())
	report.Since = *since

	if *asJSON {
		if rc := encodeJSONOrFailPrefixed(stdout, stderr, report, "fak cachevalue shapes: marshal"); rc != 0 {
			return rc
		}
		return 0
	}
	fmt.Fprint(stdout, cachevaluereport.RenderShapes(report))
	return 0
}
