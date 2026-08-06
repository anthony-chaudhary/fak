package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/modver"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// runVersionTrend is the thin shell over modver.Trend: read the append-only
// module-versions ledger that `fak version modules --stamp` writes and report
// how each module moved across the recorded window. It touches no git and does
// not snapshot the tree — it reads only what was actually stamped, so it is the
// historical companion to `fak version modules` (the live snapshot).
func runVersionTrend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("version trend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	ledger := fs.String("ledger", defaultModverLedger, "ledger path (repo-relative unless absolute)")
	asJSON := fs.Bool("json", false, "emit the trend as JSON")
	only := fs.String("only", "", "show only modules whose name has this prefix (e.g. internal/ or cmd/fak)")
	module := fs.String("module", "", "show only this exact module (e.g. internal/modver), with its score series")
	since := fs.String("since", "", "count only movement at or after this time (RFC3339 or YYYY-MM-DD)")
	sortKey := fs.String("sort", "delta", "sort order: delta|rev|velocity|name")
	top := fs.Int("top", 0, "show only the first N movers and N dormant modules (0 = all)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak version trend: unexpected args: %v\n", fs.Args())
		return 2
	}
	from, err := modver.NormalizeSince(*since)
	if err != nil {
		fmt.Fprintf(stderr, "fak version trend: %v\n", err)
		return 2
	}
	path := pathutil.ExpandTilde(*ledger)
	if !filepath.IsAbs(path) {
		root := resolveRoot(pathutil.ExpandTilde(*dir))
		if root == "" {
			fmt.Fprintln(stderr, "fak version trend: could not resolve git repo root")
			return 2
		}
		path = filepath.Join(root, path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintf(stderr, "fak version trend: no ledger at %s — run `fak version modules --stamp` first\n", path)
			return 1
		}
		fmt.Fprintf(stderr, "fak version trend: %v\n", err)
		return 1
	}
	// Cap after Select, not through it: --top means N per section, so the
	// dormant list survives the default rev-delta sort. JSON and the human
	// table then render the same capped set.
	view, verr := modver.TrendSince(b, from).SelectModule(*module).Select(*only, *sortKey, 0)
	if verr == nil {
		view = view.Cap(*top)
	}
	return emitModverView(stdout, stderr, "fak version trend", view, verr, *asJSON, func() {
		renderTrendReport(stdout, view)
	})
}

// renderTrendReport prints the human trend table in the two lists the ledger
// actually answers: the MOVERS (modules that added revisions over the window,
// fastest by revs/week first) and the DORMANT modules (no revisions since the
// window opened, stalest first). The header names the ledger window, the total
// row count, and the --since bound when one was given, so a filtered or capped
// view is never mistaken for the whole ledger. A single-module view leads with
// that module's score series — the curve the endpoints summarize.
func renderTrendReport(w io.Writer, rep modver.TrendReport) {
	window := "empty"
	if rep.Window[0] != "" {
		window = short10(rep.Window[0]) + ".." + short10(rep.Window[1])
	}
	header := fmt.Sprintf("fak version trend: %d rows over %s  %d modules",
		rep.Rows, window, len(rep.Modules))
	if rep.Since != "" {
		header += "  since " + short10(rep.Since)
	}
	fmt.Fprintln(w, header)
	if len(rep.Modules) == 1 {
		formatTrendSpark(w, rep.Modules[0])
	}
	movers := rep.TopMovers(0)
	fmt.Fprintf(w, "  movers: %d\n", len(movers))
	for _, m := range movers {
		fmt.Fprintf(w, "    %g/wk  %s\n", m.RevsPerWeek, trendModuleLine(m))
	}
	dormant := rep.DormantModules(0)
	fmt.Fprintf(w, "  dormant: %d\n", len(dormant))
	for _, m := range dormant {
		fmt.Fprintf(w, "    last %s  %s\n", short10(m.LastTS), trendModuleLine(m))
	}
}

// trendModuleLine is the shared per-module cell both lists print, so a mover
// and a dormant module read the same way apart from their leading rate/date.
func trendModuleLine(m modver.ModuleTrend) string {
	stamp := "stamps"
	if m.Stamps == 1 {
		stamp = "stamp "
	}
	line := fmt.Sprintf("Δr%+d  r%d→r%d  %d %s  %s",
		m.RevDelta, m.FirstRev, m.LastRev, m.Stamps, stamp, m.Module)
	if m.ScoreDelta != nil {
		line += fmt.Sprintf("  Δscore %+g", *m.ScoreDelta)
	}
	return line
}

// formatTrendSpark prints one module's stamps as a series — the score curve
// over the window rather than only its first/last pair.
func formatTrendSpark(w io.Writer, m modver.ModuleTrend) {
	fmt.Fprintf(w, "  series %s: %d points\n", m.Module, len(m.Series))
	for _, p := range m.Series {
		line := fmt.Sprintf("    %s  r%d", short10(p.TS), p.Rev)
		if p.Score != nil {
			line += fmt.Sprintf("  score %g", *p.Score)
		}
		fmt.Fprintln(w, line)
	}
}

// short10 trims an RFC3339 timestamp to its YYYY-MM-DD date for compact display.
func short10(ts string) string {
	if len(ts) > 10 {
		return ts[:10]
	}
	return ts
}
