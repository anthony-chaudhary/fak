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
	sortKey := fs.String("sort", "delta", "sort order: delta|rev|name")
	top := fs.Int("top", 0, "show only the first N modules after sorting (0 = all)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak version trend: unexpected args: %v\n", fs.Args())
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
	view, verr := modver.Trend(b).Select(*only, *sortKey, *top)
	return emitModverView(stdout, stderr, "fak version trend", view, verr, *asJSON, func() {
		renderTrendReport(stdout, view)
	})
}

// renderTrendReport prints the human trend table: per module, the revision
// delta over the window (first→last), the stamp count, and the score delta
// when scores were joined. The header names the ledger window and the total
// row count so a filtered/truncated view is never mistaken for the whole
// ledger.
func renderTrendReport(w io.Writer, rep modver.TrendReport) {
	window := "empty"
	if rep.Window[0] != "" {
		window = short10(rep.Window[0]) + ".." + short10(rep.Window[1])
	}
	fmt.Fprintf(w, "fak version trend: %d rows over %s  %d modules\n",
		rep.Rows, window, len(rep.Modules))
	for _, m := range rep.Modules {
		stamp := "stamps"
		if m.Stamps == 1 {
			stamp = "stamp "
		}
		line := fmt.Sprintf("  Δr%+d  r%d→r%d  %d %s  %s",
			m.RevDelta, m.FirstRev, m.LastRev, m.Stamps, stamp, m.Module)
		if m.ScoreDelta != nil {
			line += fmt.Sprintf("  Δscore %+g", *m.ScoreDelta)
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
