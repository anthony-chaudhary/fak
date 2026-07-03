package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modver"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// defaultModverLedger is where `fak version modules --stamp` appends its
// delta rows (repo-relative), beside the other nightrun ledgers.
const defaultModverLedger = "docs/nightrun/module-versions.jsonl"

// runVersionModules is the thin shell over internal/modver: snapshot the
// per-module versions, optionally join scores, print or stamp the ledger.
func runVersionModules(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("version modules", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	stamp := fs.Bool("stamp", false, "append changed-module rows to the ledger")
	ledger := fs.String("ledger", defaultModverLedger, "ledger path (repo-relative unless absolute)")
	scoresPath := fs.String("scores", "", `flat {"module": number} JSON file to join as scores`)
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak version modules: unexpected args: %v\n", fs.Args())
		return 2
	}
	root := resolveRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak version modules: could not resolve git repo root")
		return 2
	}
	rep, err := modver.Snapshot(context.Background(), root, modver.RealRunner)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 1
	}
	if *scoresPath != "" {
		b, rerr := os.ReadFile(pathutil.ExpandTilde(*scoresPath))
		if rerr != nil {
			fmt.Fprintf(stderr, "fak version modules: %v\n", rerr)
			return 2
		}
		scores, serr := modver.LoadScores(b)
		if serr != nil {
			fmt.Fprintf(stderr, "fak version modules: %v\n", serr)
			return 2
		}
		matched := rep.JoinScores(scores)
		fmt.Fprintf(stderr, "fak version modules: joined %d/%d scores\n", matched, len(scores))
	}
	if *stamp {
		return stampModverLedger(stdout, stderr, root, *ledger, rep)
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak version modules: %v\n", err)
			return 1
		}
		return 0
	}
	renderModuleReport(stdout, rep)
	return 0
}

// stampModverLedger appends the delta rows (modules whose rev/score moved
// since their last ledger row) and reports what it wrote.
func stampModverLedger(stdout, stderr io.Writer, root, ledger string, rep modver.Report) int {
	path := pathutil.ExpandTilde(ledger)
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	prev, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak version modules: read ledger: %v\n", err)
		return 1
	}
	rows := modver.DeltaRows(rep, prev, time.Now().UTC().Format(time.RFC3339))
	if len(rows) == 0 {
		fmt.Fprintf(stdout, "fak version modules: ledger current — 0 of %d modules moved\n", len(rep.Modules))
		return 0
	}
	lines, err := modver.AppendLines(rows)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 1
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 1
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 1
	}
	defer f.Close()
	if _, err := f.Write(lines); err != nil {
		fmt.Fprintf(stderr, "fak version modules: write ledger: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak version modules: stamped %d of %d modules -> %s\n",
		len(rows), len(rep.Modules), path)
	return 0
}

// renderModuleReport prints the human table: version, last-touch date, module,
// and score when joined.
func renderModuleReport(w io.Writer, rep modver.Report) {
	fmt.Fprintf(w, "fak version modules: head %s  app %s  %d modules\n",
		rep.Head, rep.AppVersion, len(rep.Modules))
	for _, m := range rep.Modules {
		date := m.LastDate
		if len(date) > 10 {
			date = date[:10]
		}
		line := fmt.Sprintf("  %-16s %s  %s", m.Version(), date, m.Name)
		if m.Score != nil {
			line += fmt.Sprintf("  score %g", *m.Score)
		}
		fmt.Fprintln(w, line)
	}
}
