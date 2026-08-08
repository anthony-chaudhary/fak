package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/maturity"
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
	coveragePath := fs.String("coverage", "", "go coverage profile (go test -coverprofile) to fold into per-module statement coverage and join as scores")
	maturityJoin := fs.Bool("maturity", false, "grade each declared capability with internal/maturity and join its lifecycle-ladder position as scores")
	only := fs.String("only", "", "show only modules whose name has this prefix (e.g. internal/, cmd/fak, or tools/)")
	sortKey := fs.String("sort", "name", "sort order for display: name|rev|date")
	top := fs.Int("top", 0, "show only the first N modules after sorting (0 = all)")
	ghosts := fs.Bool("ghosts", false, "list history-only (deleted) modules with their final rev + deletion commit, instead of the live report")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak version modules: unexpected args: %v\n", fs.Args())
		return 2
	}
	joins := 0
	for _, on := range []bool{*scoresPath != "", *coveragePath != "", *maturityJoin} {
		if on {
			joins++
		}
	}
	if joins > 1 {
		// They all write the same score column; silently letting one win would
		// make the joined number's origin unreadable in the ledger.
		fmt.Fprintln(stderr, "fak version modules: --scores, --coverage, and --maturity all set the module score; pass one")
		return 2
	}
	root := resolveRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak version modules: could not resolve git repo root")
		return 2
	}
	if *ghosts {
		// The tombstone view is the complement of the live report — a distinct
		// query with no ledger/score/display flags — so it short-circuits before
		// the live Snapshot rather than filtering it.
		return runVersionModulesGhosts(stdout, stderr, root, *asJSON)
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
	if *coveragePath != "" {
		if code := joinCoverageScores(stderr, root, *coveragePath, &rep); code != 0 {
			return code
		}
	}
	if *maturityJoin {
		if code := joinMaturityScores(stderr, root, &rep); code != 0 {
			return code
		}
	}
	if *stamp {
		// --stamp always operates on the full report: the --only/--sort/--top
		// flags are a display view only, and stamping a filtered subset would
		// make the omitted modules look permanently unchanged in the ledger.
		return stampModverLedger(stdout, stderr, root, *ledger, rep)
	}
	view, verr := rep.View(*only, *sortKey, *top)
	return emitModverView(stdout, stderr, "fak version modules", view, verr, *asJSON, func() {
		renderModuleReportN(stdout, view, len(rep.Modules))
	})
}

// emitModverView is the shared tail of the two modver report commands: a --only/--sort/--top
// selection error is a USAGE failure (exit 2, not 1), --json emits the selected view, and the
// human path defers to the command's own renderer. `fak version modules` and
// `fak version trend` differ only in which selector built the view and how it renders.
func emitModverView(stdout, stderr io.Writer, label string, view any, verr error, asJSON bool, render func()) int {
	if verr != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, verr)
		return 2
	}
	if asJSON {
		return encodeJSONOrFailPrefixed(stdout, stderr, view, label)
	}
	render()
	return 0
}

// joinCoverageScores folds a go coverage profile into per-module statement
// coverage and joins it as the report's score (#2467). Joining HERE — inside the
// same run, before --stamp reads the report — is what lets one command,
// `fak version modules --coverage coverage.out --stamp`, land a ledger stamp
// with coverage joined; the --scores path needs a separately produced file.
//
// The profile names files by import path, so the fold needs this repo's module
// path from go.mod to recover repo-relative paths. The joined score is labeled
// witnessed: it is measured off a real run's artifact, not modeled.
func joinCoverageScores(stderr io.Writer, root, profile string, rep *modver.Report) int {
	b, err := os.ReadFile(pathutil.ExpandTilde(profile))
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 2
	}
	modulePath, err := readModulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: read module path: %v\n", err)
		return 1
	}
	coverage, err := modver.CoverageScores(b, modulePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 2
	}
	matched := rep.JoinScores(modver.CoverageEntries(coverage))
	fmt.Fprintf(stderr, "fak version modules: joined %d/%d coverage scores\n", matched, len(coverage))
	return 0
}

// joinMaturityScores grades every declared capability with internal/maturity and
// joins each leaf's lifecycle-ladder position as the report's score (#2468).
// Grading HERE — inside the same run, before --stamp reads the report — is what
// lets one command, `fak version modules --maturity --stamp`, land a ledger stamp
// answering "is this leaf production-grade at its current rev?"; the --scores path
// needs a separately produced file.
//
// The scorecard is marshaled and handed to the adapter as bytes on purpose: that
// is the identical seam an operator gets by piping `fak maturity --json`, so the
// in-process path and the piped-file path cannot drift apart. The joined score is
// labeled witnessed — every rung is re-derived from evidence on disk, not modeled.
func joinMaturityScores(stderr io.Writer, root string, rep *modver.Report) int {
	b, err := json.Marshal(maturity.Build(maturity.Options{Root: root}))
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: build maturity scorecard: %v\n", err)
		return 1
	}
	scores, err := modver.MaturityScores(b)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 2
	}
	matched := rep.JoinScores(modver.MaturityEntries(scores))
	fmt.Fprintf(stderr, "fak version modules: joined %d/%d maturity scores\n", matched, len(scores))
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
	// The score-regression advisory (#2470) runs BEFORE the append: afterwards the
	// row this stamp is about to write is itself the module's last scored row, and
	// the comparison it needs — freshly joined score vs. last remembered score —
	// would compare the number against itself. It is advisory, so it only writes to
	// stderr; the stamp still happens and the exit code is untouched.
	modver.ScoreDropAdvisory(stderr, modver.ScoreDrops(rep, prev))
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
	// Guard the append with an advisory lock (#2473): on the shared multi-session
	// trunk two agents can stamp at once, and an unguarded concurrent O_APPEND can
	// interleave a torn or duplicated row into the ledger.
	if err := modver.AppendGuarded(path, lines); err != nil {
		fmt.Fprintf(stderr, "fak version modules: write ledger: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak version modules: stamped %d of %d modules -> %s\n",
		len(rows), len(rep.Modules), path)
	return 0
}

// runVersionModulesGhosts renders the tombstone report: history-only modules
// that no longer exist at HEAD, each with the final rev it reached and the commit
// that deleted it. It answers "what died and when" — the growth-shape fact the
// live report excludes by design (#2477).
func runVersionModulesGhosts(stdout, stderr io.Writer, root string, asJSON bool) int {
	ghosts, err := modver.Ghosts(context.Background(), root, modver.RealRunner)
	if err != nil {
		fmt.Fprintf(stderr, "fak version modules: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFailPrefixed(stdout, stderr, ghosts, "fak version modules")
	}
	renderGhostReport(stdout, ghosts)
	return 0
}

// renderGhostReport prints the tombstone table — the final version each deleted
// module reached, the date it died, and the module name — mirroring the live
// report's layout so a ghost row reads the same as a live one, minus the score.
func renderGhostReport(w io.Writer, ghosts []modver.Ghost) {
	fmt.Fprintf(w, "fak version modules: %d ghost modules (history-only, absent at HEAD)\n", len(ghosts))
	for _, g := range ghosts {
		date := g.DeletedDate
		if len(date) > 10 {
			date = date[:10]
		}
		fmt.Fprintf(w, "  %-16s %s  %s\n", g.Version(), date, g.Name)
	}
}

// renderModuleReport prints the human table for the full report.
func renderModuleReport(w io.Writer, rep modver.Report) {
	renderModuleReportN(w, rep, len(rep.Modules))
}

// renderModuleReportN prints the human table — version, last-touch date, module,
// and score when joined — for a (possibly filtered) view of `total` modules.
// When fewer than `total` rows are shown it says so, so a --only/--top view is
// never mistaken for the whole repo.
func renderModuleReportN(w io.Writer, rep modver.Report, total int) {
	if len(rep.Modules) == total {
		fmt.Fprintf(w, "fak version modules: head %s  app %s  %d modules\n",
			rep.Head, rep.AppVersion, len(rep.Modules))
	} else {
		fmt.Fprintf(w, "fak version modules: head %s  app %s  showing %d of %d modules\n",
			rep.Head, rep.AppVersion, len(rep.Modules), total)
	}
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
