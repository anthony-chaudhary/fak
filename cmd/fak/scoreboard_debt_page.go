package main

import (
	"flag"
	"fmt"
	"io"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// runScoreboardDebtPage implements `fak scoreboard debt-page` (and `fak scorecard debt-page`).
// It provides deterministic generation and freshness-checking for docs/scoreboard-debt.md.
func runScoreboardDebtPage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scoreboard debt-page", flag.ContinueOnError)
	fs.SetOutput(stderr)
	writeDoc := fs.Bool("write-doc", false, "regenerate docs/scoreboard-debt.md in place")
	checkDoc := fs.Bool("check-doc", false, "CI gate: red when docs/scoreboard-debt.md is stale vs scorecard baseline")
	emitBlock := fs.Bool("block", false, "emit the generated scoreboard-debt markdown block to stdout")
	emitMD := fs.Bool("markdown", false, "emit the generated scoreboard-debt markdown block to stdout")
	asJSON := fs.Bool("json", false, "emit freshness check verdict as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak scoreboard debt-page: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	if *emitBlock || *emitMD {
		baselinePath := filepath.Join(root, filepath.FromSlash(scorecardpane.BaselineRel))
		baseline := scorecardpane.LoadBaseline(baselinePath)
		var bl scorecardpane.Baseline
		if baseline != nil {
			bl = *baseline
		}
		fmt.Fprint(stdout, scorecardpane.GenerateScoreboardDebtDoc(bl))
		return 0
	}

	if !*writeDoc && !*checkDoc {
		fmt.Fprintln(stderr, "fak scoreboard debt-page: pass --write-doc, --check-doc, or --block")
		return 2
	}

	if *writeDoc {
		changed, err := scorecardpane.WriteScoreboardDebtDoc(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak scoreboard debt-page --write-doc: %v\n", err)
			return 1
		}
		if changed {
			fmt.Fprintf(stdout, "wrote %s\n", scorecardpane.ScoreboardDebtDocRel)
		} else {
			fmt.Fprintf(stdout, "%s already fresh; no change\n", scorecardpane.ScoreboardDebtDocRel)
		}
		return 0
	}

	ok, msg, err := scorecardpane.CheckScoreboardDebtDoc(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak scoreboard debt-page --check-doc: %v\n", err)
		return 1
	}
	if *asJSON {
		fmt.Fprintf(stdout, "{\"doc\":%q,\"fresh\":%t,\"message\":%q}\n", scorecardpane.ScoreboardDebtDocRel, ok, msg)
	} else {
		fmt.Fprintln(stdout, msg)
	}
	if !ok {
		return 1
	}
	return 0
}
