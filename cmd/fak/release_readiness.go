package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/releasereadiness"
)

func runReleaseReadiness(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak release readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	markdown := fs.Bool("markdown", false, "emit snapshot markdown")
	stamp := fs.String("stamp", "", "date stamp for markdown")
	check := fs.Bool("check", false, "exit non-zero when HARD release debt remains")
	compare := fs.String("compare", "", "compare score and release-debt against a prior --json baseline")
	skipGH := fs.Bool("skip-gh", false, "skip live GitHub release asset probes")
	if !parseFlags(fs, args) || fs.NArg() != 0 {
		if fs.NArg() != 0 {
			fmt.Fprintf(stderr, "fak release readiness: unexpected argument %q\n", fs.Arg(0))
		}
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	p := releasereadiness.Build(root, !*skipGH)
	if *compare != "" {
		data, err := os.ReadFile(*compare)
		if err != nil {
			fmt.Fprintf(stderr, "fak release readiness: read baseline: %v\n", err)
			return 2
		}
		var baseline struct {
			Score       float64 `json:"score"`
			ReleaseDebt int     `json:"release_debt"`
		}
		if err := json.Unmarshal(data, &baseline); err != nil {
			fmt.Fprintf(stderr, "fak release readiness: decode baseline: %v\n", err)
			return 2
		}
		fmt.Fprintf(stdout, "release-readiness delta: score %+.1f · release-debt %+d (%d -> %d)\n", p.Score-baseline.Score, p.ReleaseDebt-baseline.ReleaseDebt, baseline.ReleaseDebt, p.ReleaseDebt)
	} else if *asJSON {
		if err := writeIndentedJSON(stdout, p); err != nil {
			fmt.Fprintf(stderr, "fak release readiness: encode json: %v\n", err)
			return 2
		}
	} else if *markdown {
		fmt.Fprint(stdout, releasereadiness.Markdown(p, *stamp))
	} else {
		fmt.Fprintln(stdout, releasereadiness.Render(p))
	}
	if *check && !p.OK {
		return 1
	}
	return 0
}
