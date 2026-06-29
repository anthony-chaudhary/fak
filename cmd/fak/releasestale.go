package main

// fak release-staleness — the PUBLISH-freshness signal. It answers "is the version
// `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` would install actually
// current, or has the trunk moved far past it?" — the dual of `fak self-update`, which
// keeps a BUILT-from-source binary converged on origin/main HEAD. `@latest` resolves to
// the newest semver TAG, so if no tag is cut as work lands, `@latest` rots silently; this
// verb makes that lag a loud, gateable number (commits + days behind the latest tag) with
// a next-action that points at the real publish levers (/release, release-cadence.yml).
//
// --check turns it into a build/loop gate: non-zero exit when @latest is stale (or very
// stale). --json emits the control-pane envelope so the cadence fold (or any loop) can
// consume one record instead of re-deriving the lag.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/releasestale"
)

func cmdReleaseStaleness(argv []string) { os.Exit(runReleaseStaleness(os.Stdout, os.Stderr, argv)) }

func runReleaseStaleness(stdout, stderr io.Writer, argv []string) int {
	def := releasestale.DefaultThresholds()
	fs := flag.NewFlagSet("fak release-staleness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	check := fs.Bool("check", false, "gate: exit non-zero when @latest is stale or very stale vs HEAD")
	staleCommits := fs.Int("stale-commits", def.StaleCommits, "commits behind the latest tag that make @latest stale (0 disables)")
	staleDays := fs.Float64("stale-days", def.StaleDays, "days behind the latest tag that make @latest stale (0 disables)")
	veryStaleCommits := fs.Int("very-stale-commits", def.VeryStaleCommits, "commits behind that make @latest very stale (0 disables)")
	veryStaleDays := fs.Float64("very-stale-days", def.VeryStaleDays, "days behind that make @latest very stale (0 disables)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak release-staleness: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	// The VERSION marker at the working tree: used only to detect an untagged cut (VERSION
	// ahead of the latest tag), which changes the next-action from "cut" to "tag".
	version, _ := appversion.FromDir(root)

	facts := releasestale.Gather(context.Background(), releasestale.RealRunner, root, version)
	th := releasestale.Thresholds{
		StaleCommits:     *staleCommits,
		StaleDays:        *staleDays,
		VeryStaleCommits: *veryStaleCommits,
		VeryStaleDays:    *veryStaleDays,
	}
	p := releasestale.Compute(facts, th, root)

	if *asJSON {
		if err := writeIndentedJSON(stdout, p); err != nil {
			fmt.Fprintf(stderr, "fak release-staleness: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, releasestale.Render(p))
	}

	if *check && !p.OK {
		return 1
	}
	return 0
}
