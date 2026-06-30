package main

// fak boundary — the BOUNDARY-TELL linter run as a verb, not just a test. The
// kernel's "the part that doesn't believe the agents" discipline, turned on the
// codebase itself: source patterns where the code makes a claim about the outside
// world (the OS, the network) without the check that makes the claim true. Three
// static witnesses are run over the repo in one pass:
//
//	internal/pathlint      UNEXPANDED_USER_PATH      a -gguf/-hf/-dir path flag opened without ~ expansion
//	internal/urllint       UNVERIFIED_EXTERNAL_URL   a hardcoded download URL outside the audited chokepoint
//	internal/boundarylint  MISSING_HTTP_TIMEOUT      outbound HTTP that can hang forever
//
// These checks already existed as the package test TestBoundaryPolicy — so fak
// COULD prove them under `go test`, but fak itself never RAN them. This verb is the
// dogfood: the same three witnesses, on the running binary's path, runnable ad-hoc
// and wireable into `make ci` as a binary gate. Exit 0 = clean, 1 = at least one
// tell, 2 = usage/IO error — so it composes as a pipeline or CI step.

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/boundarylint"
	"github.com/anthony-chaudhary/fak/internal/pathlint"
	"github.com/anthony-chaudhary/fak/internal/urllint"
)

func cmdBoundary(argv []string) { os.Exit(runBoundary(os.Stdout, os.Stderr, argv)) }

// boundaryTell is one finding from any of the three witnesses, normalized to a
// single closed-vocabulary shape so the JSON view is uniform across linters.
type boundaryTell struct {
	Code   string `json:"code"`   // closed-vocabulary reason, e.g. MISSING_HTTP_TIMEOUT
	Linter string `json:"linter"` // which witness reported it: pathlint | urllint | boundarylint
	Detail string `json:"detail"` // the one-line human diagnostic (the linter's own String())
}

// boundaryReport is the control-pane envelope for `fak boundary --json`.
type boundaryReport struct {
	Schema    string         `json:"schema"`
	OK        bool           `json:"ok"`
	Workspace string         `json:"workspace"`
	Count     int            `json:"count"`
	ByLinter  map[string]int `json:"by_linter"`
	Tells     []boundaryTell `json:"tells"`
}

const boundarySchema = "fak-boundary-lint/1"

// urllintAllow is the audited-chokepoint allowlist, identical to the one
// TestBoundaryPolicy enforces: cmd/simpledemo's modelDownload is the single builder
// whose URLs the reachability test HEAD-checks.
var urllintAllow = map[string]bool{"cmd/simpledemo/main.go": true}

// runBoundary is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly.
func runBoundary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak boundary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak boundary: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	tells := []boundaryTell{} // non-nil so --json renders [] (not null) on a clean run

	// pathlint: unexpanded user-path flags under cmd/.
	pathOff, err := pathlint.ScanCmdTree(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: pathlint: %v\n", err)
		return 2
	}
	for _, o := range pathOff {
		tells = append(tells, boundaryTell{Code: pathlint.ReasonUnexpandedUserPath, Linter: "pathlint", Detail: o.String()})
	}

	// urllint: hardcoded download URLs outside the audited builder.
	urlOff, err := urllint.ScanForDownloadURLs(root, urllintAllow)
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: urllint: %v\n", err)
		return 2
	}
	for _, o := range urlOff {
		tells = append(tells, boundaryTell{Code: urllint.ReasonUnverifiedExternalURL, Linter: "urllint", Detail: o.String()})
	}

	// boundarylint: the policy family (MISSING_HTTP_TIMEOUT today) over cmd/ + internal/.
	findings, err := boundarylint.Scan([]string{root + "/cmd", root + "/internal"}, boundarylint.DefaultRules())
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: boundarylint: %v\n", err)
		return 2
	}
	for _, f := range findings {
		tells = append(tells, boundaryTell{Code: f.Code, Linter: "boundarylint", Detail: f.String()})
	}

	byLinter := map[string]int{}
	for _, t := range tells {
		byLinter[t.Linter]++
	}
	rep := boundaryReport{
		Schema:    boundarySchema,
		OK:        len(tells) == 0,
		Workspace: root,
		Count:     len(tells),
		ByLinter:  byLinter,
		Tells:     tells,
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak boundary: encode json: %v\n", err)
			return 1
		}
	} else {
		writeBoundaryHuman(stdout, rep)
	}
	if !rep.OK {
		return 1
	}
	return 0
}

func writeBoundaryHuman(w io.Writer, rep boundaryReport) {
	if rep.OK {
		fmt.Fprintln(w, "boundary: clean — no boundary tells (pathlint, urllint, boundarylint)")
		return
	}
	fmt.Fprintf(w, "boundary: %d tell(s) — %d pathlint · %d urllint · %d boundarylint\n",
		rep.Count, rep.ByLinter["pathlint"], rep.ByLinter["urllint"], rep.ByLinter["boundarylint"])
	for _, t := range rep.Tells {
		fmt.Fprintf(w, "  %s\n", t.Detail)
	}
}
