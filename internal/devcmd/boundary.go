package devcmd

// fak boundary — the BOUNDARY-TELL linter run as a verb, not just a test. The
// kernel's "the part that doesn't believe the agents" discipline, turned on the
// codebase itself: source patterns where the code makes a claim about the outside
// world (the OS, the network) without the check that makes the claim true. Three
// static witnesses are run over the repo in one pass:
//
//	internal/pathlint      UNEXPANDED_USER_PATH      a -gguf/-hf/-dir path flag opened without ~ expansion
//	internal/urllint       UNVERIFIED_EXTERNAL_URL   a hardcoded download URL outside the audited chokepoint
//	internal/boundarylint  MISSING_HTTP_TIMEOUT      outbound HTTP that can hang forever
//	internal/boundarylint  CHANGE_DETECTOR_TEST      a _test.go assertion that freezes a value instead of an invariant
//
// The change-detector family runs over _test.go files and honors the same shrink-only
// ratchet as the TestTestSuitePolicy gate: only a tell in a file NOT yet grandfathered
// (changeDetectorBaseline) is reported, so the verb stays green on the existing backlog
// and surfaces exactly the NEW change-detector an operator must convert to an invariant.
//
// Two SOFT families are reported separately (rep.Soft), never counted toward the gate:
//
//	internal/boundarylint  SKIP_DEBT          a bare t.Skip/Skipf/SkipNow with no platform/short/env guard
//	internal/boundarylint  UNPARSEABLE_SOURCE a .go file the scanner could not parse, so no rule ran over it
//
// Both are trends to watch, not build-reddeners, so they stay out of Count/OK/exit. Skip
// debt feeds the qa-process scorecard's SOFT skip_debt KPI as a work-list of skip sites.
// UNPARSEABLE_SOURCE closes a fail-OPEN blind spot: an unparseable file otherwise yields
// zero findings and is indistinguishable from a clean one, so the linter would report
// success over source it never read. It stays SOFT because this shared trunk carries
// peers' half-written files and the compiler remains the authority on real syntax errors.
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

	"github.com/anthony-chaudhary/fak/internal/boundarylint"
	"github.com/anthony-chaudhary/fak/internal/pathlint"
	"github.com/anthony-chaudhary/fak/internal/urllint"
)

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
	// Soft carries reported-not-gated tells (SKIP_DEBT today): a work-list surfaced for
	// the trend, deliberately excluded from Count/OK so it never reds the gate. The
	// qa-process scorecard folds these as a SOFT KPI.
	Soft []boundaryTell `json:"soft"`
}

const boundarySchema = "fak-boundary-lint/1"

// urllintAllow is the audited-chokepoint allowlist, identical to the one
// TestBoundaryPolicy enforces: cmd/simpledemo's modelDownload is the single builder
// whose URLs the reachability test HEAD-checks.
var urllintAllow = map[string]bool{
	"cmd/simpledemo/main.go": true, // the single audited download-url builder
	// Pinned by immutable repo revision, and main.go refuses the artifact unless
	// its bytes+sha256 match ModelBytes/ModelSHA256 — the property the chokepoint
	// exists to enforce. cmd/ mains cannot import simpledemo's builder.
	"cmd/quantdemo/contract.go": true,
}

// RunBoundary is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly.
func RunBoundary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak boundary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
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

	// boundarylint-test: the change-detector family over _test.go, ratcheted — only
	// NEW (non-grandfathered) tells are reported, so this stays green on the existing
	// backlog and surfaces exactly the change-detector an operator must convert.
	testFindings, err := boundarylint.ScanNewChangeDetectors(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: boundarylint-test: %v\n", err)
		return 2
	}
	for _, f := range testFindings {
		tells = append(tells, boundaryTell{Code: f.Code, Linter: "boundarylint-test", Detail: f.String()})
	}

	// boundarylint-soft: the SKIP_DEBT family over _test.go — a bare t.Skip/Skipf/SkipNow
	// with no platform/short/env guard. Reported as a SOFT work-list only: it feeds the
	// qa-process scorecard KPI but is kept OUT of Count/OK/exit so surfacing skip debt
	// never gates the build (unlike the change-detector family, which ratchets to green).
	soft := []boundaryTell{}
	skipFindings, err := boundarylint.ScanTests([]string{root + "/cmd", root + "/internal"}, []boundarylint.Rule{boundarylint.SkipDebt{}})
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: boundarylint-soft: %v\n", err)
		return 2
	}
	for _, f := range skipFindings {
		soft = append(soft, boundaryTell{Code: f.Code, Linter: "boundarylint-soft", Detail: f.String()})
	}

	// boundarylint-soft: UNPARSEABLE_SOURCE — a .go file the scanner could not parse, so
	// no rule ran over it. Without this the file contributes zero findings and reads as
	// clean, i.e. the linter reports success over source it never read. Also SOFT: this
	// trunk carries peers' half-written files, and the compiler owns real syntax errors.
	unparseable, err := boundarylint.ScanUnparseable([]string{root + "/cmd", root + "/internal"})
	if err != nil {
		fmt.Fprintf(stderr, "fak boundary: boundarylint-unparseable: %v\n", err)
		return 2
	}
	for _, f := range unparseable {
		soft = append(soft, boundaryTell{Code: f.Code, Linter: "boundarylint-soft", Detail: f.String()})
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
		Soft:      soft,
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
		fmt.Fprintln(w, "boundary: clean — no boundary tells (pathlint, urllint, boundarylint, boundarylint-test)")
	} else {
		fmt.Fprintf(w, "boundary: %d tell(s) — %d pathlint · %d urllint · %d boundarylint · %d boundarylint-test\n",
			rep.Count, rep.ByLinter["pathlint"], rep.ByLinter["urllint"], rep.ByLinter["boundarylint"], rep.ByLinter["boundarylint-test"])
		for _, t := range rep.Tells {
			fmt.Fprintf(w, "  %s\n", t.Detail)
		}
	}
	// The SOFT work-list prints even on a clean gate: it is a trend to watch, not a failure.
	if len(rep.Soft) > 0 {
		fmt.Fprintf(w, "soft: %d site(s) (SKIP_DEBT, UNPARSEABLE_SOURCE — reported, not gated)\n", len(rep.Soft))
		for _, t := range rep.Soft {
			fmt.Fprintf(w, "  %s\n", t.Detail)
		}
	}
}
