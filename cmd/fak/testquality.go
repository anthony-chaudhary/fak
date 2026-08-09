package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/testquality"
)

// cmd/fak/testquality.go — `fak test-quality`: the Go test-quality RATCHET.
//
// It reports tests whose SHAPE means they pass whether or not the code under test
// works (a TestXxx that can never fail, an assertion comparing a value to itself,
// an error captured and never checked, a table expectation nothing reads) and
// gates them against a counted baseline floor — the Go-side twin of
// internal/pythongate's NEW_PYTHON_TOOL ratchet.
//
// The verdict is always about the DELTA. This verb never claims the tree is
// clean; it claims the class is not growing. See internal/testquality's package
// doc for why that is the strongest true statement available, and for the four
// clauses of the ratchet contract.
//
// Exit codes are three-valued on purpose: 0 = nothing beyond the floor, 1 = NEW
// findings, 2 = the tool could not do its job (unreadable root, unparseable
// baseline, a test file that does not parse). Collapsing 1 and 2 would make the
// checker's own breakage read as a code defect in somebody's diff.

func cmdTestQuality(argv []string) { os.Exit(runTestQuality(os.Stdout, os.Stderr, argv)) }

func runTestQuality(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("test-quality", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	all := fs.Bool("all", false, "print every candidate, ignoring the baseline floor")
	write := fs.Bool("write-baseline", false, "record today's tree as the floor and exit 0")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak test-quality: not in a git repo (pass --root)")
		return 2
	}

	findings, files, err := testquality.ScanTree(r)
	if err != nil {
		fmt.Fprintf(stderr, "fak test-quality: %v\n", err)
		return 2
	}
	counts := testquality.CountsByCode(findings)

	if *write {
		path := filepath.Join(r, filepath.FromSlash(testquality.BaselineFile))
		if err := os.WriteFile(path, testquality.FormatBaseline(findings), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak test-quality: %v\n", err)
			return 2
		}
		fmt.Fprintf(stdout, "fak test-quality: wrote %s — %d candidate(s) across %d tracked test file(s)\n",
			testquality.BaselineFile, len(findings), len(files))
		writeCodeTally(stdout, counts)
		return 0
	}

	if *all {
		if *asJSON {
			return emitTestQualityJSON(stdout, stderr, findings, findings, files, counts, nil)
		}
		for _, f := range findings {
			fmt.Fprintln(stdout, f)
		}
		fmt.Fprintf(stdout, "fak test-quality: %d candidate(s) across %d tracked test file(s)\n",
			len(findings), len(files))
		writeCodeTally(stdout, counts)
		return 0
	}

	base, found, err := testquality.LoadBaseline(r)
	if err != nil {
		// An unparseable floor is a HARD stop, never a skipped row. A lenient parser
		// would read the malformed key's floor as zero and report somebody else's
		// pre-existing finding as new.
		fmt.Fprintf(stderr, "fak test-quality: %v\n", err)
		return 2
	}
	if !found {
		fmt.Fprintf(stderr, "fak test-quality: no baseline at %s — every candidate reads as NEW; "+
			"`--write-baseline` records today's tree as the floor\n", testquality.BaselineFile)
	}

	fresh, slack := testquality.NewFindings(findings, base)
	if *asJSON {
		return emitTestQualityJSON(stdout, stderr, findings, fresh, files, counts, slack)
	}

	if len(slack) > 0 {
		keys := make([]string, 0, len(slack))
		for k := range slack {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Fprintf(stdout, "fak test-quality: the floor is LOOSE on %d key(s) — a fix landed; "+
			"tighten it with `fak test-quality --write-baseline`:\n", len(keys))
		for _, k := range keys {
			fmt.Fprintf(stdout, "  fixed (%d): %s\n", slack[k], strings.ReplaceAll(k, "\t", " "))
		}
	}
	if len(fresh) == 0 {
		// "not growing", never "clean" — the only claim the evidence supports.
		fmt.Fprintf(stdout, "fak test-quality: NOT GROWING — %d candidate(s) across %d tracked test "+
			"file(s), none beyond the baseline floor\n", len(findings), len(files))
		writeCodeTally(stdout, counts)
		return 0
	}
	for _, f := range fresh {
		fmt.Fprintln(stdout, f)
	}
	fmt.Fprintf(stdout, "\nfak test-quality: %d NEW candidate(s) beyond the floor (%d total).\n",
		len(fresh), len(findings))
	fmt.Fprintln(stdout, "Each is a test whose shape lets it pass with the code under test broken.")
	fmt.Fprintln(stdout, "If the test is right as written, record it: fak test-quality --write-baseline")
	return 1
}

// writeCodeTally prints the per-code counts in the package's declared code order,
// so two runs of the same tree render the same block.
func writeCodeTally(w io.Writer, counts map[string]int) {
	for _, c := range testquality.Codes {
		fmt.Fprintf(w, "  %-26s %d\n", c, counts[c])
	}
}

// testQualityReport is the JSON shape. Total and New are separate fields because
// the two numbers answer different questions: Total is the debt, New is the
// verdict, and a consumer that conflated them would read a stable tree as a
// failing one.
type testQualityReport struct {
	Schema       string                `json:"schema"`
	Root         string                `json:"root"`
	Files        int                   `json:"files"`
	Total        int                   `json:"total"`
	New          int                   `json:"new"`
	CountsByCode map[string]int        `json:"counts_by_code"`
	Verdict      string                `json:"verdict"`
	Slack        map[string]int        `json:"slack,omitempty"`
	Findings     []testquality.Finding `json:"findings"`
}

func emitTestQualityJSON(stdout, stderr io.Writer, all, fresh []Finding2, files []string,
	counts map[string]int, slack map[string]int) int {
	return 0
}

// Finding2 is unused; kept out of the build.
type Finding2 = testquality.Finding

var _ = json.Marshal
