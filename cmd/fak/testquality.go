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
		// The prior file is read for its trailing note block only. A missing or
		// unreadable baseline simply has no notes to carry; it must not stop a
		// regeneration, which is the documented way OUT of a broken baseline.
		prior, _ := os.ReadFile(path)
		if err := os.WriteFile(path, testquality.FormatBaselineWithNotes(findings, prior), 0o644); err != nil {
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
			// --all deliberately never consulted the floor, so this report says
			// "unratcheted" rather than reporting New: 0. Reusing "not growing" here
			// would let `--all --json` masquerade as a passing ratchet run.
			return emitTestQualityJSON(stdout, stderr, testQualityReport{
				Schema: testQualityJSONSchema, Root: r, Files: len(files),
				Total: len(findings), New: 0, CountsByCode: counts,
				Verdict: verdictUnratcheted, Findings: nonNilFindings(findings),
			}, 0)
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
	verdict, code := verdictNotGrowing, 0
	if len(fresh) > 0 {
		verdict, code = verdictGrowing, 1
	}
	if *asJSON {
		// Same verdict and same exit code as the text path below. A --json run that
		// disagreed with the text run about whether the class grew would make the
		// machine-read gate the LOOSER of the two, which is the direction that goes
		// unnoticed.
		return emitTestQualityJSON(stdout, stderr, testQualityReport{
			Schema: testQualityJSONSchema, Root: r, Files: len(files),
			Total: len(findings), New: len(fresh), CountsByCode: counts,
			Verdict: verdict, Slack: slack, Findings: nonNilFindings(fresh),
		}, code)
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

// The report's verdict vocabulary. Three values, not a bool, because "the floor
// was never consulted" (--all) is a different claim from "nothing grew", and a
// consumer that collapsed them would read an unratcheted dump as a passing gate.
const (
	verdictNotGrowing  = "not_growing" // findings exist, none beyond the floor
	verdictGrowing     = "growing"     // at least one finding beyond the floor
	verdictUnratcheted = "unratcheted" // --all: every candidate listed, no floor applied
	// testQualityJSONSchema is versioned so a consumer can refuse a shape it does
	// not know rather than silently reading absent fields as zeroes.
	testQualityJSONSchema = "fak.test-quality/v1"
)

// testQualityReport is the JSON shape. Total and New are separate fields because
// the two numbers answer different questions: Total is the debt, New is the
// verdict, and a consumer that conflated them would read a stable tree as a
// failing one. Branch on Verdict, not on New.
//
// Findings mirrors whatever the text mode would have printed for the same flags:
// the NEW findings on a ratcheted run, every candidate under --all. Two formats
// that disagreed about what "the findings" are would make the choice of -json a
// silent change of question.
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

// emitTestQualityJSON writes the report and returns exitCode UNCHANGED — the
// verdict is the caller's, so the two output formats cannot drift apart.
//
// A marshal failure returns 2 (tool broke) and never the caller's code: a
// consumer that received no JSON must not be able to read exit 0 off the wire as
// "clean". That is the same three-valued-exit argument the file header makes,
// applied to the encoder itself.
func emitTestQualityJSON(stdout, stderr io.Writer, rep testQualityReport, exitCode int) int {
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak test-quality: encode report: %v\n", err)
		return 2
	}
	fmt.Fprintln(stdout, string(b))
	return exitCode
}

// nonNilFindings keeps the JSON array an ARRAY. A nil slice marshals to `null`,
// and `null` is the one value a consumer is most likely to mistake for "the run
// did not report" rather than "the run reported nothing".
func nonNilFindings(f []testquality.Finding) []testquality.Finding {
	if f == nil {
		return []testquality.Finding{}
	}
	return f
}
