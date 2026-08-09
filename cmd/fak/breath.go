package main

// breath.go — `fak breath`, the advisory/count-ratchet front end for fak's
// named `In one breath` documentation contract (#5939).

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/promptlint/breath"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdBreath(argv []string) { os.Exit(runBreath(os.Stdout, os.Stderr, argv)) }

type breathReport struct {
	Schema        string           `json:"schema"`
	Root          string           `json:"root"`
	Advisory      bool             `json:"advisory"`
	Gate          bool             `json:"gate"`
	Baseline      string           `json:"baseline"`
	Census        breath.Census    `json:"census"`
	TotalFindings int              `json:"total_findings"`
	NewFindings   int              `json:"new_findings"`
	Verdict       string           `json:"verdict"`
	Findings      []breath.Finding `json:"findings"`
	FreshFindings []breath.Finding `json:"fresh_findings"`
}

// runBreath returns 0 for an advisory census or a ratchet with no growth, 1 for
// new findings when --gate is armed, and 2 when the checker could not establish
// its own inputs. The default is deliberately advisory: the corpus predates the
// contract, and the only enforceable claim is that its counted debt is not growing.
func runBreath(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("breath", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rootFlag := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	baselineFlag := fs.String("baseline", breath.BaselineFile, "repo-relative counted-ratchet baseline")
	gate := fs.Bool("gate", false, "exit 1 when findings exceed the counted floor (default remains advisory)")
	all := fs.Bool("all", false, "print every finding instead of only findings above the floor")
	emitBaseline := fs.Bool("emit-baseline", false, "write the current counted floor to stdout and exit")
	asJSON := fs.Bool("json", false, "emit the census and ratchet result as JSON")
	floor := fs.Int("floor", breath.DefaultFloor, "minimum pages required for a trustworthy absence census")
	var allow repeatedStringFlag
	fs.Var(&allow, "allow-acronym", "capitalized name that is not an acronym (repeatable)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *floor < 0 {
		fmt.Fprintln(stderr, "fak breath: --floor must be non-negative")
		return 2
	}
	if *emitBaseline && (*asJSON || *gate || *all) {
		fmt.Fprintln(stderr, "fak breath: --emit-baseline cannot be combined with --json, --gate, or --all")
		return 2
	}

	root := resolveRoot(*rootFlag)
	if strings.TrimSpace(root) == "" {
		fmt.Fprintln(stderr, "fak breath: not in a git repository (pass --root)")
		return 2
	}
	root, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak breath: resolve root: %v\n", err)
		return 2
	}

	contract := breath.DefaultContract()
	contract.Floor = *floor
	contract.AllowAcronyms = allow.Values()
	rels, err := breathTrackedPaths(root, contract)
	if err != nil {
		fmt.Fprintf(stderr, "fak breath: %v\n", err)
		return 2
	}
	docs, err := breath.Collect(root, rels)
	if err != nil {
		fmt.Fprintf(stderr, "fak breath: %v\n", err)
		return 2
	}
	census, findings := contract.Scan(docs)
	if hasBreathKind(findings, breath.BreathScanFloor) {
		writeBreathFailure(stderr, census, findings[0])
		return 2
	}
	if *emitBaseline {
		fmt.Fprint(stdout, breath.FormatBaseline(findings))
		return 0
	}

	baselinePath := *baselineFlag
	if !filepath.IsAbs(baselinePath) {
		baselinePath = filepath.Join(root, filepath.FromSlash(baselinePath))
	}
	f, err := os.Open(baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak breath: read counted baseline %s: %v (regenerate with `fak breath --emit-baseline > %s`)\n",
			*baselineFlag, err, breath.BaselineFile)
		return 2
	}
	baseline, parseErr := breath.ParseBaseline(f)
	closeErr := f.Close()
	if parseErr != nil {
		fmt.Fprintf(stderr, "fak breath: %v\n", parseErr)
		return 2
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "fak breath: close counted baseline %s: %v\n", *baselineFlag, closeErr)
		return 2
	}
	fresh := breath.Ratchet(findings, baseline)

	verdict := "NOT_GROWING"
	if len(fresh) > 0 {
		verdict = "ADVISORY"
		if *gate {
			verdict = "GROWING"
		}
	}
	report := breathReport{
		Schema: "fak.breath.v1", Root: root, Advisory: !*gate, Gate: *gate,
		Baseline: filepath.ToSlash(*baselineFlag), Census: census,
		TotalFindings: len(findings), NewFindings: len(fresh), Verdict: verdict,
		Findings: findings, FreshFindings: fresh,
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, report, "fak breath"); code != 0 {
			return code
		}
	} else {
		shown := fresh
		if *all {
			shown = findings
		}
		writeBreathReport(stdout, report, shown, *all)
	}
	if *gate && len(fresh) > 0 {
		return 1
	}
	return 0
}

// breathTrackedPaths names the contract's input from git's tracked-file view.
// An untracked peer page therefore cannot red another worker's gate. Collect
// still hard-fails if a tracked page is unreadable, so an empty/mismatched view
// never masquerades as a clean corpus.
func breathTrackedPaths(root string, contract breath.Contract) ([]string, error) {
	args := []string{"-C", root, "ls-files", "--"}
	args = append(args, contract.Roots...)
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return contract.Filter(strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n")), nil
}

func writeBreathFailure(w io.Writer, census breath.Census, finding breath.Finding) {
	fmt.Fprintf(w, "fak breath: SCAN_INVALID — %d page(s) examined; %s: %s\n",
		census.Pages, finding.Kind, finding.Detail)
	fmt.Fprintln(w, breath.ScopeNotice)
}

func writeBreathReport(w io.Writer, r breathReport, shown []breath.Finding, all bool) {
	fmt.Fprintf(w, "fak breath: %s — %d pages: %d conforming, %d failing, %d missing; "+
		"%d finding(s), %d above the counted floor\n",
		r.Verdict, r.Census.Pages, r.Census.Conforming, r.Census.Failing, r.Census.Missing,
		r.TotalFindings, r.NewFindings)
	if len(shown) > 0 {
		label := "new findings"
		if all {
			label = "all findings"
		}
		fmt.Fprintf(w, "%s:\n", label)
		for _, f := range shown {
			fmt.Fprintf(w, "  %s %s:%d — %s\n", f.Kind, f.Path, f.Line, f.Detail)
		}
	}
	fmt.Fprintln(w, breath.ScopeNotice)
}

func hasBreathKind(findings []breath.Finding, kind breath.Kind) bool {
	for _, f := range findings {
		if f.Kind == kind {
			return true
		}
	}
	return false
}
