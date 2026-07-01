package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const testDurationLedgerSchema = "fak.test_duration_ledger.v1"

type goTestJSONEvent struct {
	Action  string  `json:"Action"`
	Package string  `json:"Package"`
	Test    string  `json:"Test,omitempty"`
	Elapsed float64 `json:"Elapsed,omitempty"`
}

type testDurationOptions struct {
	Source        string
	Command       []string
	TestExitCode  int
	PackageBudget time.Duration
	TestBudget    time.Duration
}

type testDurationLedger struct {
	Schema          string                `json:"schema"`
	Source          string                `json:"source,omitempty"`
	Command         []string              `json:"command,omitempty"`
	TestExitCode    int                   `json:"test_exit_code,omitempty"`
	PackageBudgetMS int64                 `json:"package_budget_ms,omitempty"`
	TestBudgetMS    int64                 `json:"test_budget_ms,omitempty"`
	Summary         testDurationSummary   `json:"summary"`
	Packages        []testDurationPackage `json:"packages"`
	Tests           []testDurationTest    `json:"tests"`
	Findings        []testDurationFinding `json:"findings,omitempty"`
}

type testDurationSummary struct {
	Packages              int    `json:"packages"`
	Tests                 int    `json:"tests"`
	TotalPackageElapsedMS int64  `json:"total_package_elapsed_ms"`
	SlowestPackage        string `json:"slowest_package,omitempty"`
	SlowestPackageMS      int64  `json:"slowest_package_ms,omitempty"`
	SlowestTest           string `json:"slowest_test,omitempty"`
	SlowestTestMS         int64  `json:"slowest_test_ms,omitempty"`
	Findings              int    `json:"findings"`
}

type testDurationPackage struct {
	Package       string `json:"package"`
	Action        string `json:"action"`
	Runs          int    `json:"runs,omitempty"`
	ElapsedMS     int64  `json:"elapsed_ms"`
	Tests         int    `json:"tests"`
	SlowestTest   string `json:"slowest_test,omitempty"`
	SlowestTestMS int64  `json:"slowest_test_ms,omitempty"`
}

type testDurationTest struct {
	Package   string `json:"package"`
	Test      string `json:"test"`
	Action    string `json:"action"`
	Runs      int    `json:"runs,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms"`
}

type testDurationFinding struct {
	Rank         int    `json:"rank"`
	Kind         string `json:"kind"`
	Target       string `json:"target"`
	ElapsedMS    int64  `json:"elapsed_ms"`
	BudgetMS     int64  `json:"budget_ms"`
	OverBudgetMS int64  `json:"over_budget_ms"`
	Action       string `json:"action"`
}

type testDurationPackageAccum struct {
	row testDurationPackage
}

type testDurationTestAccum struct {
	row testDurationTest
}

type testDurationRunPlan struct {
	Target  string
	GoArgs  []string
	Argv    []string
	ViaWSL  bool
	Source  string
	Command []string
}

var testDurationRunCommand = runTestDurationCommand

func runTestDurations(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak test durations", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "go test -json stream to read (default: stdin)")
	runTarget := fs.String("run", "", "run a fak test tier/package through go test -json before folding (fast, full, race, ./pkg)")
	outPath := fs.String("out", "", "write the duration ledger JSON to this path in addition to stdout")
	source := fs.String("source", "", "source label for the ledger (default: input path or stdin)")
	packageBudget := fs.Duration("package-budget", 0, "rank packages whose elapsed time exceeds this duration")
	testBudget := fs.Duration("test-budget", 0, "rank tests whose elapsed time exceeds this duration")
	check := fs.Bool("check", false, "exit non-zero when any budget finding is emitted")
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak test durations -- fold go test -json into a duration ledger

  go test -json ./... | fak test durations --package-budget 30s --test-budget 5s
  fak test durations --input go-test.jsonl --check
  fak test durations --run fast --out .fak/test-duration-ledger.json

The output schema is fak.test_duration_ledger.v1. Budget findings are ranked by
over-budget time so agents see the next slow package or test first.
`)
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak test durations: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *packageBudget < 0 || *testBudget < 0 {
		fmt.Fprintln(stderr, "fak test durations: budgets must be non-negative")
		return 2
	}
	if *input != "" && *runTarget != "" {
		fmt.Fprintln(stderr, "fak test durations: --input and --run are mutually exclusive")
		return 2
	}

	var r io.Reader = os.Stdin
	src := *source
	var command []string
	testExitCode := 0
	if *runTarget != "" {
		plan, err := planTestDurationRun(runtimeGOOS(), *runTarget)
		if err != nil {
			fmt.Fprintf(stderr, "fak test durations: %v\n", err)
			return 2
		}
		fmt.Fprintf(stderr, "fak test durations: running %s\n", strings.Join(plan.Argv, " "))
		raw, code, err := testDurationRunCommand(plan.Argv[0], plan.Argv[1:], stderr)
		if err != nil {
			fmt.Fprintf(stderr, "fak test durations: run %s: %v\n", strings.Join(plan.Argv, " "), err)
			return 1
		}
		r = bytes.NewReader(raw)
		command = plan.Command
		testExitCode = code
		if src == "" {
			src = plan.Source
		}
	} else if *input != "" {
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintf(stderr, "fak test durations: open input: %v\n", err)
			return 1
		}
		defer f.Close()
		r = f
		if src == "" {
			src = *input
		}
	}
	if src == "" {
		src = "stdin"
	}

	ledger, err := parseTestDurationLedger(r, testDurationOptions{
		Source:        src,
		Command:       command,
		TestExitCode:  testExitCode,
		PackageBudget: *packageBudget,
		TestBudget:    *testBudget,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak test durations: %v\n", err)
		return 1
	}
	if *outPath != "" {
		if err := writeTestDurationLedgerFile(*outPath, ledger); err != nil {
			fmt.Fprintf(stderr, "fak test durations: write ledger: %v\n", err)
			return 1
		}
	}
	if err := writeIndentedJSONNoEscape(stdout, ledger); err != nil {
		fmt.Fprintf(stderr, "fak test durations: encode json: %v\n", err)
		return 1
	}
	if testExitCode != 0 {
		return testExitCode
	}
	if *check && len(ledger.Findings) > 0 {
		return 1
	}
	return 0
}

func parseTestDurationLedger(r io.Reader, opts testDurationOptions) (testDurationLedger, error) {
	packages := map[string]*testDurationPackageAccum{}
	tests := map[string]*testDurationTestAccum{}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev goTestJSONEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return testDurationLedger{}, fmt.Errorf("line %d: parse go test json: %w", lineNo, err)
		}
		if ev.Package == "" || !isTerminalTestAction(ev.Action) {
			continue
		}
		elapsedMS := elapsedSecondsToMS(ev.Elapsed)
		if ev.Test == "" {
			acc := packages[ev.Package]
			if acc == nil {
				acc = &testDurationPackageAccum{row: testDurationPackage{Package: ev.Package}}
				packages[ev.Package] = acc
			}
			acc.row.Action = ev.Action
			acc.row.Runs++
			acc.row.ElapsedMS += elapsedMS
			continue
		}

		key := ev.Package + "\x00" + ev.Test
		acc := tests[key]
		if acc == nil {
			acc = &testDurationTestAccum{row: testDurationTest{Package: ev.Package, Test: ev.Test}}
			tests[key] = acc
		}
		acc.row.Action = ev.Action
		acc.row.Runs++
		acc.row.ElapsedMS += elapsedMS
	}
	if err := sc.Err(); err != nil {
		return testDurationLedger{}, err
	}

	testRows := make([]testDurationTest, 0, len(tests))
	testsByPackage := map[string][]testDurationTest{}
	for _, acc := range tests {
		testRows = append(testRows, acc.row)
		testsByPackage[acc.row.Package] = append(testsByPackage[acc.row.Package], acc.row)
	}
	sort.Slice(testRows, func(i, j int) bool {
		return durationTestLess(testRows[i], testRows[j])
	})

	packageRows := make([]testDurationPackage, 0, len(packages))
	for _, acc := range packages {
		row := acc.row
		for _, tr := range testsByPackage[row.Package] {
			row.Tests += tr.Runs
			if tr.ElapsedMS > row.SlowestTestMS || (tr.ElapsedMS == row.SlowestTestMS && tr.Test < row.SlowestTest) {
				row.SlowestTest = tr.Test
				row.SlowestTestMS = tr.ElapsedMS
			}
		}
		packageRows = append(packageRows, row)
	}
	sort.Slice(packageRows, func(i, j int) bool {
		return durationPackageLess(packageRows[i], packageRows[j])
	})

	ledger := testDurationLedger{
		Schema:       testDurationLedgerSchema,
		Source:       opts.Source,
		Command:      append([]string(nil), opts.Command...),
		TestExitCode: opts.TestExitCode,
		Packages:     packageRows,
		Tests:        testRows,
	}
	if opts.PackageBudget > 0 {
		ledger.PackageBudgetMS = opts.PackageBudget.Milliseconds()
	}
	if opts.TestBudget > 0 {
		ledger.TestBudgetMS = opts.TestBudget.Milliseconds()
	}
	ledger.Findings = buildTestDurationFindings(packageRows, testRows, opts)
	ledger.Summary = summarizeTestDurations(packageRows, testRows, ledger.Findings)
	return ledger, nil
}

func planTestDurationRun(goos, target string) (testDurationRunPlan, error) {
	if target == "" {
		target = "fast"
	}
	p, err := planTest(goos, []string{target})
	if err != nil {
		return testDurationRunPlan{}, err
	}
	goArgs := append([]string{"-json"}, p.GoArgs...)
	argv, viaWSL := resolveGoTestArgv(goos, goArgs)
	source := "go test " + strings.Join(goArgs, " ")
	return testDurationRunPlan{
		Target:  target,
		GoArgs:  goArgs,
		Argv:    argv,
		ViaWSL:  viaWSL,
		Source:  source,
		Command: append([]string(nil), argv...),
	}, nil
}

func resolveGoTestArgv(goos string, goArgs []string) ([]string, bool) {
	if goos == "windows" {
		argv := append([]string{"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1"}, goArgs...)
		return argv, true
	}
	argv := append([]string{"go", "test"}, goArgs...)
	return argv, false
}

func runtimeGOOS() string { return runtime.GOOS }

func runTestDurationCommand(name string, args []string, stderr io.Writer) ([]byte, int, error) {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdin = os.Stdin
	cmd.Stderr = stderr
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return out.Bytes(), ee.ExitCode(), nil
		}
		return out.Bytes(), 1, err
	}
	return out.Bytes(), 0, nil
}

func writeTestDurationLedgerFile(path string, ledger testDurationLedger) error {
	return writeIndentedJSONFile(path, ledger)
}

func isTerminalTestAction(action string) bool {
	return action == "pass" || action == "fail" || action == "skip"
}

func elapsedSecondsToMS(seconds float64) int64 {
	if seconds <= 0 {
		return 0
	}
	return int64(math.Round(seconds * 1000))
}

func durationPackageLess(a, b testDurationPackage) bool {
	if a.ElapsedMS != b.ElapsedMS {
		return a.ElapsedMS > b.ElapsedMS
	}
	return a.Package < b.Package
}

func durationTestLess(a, b testDurationTest) bool {
	if a.ElapsedMS != b.ElapsedMS {
		return a.ElapsedMS > b.ElapsedMS
	}
	if a.Package != b.Package {
		return a.Package < b.Package
	}
	return a.Test < b.Test
}

func buildTestDurationFindings(packages []testDurationPackage, tests []testDurationTest, opts testDurationOptions) []testDurationFinding {
	var findings []testDurationFinding
	if opts.PackageBudget > 0 {
		budgetMS := opts.PackageBudget.Milliseconds()
		for _, row := range packages {
			if row.ElapsedMS > budgetMS {
				findings = append(findings, testDurationFinding{
					Kind:         "package_over_budget",
					Target:       row.Package,
					ElapsedMS:    row.ElapsedMS,
					BudgetMS:     budgetMS,
					OverBudgetMS: row.ElapsedMS - budgetMS,
					Action:       "split, shard, or investigate the slow package before widening the full gate",
				})
			}
		}
	}
	if opts.TestBudget > 0 {
		budgetMS := opts.TestBudget.Milliseconds()
		for _, row := range tests {
			if row.ElapsedMS > budgetMS {
				findings = append(findings, testDurationFinding{
					Kind:         "test_over_budget",
					Target:       row.Package + "::" + row.Test,
					ElapsedMS:    row.ElapsedMS,
					BudgetMS:     budgetMS,
					OverBudgetMS: row.ElapsedMS - budgetMS,
					Action:       "isolate the slow test, add a smaller fixture, or mark the cost intentionally budgeted",
				})
			}
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].OverBudgetMS != findings[j].OverBudgetMS {
			return findings[i].OverBudgetMS > findings[j].OverBudgetMS
		}
		if findings[i].ElapsedMS != findings[j].ElapsedMS {
			return findings[i].ElapsedMS > findings[j].ElapsedMS
		}
		return findings[i].Target < findings[j].Target
	})
	for i := range findings {
		findings[i].Rank = i + 1
	}
	return findings
}

func summarizeTestDurations(packages []testDurationPackage, tests []testDurationTest, findings []testDurationFinding) testDurationSummary {
	var s testDurationSummary
	s.Packages = len(packages)
	s.Tests = len(tests)
	s.Findings = len(findings)
	for _, row := range packages {
		s.TotalPackageElapsedMS += row.ElapsedMS
		if row.ElapsedMS > s.SlowestPackageMS || (row.ElapsedMS == s.SlowestPackageMS && row.Package < s.SlowestPackage) {
			s.SlowestPackage = row.Package
			s.SlowestPackageMS = row.ElapsedMS
		}
	}
	for _, row := range tests {
		target := row.Package + "::" + row.Test
		if row.ElapsedMS > s.SlowestTestMS || (row.ElapsedMS == s.SlowestTestMS && target < s.SlowestTest) {
			s.SlowestTest = target
			s.SlowestTestMS = row.ElapsedMS
		}
	}
	return s
}
