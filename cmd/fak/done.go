package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopgate"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	doneVerdictGreen = "GREEN"
	doneVerdictRed   = "RED"
)

type doneRunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
	Err    error
}

type doneRunner func(ctx context.Context, dir, name string, args ...string) doneRunResult

type doneOptions struct {
	Dir        string
	Paths      []string
	TestTarget string
	Witness    string
	JSON       bool
}

type doneReport struct {
	OK             bool        `json:"ok"`
	Verdict        string      `json:"verdict"`
	MissingWitness string      `json:"missing_witness,omitempty"`
	NextStep       string      `json:"next_step,omitempty"`
	Checks         []doneCheck `json:"checks"`
}

type doneCheck struct {
	Name           string   `json:"name"`
	OK             bool     `json:"ok"`
	Command        []string `json:"command,omitempty"`
	ExitCode       int      `json:"exit_code,omitempty"`
	Detail         string   `json:"detail,omitempty"`
	MissingWitness string   `json:"missing_witness,omitempty"`
	NextStep       string   `json:"next_step,omitempty"`
}

var doneRunnerForCommand doneRunner = doneRealRunner

func cmdDone(argv []string) { os.Exit(runDone(os.Stdout, os.Stderr, argv)) }

func runDone(stdout, stderr io.Writer, argv []string) int {
	return runDoneWithRunner(stdout, stderr, argv, doneRunnerForCommand)
}

func runDoneWithRunner(stdout, stderr io.Writer, argv []string, runner doneRunner) int {
	fs := flag.NewFlagSet("done", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths pathList
	fs.Var(&paths, "paths", "repo-relative pathspec/glob to require clean before claiming done (repeatable)")
	fs.Var(&paths, "path", "alias for --paths")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	testTarget := fs.String("test", "fast", "fak test target to run before claiming done (use 'none' to skip only when another witness covers tests)")
	witness := fs.String("witness", "commit-audit HEAD", "loopgate witness criterion: commit-audit [REF], verify PLAN PHASE, test-witness BASE CAND, witness SOURCE SUBJECT")
	asJSON := fs.Bool("json", false, "emit the done report as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak done: unexpected positional arguments")
		return 2
	}
	root := resolveRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak done: could not resolve git repo root")
		return 2
	}
	report := runDoneCheck(context.Background(), doneOptions{
		Dir:        root,
		Paths:      paths,
		TestTarget: *testTarget,
		Witness:    *witness,
		JSON:       *asJSON,
	}, runner)
	if *asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak done: %v\n", err)
			return 1
		}
	} else {
		renderDoneReport(stdout, report)
	}
	if !report.OK {
		return 1
	}
	return 0
}

func runDoneCheck(ctx context.Context, opt doneOptions, runner doneRunner) doneReport {
	if runner == nil {
		runner = doneRealRunner
	}
	var checks []doneCheck
	checks = append(checks, doneCleanCheck(ctx, opt, runner))
	if strings.TrimSpace(opt.TestTarget) != "" && !strings.EqualFold(strings.TrimSpace(opt.TestTarget), "none") {
		checks = append(checks, doneCommandCheck(ctx, opt.Dir, runner, "tests_green", "tests_green",
			"run fak test "+strings.TrimSpace(opt.TestTarget)+" and fix the failing test/build witness",
			doneFakTestCommand(strings.TrimSpace(opt.TestTarget))...))
	}
	checks = append(checks, doneCommandCheck(ctx, opt.Dir, runner, "claims_lint", "claims_lint",
		"run make claims-lint and fix every CLAIMS.md tag violation",
		"make", "claims-lint"))
	checks = append(checks, doneWitnessCheck(ctx, opt, runner))
	return summarizeDoneChecks(checks)
}

func doneCleanCheck(ctx context.Context, opt doneOptions, runner doneRunner) doneCheck {
	args := []string{"status", "--porcelain", "--"}
	args = append(args, opt.Paths...)
	res := runner(ctx, opt.Dir, "git", args...)
	check := doneCheck{
		Name:     "path_clean",
		Command:  append([]string{"git"}, args...),
		ExitCode: res.Code,
	}
	if res.Err != nil {
		check.Detail = trimDoneDetail(res.Err.Error())
		check.MissingWitness = "path_clean"
		check.NextStep = "make git status readable, then rerun fak done"
		return check
	}
	if res.Code != 0 {
		check.Detail = firstDoneDetail(res.Stderr, res.Stdout)
		check.MissingWitness = "path_clean"
		check.NextStep = "make git status readable, then rerun fak done"
		return check
	}
	dirty := strings.TrimSpace(string(res.Stdout))
	if dirty != "" {
		check.Detail = dirty
		check.MissingWitness = "clean_path_state"
		check.NextStep = "commit the intended paths with fak commit, narrow --paths, or remove unrelated dirty work before claiming done"
		return check
	}
	check.OK = true
	check.Detail = "selected paths are clean"
	return check
}

func doneCommandCheck(ctx context.Context, dir string, runner doneRunner, name, missing, next string, argv ...string) doneCheck {
	if len(argv) == 0 {
		return doneCheck{Name: name, MissingWitness: missing, NextStep: next, Detail: "empty command"}
	}
	res := runner(ctx, dir, argv[0], argv[1:]...)
	check := doneCheck{
		Name:     name,
		Command:  argv,
		ExitCode: res.Code,
	}
	if res.Err == nil && res.Code == 0 {
		check.OK = true
		check.Detail = firstDoneDetail(res.Stdout, res.Stderr)
		if check.Detail == "" {
			check.Detail = "passed"
		}
		return check
	}
	check.Detail = firstDoneDetail(res.Stderr, res.Stdout)
	if check.Detail == "" && res.Err != nil {
		check.Detail = trimDoneDetail(res.Err.Error())
	}
	check.MissingWitness = missing
	check.NextStep = next
	return check
}

func doneWitnessCheck(ctx context.Context, opt doneOptions, runner doneRunner) doneCheck {
	criterion, err := loopDriveGateCriterion(opt.Witness)
	check := doneCheck{Name: "done_witness"}
	if err != nil {
		check.MissingWitness = "witness_criterion"
		check.NextStep = "use --witness 'commit-audit HEAD' or --witness 'verify PLAN PHASE'"
		check.Detail = err.Error()
		return check
	}
	decision := loopgate.Adjudicate(ctx, loopgate.Turn{
		ClaimedDone: true,
		Claim:       "fak done pre-claim check",
		HeadRef:     "HEAD",
		Criterion:   criterion,
	}, func(ctx context.Context, req loopgate.Request) (loopgate.WitnessResult, error) {
		check.Command = append([]string{"dos"}, req.Argv()...)
		res := runner(ctx, opt.Dir, "dos", req.Argv()...)
		check.ExitCode = res.Code
		parsed, parseErr := parseDOSLoopGateWitness(req, res.Stdout)
		if parseErr == nil {
			return parsed, nil
		}
		if res.Err != nil {
			return loopgate.WitnessResult{}, res.Err
		}
		if res.Code != 0 {
			return loopgate.WitnessResult{}, fmt.Errorf("%s exited %d: %s", strings.Join(check.Command, " "), res.Code, firstDoneDetail(res.Stderr, res.Stdout))
		}
		return loopgate.WitnessResult{}, parseErr
	})
	check.Detail = decision.Summary
	if decision.Verdict == loopgate.VerdictWitnessed {
		check.OK = true
		return check
	}
	check.MissingWitness = firstNonEmptyString(decision.Reason, "done_witness")
	check.NextStep = doneWitnessNextStep(decision)
	return check
}

func doneWitnessNextStep(decision loopgate.Decision) string {
	if len(decision.Request.Argv()) == 0 {
		return "provide a checkable witness criterion and rerun fak done"
	}
	return "satisfy the witness, then rerun: dos " + strings.Join(decision.Request.Argv(), " ")
}

func summarizeDoneChecks(checks []doneCheck) doneReport {
	report := doneReport{OK: true, Verdict: doneVerdictGreen, Checks: checks}
	for _, check := range checks {
		if check.OK {
			continue
		}
		report.OK = false
		report.Verdict = doneVerdictRed
		report.MissingWitness = check.MissingWitness
		report.NextStep = check.NextStep
		break
	}
	return report
}

func renderDoneReport(w io.Writer, report doneReport) {
	fmt.Fprintf(w, "fak done: %s\n", report.Verdict)
	if !report.OK {
		fmt.Fprintf(w, "missing witness: %s\n", report.MissingWitness)
		fmt.Fprintf(w, "next step: %s\n", report.NextStep)
	}
	for _, check := range report.Checks {
		status := "FAIL"
		if check.OK {
			status = "PASS"
		}
		fmt.Fprintf(w, "  %s %s", status, check.Name)
		if check.Detail != "" {
			fmt.Fprintf(w, " - %s", check.Detail)
		}
		fmt.Fprintln(w)
	}
}

func doneFakTestCommand(target string) []string {
	exe, err := os.Executable()
	if err != nil || strings.TrimSpace(exe) == "" {
		exe = "fak"
	}
	return []string{exe, "test", target}
}

func doneRealRunner(ctx context.Context, dir, name string, args ...string) doneRunResult {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			code = exit.ExitCode()
			err = nil
		} else {
			code = 1
		}
	}
	return doneRunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code, Err: err}
}

func firstDoneDetail(streams ...[]byte) string {
	for _, stream := range streams {
		if s := trimDoneDetail(string(stream)); s != "" {
			return s
		}
	}
	return ""
}

func trimDoneDetail(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 300 {
		return s[:297] + "..."
	}
	return s
}

func firstNonEmptyString(xs ...string) string {
	for _, x := range xs {
		if s := strings.TrimSpace(x); s != "" {
			return s
		}
	}
	return ""
}
