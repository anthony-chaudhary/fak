package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
)

type validateArgs struct {
	root           string
	ref            string
	asJSON         bool
	timeout        time.Duration
	progress       bool
	testOnly       bool
	wslTests       bool
	testRun        string
	auditSelection bool
	mine           pathList
}

func parseValidateArgs(stderr io.Writer, argv []string) (*validateArgs, int) {
	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	ref := fs.String("ref", "HEAD", "committed base ref or sha")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	timeout := fs.Duration("timeout", defaultValidateTimeout, "maximum total validation time")
	progress := fs.Bool("progress", validateWriterIsTerminal(stderr), "emit phase progress to stderr (default on when stderr is a TTY)")
	testOnly := fs.Bool("test-only", false, "skip affected-package build/vet and run only affected tests in the isolated checkout")
	wslTests := fs.Bool("wsl-tests", defaultValidateWSLTests(runtime.GOOS), "run isolated affected tests through WSL (default on Windows hosts)")
	testRun := fs.String("test-run", "", "go test -run expression for isolated affected tests")
	auditSelection := fs.Bool("audit-selection", false, "compare affected tests with a full-suite truth run")
	var mine pathList
	fs.Var(&mine, "mine", "owned changed path to overlay (repeatable; files and directories accepted)")
	positional, parseErr := interspersedflags.Parse(fs, argv)
	if parseErr != nil {
		return nil, 2
	}
	for _, p := range positional {
		if p = strings.TrimSpace(p); p != "" {
			mine = append(mine, p)
		}
	}
	if len(mine) == 0 {
		fmt.Fprintln(stderr, "fak validate: at least one --mine path is required; ownership is never inferred from a peer-dirty tree")
		return nil, 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak validate: --timeout must be greater than zero")
		return nil, 2
	}
	return &validateArgs{
		root:           *root,
		ref:            *ref,
		asJSON:         *asJSON,
		timeout:        *timeout,
		progress:       *progress,
		testOnly:       *testOnly,
		wslTests:       *wslTests,
		testRun:        *testRun,
		auditSelection: *auditSelection,
		mine:           mine,
	}, 0
}

func writeValidateTestContext(w io.Writer, res validateResult) {
	if res.Runner != "" {
		fmt.Fprintf(w, "runner: %s\n", res.Runner)
	}
	if res.TestScope != "" {
		fmt.Fprintf(w, "tests: %s (%s)\n", res.TestScope, res.TestRun)
	}
}

func recordValidateFailure(res *validateResult, phase validateActivePhase, step, detail string, cause error) {
	phase.finishAs("failed", cause.Error())
	res.OK = false
	res.Failures = append(res.Failures, ciPreflightFailure{Step: step, Detail: detail})
}

func finishValidatePhaseOrTimeout(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, err error, asJSON bool) (int, bool) {
	phase.finish(err)
	if recorder.ctx.Err() == nil {
		return 0, false
	}
	return finishValidateTimeout(stdout, res, recorder, name, asJSON), true
}

func finishValidateRequiredPhase(stdout, stderr io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, err error, asJSON bool, failureMessage string) (int, bool) {
	if code, timedOut := finishValidatePhaseOrTimeout(stdout, res, recorder, phase, name, err, asJSON); timedOut {
		return code, true
	}
	if err == nil {
		return 0, false
	}
	fmt.Fprintln(stderr, failureMessage)
	return 2, true
}

func finishValidateContextPhase(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, asJSON bool) (int, bool) {
	if recorder.ctx.Err() == nil {
		return 0, false
	}
	phase.finish(recorder.ctx.Err())
	return finishValidateTimeout(stdout, res, recorder, name, asJSON), true
}

func runValidateCheckPhase(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, failure error, asJSON bool, run func() (string, bool)) (int, bool) {
	detail, ok := run()
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, name, asJSON); timedOut {
		return code, true
	}
	if ok {
		phase.finish(nil)
	} else {
		recordValidateFailure(res, phase, name, detail, failure)
	}
	return 0, false
}

func validateTimeoutPhase(res validateResult) string {
	for i := len(res.Phases) - 1; i >= 0; i-- {
		if res.Phases[i].Status == "timeout" {
			return res.Phases[i].Name
		}
	}
	return "unknown"
}

func finishValidateTimeout(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase string, asJSON bool) int {
	res.OK = false
	res.Partial = true
	res.TimedOut = true
	res.Reason = "TIMEOUT"
	res.Failures = append(res.Failures, ciPreflightFailure{
		Step: phase, Detail: fmt.Sprintf("validation timeout after %s", time.Duration(res.TimeoutMS)*time.Millisecond),
	})
	res.Overlays.Skipped = subtractValidatePaths(res.Mine, res.Overlays.Checked)
	completed := make(map[string]bool, len(res.Phases))
	for _, timing := range res.Phases {
		completed[timing.Name] = true
	}
	for _, name := range recorder.phaseOrder {
		if !completed[name] {
			res.SkippedPhases = append(res.SkippedPhases, name)
		}
	}
	recorder.finish()
	emitValidateResult(stdout, *res, asJSON)
	return 1
}

func finishValidateWSLCapabilityRefusal(stdout, stderr io.Writer, res *validateResult, recorder *validateRecorder, verdict validateWSLCapabilityVerdict, asJSON bool) int {
	reason := "WSL_CAPABILITY_MISSING"
	if verdict.Status == "unavailable" {
		reason = "WSL_CAPABILITY_PREFLIGHT_FAILED"
	}
	res.OK = false
	res.Partial = true
	res.Reason = reason
	res.Failures = append(res.Failures, ciPreflightFailure{Step: "wsl_preflight", Detail: verdict.Detail})
	completed := make(map[string]bool, len(res.Phases))
	for _, timing := range res.Phases {
		completed[timing.Name] = true
	}
	for _, name := range recorder.phaseOrder {
		if !completed[name] {
			res.SkippedPhases = append(res.SkippedPhases, name)
		}
	}
	recorder.finish()
	emitValidateResult(stdout, *res, asJSON)
	fmt.Fprintln(stderr, "fak validate: "+verdict.Detail)
	return 2
}

func emitValidateResult(stdout io.Writer, res validateResult, asJSON bool) {
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}
	renderValidate(stdout, res)
}

func renderValidate(w io.Writer, res validateResult) {
	if res.TimedOut {
		fmt.Fprintf(w, "PARTIAL: validation timed out after %s during %s\n",
			time.Duration(res.ElapsedMS)*time.Millisecond, validateTimeoutPhase(res))
		fmt.Fprintf(w, "  overlays checked: %d; skipped: %d\n", len(res.Overlays.Checked), len(res.Overlays.Skipped))
		if len(res.SkippedPhases) > 0 {
			fmt.Fprintf(w, "  phases not run: %s\n", strings.Join(res.SkippedPhases, ", "))
		}
		return
	}
	if res.OK {
		writeValidateTestContext(w, res)
		if res.Mode == "test-only" {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) changed-package tests clean (isolated test-only mode)\n", short(res.Tip), len(res.Mine))
		} else {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) importer build/vet and changed-package tests clean\n", short(res.Tip), len(res.Mine))
		}
		return
	}
	writeValidateTestContext(w, res)
	fmt.Fprintf(w, "RED: committed tip %s + owned delta failed\n", short(res.Tip))
	for _, f := range res.Failures {
		fmt.Fprintf(w, "  %s", f.Step)
		if f.Detail != "" {
			fmt.Fprintf(w, ": %s", f.Detail)
		}
		if len(f.Files) > 0 {
			fmt.Fprintf(w, ": %s", strings.Join(f.Files, ", "))
		}
		fmt.Fprintln(w)
	}
}

func runValidateSelectionAudit(stdout io.Writer, ctx context.Context, r, dir, tip string, paths []string, fileToPkg map[string]string, selectedObservation affectedtests.TestObservation, res *validateResult, recorder *validateRecorder, args *validateArgs, wslWorkspace bool) (int, bool) {
	fullPackages := validateAllPackages(fileToPkg)
	phase := recorder.start("test_audit_full")
	fullArgs := validateJSONTestArgs(validateTestArgs("", packagePatternsForRoot(dir, fullPackages, fileToPkg)))
	detail, _ := runValidateTestCommand(ctx, r, dir, tip, fullArgs, args.wslTests, wslWorkspace)
	fullObservation := parseValidateTestObservation(detail, fullPackages, ctx.Err() == nil)
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "test_audit_full", args.asJSON); timedOut {
		return code, true
	}
	audit := affectedtests.AuditSelection(selectedObservation, fullObservation)
	selectedPackages := append([]string(nil), res.Tested...)
	sort.Strings(selectedPackages)
	res.SelectionAudit = &validateSelectionAudit{
		Base: tip, Head: validateAuditHead(r, tip, paths),
		SelectedPackages: selectedPackages, SelectionAudit: audit,
	}
	if audit.Sound {
		phase.finish(nil)
	} else {
		detail := "affected-test selection was not sound"
		if !audit.Complete {
			detail = "affected-test selection audit was incomplete"
		}
		phase.finishAs("failed", detail)
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "test-audit-selection", Detail: detail})
	}
	return 0, false
}
