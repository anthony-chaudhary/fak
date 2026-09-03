package main

import (
	"fmt"
	"io"
)

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
