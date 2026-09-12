package main

import (
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/armtracking"
)

// runBaselineArms is the `fak baseline-arms` verb shell; it dispatches to the
// tested internal/armtracking CLI (record/compare/leaderboard/audit).
func runBaselineArms(stdout, stderr io.Writer, args []string) int {
	return armtracking.RunCLI(stdout, stderr, args)
}

// baselineArmsStdout/Stderr and baselineArmsExit are the production sinks for
// the verb shell. They exist so the router-level reachability test can
// capture real dispatch output and exit codes in-process; tests swap them
// for buffers and a recorder via t.Cleanup.
var (
	baselineArmsStdout io.Writer = os.Stdout
	baselineArmsStderr io.Writer = os.Stderr
	baselineArmsExit             = os.Exit
)

// cmdBaselineArms exits only for non-zero codes: help/usage (exit 0) returns
// to main for its normal recordUsage+return flow, which is behavior-identical
// for users while letting the router-level dispatch test run in-process
// (os.Exit(0) during a test fails the run).
func cmdBaselineArms(args []string) {
	if code := runBaselineArms(baselineArmsStdout, baselineArmsStderr, args); code != 0 {
		baselineArmsExit(code)
	}
}
