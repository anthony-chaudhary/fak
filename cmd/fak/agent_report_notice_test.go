package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAgentOfflineAnnouncesReportPath is the #5473 witness.
//
// `fak agent --offline` is the first command the README hands a stranger, and
// the natural place to run it is wherever that stranger's shell happens to be.
// Its --out default is the bare name "agent-report.json", so the run drops a
// file into that directory. The run summary said only
// "report written: agent-report.json" — a filename with no directory in it —
// which is not enough to find or delete the file afterwards.
//
// This test pins the stderr notice that names the absolute path actually
// written plus the flag that redirects it. It FAILS before the fix, where
// nothing at all reached stderr on the offline path.
func TestAgentOfflineAnnouncesReportPath(t *testing.T) {
	t.Chdir(t.TempDir())
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}

	stdout, stderr := captureAgentStdio(t, func() { cmdAgent([]string{"--offline"}) })

	want := filepath.Join(cwd, "agent-report.json")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("fak agent --offline must write %s: %v", want, err)
	}
	if !strings.Contains(stderr, want) {
		t.Errorf("fak agent --offline must announce the FULL path of the file it drops in the caller's cwd (#5473).\n"+
			"want stderr to contain: %s\ngot stderr:\n%s", want, stderr)
	}
	if !strings.Contains(stderr, "--out") {
		t.Errorf("the announcement must name the flag that redirects the file, so the user can put it elsewhere (#5473).\n"+
			"got stderr:\n%s", stderr)
	}
	// The stdout summary is a published transcript (GETTING-STARTED.md,
	// docs/fak/tutorial.md, examples/agent-ab/EXAMPLE-OUTPUT.md) and is what any
	// script parses, so the notice has to be additive rather than a rewrite of
	// the existing line.
	if !strings.Contains(stdout, "report written: agent-report.json") {
		t.Errorf("the stdout run summary must stay byte-identical; got stdout:\n%s", stdout)
	}
}

// TestAgentReportNoticeHonoursExplicitOut keeps step 3 of the ticket honest: an
// explicit --out is written exactly where the caller asked (no relocation), and
// the notice reports that same path.
func TestAgentReportNoticeHonoursExplicitOut(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "chosen-report.json")

	_, stderr := captureAgentStdio(t, func() { cmdAgent([]string{"--offline", "--out", out}) })

	if _, err := os.Stat(out); err != nil {
		t.Fatalf("an explicit --out must still be honoured verbatim: %v", err)
	}
	if !strings.Contains(stderr, out) {
		t.Errorf("the notice must report the explicit --out path.\nwant stderr to contain: %s\ngot stderr:\n%s", out, stderr)
	}
}

// captureAgentStdio swaps the process stdio for pipes while fn runs and returns
// what fn wrote to each. cmdAgent addresses os.Stdout/os.Stderr at call time, so
// swapping the variables is enough; the readers run concurrently so a report
// larger than the pipe buffer cannot deadlock.
func captureAgentStdio(t *testing.T, fn func()) (stdout, stderr string) {
	t.Helper()
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	outDone := make(chan string, 1)
	errDone := make(chan string, 1)
	go func() { b, _ := io.ReadAll(outR); outDone <- string(b) }()
	go func() { b, _ := io.ReadAll(errR); errDone <- string(b) }()

	origOut, origErr := os.Stdout, os.Stderr
	func() {
		defer func() {
			os.Stdout, os.Stderr = origOut, origErr
			_ = outW.Close()
			_ = errW.Close()
		}()
		os.Stdout, os.Stderr = outW, errW
		fn()
	}()
	return <-outDone, <-errDone
}
