package selfupdatecmd

import (
	"bytes"
	"io"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestProgressBarFormatAndGlyphs(t *testing.T) {
	// Test at 40%
	got40 := formatSelfUpdateProgressBar(40, "building fak-dev companion", 80)
	want40 := "fak self-update · [████████░░░░░░░░░░░░]  40%  building fak-dev companion"
	if got40 != want40 {
		t.Fatalf("format 40%% mismatch:\ngot:  %q\nwant: %q", got40, want40)
	}

	// Test at 0%
	got0 := formatSelfUpdateProgressBar(0, "starting", 80)
	want0 := "fak self-update · [░░░░░░░░░░░░░░░░░░░░]   0%  starting"
	if got0 != want0 {
		t.Fatalf("format 0%% mismatch:\ngot:  %q\nwant: %q", got0, want0)
	}

	// Test at 100%
	got100 := formatSelfUpdateProgressBar(100, "done", 80)
	want100 := "fak self-update · [████████████████████] 100%  done"
	if got100 != want100 {
		t.Fatalf("format 100%% mismatch:\ngot:  %q\nwant: %q", got100, want100)
	}

	// Out of bounds: negative clamped to 0%
	gotNeg := formatSelfUpdateProgressBar(-5, "prep", 80)
	if gotNeg != want0[:strings.LastIndex(want0, "starting")]+"prep" {
		t.Fatalf("format -5%% mismatch: got %q", gotNeg)
	}

	// Out of bounds: >100 clamped to 100%
	gotOver := formatSelfUpdateProgressBar(150, "done", 80)
	if gotOver != want100 {
		t.Fatalf("format 150%% mismatch: got %q", gotOver)
	}
}

func TestProgressBarClampingToTerminalWidth(t *testing.T) {
	full := formatSelfUpdateProgressBar(40, "building fak-dev companion", 0)
	fullRunes := utf8.RuneCountInString(full)

	for _, width := range []int{10, 30, 46, 50, 60, fullRunes - 5, fullRunes} {
		clamped := formatSelfUpdateProgressBar(40, "building fak-dev companion", width)
		runeCount := utf8.RuneCountInString(clamped)
		if runeCount > width {
			t.Fatalf("width %d exceeded: rune count %d (%q)", width, runeCount, clamped)
		}
		if runeCount != width {
			t.Fatalf("width %d not fully filled: got %d runes (%q)", width, runeCount, clamped)
		}
	}
}

func TestProgressBarTTYDrawingPrefixAndNoNewline(t *testing.T) {
	var buf bytes.Buffer
	oldProgress := selfUpdateProgress
	oldTerminal := selfUpdateWriterIsTerminal
	oldWidth := selfUpdateTerminalWidth
	selfUpdateProgress = &buf
	selfUpdateWriterIsTerminal = func(_ io.Writer) bool { return true }
	selfUpdateTerminalWidth = func(_ io.Writer) int { return 80 }
	setSelfUpdateVerbose(false)
	resetSelfUpdateProgressForTest()

	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateWriterIsTerminal = oldTerminal
		selfUpdateTerminalWidth = oldWidth
		setSelfUpdateVerbose(false)
		resetSelfUpdateProgressForTest()
	})

	reportSelfUpdateProgress(40, "building fak-dev companion")
	got := buf.String()

	if !strings.HasPrefix(got, "\r\x1b[2K") {
		t.Fatalf("TTY progress bar must start with \\r\\x1b[2K, got: %q", got)
	}
	if strings.HasSuffix(got, "\n") {
		t.Fatalf("TTY progress bar must not have a trailing newline, got: %q", got)
	}
	wantText := "fak self-update · [████████░░░░░░░░░░░░]  40%  building fak-dev companion"
	if !strings.Contains(got, wantText) {
		t.Fatalf("TTY progress bar missing content:\ngot:  %q\nwant: %q", got, wantText)
	}

	// Finishing progress clears bar cleanly
	buf.Reset()
	finishSelfUpdateProgress(outcomeInstalled)
	if buf.String() != "\r\x1b[2K" {
		t.Fatalf("finishSelfUpdateProgress must clear line with \\r\\x1b[2K, got: %q", buf.String())
	}
}

func TestProgressBarNonTTYEmitsStandardLines(t *testing.T) {
	var buf bytes.Buffer
	oldProgress := selfUpdateProgress
	oldTerminal := selfUpdateWriterIsTerminal
	selfUpdateProgress = &buf
	selfUpdateWriterIsTerminal = func(_ io.Writer) bool { return false }
	setSelfUpdateVerbose(false)
	resetSelfUpdateProgressForTest()

	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateWriterIsTerminal = oldTerminal
		setSelfUpdateVerbose(false)
		resetSelfUpdateProgressForTest()
	})

	reportSelfUpdateProgress(40, "building fak-dev companion")
	want := "self-update: progress=40% operation=\"building fak-dev companion\"\n"
	if got := buf.String(); got != want {
		t.Fatalf("non-TTY progress mismatch:\ngot:  %q\nwant: %q", got, want)
	}

	buf.Reset()
	finishSelfUpdateProgress(outcomeInstalled)
	wantFinish := "self-update: progress=100% operation=\"terminal outcome: installed\"\n"
	if got := buf.String(); got != wantFinish {
		t.Fatalf("non-TTY finish mismatch:\ngot:  %q\nwant: %q", got, wantFinish)
	}
}

func TestProgressBarVerboseModeEmitsStandardLinesEvenOnTTY(t *testing.T) {
	var buf bytes.Buffer
	oldProgress := selfUpdateProgress
	oldTerminal := selfUpdateWriterIsTerminal
	selfUpdateProgress = &buf
	selfUpdateWriterIsTerminal = func(_ io.Writer) bool { return true }
	setSelfUpdateVerbose(true)
	resetSelfUpdateProgressForTest()

	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateWriterIsTerminal = oldTerminal
		setSelfUpdateVerbose(false)
		resetSelfUpdateProgressForTest()
	})

	reportSelfUpdateProgress(40, "building fak-dev companion")
	want := "self-update: progress=40% operation=\"building fak-dev companion\"\n"
	if got := buf.String(); got != want {
		t.Fatalf("verbose TTY progress mismatch:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestProgressBarSilencesAuxiliaryMessagesInInteractiveMode(t *testing.T) {
	var buf bytes.Buffer
	oldProgress := selfUpdateProgress
	oldTerminal := selfUpdateWriterIsTerminal
	selfUpdateProgress = &buf
	selfUpdateWriterIsTerminal = func(_ io.Writer) bool { return true }
	setSelfUpdateVerbose(false)
	resetSelfUpdateProgressForTest()

	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateWriterIsTerminal = oldTerminal
		setSelfUpdateVerbose(false)
		resetSelfUpdateProgressForTest()
	})

	// Timing should be silenced
	reportSelfUpdateTiming(selfUpdateTimingSnapshot{totalMS: 100, dominantPhase: "build", dominantMS: 80})
	if buf.Len() != 0 {
		t.Fatalf("timing must be silenced in interactive mode, got: %q", buf.String())
	}

	// Aside notes should be silenced
	reportAsideFootprint("fak.exe")
	if buf.Len() != 0 {
		t.Fatalf("reap notes must be silenced in interactive mode, got: %q", buf.String())
	}

	// With verbose enabled, timing is printed
	setSelfUpdateVerbose(true)
	reportSelfUpdateTiming(selfUpdateTimingSnapshot{totalMS: 100, dominantPhase: "build", dominantMS: 80})
	if !strings.Contains(buf.String(), "self-update: timing") {
		t.Fatalf("timing must be printed in verbose mode, got: %q", buf.String())
	}
}

func TestProgressBarHeartbeatInInteractiveModeRefreshesWithoutDuplicateLines(t *testing.T) {
	var buf bytes.Buffer
	oldProgress, oldWait := selfUpdateProgress, selfUpdateHeartbeatWait
	oldTerminal := selfUpdateWriterIsTerminal
	selfUpdateProgress = &buf
	selfUpdateWriterIsTerminal = func(_ io.Writer) bool { return true }
	selfUpdateTerminalWidth = func(_ io.Writer) int { return 80 }
	setSelfUpdateVerbose(false)
	resetSelfUpdateProgressForTest()

	ready := make(chan struct{})
	calls := 0
	selfUpdateHeartbeatWait = func(stop <-chan struct{}, interval time.Duration) bool {
		calls++
		if calls <= 2 {
			return true
		}
		close(ready)
		<-stop
		return false
	}
	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateHeartbeatWait = oldWait
		selfUpdateWriterIsTerminal = oldTerminal
		setSelfUpdateVerbose(false)
		resetSelfUpdateProgressForTest()
	})

	stop := startSelfUpdateHeartbeat(55, "building fak candidate")
	<-ready
	stop()

	got := buf.String()
	// Must not contain heartbeat=true line
	if strings.Contains(got, "heartbeat=true") {
		t.Fatalf("interactive heartbeat must not emit heartbeat=true lines, got: %q", got)
	}
	// Must not have newlines
	if strings.Contains(got, "\n") {
		t.Fatalf("interactive heartbeat must not emit newlines, got: %q", got)
	}
	// Must contain prefix \r\x1b[2K
	if !strings.HasPrefix(got, "\r\x1b[2K") {
		t.Fatalf("interactive heartbeat must emit \\r\\x1b[2K, got: %q", got)
	}
}

func TestProgressReporterStructAPI(t *testing.T) {
	var buf bytes.Buffer
	reporter := NewProgressReporter(&buf, false)
	reporter.isTerminal = func(_ io.Writer) bool { return true }
	reporter.termWidth = func(_ io.Writer) int { return 80 }

	if !reporter.IsInteractive() {
		t.Fatalf("reporter must be interactive")
	}

	reporter.Report(40, "building candidate")
	if !strings.HasPrefix(buf.String(), "\r\x1b[2K") {
		t.Fatalf("reporter output missing \\r\\x1b[2K: %q", buf.String())
	}
	if strings.HasSuffix(buf.String(), "\n") {
		t.Fatalf("reporter output should not have trailing newline: %q", buf.String())
	}

	buf.Reset()
	reporter.Clear()
	if buf.String() != "\r\x1b[2K" {
		t.Fatalf("reporter.Clear() mismatch: %q", buf.String())
	}

	buf.Reset()
	reporter.barDrawn = true
	reporter.Settle()
	if buf.String() != "\n" {
		t.Fatalf("reporter.Settle() mismatch: %q", buf.String())
	}
}

func TestSelfUpdateVerboseFlags(t *testing.T) {
	for _, tc := range []struct {
		argv []string
		env  string
		want bool
	}{
		{argv: nil, env: "", want: false},
		{argv: []string{"--verbose"}, env: "", want: true},
		{argv: []string{"-v"}, env: "", want: true},
		{argv: nil, env: "1", want: true},
		{argv: nil, env: "true", want: true},
		{argv: nil, env: "0", want: false},
		{argv: []string{"--verbose"}, env: "0", want: true}, // flag overrides env
	} {
		t.Run(strings.Join(tc.argv, " ")+"/env="+tc.env, func(t *testing.T) {
			t.Setenv("FAK_SELF_UPDATE_VERBOSE", tc.env)
			isVerbose := false
			for _, a := range tc.argv {
				if a == "--verbose" || a == "-v" {
					isVerbose = true
				}
			}
			if !isVerbose {
				if v := tc.env; v != "" {
					if v == "1" || v == "true" {
						isVerbose = true
					}
				}
			}
			if isVerbose != tc.want {
				t.Fatalf("verbose parsed = %v, want %v", isVerbose, tc.want)
			}
		})
	}
}
