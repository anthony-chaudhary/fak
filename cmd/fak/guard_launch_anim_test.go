package main

import (
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// TestGuardLaunchAnimEnabled pins the render-site gate: the animation plays ONLY for the
// resolved animate mode into a real color TTY that has not opted out. Every other shape
// (another banner mode, a piped/redirected stderr, NO_COLOR, FAK_GUARD_LAUNCH_ANIM=off) must
// fall back to the byte-clean static banner — the piped/NO_COLOR cases are the ones that keep a
// captured log free of the carriage-return redraw.
func TestGuardLaunchAnimEnabled(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mode      string
		stderrTTY bool
		noColor   bool
		env       string
		want      bool
	}{
		{name: "animate + color TTY plays", mode: guardBannerAnimate, stderrTTY: true, want: true},
		{name: "compact mode never animates", mode: guardBannerCompact, stderrTTY: true, want: false},
		{name: "full mode never animates", mode: guardBannerFull, stderrTTY: true, want: false},
		{name: "piped stderr falls back", mode: guardBannerAnimate, stderrTTY: false, want: false},
		{name: "NO_COLOR falls back", mode: guardBannerAnimate, stderrTTY: true, noColor: true, want: false},
		{name: "env off falls back", mode: guardBannerAnimate, stderrTTY: true, env: "off", want: false},
		{name: "env OFF case-insensitive", mode: guardBannerAnimate, stderrTTY: true, env: "  OFF ", want: false},
		{name: "env on is not an opt-out", mode: guardBannerAnimate, stderrTTY: true, env: "on", want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardLaunchAnimEnabled(tc.mode, tc.stderrTTY, tc.noColor, tc.env); got != tc.want {
				t.Fatalf("guardLaunchAnimEnabled(%q, tty=%v, noColor=%v, env=%q) = %v, want %v",
					tc.mode, tc.stderrTTY, tc.noColor, tc.env, got, tc.want)
			}
		})
	}
}

// TestGuardLaunchAnimFramesAreCleanAndFill witnesses the frame builder's two invariants: every
// frame is byte-clean (no escape/carriage-return bytes — color and cursor motion are the
// player's job, so the frames stay assertable and width math stays exact), and the capability
// floor visibly fills from empty to full across the sequence.
func TestGuardLaunchAnimFramesAreCleanAndFill(t *testing.T) {
	frames := guardLaunchAnimFrames(80)
	if len(frames) < 2 {
		t.Fatalf("want at least 2 frames, got %d", len(frames))
	}
	for i, f := range frames {
		if strings.ContainsAny(f, "\x1b\r\n") {
			t.Fatalf("frame %d carries escape/CR/LF bytes (%q); frames must be byte-clean", i, f)
		}
		if !strings.Contains(f, "fak "+appversion.Current()) {
			t.Fatalf("frame %d %q lost the fak version mark", i, f)
		}
	}
	first, last := frames[0], frames[len(frames)-1]
	if strings.ContainsRune(first, guardLaunchFilledCell) {
		t.Errorf("first frame should start with an empty floor, got %q", first)
	}
	if strings.ContainsRune(last, guardLaunchEmptyCell) {
		t.Errorf("last frame should show a full floor, got %q", last)
	}
	if strings.Count(last, string(guardLaunchFilledCell)) != guardLaunchTrackCells {
		t.Errorf("last frame floor = %d filled cells, want %d (%q)",
			strings.Count(last, string(guardLaunchFilledCell)), guardLaunchTrackCells, last)
	}
}

// TestGuardLaunchAnimFramesWidthCapped proves a narrow terminal can never make a frame wrap
// onto a second row (which would scroll and destroy the one-line motif): every frame fits the
// requested display width.
func TestGuardLaunchAnimFramesWidthCapped(t *testing.T) {
	const width = 12
	for i, f := range guardLaunchAnimFrames(width) {
		if w := dispWidthTUI(f); w > width {
			t.Fatalf("frame %d width %d exceeds cap %d (%q)", i, w, width, f)
		}
	}
}

// TestGuardLaunchSettleLines pins the resting state the animation lands on: an iconic identity
// line carrying the version + build + agent + gateway, then the `fak info --startup` pointer
// that also names the launch-failure spill, then the prior-run refusal carry-forward VERBATIM
// when present (the block compaction must never hide).
func TestGuardLaunchSettleLines(t *testing.T) {
	lines := guardLaunchSettleLines("9.9.9", "abc123", "claude", "http://127.0.0.1:8080", 200, nil)
	if len(lines) != 2 {
		t.Fatalf("want 2 settle lines with no refusals, got %d: %#v", len(lines), lines)
	}
	head := lines[0]
	for _, want := range []string{"9.9.9", "abc123", "claude", "kernel floor", "http://127.0.0.1:8080"} {
		if !strings.Contains(head, want) {
			t.Errorf("identity line %q missing %q", head, want)
		}
	}
	if !strings.Contains(lines[1], "fak info --startup") {
		t.Errorf("pointer line %q must name `fak info --startup`", lines[1])
	}
	if !strings.Contains(lines[1], "launch fails") {
		t.Errorf("pointer line %q must state the launch-failure spill", lines[1])
	}

	// No build stamp: the identity must SAY so rather than look current (the stale-binary tell).
	noStamp := guardLaunchSettleLines("9.9.9", "", "claude", "http://x", 200, nil)
	if !strings.Contains(noStamp[0], "no stamp") {
		t.Errorf("missing build stamp must surface in the identity line, got %q", noStamp[0])
	}

	// Refusal history must not grow the attended resting state; full detail stays in the startup report.
	withRefusals := guardLaunchSettleLines("9.9.9", "abc123", "claude", "http://x", 200,
		[]guardRefusalCarry{{Reason: "FS_WRITE_OUTSIDE_ROOT", Count: 2}})
	if len(withRefusals) != 2 {
		t.Fatalf("refusal history grew attended settle lines: got %d: %#v", len(withRefusals), withRefusals)
	}
	if strings.Contains(strings.Join(withRefusals, "\n"), "FS_WRITE_OUTSIDE_ROOT") {
		t.Errorf("refusal history leaked into attended settle lines: %#v", withRefusals)
	}
}

// TestPlayGuardLaunchAnimationInPlace witnesses the player: each frame is preceded by the
// clear-line control so the redraw stays on ONE row (no per-frame newline scroll), the settle
// lines land on their own rows, the injected sleep is called once per frame, and the color flag
// gates only the color SGR (the cursor control is always present — the gate, not the player, is
// what keeps a non-TTY stderr from ever seeing this).
func TestPlayGuardLaunchAnimationInPlace(t *testing.T) {
	frames := []string{"f0", "f1", "f2"}
	settle := []string{"identity", "pointer"}

	var mono strings.Builder
	sleeps := 0
	playGuardLaunchAnimation(&mono, frames, settle, false, func(time.Duration) { sleeps++ })
	out := mono.String()

	if sleeps != len(frames) {
		t.Errorf("sleep called %d times, want one per frame (%d)", sleeps, len(frames))
	}
	if strings.Contains(out, tuiSGRCyanBold) || strings.Contains(out, tuiSGRDim) {
		t.Errorf("non-color playback must emit no color SGR, got %q", out)
	}
	if got := strings.Count(out, guardLaunchClearLine); got != len(frames)+1 {
		t.Errorf("clear-line count = %d, want one per frame plus the pre-settle clear (%d)", got, len(frames)+1)
	}
	// Only the settle lines produce newlines: the moving frames redraw in place.
	if got := strings.Count(out, "\n"); got != len(settle) {
		t.Errorf("newline count = %d, want one per settle line (%d) — frames must not scroll", got, len(settle))
	}
	if !strings.HasSuffix(out, "identity\npointer\n") {
		t.Errorf("playback must land on the settle lines, got tail %q", out)
	}

	var color strings.Builder
	playGuardLaunchAnimation(&color, frames, settle, true, func(time.Duration) {})
	if !strings.Contains(color.String(), tuiSGRCyanBold) {
		t.Errorf("color playback must emit the cyan-bold frame hue")
	}
	if !strings.Contains(color.String(), tuiSGRDim) {
		t.Errorf("color playback must dim the settle pointer line")
	}
}

// TestGuardLaunchAgentName pins the short-name extraction used in the settle line: a bare name,
// a path, and a Windows .exe all collapse to the lowercase base with no extension.
func TestGuardLaunchAgentName(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"claude", "claude"},
		{`C:\tools\Claude.exe`, "claude"},
		{"/usr/local/bin/claude", "claude"},
		{"  codex.CMD ", "codex"},
	} {
		if got := guardLaunchAgentName(tc.in); got != tc.want {
			t.Errorf("guardLaunchAgentName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGuardChildIsLaunchFailure pins the classifier that decides whether to spill the full
// report: a spawn/exec error (child never started) is a launch failure; a normal non-zero exit
// (*exec.ExitError — the child ran) is NOT; a nil error is a clean run. The ExitError witness
// uses the zero value: the classifier keys on the error TYPE, so no subprocess is needed.
func TestGuardChildIsLaunchFailure(t *testing.T) {
	if guardChildIsLaunchFailure(nil) {
		t.Error("nil error is a clean run, not a launch failure")
	}
	if guardChildIsLaunchFailure(&exec.ExitError{}) {
		t.Error("an *exec.ExitError means the child ran and exited — not a launch failure")
	}
	if !guardChildIsLaunchFailure(exec.ErrNotFound) {
		t.Error("exec.ErrNotFound (binary missing) is the canonical launch failure")
	}
	if !guardChildIsLaunchFailure(errors.New("fork/exec: permission denied")) {
		t.Error("a non-ExitError spawn error must classify as a launch failure")
	}
}

// TestGuardWriteLaunchFailReport pins the failure spill: an enabled, non-empty report prints
// under the "launch failed" header; a disabled call (the --banner=full case that already
// streamed the report at boot) and an empty report are both silent no-ops.
func TestGuardWriteLaunchFailReport(t *testing.T) {
	const report = "fak guard 9.9.9 — kernel-adjudicated: claude\n  floor : built-in guard floor"

	var on strings.Builder
	guardWriteLaunchFailReport(&on, report, true)
	if !strings.Contains(on.String(), "launch failed") || !strings.Contains(on.String(), "built-in guard floor") {
		t.Errorf("enabled spill must print the header and the report body, got %q", on.String())
	}

	var off strings.Builder
	guardWriteLaunchFailReport(&off, report, false)
	if off.Len() != 0 {
		t.Errorf("disabled spill must be silent (report already streamed at boot), got %q", off.String())
	}

	var empty strings.Builder
	guardWriteLaunchFailReport(&empty, "   \n", true)
	if empty.Len() != 0 {
		t.Errorf("an empty/unrecorded report must be a no-op, got %q", empty.String())
	}
}

// TestGuardDumpStartupReportOnLaunchFailNilServerSafe pins the nil-Server contract: a launch
// failure before the gateway exists (or in a test with no server) must not panic.
func TestGuardDumpStartupReportOnLaunchFailNilServerSafe(t *testing.T) {
	var out strings.Builder
	guardDumpStartupReportOnLaunchFail(&out, nil, true)
	if out.Len() != 0 {
		t.Errorf("nil Server must produce no output, got %q", out.String())
	}
}
