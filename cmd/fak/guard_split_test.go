package main

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// envFunc builds a getenv closure over a fixed map for the pure plan/decision tests.
func envFunc(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func lookPathOK(string) (string, error)   { return "wt", nil }
func lookPathFail(string) (string, error) { return "", errors.New("not found") }

func guardOverlayArgs() []string {
	return []string{"info", "--gateway-url", "http://127.0.0.1:5000", "--interval", "2s"}
}

func TestBuildGuardSplitPlanTmuxBottom(t *testing.T) {
	plan, err := buildGuardSplitPlan("linux", envFunc(map[string]string{"TMUX": "/tmp/tmux-1/default,1,0"}), lookPathFail, "fak", "bottom", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Host != "tmux" {
		t.Fatalf("host = %q, want tmux", plan.Host)
	}
	// -d keeps the agent pane active (the overlay never steals focus).
	want := []string{"tmux", "split-window", "-v", "-d", "-l", "20%", "--", "fak", "info", "--gateway-url", "http://127.0.0.1:5000", "--interval", "2s"}
	if strings.Join(plan.Spawn, " ") != strings.Join(want, " ") {
		t.Fatalf("spawn = %v\nwant   %v", plan.Spawn, want)
	}
	if got := strings.Join(plan.Overlay, " "); got != "fak info --gateway-url http://127.0.0.1:5000 --interval 2s" {
		t.Fatalf("overlay = %q", got)
	}
}

func TestBuildGuardSplitPlanTmuxRightIsHorizontalSplit(t *testing.T) {
	plan, err := buildGuardSplitPlan("darwin", envFunc(map[string]string{"TMUX": "x"}), lookPathFail, "fak", "right", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Spawn[2] != "-h" {
		t.Fatalf("right column should use tmux -h, got %v", plan.Spawn)
	}
	// The overlay must not steal focus from the agent pane regardless of orientation.
	if plan.Spawn[3] != "-d" {
		t.Fatalf("tmux split must pass -d to keep the agent pane focused, got %v", plan.Spawn)
	}
}

func TestBuildGuardSplitPlanWindowsTerminalCurrentWindow(t *testing.T) {
	plan, err := buildGuardSplitPlan("windows", envFunc(map[string]string{"WT_SESSION": "abc-123"}), lookPathOK, "fak.exe", "bottom", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Host != "wt" {
		t.Fatalf("host = %q, want wt", plan.Host)
	}
	// `; move-focus up` returns the cursor to the agent pane (above the bottom strip) after
	// split-pane focuses the new overlay pane.
	want := []string{"wt", "-w", "0", "split-pane", "-H", "-s", "0.2", "fak.exe", "info", "--gateway-url", "http://127.0.0.1:5000", "--interval", "2s", ";", "move-focus", "up"}
	if strings.Join(plan.Spawn, " ") != strings.Join(want, " ") {
		t.Fatalf("spawn = %v\nwant   %v", plan.Spawn, want)
	}
}

func TestBuildGuardSplitPlanWindowsTerminalRightColumn(t *testing.T) {
	plan, err := buildGuardSplitPlan("windows", envFunc(map[string]string{"WT_SESSION": "x"}), lookPathOK, "fak.exe", "right", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Spawn[4] != "-V" {
		t.Fatalf("right column should use wt -V, got %v", plan.Spawn)
	}
	// A right-column overlay sits to the RIGHT of the agent, so focus returns LEFT.
	if got := strings.Join(plan.Spawn, " "); !strings.HasSuffix(got, "; move-focus left") {
		t.Fatalf("right column should return focus with `; move-focus left`, got %v", plan.Spawn)
	}
}

func TestBuildGuardSplitPlanWindowsNoWTSessionFallsThrough(t *testing.T) {
	// On Windows WITHOUT $WT_SESSION (e.g. a bare conhost), there is no current WT window to
	// split — must NOT spawn, even if `wt` is on PATH (a new window would orphan the gateway).
	plan, err := buildGuardSplitPlan("windows", envFunc(nil), lookPathOK, "fak.exe", "bottom", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Host != "none" {
		t.Fatalf("host = %q, want none", plan.Host)
	}
	if plan.Spawn != nil {
		t.Fatalf("expected no spawn, got %v", plan.Spawn)
	}
}

func TestBuildGuardSplitPlanNoMultiplexerFallback(t *testing.T) {
	plan, err := buildGuardSplitPlan("linux", envFunc(nil), lookPathFail, "fak", "bottom", guardOverlayArgs())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Host != "none" {
		t.Fatalf("host = %q, want none", plan.Host)
	}
	if !strings.Contains(plan.Fallback, "fak info --gateway-url http://127.0.0.1:5000") {
		t.Fatalf("fallback should print the exact overlay command, got:\n%s", plan.Fallback)
	}
}

func TestBuildGuardSplitPlanInvalidWhere(t *testing.T) {
	if _, err := buildGuardSplitPlan("linux", envFunc(map[string]string{"TMUX": "x"}), lookPathFail, "fak", "diagonal", guardOverlayArgs()); err == nil {
		t.Fatal("expected an error for an invalid --split-where value")
	}
}

// TestGuardInfoPaneOverlayArgsSingleSource proves the pane's `fak info` argv carries the live
// gateway URL and interval and nothing else — the single shape both the dry-run preview and
// the live spawn read, so the two can never drift.
func TestGuardInfoPaneOverlayArgsSingleSource(t *testing.T) {
	got := guardInfoPaneOverlayArgs("http://127.0.0.1:8080", 3*time.Second)
	want := []string{"info", "--gateway-url", "http://127.0.0.1:8080", "--interval", "3s"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("overlay args = %v, want %v", got, want)
	}
}

// TestRenderGuardInfoPaneDryRun proves --split-dry-run renders the resolved plan (multiplexer,
// geometry, and the exact pane command) with exit 0, and returns a non-zero code with a message
// on a bad --split-where — the preview surface an operator reads before launching the split. It
// pins the detection seams so the test does not depend on the host's real multiplexer.
func TestRenderGuardInfoPaneDryRun(t *testing.T) {
	savedGOOS, savedLook := guardSplitGOOS, guardSplitLookPath
	t.Cleanup(func() { guardSplitGOOS, guardSplitLookPath = savedGOOS, savedLook })
	guardSplitGOOS, guardSplitLookPath = "windows", lookPathOK

	out, code := renderGuardInfoPaneDryRun(envFunc(map[string]string{"WT_SESSION": "x"}), "right", "http://127.0.0.1:9", 2*time.Second)
	if code != 0 {
		t.Fatalf("dry-run code = %d, want 0; out=%s", code, out)
	}
	for _, want := range []string{"host: wt", "right column", "split-pane", "-V", "--gateway-url http://127.0.0.1:9"} {
		if !strings.Contains(out, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, out)
		}
	}

	badOut, badCode := renderGuardInfoPaneDryRun(envFunc(nil), "sideways", "http://127.0.0.1:9", 2*time.Second)
	if badCode == 0 || !strings.Contains(badOut, "split-where") {
		t.Fatalf("bad --split-where: code=%d out=%q", badCode, badOut)
	}
}

func TestGuardSplitEnabled(t *testing.T) {
	cases := []struct {
		name             string
		mode             string
		env              map[string]string
		stdinInteractive bool
		childInteractive bool
		want             bool
		wantErr          bool
	}{
		{"auto in WT enables", "auto", map[string]string{"WT_SESSION": "x"}, true, true, true, false},
		{"auto in tmux enables", "auto", map[string]string{"TMUX": "y"}, true, true, true, false},
		{"auto with no multiplexer no-ops", "auto", nil, true, true, false, false},
		{"auto nested never re-splits", "auto", map[string]string{"WT_SESSION": "x", "FAK_GUARD_SPLIT": "1"}, true, true, false, false},
		{"auto non-interactive stdin no-ops", "auto", map[string]string{"WT_SESSION": "x"}, false, true, false, false},
		{"auto headless child no-ops", "auto", map[string]string{"WT_SESSION": "x"}, true, false, false, false},
		{"empty defaults to auto", "", map[string]string{"TMUX": "y"}, true, true, true, false},
		{"on forces enable even bare", "on", nil, false, false, true, false},
		{"off disables even in WT", "off", map[string]string{"WT_SESSION": "x"}, true, true, false, false},
		{"bogus errors", "sideways", nil, true, true, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := guardSplitEnabled(tc.mode, envFunc(tc.env), tc.stdinInteractive, tc.childInteractive)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGuardChildInteractive(t *testing.T) {
	cases := []struct {
		command []string
		want    bool
	}{
		{[]string{"claude", "--settings", "{}"}, true},
		{[]string{"claude"}, true},
		{[]string{"claude", "-p", "do a thing"}, false},
		{[]string{"claude", "--print"}, false},
		{[]string{"claude", "--print=json"}, false},
		{[]string{"codex", "exec"}, true},
	}
	for _, tc := range cases {
		if got := guardChildInteractive(tc.command); got != tc.want {
			t.Fatalf("guardChildInteractive(%v) = %v, want %v", tc.command, got, tc.want)
		}
	}
}
