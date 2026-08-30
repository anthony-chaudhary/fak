package main

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// formatGuardResumeGuidance is printed when the wrapped agent exits abnormally. It must name
// the agent and exit code, point at the same-command resume, surface the replayable decision
// journal, and carry the bare-resume recovery for the "upstream model error" failure mode —
// so an operator whose `fak guard -- claude` session crashed knows exactly how to get back in.
func TestFormatGuardResumeGuidance(t *testing.T) {
	out := formatGuardResumeGuidance("claude", 1)
	for _, want := range []string{
		"claude",               // the wrapped agent is named
		"code 1",               // the abnormal exit code is surfaced
		"fak guard --",         // the resume re-run command
		"--continue",           // the agent's own resume/continue flag
		"fak audit verify",     // the journal is replayable
		"WITHOUT fak guard",    // the bare-resume recovery
		"upstream model error", // the specific failure it recovers
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %q\n--- guidance ---\n%s", want, out)
		}
	}
}

func TestFormatGuardResumeGuidanceSurfacesGuardActivity(t *testing.T) {
	out := formatGuardResumeGuidanceWithRefusals("claude", 1, []guardRefusalCarry{{
		Reason: "OFF_TRUNK",
		Count:  2,
		Fix:    "commit directly to main",
	}})
	for _, want := range []string{
		"guard activity",
		"recovery/debugging",
		"OFF_TRUNK x2",
		"commit directly to main",
		"do not retry the same refused call unchanged",
		"WITHOUT fak guard",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %q\n--- guidance ---\n%s", want, out)
		}
	}
}

func TestFormatGuardSessionResumeCommandUsesExactClaudeSession(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	const sessionID = "11111111-1111-4111-8111-111111111111"
	if err := resume.AppendIdentityRow(regDir, resume.IdentityRow{
		UUID: sessionID, Trace: "trace-claude", Provider: "claude",
	}); err != nil {
		t.Fatal(err)
	}

	got := formatGuardSessionResumeCommand("/usr/local/bin/claude", "trace-claude")
	want := "\nfak guard: resume this session with:\n  fak guard -- claude --resume " + sessionID + "\n"
	if got != want {
		t.Fatalf("resume output = %q, want %q", got, want)
	}
	for _, forbidden := range []string{"<session", "…", "--continue"} {
		if strings.Contains(got, forbidden) {
			t.Fatalf("resume output contains non-exact fallback %q: %q", forbidden, got)
		}
	}
}

func TestFormatGuardSessionResumeCommandUsesExactCodexThread(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	const sessionID = "22222222-2222-4222-8222-222222222222"
	if err := resume.AppendIdentityRow(regDir, resume.IdentityRow{
		UUID: sessionID, Trace: "trace-codex", Provider: "codex",
	}); err != nil {
		t.Fatal(err)
	}

	got := formatGuardSessionResumeCommand("C:\\tools\\codex.exe", "trace-codex")
	want := "\nfak guard: resume this session with:\n  fak guard -- codex resume " + sessionID + "\n"
	if got != want {
		t.Fatalf("resume output = %q, want %q", got, want)
	}
}

func TestFormatGuardSessionResumeCommandNeverGuesses(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	rows := []resume.IdentityRow{
		{UUID: "33333333-3333-4333-8333-333333333333", Trace: "other-trace", Provider: "claude"},
		{UUID: "not-a-provider-session-id", Trace: "bad-id", Provider: "claude"},
		{UUID: "44444444-4444-4444-8444-444444444444", Trace: "wrong-provider", Provider: "codex"},
	}
	for _, row := range rows {
		if err := resume.AppendIdentityRow(regDir, row); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct{ agent, trace string }{
		{"claude", "missing"},
		{"claude", "bad-id"},
		{"claude", "wrong-provider"},
		{"opencode", "other-trace"},
	} {
		if got := formatGuardSessionResumeCommand(tc.agent, tc.trace); got != "" {
			t.Errorf("formatGuardSessionResumeCommand(%q, %q) = %q, want empty", tc.agent, tc.trace, got)
		}
	}
}

// This pins the production funnel, not only the pure formatter: every terminal
// branch must emit after the exact current provider trace has been resolved, and
// the quiet gate must remain inside the emission closure so hook stdout is untouched.
func TestGuardTerminalFunnelEmitsExactResumeCommand(t *testing.T) {
	raw, err := os.ReadFile("guard_child_supervision.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	resolve := strings.Index(text, "resumeCommand := formatGuardSessionResumeCommand(agentName, srv.DefaultTraceID())")
	quiet := strings.Index(text, "emitResumeCommand := func() {\n\t\tif !quiet {")
	if resolve < 0 || quiet < resolve {
		t.Fatal("terminal funnel no longer resolves the exact current provider session behind the quiet stderr gate")
	}
	terminal := text[resolve:]
	if got := strings.Count(terminal, "emitResumeCommand()"); got != 4 {
		t.Fatalf("terminal funnel has %d resume emissions, want clean exit plus all three error exits", got)
	}
	if firstExit := strings.Index(terminal, "os.Exit("); firstExit < 0 || strings.Index(terminal[:firstExit], "emitResumeCommand()") < 0 {
		t.Fatal("the first terminal error branch exits before printing the exact resume command")
	}
}
