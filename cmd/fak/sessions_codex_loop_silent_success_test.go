package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A command that succeeds and prints nothing carries no content to tell it apart from any
// other command that succeeds and prints nothing. The outcome digest therefore collides
// across wholly unrelated calls, and the launch gate read that collision as a repetition
// loop: a live `fak guard codex` was refused on shell_command count=4 longest_run=1
// args_digests=4, whose entire "repeated" output was `Exit code: 0 Wall time: 0.3 seconds
// Output:`. Four distinct commands that each worked and said nothing is the opposite of a
// no-progress loop. These tests pin the discrimination — and the arguments are what carry
// it, so a tool re-running the SAME command to silence must stay fused.

func TestCodexOutcomeContentFreeSuccessDiscriminates(t *testing.T) {
	const silent = "Exit code: 0 Wall time: 0.3 seconds Output:"

	distinct := codexRepeatedOutcome{Tool: "shell_command", OutputExcerpt: silent, Count: 4, ArgsDigestCount: 4}
	if !codexOutcomeIsContentFreeSuccess(distinct) || !codexOutcomeIsForwardProgress(distinct) {
		t.Fatalf("distinct silent successes must not be a loop signal: %+v", distinct)
	}

	// Same command re-run to silence: args_digests < count is thrash, still a loop.
	thrash := codexRepeatedOutcome{Tool: "shell_command", OutputExcerpt: silent, Count: 4, ArgsDigestCount: 1}
	if codexOutcomeIsForwardProgress(thrash) {
		t.Fatalf("a re-run of the same silent command must stay a loop signal: %+v", thrash)
	}

	// A repeated FAILURE envelope is the loop this fuse exists for: only exit 0 is exempt.
	failed := codexRepeatedOutcome{
		Tool: "shell_command", Count: 4, ArgsDigestCount: 4,
		OutputExcerpt: "Exit code: 1 Wall time: 0.3 seconds Output:",
	}
	if codexOutcomeIsForwardProgress(failed) {
		t.Fatalf("a repeated non-zero exit must stay a loop signal: %+v", failed)
	}

	// Real output body: judged normally, however short. The regex is anchored so no
	// content-bearing envelope can be mistaken for the empty one.
	content := codexRepeatedOutcome{
		Tool: "shell_command", Count: 4, ArgsDigestCount: 4,
		OutputExcerpt: silent + " same answer every time",
	}
	if codexOutcomeIsForwardProgress(content) {
		t.Fatalf("an identical non-empty output must stay a loop signal: %+v", content)
	}

	// A silent-success class must never mask a concurrent real loop.
	top, ok := codexTopLoopDrivingOutcome([]codexRepeatedOutcome{distinct, thrash})
	if !ok || top.ArgsDigestCount != 1 {
		t.Fatalf("silent-success outcome masked the real loop: top=%+v ok=%v", top, ok)
	}
}

// codexSilentSuccessFixture writes a rollout replaying the refused session's shape: one
// guarded (model_provider=fak) session whose shell_command traffic is n DISTINCT commands
// that each exited 0 and printed nothing, never two in a row.
func codexSilentSuccessFixture(t *testing.T, home string, n int) string {
	t.Helper()
	sessionsDir := filepath.Join(home, "sessions", "2026", "08", "11")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sessionsDir, "rollout-2026-08-11T13-44-21-silent.jsonl")
	lines := []string{
		`{"timestamp":"2026-08-11T20:44:21.000Z","type":"session_meta","payload":{"session_id":"silent-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"fak","git":{"commit_hash":"abc1234","branch":"main"}}}`,
	}
	for i := 0; i < n; i++ {
		call := fmt.Sprintf("sh_%d", i+1)
		lines = append(lines,
			fmt.Sprintf(`{"timestamp":"2026-08-11T20:5%d:00.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"Add-Content -LiteralPath overlay-%d.md -Value step-%d\"}","call_id":"%s"}}`, i, i, i, call),
			// Interleave a content-bearing outcome so no two silent successes are ever
			// adjacent -- exactly the longest_run=1 shape the live refusal reported.
			fmt.Sprintf(`{"timestamp":"2026-08-11T20:5%d:01.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"Exit code: 0\nWall time: 0.3 seconds\nOutput:\n"}}`, i, call),
			fmt.Sprintf(`{"timestamp":"2026-08-11T20:5%d:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"git status --short -- overlay-%d.md\"}","call_id":"%s_probe"}}`, i, i, call),
			fmt.Sprintf(`{"timestamp":"2026-08-11T20:5%d:03.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"%s_probe","output":"Exit code: 0\nWall time: 0.4 seconds\nOutput:\n?? overlay-%d.md"}}`, i, call, i),
		)
	}
	writeCodexLoopFixture(t, path, lines)
	return path
}

func TestDiagnoseCodexLoopClassifiesDistinctSuccessfulWrapperOutputsAsOK(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	path := codexSilentSuccessFixture(t, home, 6)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b = bytes.ReplaceAll(b, []byte(`"output":"Exit code: 0\nWall time: 0.0 seconds\nOutput:"`), []byte(`"output":"Script completed\nWall time: 0.0 seconds\nOutput: {}"`))
	d, err := diagnoseCodexLoop(bytes.NewReader(b), path)
	if err != nil {
		t.Fatal(err)
	}
	if d.Verdict != "OK" || d.Reason != "repeated_content_free_success_no_loop" {
		t.Fatalf("verdict=%s reason=%s, want OK/repeated_content_free_success_no_loop", d.Verdict, d.Reason)
	}
}

func TestSessionsCodexLoopTreatsDistinctSilentSuccessesAsProgress(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	path := codexSilentSuccessFixture(t, home, 5)
	// Witness the guarded launch, so the exemption's own next action is the one under
	// test rather than the direct-provider guidance every unguarded session gets.
	if err := writeCodexGuardWitness(home, "silent-session"); err != nil {
		t.Fatal(err)
	}

	d, err := diagnoseCodexLoopPath(path)
	if err != nil {
		t.Fatal(err)
	}
	bindCodexGuardWitness(&d, home, d.SessionID)
	if d.Verdict != "OK" {
		t.Fatalf("distinct silent successes classified %s (reason=%q), want OK: %+v", d.Verdict, d.Reason, d.RepeatedOutcomes)
	}
	if d.Reason != "repeated_content_free_success_no_loop" {
		t.Fatalf("reason = %q, want repeated_content_free_success_no_loop", d.Reason)
	}
	if !strings.Contains(d.NextAction, "not a stuck operation") {
		t.Fatalf("next action must name WHY no fuse is needed, got %q", d.NextAction)
	}
	// The traffic still has to be visible: this is an exemption from the verdict, not
	// from observability.
	if len(d.RepeatedOutcomes) == 0 {
		t.Fatal("silent-success traffic must stay reported for observability")
	}
}

func TestRunCodexLoopGateAllowsDistinctSilentSuccesses(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := filepath.Join(t.TempDir(), "codex-home")
	codexSilentSuccessFixture(t, home, 5)
	if err := writeCodexGuardWitness(home, "silent-session"); err != nil {
		t.Fatal(err)
	}

	orig := codexLaunchRun
	spawned := false
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		spawned = true
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--", "exec", "keep working",
	})
	if rc != 17 || !spawned {
		t.Fatalf("silent-success traffic blocked the launch: rc=%d spawned=%v stderr=%s", rc, spawned, errb.String())
	}
	if strings.Contains(errb.String(), "loop gate REFUSE") {
		t.Fatalf("distinct silent successes poisoned the guarded relaunch:\n%s", errb.String())
	}
}

// A guard-witnessed session that really did loop must be told what a guarded session can
// act on. The witness used to be bound AFTER classification, so the classifier only ever
// saw GuardWitnessed=false and handed every looping session the direct-provider remedy --
// "launch future Codex sessions through `fak codex`" -- which a session that already
// entered through fak guard has no way to act on and no way to clear.
func TestSessionsCodexLoopGuardedLoopGetsGuardedNextAction(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "08", "11")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sessionsDir, "rollout-2026-08-11T14-00-00-loop.jsonl")
	lines := []string{
		`{"timestamp":"2026-08-11T21:00:00.000Z","type":"session_meta","payload":{"session_id":"guarded-loop","originator":"codex-tui","model_provider":"fak"}}`,
	}
	for i := 1; i <= 3; i++ {
		call := fmt.Sprintf("sh_%d", i)
		lines = append(lines,
			fmt.Sprintf(`{"timestamp":"2026-08-11T21:0%d:00.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"go build ./...\"}","call_id":"%s"}}`, i, call),
			fmt.Sprintf(`{"timestamp":"2026-08-11T21:0%d:01.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"Exit code: 1\nWall time: 2.0 seconds\nOutput:\nundefined: thing"}}`, i, call),
		)
	}
	writeCodexLoopFixture(t, path, lines)
	if err := writeCodexGuardWitness(home, "guarded-loop"); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := sessionsCodexLoop(&stdout, &stderr, []string{"--path", path, "--codex-home", home, "--fail-on", "loop"})
	if rc != 1 {
		t.Fatalf("a real repeated-failure loop must still refuse: rc=%d stdout=%s stderr=%s", rc, stdout.String(), stderr.String())
	}
	got := stdout.String()
	if strings.Contains(got, "launch future Codex sessions through") {
		t.Fatalf("a guard-witnessed session was handed the direct-provider remedy it already followed:\n%s", got)
	}
	if !strings.Contains(got, "stop re-calling the same tool") {
		t.Fatalf("guarded loop must get the guarded next action:\n%s", got)
	}
}
