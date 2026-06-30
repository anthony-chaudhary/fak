package dispatchpost

import (
	"strings"
	"testing"
)

func TestResolveToken_EnvWins(t *testing.T) {
	t.Setenv("FAK_DISPATCH_TOKEN", "xoxb-dispatch")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard")
	if got := ResolveToken(); got != "xoxb-dispatch" {
		t.Fatalf("dispatch token should win: got %q", got)
	}
}

func TestResolveToken_FallsBackToScoreboard(t *testing.T) {
	// No dispatch token, no .env.slack.local in a temp cwd: fall back to scoreboard.
	t.Setenv("FAK_DISPATCH_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard")
	dir := t.TempDir()
	chdir(t, dir)
	if got := ResolveToken(); got != "xoxb-scoreboard" {
		t.Fatalf("expected scoreboard fallback, got %q", got)
	}
}

func TestResolveChannel_EnvThenFile(t *testing.T) {
	t.Setenv("FAK_DISPATCH_CHANNEL", "C_ENV")
	if got := ResolveChannel(); got != "C_ENV" {
		t.Fatalf("env channel: got %q", got)
	}
	t.Setenv("FAK_DISPATCH_CHANNEL", "")
	dir := t.TempDir()
	writeFile(t, dir, ".env.slack.local", "export FAK_DISPATCH_CHANNEL=C_FILE\n")
	chdir(t, dir)
	if got := ResolveChannel(); got != "C_FILE" {
		t.Fatalf("file channel: got %q", got)
	}
}

func TestResolveChannel_UnsetIsEmpty(t *testing.T) {
	t.Setenv("FAK_DISPATCH_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != "" {
		t.Fatalf("unset channel must be empty (so the caller skips the post), got %q", got)
	}
}

func TestResult_Shipped_RendersCommits(t *testing.T) {
	r := Result{
		LoopID:     "nightly-fix",
		RunID:      "run-001",
		ExitCode:   0,
		DurationMS: 125000,
		Command:    "fak-agent",
		HeadBefore: "aaaa111",
		HeadAfter:  "bbbb222",
		Commits:    []string{"bbbb222 fix(gateway): treat same-tick ready as positive (fak gateway)"},
		Source:     "cron",
	}
	if !r.Shipped() {
		t.Fatal("a run with commits must be Shipped()")
	}
	txt := r.Text()
	for _, want := range []string{":white_check_mark:", "shipped", "exit 0", "HEAD aaaa111→bbbb222", "fix(gateway)", "2m5s", "posted by cron"} {
		if !strings.Contains(txt, want) {
			t.Errorf("shipped card missing %q in:\n%s", want, txt)
		}
	}
	if strings.Contains(txt, "S/N self-score") {
		t.Errorf("dispatch result should not include noisy S/N footer:\n%s", txt)
	}
}

func TestResult_GreenButNoCommit_IsHonest(t *testing.T) {
	// The load-bearing case: exit 0 but no git delta. The card must NOT claim shipped.
	r := Result{
		LoopID:     "nightly-fix",
		ExitCode:   0,
		DurationMS: 3000,
		HeadBefore: "aaaa111",
		HeadAfter:  "aaaa111",
	}
	if r.Shipped() {
		t.Fatal("exit 0 with no commit delta must NOT be Shipped()")
	}
	txt := r.Text()
	if strings.Contains(txt, ":white_check_mark:") {
		t.Errorf("no-commit run must not use the shipped check glyph:\n%s", txt)
	}
	for _, want := range []string{":large_green_circle:", "passed; no code shipped", "exit 0", "passed: command exited 0; this was a check/test result, not shipped work", "HEAD aaaa111 unchanged"} {
		if !strings.Contains(txt, want) {
			t.Errorf("no-commit card missing %q in:\n%s", want, txt)
		}
	}
	if strings.Contains(txt, "S/N self-score") {
		t.Errorf("no-commit card should not include noisy S/N footer:\n%s", txt)
	}
}

func TestResult_NonZeroExit_IsRedRegardlessOfCommits(t *testing.T) {
	r := Result{
		LoopID:   "nightly-fix",
		ExitCode: 2,
		Commits:  []string{"cccc333 wip"}, // even with a commit, a failed run is red
	}
	txt := r.Text()
	if !strings.Contains(txt, ":red_circle:") {
		t.Errorf("non-zero exit must be red:\n%s", txt)
	}
	if !strings.Contains(txt, "FAILED (exit 2)") {
		t.Errorf("failed card must name the exit code:\n%s", txt)
	}
}

func TestResult_FailedNoCommit_IsUnambiguous(t *testing.T) {
	r := Result{
		LoopID:     "scheduler/test",
		RunID:      "run-fail",
		ExitCode:   7,
		DurationMS: 5,
		Command:    "fak.test",
		HeadBefore: "e8fcc66e",
		HeadAfter:  "e8fcc66e",
	}
	txt := r.Text()
	for _, want := range []string{":red_circle:", "FAILED (exit 7)", "exit 7", "HEAD e8fcc66e unchanged", "FAILED: exit 7; no code shipped"} {
		if !strings.Contains(txt, want) {
			t.Errorf("failed no-commit card missing %q in:\n%s", want, txt)
		}
	}
	if strings.Contains(txt, "S/N self-score") {
		t.Errorf("failed card should not include noisy S/N footer:\n%s", txt)
	}
}

func TestResult_Blocks_CarrySameFacts(t *testing.T) {
	r := Result{LoopID: "L", ExitCode: 0, Commits: []string{"abc shipped it"}}
	blocks := r.Blocks()
	if len(blocks) == 0 {
		t.Fatal("blocks must be non-empty")
	}
	// Flatten the block text and assert the commit appears.
	flat := flattenBlocks(blocks)
	if !strings.Contains(flat, "shipped it") {
		t.Errorf("blocks missing commit subject; flat=%s", flat)
	}
	if strings.Contains(flat, "S/N self-score") {
		t.Errorf("dispatch blocks should not include noisy S/N footer; flat=%s", flat)
	}
}

func TestHumaniseDuration(t *testing.T) {
	cases := map[int64]string{
		320:     "320ms",
		5000:    "5s",
		125000:  "2m5s",
		7384000: "2h3m",
	}
	for ms, want := range cases {
		got := humaniseDuration(durationFromMS(ms))
		if got != want {
			t.Errorf("humaniseDuration(%dms) = %q, want %q", ms, got, want)
		}
	}
}
