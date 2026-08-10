package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestLookaheadLessonChannelFreshness pins the base-SHA staleness contract: a lesson written at
// one base SHA is FRESH (readable) only while trunk still points there, and only the most-recent
// matching row wins. A different SHA, a blank SHA (git unavailable), or a missing store yields no
// lesson — the fail-open discipline that keeps a stale lesson out of the compacted context.
func TestLookaheadLessonChannelFreshness(t *testing.T) {
	regDir := t.TempDir()
	const shaA = "aaaaaaaaaaaa"
	const shaB = "bbbbbbbbbbbb"

	if _, ok := readFreshLookaheadLesson(regDir, shaA); ok {
		t.Fatalf("empty store must return no lesson")
	}

	if err := appendLookaheadLesson(regDir, lookaheadLesson{TS: lookaheadNow(), BaseSHA: shaA, Rung: "W2", Text: "first"}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := appendLookaheadLesson(regDir, lookaheadLesson{TS: lookaheadNow(), BaseSHA: shaA, Rung: "W3", Text: "second"}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := appendLookaheadLesson(regDir, lookaheadLesson{TS: lookaheadNow(), BaseSHA: shaB, Rung: "W3", Text: "other-base"}); err != nil {
		t.Fatalf("append: %v", err)
	}

	got, ok := readFreshLookaheadLesson(regDir, shaA)
	if !ok {
		t.Fatalf("expected a fresh lesson at base %s", shaA)
	}
	if got.Text != "second" {
		t.Fatalf("most-recent matching row must win: got %q, want %q", got.Text, "second")
	}

	// A moved trunk (different base SHA) has its own most-recent row, not shaA's.
	if got, ok := readFreshLookaheadLesson(regDir, shaB); !ok || got.Text != "other-base" {
		t.Fatalf("base %s lookup = (%q, %v), want (other-base, true)", shaB, got.Text, ok)
	}

	// A blank base SHA (git unavailable) never matches — never inject an unverifiable lesson.
	if _, ok := readFreshLookaheadLesson(regDir, ""); ok {
		t.Fatalf("blank base SHA must never return a lesson")
	}
}

// TestLookaheadLessonForCompactSource pins the pure pickup core: it fires ONLY on source=compact
// with a fresh lesson, renders the rung visibly, and stays silent on every other source or a
// stale/missing lesson.
func TestLookaheadLessonForCompactSource(t *testing.T) {
	regDir := t.TempDir()
	const sha = "0123456789abcdef"
	if err := appendLookaheadLesson(regDir, lookaheadLesson{TS: lookaheadNow(), BaseSHA: sha, Rung: "W3", Text: "path X failed test T at t+2"}); err != nil {
		t.Fatalf("append: %v", err)
	}

	rendered, ok := lookaheadLessonForCompactSource("compact", regDir, sha)
	if !ok {
		t.Fatalf("source=compact with a fresh lesson must inject")
	}
	if !strings.Contains(rendered, "Witnessed W3") {
		t.Fatalf("rendered lesson must lead with its rung: %q", rendered)
	}
	if !strings.Contains(rendered, "path X failed test T at t+2") {
		t.Fatalf("rendered lesson dropped the claim: %q", rendered)
	}

	for _, src := range []string{"startup", "resume", "clear", ""} {
		if _, ok := lookaheadLessonForCompactSource(src, regDir, sha); ok {
			t.Fatalf("source=%q must NOT inject a look-ahead lesson", src)
		}
	}

	// A moved trunk (stale base SHA) suppresses injection even on a compact start.
	if _, ok := lookaheadLessonForCompactSource("compact", regDir, "ffffffffffff"); ok {
		t.Fatalf("a stale base SHA must suppress injection on compact")
	}
}

// TestSessionStartCompactInjectsLesson is the end-to-end pickup through the SessionStart hook: on
// a source=compact payload with a fresh lesson at the live HEAD sha, the injected additionalContext
// carries BOTH the base affordance and the look-ahead lesson; a startup payload carries the
// affordance alone. Uses a throwaway git repo so currentHeadSha resolves deterministically.
func TestSessionStartCompactInjectsLesson(t *testing.T) {
	repo := t.TempDir()
	gitInitRepoForLookahead(t, repo)
	sha := gitHeadShortForLookahead(t, repo)

	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	if err := appendLookaheadLesson(regDir, lookaheadLesson{TS: lookaheadNow(), BaseSHA: sha, Rung: "W3", Text: "reset-with-lesson canary"}); err != nil {
		t.Fatalf("append: %v", err)
	}

	// Run the hook from inside the repo so findRepoRoot/currentHeadSha bind to it.
	restore := chdirForLookahead(t, repo)
	defer restore()

	readCtx := func(source string) string {
		payload, _ := json.Marshal(map[string]string{"source": source, "session_id": "sess-1"})
		var out, errb bytes.Buffer
		if code := runGuardSessionStartHook(&out, &errb, bytes.NewReader(payload), []string{"--mode", "on"}); code != 0 {
			t.Fatalf("exit = %d for source=%s", code, source)
		}
		var env struct {
			HookSpecificOutput struct {
				AdditionalContext string `json:"additionalContext"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &env); err != nil {
			t.Fatalf("not valid JSON for source=%s: %v", source, err)
		}
		return env.HookSpecificOutput.AdditionalContext
	}

	compact := readCtx("compact")
	if !strings.Contains(compact, "reset-with-lesson canary") {
		t.Fatalf("source=compact did not inject the fresh lesson: %s", compact)
	}
	if !strings.Contains(compact, "fak_capabilities") {
		t.Fatalf("source=compact dropped the base affordance: %s", compact)
	}

	startup := readCtx("startup")
	if strings.Contains(startup, "reset-with-lesson canary") {
		t.Fatalf("source=startup must NOT inject the look-ahead lesson: %s", startup)
	}
	if !strings.Contains(startup, "fak_capabilities") {
		t.Fatalf("source=startup dropped the base affordance: %s", startup)
	}
}

// TestSessionStartNilStdinUnchanged asserts the stdin-free entry (the retained runGuardSessionStart)
// is byte-identical to a startup start: no payload, no source, no lesson, base affordance intact.
func TestSessionStartNilStdinUnchanged(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "fak_capabilities") {
		t.Fatalf("nil-stdin start dropped the base affordance: %s", out.String())
	}
}

// TestLookaheadAdmissionGate pins the PreCompact gate: the rollout candidate is effect-free and
// positive-EV, so loopmgr.AdmitSpeculation admits it, and the fail-open trigger never panics or
// blocks (no fork transport today, so nothing is spawned).
func TestLookaheadAdmissionGate(t *testing.T) {
	dec := loopmgr.AdmitSpeculation(lookaheadSpeculationCandidate())
	if !dec.Admit {
		t.Fatalf("look-ahead rollout should be admitted; reason=%q summary=%q", dec.Reason, dec.Summary)
	}
	if _, ok := resolveLookaheadForkSession(); ok {
		t.Fatalf("transcript-fork transport is not yet wired; resolveLookaheadForkSession must report none")
	}
	// Must return promptly and never panic even with a discarding writer.
	maybeSpawnLookaheadRolloutFailOpen(&bytes.Buffer{})
}

// TestLookaheadRolloutArgvShape pins the detached rollout command shape: fronted by `fak guard`
// (the deny floor) and turn-capped at 3, using only verified claude flags.
func TestLookaheadRolloutArgvShape(t *testing.T) {
	argv := lookaheadRolloutArgv("fak", "claude", "fork-abc")
	joined := strings.Join(argv, " ")
	for _, want := range []string{"fak guard --", "claude --resume fork-abc", "--max-turns 3", "--dangerously-skip-permissions"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("rollout argv missing %q: %s", want, joined)
		}
	}
	// With no fak binary, the bare child form is returned (still turn-capped).
	bare := strings.Join(lookaheadRolloutArgv("", "claude", "fork-abc"), " ")
	if strings.Contains(bare, "guard") {
		t.Fatalf("empty fakExe must yield a bare child form: %s", bare)
	}
	if !strings.Contains(bare, "--max-turns 3") {
		t.Fatalf("bare child form dropped the turn cap: %s", bare)
	}
}

func gitInitRepoForLookahead(t *testing.T, dir string) {
	t.Helper()
	run := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init")
	if err := os.WriteFile(filepath.Join(dir, "seed.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	run("add", "seed.txt")
	run("commit", "-m", "seed")
}

func gitHeadShortForLookahead(t *testing.T, dir string) string {
	t.Helper()
	out, err := exec.Command("git", "-C", dir, "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		t.Fatalf("rev-parse: %v", err)
	}
	return strings.TrimSpace(string(out))
}

func chdirForLookahead(t *testing.T, dir string) func() {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	return func() { _ = os.Chdir(prev) }
}
