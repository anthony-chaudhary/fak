package main

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// guard_auth_crash_test.go — the mid-session counterpart to guard_rehydrate_test.go's #1834
// witness. That file proves a headless launch self-heals a stale credential BEFORE the child
// ever spawns; this one proves the OLD absence on the other side of the crash: before
// guardMaybeRecoverAuthCrash existed, a wrapped agent that exited abnormally because its pinned
// subscription token expired mid-session had NO automatic path back — cmdGuard printed
// formatGuardResumeGuidance and left the operator to manually re-run `fak guard -- claude
// --continue`. These tests exercise guardContinueFlagForAgent / guardClassifyAuthCrash /
// guardMaybeRecoverAuthCrash directly against that OLD absence: before the wiring, none of these
// functions existed to call at all, so a crash correlated with an expired-but-recoverable
// credential could never auto-relaunch — it fails against a no-op and passes against the real
// decision.

// execCommandExit returns a *exec.Cmd that, when Run, exits with exactly `code` and no other
// side effects — a portable way to manufacture a real *exec.ExitError (which has no public
// constructor) without depending on any fak binary or repo state.
func execCommandExit(code int) *exec.Cmd {
	if runtime.GOOS == "windows" {
		return exec.Command("cmd", "/c", "exit", strconv.Itoa(code))
	}
	return exec.Command("sh", "-c", fmt.Sprintf("exit %d", code))
}

func TestGuardContinueFlagForAgent(t *testing.T) {
	cases := []struct {
		name      string
		agentName string
		wantFlag  string
		wantOK    bool
	}{
		{"bare claude", "claude", "--continue", true},
		{"absolute path", "/usr/local/bin/claude", "--continue", true},
		{"windows exe suffix", "claude.exe", "--continue", true},
		{"windows cmd suffix", "claude.cmd", "--continue", true},
		{"case insensitive", "CLAUDE", "--continue", true},
		{"unrecognized agent", "codex", "", false},
		{"empty", "", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			flag, ok := guardContinueFlagForAgent(c.agentName)
			if flag != c.wantFlag || ok != c.wantOK {
				t.Fatalf("guardContinueFlagForAgent(%q) = (%q, %v), want (%q, %v)", c.agentName, flag, ok, c.wantFlag, c.wantOK)
			}
		})
	}
}

// TestGuardContinueFlagForAgent_UnknownNeverGuesses is the specific safety witness: fak must
// NEVER auto-relaunch a wrapped agent it does not recognize, because guessing its continuation
// syntax wrong would silently drop the conversation instead of resuming it — worse than today's
// manual guidance, which at least tells the operator to look it up themselves.
func TestGuardContinueFlagForAgent_UnknownNeverGuesses(t *testing.T) {
	for _, name := range []string{"codex", "aider", "cursor-agent", "gemini"} {
		if _, ok := guardContinueFlagForAgent(name); ok {
			t.Fatalf("guardContinueFlagForAgent(%q) = ok=true; an unrecognized wrapped agent must never get a guessed continuation flag", name)
		}
	}
}

func TestGuardAppendContinueFlag(t *testing.T) {
	t.Run("appends_when_absent", func(t *testing.T) {
		command := []string{"claude", "-p"}
		got := guardAppendContinueFlag(command, "--continue")
		want := []string{"claude", "-p", "--continue"}
		if len(got) != len(want) {
			t.Fatalf("got %v, want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("got %v, want %v", got, want)
			}
		}
		if len(command) != 2 {
			t.Fatalf("input command mutated in place: %v", command)
		}
	})

	t.Run("idempotent_when_already_present", func(t *testing.T) {
		command := []string{"claude", "--continue"}
		got := guardAppendContinueFlag(command, "--continue")
		if len(got) != 2 {
			t.Fatalf("a repeated auth-crash-and-recover cycle stacked the flag: got %v", got)
		}
	})
}

func TestGuardClassifyAuthCrash(t *testing.T) {
	alwaysFresh := func(context.Context) (bool, bool) { return true, false }
	staleAndRecovered := func(context.Context) (bool, bool) { return false, true }
	staleAndNeverRecovered := func(context.Context) (bool, bool) { return false, false }

	cases := []struct {
		name          string
		hasCredential bool
		check         func(context.Context) (bool, bool)
		wantCorr      bool
		wantRecovered bool
	}{
		{"no_credential_never_correlates_even_if_check_would_say_stale", false, staleAndNeverRecovered, false, false},
		{"live_credential_is_not_an_auth_crash", true, alwaysFresh, false, false},
		{"stale_and_recovered_is_a_correlated_recovered_crash", true, staleAndRecovered, true, true},
		{"stale_and_unrecovered_is_a_correlated_unrecovered_crash", true, staleAndNeverRecovered, true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			corr, recovered := guardClassifyAuthCrash(context.Background(), c.hasCredential, c.check)
			if corr != c.wantCorr || recovered != c.wantRecovered {
				t.Fatalf("guardClassifyAuthCrash(hasCredential=%v) = (%v, %v), want (%v, %v)", c.hasCredential, corr, recovered, c.wantCorr, c.wantRecovered)
			}
		})
	}
}

// TestGuardMaybeRecoverAuthCrash is the end-to-end #GOAL witness: cross-session auth resume, at
// the level cmdGuard actually calls it. Before this wiring existed, EVERY one of these cases
// (including the recoverable one) fell through to the exact same generic
// formatGuardResumeGuidance message and a bare os.Exit(code) — a live guarded session that hit
// an expired subscription token mid-run always needed a human to notice and manually re-run with
// --continue. These assertions are the fail-before/pass-after boundary: they fail against a
// no-op wiring (relaunch never happens, even when the credential visibly recovers) and pass
// against the real decision.
func TestGuardMaybeRecoverAuthCrash(t *testing.T) {
	command := []string{"claude", "-p", "hello"}

	t.Run("clean_exit_never_recovers", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())

		next, ok := guardMaybeRecoverAuthCrash(nil, command, credPath, "claude", true, nil)
		if ok || next != nil {
			t.Fatalf("a nil (clean) exit must never trigger a relaunch; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("unrecognized_agent_never_auto_relaunches", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "50ms")

		runErr := execCommandExit(1).Run()
		next, ok := guardMaybeRecoverAuthCrash(runErr, []string{"codex", "exec"}, credPath, "codex", true, nil)
		if ok || next != nil {
			t.Fatalf("an unrecognized wrapped agent must never auto-relaunch even with a correlated stale credential; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("no_credential_file_is_not_diagnosed_as_auth_crash", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json") // never written
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "50ms")

		runErr := execCommandExit(1).Run()
		next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, "claude", true, nil)
		if ok || next != nil {
			t.Fatalf("a crash with no credential file to correlate against must not be guessed as an auth crash; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("live_credential_at_crash_time_is_not_auth_related", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-live", time.Now().Add(time.Hour).UnixMilli())

		runErr := execCommandExit(1).Run()
		next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, "claude", true, nil)
		if ok || next != nil {
			t.Fatalf("a live credential at crash time means the crash was NOT auth-caused; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("expired_credential_that_recovers_relaunches_with_continue", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-expired", time.Now().Add(-time.Hour).UnixMilli())
		// Give the poll a real window, then rotate the credential in shortly after the call
		// starts — guardCredCheckWithWindow polls every 150ms, so 400ms gives it a couple of
		// chances to observe the rotation before the (shrunk) deadline.
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "2s")
		go func() {
			time.Sleep(300 * time.Millisecond)
			writeCred(t, credPath, "sk-ant-oat01-rotated", time.Now().Add(time.Hour).UnixMilli())
		}()

		runErr := execCommandExit(1).Run()
		next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, "claude", true, nil)
		if !ok {
			t.Fatal("a credential that recovers within the window must trigger an auto-relaunch")
		}
		want := append(append([]string{}, command...), "--continue")
		if len(next) != len(want) {
			t.Fatalf("relaunch command = %v, want %v", next, want)
		}
		for i := range want {
			if next[i] != want[i] {
				t.Fatalf("relaunch command = %v, want %v", next, want)
			}
		}
	})

	t.Run("expired_credential_that_never_recovers_falls_back_to_manual", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "50ms") // never rewritten within this window

		runErr := execCommandExit(1).Run()
		next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, "claude", true, nil)
		if ok || next != nil {
			t.Fatalf("a credential that never recovers must fall back to the manual formatGuardResumeGuidance path, not auto-relaunch; got next=%v ok=%v", next, ok)
		}
	})
}

func TestGuardAuthCrashRecoverWindowDuration(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		if got := guardAuthCrashRecoverWindowDuration(); got != guardAuthCrashRecoverWindow {
			t.Fatalf("default window = %s, want %s", got, guardAuthCrashRecoverWindow)
		}
	})

	t.Run("env_override_clamped_to_ceiling", func(t *testing.T) {
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "1h")
		if got := guardAuthCrashRecoverWindowDuration(); got != maxGuardAuthCrashRecoverWindow {
			t.Fatalf("a value above the ceiling must clamp to it: got %s, want %s", got, maxGuardAuthCrashRecoverWindow)
		}
	})

	t.Run("env_override_within_range_is_honored", func(t *testing.T) {
		t.Setenv("FAK_GUARD_AUTH_RECOVER_WINDOW", "90s")
		if got := guardAuthCrashRecoverWindowDuration(); got != 90*time.Second {
			t.Fatalf("got %s, want 90s", got)
		}
	})
}
