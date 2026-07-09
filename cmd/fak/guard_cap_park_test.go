package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// guard_cap_park_test.go — the cap-crash sibling of guard_auth_crash_test.go. Where that file
// proves an EXPIRED-token crash self-heals, this proves the OTHER terminal-upstream class named
// on #2256: a wrapped agent that exits because the account hit a usage/session/weekly CAP whose
// reset is known. Before guardClassifyCapCrash existed the guard had no inline park-then-resume
// for this — the crash fell through to formatGuardResumeGuidance and the external `fak resume`
// sweep was the only path back. These exercise the pure decision core directly; the transcript
// witness that feeds it (resume.Diagnose over the child's .jsonl) is the #2256 wiring rung.

// capDiag builds a rate-limit crash Diagnosis carrying the given closed cap token, the shape
// resume.Diagnose returns for a session that died on a cap and never resumed.
func capDiag(reason string) resume.Diagnosis {
	return resume.Diagnosis{Crash: resume.CrashRateLimit, LimitReason: reason, Unresumed: true, NeedsRestart: true}
}

func TestGuardCapParkResetWindow(t *testing.T) {
	cases := []struct {
		reason string
		want   time.Duration
	}{
		{resume.LimitSession, time.Duration(resume.SessionLimitResetSeconds) * time.Second},
		{resume.LimitUsage, time.Duration(resume.SessionLimitResetSeconds) * time.Second}, // usage takes the session floor
		{resume.LimitWeekly, time.Duration(resume.WeeklyLimitResetSeconds) * time.Second},
		{resume.LimitRate, 0}, // a burst has no wall-clock reset to park on
		{"", 0},
		{"nonsense", 0},
	}
	for _, tc := range cases {
		if got := guardCapParkResetWindow(tc.reason); got != tc.want {
			t.Errorf("guardCapParkResetWindow(%q) = %s, want %s", tc.reason, got, tc.want)
		}
	}
}

func TestGuardClassifyCapCrash(t *testing.T) {
	cmd := []string{"claude", "-p", "do the thing"}

	t.Run("session cap with unknown idle parks the full window and relaunches --continue", func(t *testing.T) {
		dec := guardClassifyCapCrash(capDiag(resume.LimitSession), -1, cmd, "claude")
		if !dec.Park {
			t.Fatalf("Park = false, want true; reason=%q", dec.Reason)
		}
		if dec.LimitReason != resume.LimitSession {
			t.Errorf("LimitReason = %q, want %q", dec.LimitReason, resume.LimitSession)
		}
		if want := time.Duration(resume.SessionLimitResetSeconds) * time.Second; dec.Wait != want {
			t.Errorf("Wait = %s, want the full window %s (idle unknown must not shorten it)", dec.Wait, want)
		}
		if got := strings.Join(dec.Relaunch, " "); got != "claude -p do the thing --continue" {
			t.Errorf("Relaunch = %q, want the command with --continue appended", got)
		}
	})

	t.Run("weekly cap uses the 7d window", func(t *testing.T) {
		dec := guardClassifyCapCrash(capDiag(resume.LimitWeekly), -1, cmd, "claude")
		if !dec.Park || dec.Wait != time.Duration(resume.WeeklyLimitResetSeconds)*time.Second {
			t.Fatalf("weekly park = (park=%v, wait=%s), want park with the 7d window", dec.Park, dec.Wait)
		}
	})

	t.Run("partial idle shortens the remaining wait", func(t *testing.T) {
		window := resume.SessionLimitResetSeconds
		idle := window / 4
		dec := guardClassifyCapCrash(capDiag(resume.LimitSession), idle, cmd, "claude")
		if !dec.Park {
			t.Fatalf("Park = false, want true")
		}
		if want := time.Duration(window-idle) * time.Second; dec.Wait != want {
			t.Errorf("Wait = %s, want window-idle = %s", dec.Wait, want)
		}
	})

	t.Run("reset already elapsed relaunches now with no park", func(t *testing.T) {
		dec := guardClassifyCapCrash(capDiag(resume.LimitSession), resume.SessionLimitResetSeconds+1, cmd, "claude")
		if dec.Park {
			t.Errorf("Park = true, want false (the window already passed)")
		}
		if dec.Wait != 0 {
			t.Errorf("Wait = %s, want 0", dec.Wait)
		}
		if got := strings.Join(dec.Relaunch, " "); got != "claude -p do the thing --continue" {
			t.Errorf("Relaunch = %q, want the command relaunch-ready even with no wait", got)
		}
	})

	t.Run("a burst (rate_limited) is not a park case", func(t *testing.T) {
		dec := guardClassifyCapCrash(capDiag(resume.LimitRate), -1, cmd, "claude")
		if dec.Park || dec.Relaunch != nil {
			t.Errorf("burst = (park=%v, relaunch=%v), want no park / no relaunch — admission owns bursts", dec.Park, dec.Relaunch)
		}
	})

	t.Run("a non-rate-limit crash is not our case", func(t *testing.T) {
		other := resume.Diagnosis{Crash: resume.CrashOther}
		if dec := guardClassifyCapCrash(other, -1, cmd, "claude"); dec.Park || dec.Relaunch != nil {
			t.Errorf("CrashOther = (park=%v, relaunch=%v), want no park", dec.Park, dec.Relaunch)
		}
		clean := resume.Diagnosis{Crash: resume.CrashNone}
		if dec := guardClassifyCapCrash(clean, -1, cmd, "claude"); dec.Park {
			t.Errorf("CrashNone parked, want no park")
		}
	})

	t.Run("an unrecognized agent never auto-relaunches", func(t *testing.T) {
		dec := guardClassifyCapCrash(capDiag(resume.LimitSession), -1, []string{"codex", "exec"}, "codex")
		if dec.Park || dec.Relaunch != nil {
			t.Errorf("codex cap = (park=%v, relaunch=%v), want no park / no relaunch (no known safe resume flag)", dec.Park, dec.Relaunch)
		}
		if dec.LimitReason != resume.LimitSession {
			t.Errorf("LimitReason = %q, want the cap still named for the report", dec.LimitReason)
		}
	})

	t.Run("--continue is never stacked twice", func(t *testing.T) {
		already := []string{"claude", "-p", "x", "--continue"}
		dec := guardClassifyCapCrash(capDiag(resume.LimitSession), -1, already, "claude")
		if n := strings.Count(strings.Join(dec.Relaunch, " "), "--continue"); n != 1 {
			t.Errorf("--continue appears %d times in %q, want exactly 1", n, dec.Relaunch)
		}
	})
}

func TestGuardCapParkWait(t *testing.T) {
	dec := guardClassifyCapCrash(capDiag(resume.LimitSession), -1, []string{"claude", "-p", "x"}, "claude")

	t.Run("waits the decision window with an injected clock, prints park + outcome", func(t *testing.T) {
		var slept time.Duration
		base := time.Unix(1_700_000_000, 0)
		calls := 0
		now := func() time.Time {
			// first call = start, second = after sleep (advanced by slept)
			calls++
			if calls == 1 {
				return base
			}
			return base.Add(slept)
		}
		sleep := func(d time.Duration) { slept = d }
		var sb strings.Builder
		elapsed := guardCapParkWait(dec, 0, now, sleep, &sb)
		if slept != dec.Wait {
			t.Errorf("slept %s, want the full decision wait %s", slept, dec.Wait)
		}
		if elapsed != dec.Wait {
			t.Errorf("elapsed = %s, want %s", elapsed, dec.Wait)
		}
		out := sb.String()
		if !strings.Contains(out, "parked") || !strings.Contains(out, "relaunching") {
			t.Errorf("park log missing a park and/or outcome line:\n%s", out)
		}
	})

	t.Run("budget clamps the wait down", func(t *testing.T) {
		var slept time.Duration
		sleep := func(d time.Duration) { slept = d }
		budget := time.Hour
		guardCapParkWait(dec, budget, func() time.Time { return time.Unix(0, 0) }, sleep, nil)
		if slept != budget {
			t.Errorf("slept %s, want it clamped to the budget %s", slept, budget)
		}
	})

	t.Run("a no-park decision sleeps nothing", func(t *testing.T) {
		nopark := guardClassifyCapCrash(capDiag(resume.LimitRate), -1, []string{"claude"}, "claude")
		slept := false
		guardCapParkWait(nopark, 0, nil, func(time.Duration) { slept = true }, nil)
		if slept {
			t.Errorf("a no-park decision slept, want no sleep")
		}
	})
}

// writeTranscriptAt writes a `.jsonl` at the given path with the given mtime, creating parents.
func writeTranscriptAt(t *testing.T, path string, mod time.Time, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
}

func TestGuardCapWitnessTranscript(t *testing.T) {
	base := time.Unix(1_700_000_000, 0)
	childStarted := base.Add(10 * time.Minute)

	t.Run("picks the freshest transcript written at or after the child started", func(t *testing.T) {
		root := t.TempDir()
		proj := filepath.Join(root, "C--work-fak")
		// A stale sibling (before the child), and two fresh ones — the freshest must win.
		writeTranscriptAt(t, filepath.Join(proj, "stale.jsonl"), base, "{}")
		writeTranscriptAt(t, filepath.Join(proj, "older-live.jsonl"), childStarted.Add(1*time.Minute), "{}")
		writeTranscriptAt(t, filepath.Join(proj, "newest.jsonl"), childStarted.Add(5*time.Minute), "{}")

		got := guardCapWitnessTranscript(root, childStarted)
		if filepath.Base(got) != "newest.jsonl" {
			t.Errorf("witness = %q, want newest.jsonl (freshest at/after childStarted)", got)
		}
	})

	t.Run("returns empty when every transcript predates the child (stale siblings only)", func(t *testing.T) {
		root := t.TempDir()
		proj := filepath.Join(root, "C--work-fak")
		writeTranscriptAt(t, filepath.Join(proj, "a.jsonl"), base, "{}")
		writeTranscriptAt(t, filepath.Join(proj, "b.jsonl"), base.Add(1*time.Minute), "{}")
		if got := guardCapWitnessTranscript(root, childStarted); got != "" {
			t.Errorf("witness = %q, want \"\" (no transcript at/after childStarted)", got)
		}
	})

	t.Run("returns empty for a missing or empty store", func(t *testing.T) {
		if got := guardCapWitnessTranscript(filepath.Join(t.TempDir(), "nope"), childStarted); got != "" {
			t.Errorf("missing store witness = %q, want \"\"", got)
		}
		if got := guardCapWitnessTranscript(t.TempDir(), childStarted); got != "" {
			t.Errorf("empty store witness = %q, want \"\"", got)
		}
	})
}

func TestGuardCapRecoverFromEvents(t *testing.T) {
	cmd := []string{"claude", "-p", "x"}
	capEvents := []resume.Event{
		{Kind: resume.EventRealAssistant, PromptTokens: 120000},
		{Kind: resume.EventRateLimitError, LimitReason: resume.LimitSession},
	}

	t.Run("a witnessed session cap with fresh idle parks", func(t *testing.T) {
		last := int64(1_700_000_000)
		now := last + 60 // 1 minute idle — well inside the 5h window
		dec := guardCapRecoverFromEvents(capEvents, last, now, cmd, "claude")
		if !dec.Park || dec.LimitReason != resume.LimitSession {
			t.Fatalf("dec = (park=%v, reason=%q), want a session-cap park", dec.Park, dec.LimitReason)
		}
	})

	t.Run("unknown timestamps yield idle -1 and still park the full window", func(t *testing.T) {
		dec := guardCapRecoverFromEvents(capEvents, 0, 0, cmd, "claude")
		if !dec.Park {
			t.Fatalf("dec.Park = false, want true when idle is unknown")
		}
		if want := time.Duration(resume.SessionLimitResetSeconds) * time.Second; dec.Wait != want {
			t.Errorf("Wait = %s, want the full window %s", dec.Wait, want)
		}
	})

	t.Run("a clean transcript (no cap) does not park", func(t *testing.T) {
		clean := []resume.Event{{Kind: resume.EventRealAssistant, PromptTokens: 1000}}
		if dec := guardCapRecoverFromEvents(clean, 1_700_000_000, 1_700_000_060, cmd, "claude"); dec.Park {
			t.Errorf("clean transcript parked, want no park")
		}
	})
}

func TestGuardCapParkEnabled(t *testing.T) {
	cases := map[string]bool{"": true, "1": true, "true": true, "yes": true, "0": false, "false": false, "off": false, "no": false}
	for val, want := range cases {
		t.Setenv("FAK_GUARD_CAP_PARK", val)
		if got := guardCapParkEnabled(); got != want {
			t.Errorf("FAK_GUARD_CAP_PARK=%q -> %v, want %v", val, got, want)
		}
	}
}
