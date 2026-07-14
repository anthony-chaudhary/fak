package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// guard_wire_crash_test.go — the #3514 witness. Before this wiring, a dispatch worker that
// exited non-zero on a SINGLE transient upstream wire error (a mid-flight reset/timeout the
// in-handler retry could not absorb) was never retried: the guard recorded the crash and
// os.Exit()'d, and only a human re-running `fak guard -- claude --continue` rescued the live
// session. These tests exercise guardMaybeRetryTransientWireCrash directly (the decision core
// both runGuardChildAndReport and runGuardChildSupervisedAndReport now gate on), mirroring the
// guardMaybeRecoverAuthCrash test shape: they fail against a no-op (a relaunch that never
// happens even with a transient error visibly observed in-window) and pass against the real
// decision. The load-bearing property — a crash with NO observed transient is NOT retried, so a
// systematic crash is never masked — is pinned by the no_transient case.

// wireCrashExitErr manufactures a real, non-zero *exec.ExitError (NONZERO_EXIT) so
// guardClassifyChildCrash inside the decision sees a genuine crash, not a synthetic error.
// Reuses execCommandExit from guard_auth_crash_test.go.
func wireCrashExitErr(t *testing.T) error {
	t.Helper()
	return execCommandExit(1).Run()
}

func TestGuardMaybeRetryTransientWireCrash(t *testing.T) {
	command := []string{"claude", "-p", "hello"}

	t.Run("transient_observed_crash_retries_with_resume_flag", func(t *testing.T) {
		runErr := wireCrashExitErr(t)
		next, ok := guardMaybeRetryTransientWireCrash(runErr, nil, command, "claude", true, 0, 2, true, nil)
		if !ok {
			t.Fatalf("a NONZERO_EXIT with a transient wire error observed in-window must be retried; got ok=false")
		}
		if len(next) == 0 || next[len(next)-1] != "--continue" {
			t.Fatalf("the relaunch must carry the resume flag so the session continues; got %v", next)
		}
	})

	t.Run("no_transient_observed_is_not_retried", func(t *testing.T) {
		// The load-bearing gate: an identical crash with NO transient wire evidence must fall
		// through untouched — loosening to a bare NONZERO_EXIT would re-mask a systematic crash.
		runErr := wireCrashExitErr(t)
		next, ok := guardMaybeRetryTransientWireCrash(runErr, nil, command, "claude", false, 0, 2, true, nil)
		if ok || next != nil {
			t.Fatalf("a crash with no observed transient wire error must NOT be retried; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("clean_exit_never_retries", func(t *testing.T) {
		next, ok := guardMaybeRetryTransientWireCrash(nil, nil, command, "claude", true, 0, 2, true, nil)
		if ok || next != nil {
			t.Fatalf("a clean (nil) exit is a completed session, never a wire retry; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("unrecognized_agent_never_auto_relaunches", func(t *testing.T) {
		runErr := wireCrashExitErr(t)
		next, ok := guardMaybeRetryTransientWireCrash(runErr, nil, []string{"codex", "exec"}, "codex", true, 0, 2, true, nil)
		if ok || next != nil {
			t.Fatalf("fak must never guess a foreign agent's resume flag; got next=%v ok=%v", next, ok)
		}
	})

	t.Run("zero_limit_disables_the_arm", func(t *testing.T) {
		runErr := wireCrashExitErr(t)
		next, ok := guardMaybeRetryTransientWireCrash(runErr, nil, command, "claude", true, 0, 0, true, nil)
		if ok || next != nil {
			t.Fatalf("--wire-retry-limit=0 must disable the arm; got next=%v ok=%v", next, ok)
		}
	})

	// The retry budget is exact: with limit N, exactly N relaunches fire (retries 0..N-1), then
	// the arm stops — a child that keeps crashing on wire blips can never loop unbounded.
	t.Run("retries_stop_at_limit", func(t *testing.T) {
		const limit = 2
		runErr := wireCrashExitErr(t)
		fired := 0
		for retries := 0; retries < limit+2; retries++ {
			_, ok := guardMaybeRetryTransientWireCrash(runErr, nil, command, "claude", true, retries, limit, true, nil)
			if ok {
				fired++
			}
		}
		if fired != limit {
			t.Fatalf("with --wire-retry-limit=%d exactly %d relaunches must fire; got %d", limit, limit, fired)
		}
	})

	// Convergence: a transient wire crash relaunches ONCE, then the resumed child exits cleanly —
	// exactly one relaunch, no further retries, mirroring the acceptance's fake-child loop.
	t.Run("converges_after_one_relaunch", func(t *testing.T) {
		relaunches := 0
		// Iteration 1: the child crashed with a transient wire error observed → one relaunch.
		if _, ok := guardMaybeRetryTransientWireCrash(wireCrashExitErr(t), nil, command, "claude", true, relaunches, 2, true, nil); ok {
			relaunches++
		}
		// Iteration 2: the resumed child exits cleanly (nil) → no retry, the loop converges.
		if _, ok := guardMaybeRetryTransientWireCrash(nil, nil, command, "claude", true, relaunches, 2, true, nil); ok {
			relaunches++
		}
		if relaunches != 1 {
			t.Fatalf("a transient wire crash then a clean exit must be exactly ONE relaunch; got %d", relaunches)
		}
	})
}

// TestGuardWireRetryHop proves the wire retry is not an invisible relaunch: it emits a
// correlated RESTART_HOP that folds into the same restart chain (and `fak guard restart-audit`)
// as a budget restart — a --continue reattach under the SAME trace, handback "continue", ok.
func TestGuardWireRetryHop(t *testing.T) {
	hop := guardWireRetryHop("guard-abc", "claude", 1)
	if hop.Schema != journal.RestartChainSchema {
		t.Fatalf("hop schema = %q, want %q", hop.Schema, journal.RestartChainSchema)
	}
	if hop.Hop != 1 {
		t.Fatalf("hop ordinal = %d, want 1", hop.Hop)
	}
	if hop.FromTrace != "guard-abc" || hop.ToTrace != "guard-abc" || hop.Child != "guard-abc" {
		t.Fatalf("a --continue reattach stays on the same trace; got from=%q to=%q child=%q", hop.FromTrace, hop.ToTrace, hop.Child)
	}
	if hop.Handback != guardRestartHandbackContinue || hop.Status != journal.RestartHopOK {
		t.Fatalf("a recognized-agent wire retry must be handback=continue status=ok; got handback=%q status=%q", hop.Handback, hop.Status)
	}

	// An unrecognized agent degrades to the ORPHANED/inert shape (the arm never actually fires
	// there, but the hop shape must stay honest for symmetry with the budget-restart hop).
	inert := guardWireRetryHop("guard-xyz", "codex", 3)
	if inert.Handback != guardRestartHandbackOrphaned || inert.Status != journal.RestartHopInert {
		t.Fatalf("an unrecognized agent hop must be ORPHANED/inert; got handback=%q status=%q", inert.Handback, inert.Status)
	}
}
