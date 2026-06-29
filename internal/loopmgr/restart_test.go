package loopmgr

import (
	"testing"
	"time"
)

func TestRestartPolicySchedulesFirstAttempt(t *testing.T) {
	p := RestartPolicy{MaxAttempts: 3, BaseDelay: time.Second, MaxDelay: 10 * time.Second}
	lastFailure := time.Unix(100, 0).UTC()

	d := p.Decide(0, lastFailure, lastFailure, nil)
	if !d.Restart || d.GiveUp {
		t.Fatalf("first failure should schedule a restart, got %+v", d)
	}
	if d.Reason != ReasonRestartScheduled || !ValidRestartReason(d.Reason) {
		t.Fatalf("reason = %q, want %q", d.Reason, ReasonRestartScheduled)
	}
	if d.Attempt != 1 || d.After != time.Second || d.AfterNanos != int64(time.Second) {
		t.Fatalf("attempt/after = %d/%s/%d, want 1/1s/%d", d.Attempt, d.After, d.AfterNanos, int64(time.Second))
	}
	if got, want := d.RestartAtUnixNano, lastFailure.Add(time.Second).UnixNano(); got != want {
		t.Fatalf("restart_at = %d, want %d", got, want)
	}
}

func TestRestartPolicyBackoffCapsAndJittersDeterministically(t *testing.T) {
	p := RestartPolicy{
		MaxAttempts: 5,
		BaseDelay:   time.Second,
		MaxDelay:    5 * time.Second,
		Jitter:      500 * time.Millisecond,
	}
	var calls []uint64
	jitter := func(attempt uint64, window time.Duration) time.Duration {
		calls = append(calls, attempt)
		if window != 500*time.Millisecond {
			t.Fatalf("jitter window = %s, want 500ms", window)
		}
		return 250 * time.Millisecond
	}

	if got := p.BackoffDelay(0, jitter); got != 1250*time.Millisecond {
		t.Fatalf("attempt 1 delay = %s, want 1.25s", got)
	}
	if got := p.BackoffDelay(1, jitter); got != 2250*time.Millisecond {
		t.Fatalf("attempt 2 delay = %s, want 2.25s", got)
	}
	if got := p.BackoffDelay(3, jitter); got != 5*time.Second {
		t.Fatalf("capped delay = %s, want 5s", got)
	}
	if len(calls) != 3 || calls[0] != 1 || calls[1] != 2 || calls[2] != 4 {
		t.Fatalf("jitter attempts = %v, want [1 2 4]", calls)
	}

	tooLarge := func(uint64, time.Duration) time.Duration { return time.Hour }
	if got := p.BackoffDelay(0, tooLarge); got != 1500*time.Millisecond {
		t.Fatalf("clamped jitter delay = %s, want 1.5s", got)
	}
}

func TestRestartPolicyDebounceReturnsPendingRestart(t *testing.T) {
	p := RestartPolicy{MaxAttempts: 3, BaseDelay: 3 * time.Second, MaxDelay: 10 * time.Second}
	lastFailure := time.Unix(200, 0).UTC()

	first := p.Decide(0, lastFailure, lastFailure, nil)
	again := p.Decide(0, lastFailure, lastFailure.Add(time.Second), nil)

	if !again.Restart || again.GiveUp {
		t.Fatalf("debounced attempt should keep a pending restart, got %+v", again)
	}
	if again.Reason != ReasonRestartDebounced {
		t.Fatalf("reason = %q, want %q", again.Reason, ReasonRestartDebounced)
	}
	if again.RestartAtUnixNano != first.RestartAtUnixNano {
		t.Fatalf("debounce changed restart time: first=%d again=%d", first.RestartAtUnixNano, again.RestartAtUnixNano)
	}
	if again.Attempt != first.Attempt {
		t.Fatalf("debounce changed attempt: first=%d again=%d", first.Attempt, again.Attempt)
	}
	if again.After != 2*time.Second {
		t.Fatalf("remaining backoff = %s, want 2s", again.After)
	}
}

func TestRestartPolicyGiveUpAtMaxAttempts(t *testing.T) {
	p := RestartPolicy{MaxAttempts: 2, BaseDelay: time.Second, MaxDelay: time.Minute}

	d := p.Decide(2, time.Unix(300, 0).UTC(), time.Unix(300, 0).UTC(), nil)
	if !d.GiveUp || d.Restart {
		t.Fatalf("max attempts should give up, got %+v", d)
	}
	if d.Reason != ReasonRestartExhausted || !ValidRestartReason(d.Reason) {
		t.Fatalf("reason = %q, want %q", d.Reason, ReasonRestartExhausted)
	}

	bad := (RestartPolicy{}).Decide(0, time.Time{}, time.Unix(0, 0).UTC(), nil)
	if !bad.GiveUp || bad.Reason != ReasonRestartPolicyInvalid {
		t.Fatalf("invalid policy should give up with typed reason, got %+v", bad)
	}
}

func TestRestartPolicyResetOnSuccessAfterClearWindow(t *testing.T) {
	p := RestartPolicy{MaxAttempts: 3, BaseDelay: time.Second, MaxDelay: time.Minute, ResetAfter: 10 * time.Second}
	start := time.Unix(400, 0).UTC()

	if got := p.AttemptsAfterSuccess(2, start, start.Add(9*time.Second)); got != 2 {
		t.Fatalf("short success reset attempts to %d, want 2", got)
	}
	if got := p.AttemptsAfterSuccess(2, start, start.Add(10*time.Second)); got != 0 {
		t.Fatalf("clear-window success left attempts = %d, want 0", got)
	}

	anySuccess := RestartPolicy{MaxAttempts: 3, BaseDelay: time.Second, MaxDelay: time.Minute}
	if got := anySuccess.AttemptsAfterSuccess(1, start, start); got != 0 {
		t.Fatalf("zero reset_after should reset on any success, got %d", got)
	}
}
