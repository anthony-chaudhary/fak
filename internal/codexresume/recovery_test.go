package codexresume

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRecoverLoggedOutRefreshSuccess(t *testing.T) {
	sourceHome := filepath.Join(t.TempDir(), "source")
	targetHome := filepath.Join(t.TempDir(), "target")
	rollout := writeSyntheticRollout(t, sourceHome, "2026/08/31", "rollout.jsonl", "thread-recover", `{"type":"event_msg"}`)
	writeFile(t, filepath.Join(sourceHome, "auth.json"), []byte("source-secret"))
	writeFile(t, filepath.Join(targetHome, "auth.json"), []byte("target-secret"))
	binding, err := NewThreadBinding("thread-recover", sourceHome, "account-a", rollout, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	failed := loggedOutFailure()
	calls := 0
	attempt, err := RecoverLoggedOutRefresh(context.Background(), failed, binding, []RecoveryHome{
		{Home: filepath.Join(t.TempDir(), "unhealthy"), AccountKey: "account-b", Eligible: true, Healthy: false},
		{Home: targetHome, AccountKey: "account-b", Eligible: true, Healthy: true},
	}, func(_ context.Context, target RecoveryTarget) (Result, error) {
		calls++
		if target.Binding.AccountKeyDigest != AccountKeyDigest("account-b") {
			t.Fatalf("target digest = %q", target.Binding.AccountKeyDigest)
		}
		if _, err := os.Stat(target.RolloutPath); err != nil {
			t.Fatalf("copied rollout: %v", err)
		}
		return Result{Outcome: OutcomeCompleted, TaskCompleted: true}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || !attempt.Attempted || !attempt.Recovered || !attempt.Result.TaskCompleted {
		t.Fatalf("attempt = %#v calls=%d", attempt, calls)
	}
	gotAuth, err := os.ReadFile(filepath.Join(targetHome, "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotAuth) != "target-secret" {
		t.Fatalf("target auth changed: %q", gotAuth)
	}
}

func TestRecoverLoggedOutRefreshNoTargetPreservesOriginal(t *testing.T) {
	sourceHome := filepath.Join(t.TempDir(), "source")
	rollout := writeSyntheticRollout(t, sourceHome, "2026/08/31", "rollout.jsonl", "thread-no-target", "")
	binding, err := NewThreadBinding("thread-no-target", sourceHome, "account-a", rollout, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	failed := loggedOutFailure()
	calls := 0
	attempt, err := RecoverLoggedOutRefresh(context.Background(), failed, binding, []RecoveryHome{
		{Home: sourceHome, AccountKey: "account-a", Eligible: true, Healthy: true},
		{Home: filepath.Join(t.TempDir(), "ineligible"), AccountKey: "account-b", Eligible: false, Healthy: true},
	}, func(context.Context, RecoveryTarget) (Result, error) {
		calls++
		return Result{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 0 || attempt.Attempted || attempt.Recovered || attempt.Result.TurnError != failed.TurnError {
		t.Fatalf("attempt = %#v calls=%d", attempt, calls)
	}
}

func TestRecoverLoggedOutRefreshRetryFailurePreservesOriginal(t *testing.T) {
	sourceHome := filepath.Join(t.TempDir(), "source")
	targetHome := filepath.Join(t.TempDir(), "target")
	rollout := writeSyntheticRollout(t, sourceHome, "2026/08/31", "rollout.jsonl", "thread-retry-fail", "")
	binding, err := NewThreadBinding("thread-retry-fail", sourceHome, "account-a", rollout, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	failed := loggedOutFailure()
	retryCause := errors.New("retry transport failed")
	calls := 0
	attempt, err := RecoverLoggedOutRefresh(context.Background(), failed, binding, []RecoveryHome{
		{Home: targetHome, AccountKey: "account-b", Eligible: true, Healthy: true},
	}, func(context.Context, RecoveryTarget) (Result, error) {
		calls++
		return Result{Outcome: OutcomeUpstreamInterrupted}, retryCause
	})
	if !errors.Is(err, retryCause) {
		t.Fatalf("error = %v, want retry cause", err)
	}
	if calls != 1 || !attempt.Attempted || attempt.Recovered || attempt.Result.TurnError != failed.TurnError {
		t.Fatalf("attempt = %#v calls=%d", attempt, calls)
	}
	if attempt.RetryResult == nil || attempt.RetryResult.Outcome != OutcomeUpstreamInterrupted {
		t.Fatalf("retry result = %#v", attempt.RetryResult)
	}
}

func TestRecoverLoggedOutRefreshUnrelatedErrorDoesNotRehome(t *testing.T) {
	sourceHome := filepath.Join(t.TempDir(), "source")
	targetHome := filepath.Join(t.TempDir(), "target")
	rollout := writeSyntheticRollout(t, sourceHome, "2026/08/31", "rollout.jsonl", "thread-unrelated", "")
	binding, err := NewThreadBinding("thread-unrelated", sourceHome, "account-a", rollout, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	failed := loggedOutFailure()
	failed.TurnError = &TurnError{Message: LoggedOutRefreshFailure + " "}
	calls := 0
	attempt, err := RecoverLoggedOutRefresh(context.Background(), failed, binding, []RecoveryHome{
		{Home: targetHome, AccountKey: "account-b", Eligible: true, Healthy: true},
	}, func(context.Context, RecoveryTarget) (Result, error) {
		calls++
		return Result{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 0 || attempt.Attempted || attempt.Recovered {
		t.Fatalf("attempt = %#v calls=%d", attempt, calls)
	}
	if _, err := os.Stat(filepath.Join(targetHome, filepath.FromSlash(binding.RelativeRolloutPath))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unrelated error copied rollout: %v", err)
	}
}

func loggedOutFailure() Result {
	return Result{
		Outcome:    OutcomeTurnFailed,
		TurnStatus: "failed",
		TurnError:  &TurnError{Type: "response_error", Message: LoggedOutRefreshFailure, Status: 401},
	}
}
