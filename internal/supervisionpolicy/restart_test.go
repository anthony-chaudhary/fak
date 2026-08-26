package supervisionpolicy

import (
	"path/filepath"
	"testing"
	"time"
)

func TestCrashLoopBudgetPersistsAndContainsSiblings(t *testing.T) {
	now := time.Date(2026, 8, 26, 10, 0, 0, 0, time.UTC)
	store := FileStore{Path: filepath.Join(t.TempDir(), "restart.json")}
	spec := IndependentChildSpec("orchestration", "task-a")
	spec.MaxAttempts, spec.BaseBackoff, spec.Jitter = 2, 0, 0

	for attempt := 0; attempt < 2; attempt++ {
		decision, err := RecordExit(store, "worker-a", spec, FailureTransient, now, now.Add(time.Duration(attempt)*time.Second), "receipt-a")
		if err != nil || decision.Action != ActionRestart {
			t.Fatalf("attempt %d: decision=%+v err=%v", attempt+1, decision, err)
		}
	}
	// A new FileStore models a restarted coordinator reading durable accounting.
	exhausted, err := RecordExit(FileStore{Path: store.Path}, "worker-a", spec, FailureTransient, now, now.Add(2*time.Second), "last-a")
	if err != nil {
		t.Fatal(err)
	}
	if exhausted.Action != ActionHold || exhausted.Outcome != OutcomeBudgetExhausted || !exhausted.State.Quarantined || exhausted.State.Reason != "restart_exhausted" || exhausted.State.LastReceipt != "last-a" {
		t.Fatalf("exhausted decision = %+v", exhausted)
	}
	b, err := store.Load("worker-b")
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Failures) != 0 || b.Quarantined {
		t.Fatalf("sibling state changed: %+v", b)
	}
}

func TestRestartClassesStableResetAndDeterministicQuarantine(t *testing.T) {
	now := time.Date(2026, 8, 26, 11, 0, 0, 0, time.UTC)
	tests := []struct {
		name   string
		class  RestartClass
		kind   FailureKind
		want   Action
		reason string
	}{
		{"permanent clean", RestartPermanent, FailureClean, ActionRestart, ""},
		{"transient clean", RestartTransient, FailureClean, ActionHold, "restart_not_requested"},
		{"temporary failure", RestartTemporary, FailureTransient, ActionHold, "restart_not_requested"},
		{"deterministic", RestartPermanent, FailureDeterministic, ActionHold, "deterministic_failure"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := FileStore{Path: filepath.Join(t.TempDir(), "state.json")}
			spec := IndependentChildSpec("orchestration", "task")
			spec.Restart, spec.BaseBackoff, spec.Jitter = tt.class, 0, 0
			d, err := RecordExit(store, "worker", spec, tt.kind, now, now, "receipt")
			if err != nil || d.Action != tt.want || d.State.Reason != tt.reason {
				t.Fatalf("decision=%+v err=%v", d, err)
			}
		})
	}

	store := FileStore{Path: filepath.Join(t.TempDir(), "stable.json")}
	spec := IndependentChildSpec("orchestration", "task")
	spec.MaxAttempts, spec.StableReset, spec.BaseBackoff, spec.Jitter = 1, time.Minute, 0, 0
	first, err := RecordExit(store, "worker", spec, FailureTransient, now, now, "first")
	if err != nil || first.Action != ActionRestart {
		t.Fatalf("first=%+v err=%v", first, err)
	}
	stable, err := RecordExit(store, "worker", spec, FailureTransient, now, now.Add(spec.StableReset), "stable")
	if err != nil || stable.Action != ActionRestart || len(stable.State.Failures) != 1 {
		t.Fatalf("stable reset=%+v err=%v", stable, err)
	}
}

func TestIndependentChildSpecIsVisibleOneForOneContract(t *testing.T) {
	spec := IndependentChildSpec("orchestration", "task")
	if spec.Strategy != StrategyOneForOne || spec.Escalation != EscalateChild || spec.Restart != RestartTransient {
		t.Fatalf("unsafe default: %+v", spec)
	}
}
