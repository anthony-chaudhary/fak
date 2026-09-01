package orchestration

import (
	"encoding/json"
	"testing"
	"time"
)

func TestReconcileWaitLifecycleFixtures(t *testing.T) {
	t.Parallel()
	const timeout = 30 * time.Second
	tests := []struct {
		name    string
		event   WaitLifecycleEvent
		wantErr bool
		check   func(*testing.T, WaitLifecycleState)
	}{
		{name: "one target started", event: WaitLifecycleEvent{Phase: WaitPhaseStarted, TargetIDs: []string{"worker-1"}, RequestedTimeout: timeout, EffectiveTimeout: timeout}},
		{name: "multiple targets complete", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-2", "worker-1"}, RequestedTimeout: timeout, EffectiveTimeout: timeout, Targets: []WaitTarget{{ID: "worker-2", Status: WaitTargetCompleted}, {ID: "worker-1", Status: WaitTargetCompleted}}}, check: func(t *testing.T, got WaitLifecycleState) {
			if got.TargetIDs[0] != "worker-1" || got.Targets[0].ID != "worker-1" {
				t.Fatalf("targets not canonicalized: %+v", got)
			}
		}},
		{name: "partial completion before deadline", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1", "worker-2"}, EffectiveTimeout: timeout, Targets: []WaitTarget{{ID: "worker-1", Status: WaitTargetCompleted}, {ID: "worker-2", Status: WaitTargetRunning}}}},
		{name: "pure timeout remains bounded wait", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1", "worker-2"}, EffectiveTimeout: timeout, TimedOut: true, Targets: []WaitTarget{{ID: "worker-1", Status: WaitTargetRunning}, {ID: "worker-2", Status: WaitTargetRunning}}}, check: func(t *testing.T, got WaitLifecycleState) {
			if !got.TimedOut || got.Authority != WaitAuthorityNative {
				t.Fatalf("timeout lost authority: %+v", got)
			}
			for _, target := range got.Targets {
				if target.Status != WaitTargetRunning {
					t.Fatalf("timeout changed target status: %+v", got)
				}
			}
		}},
		{name: "terminal failure", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1"}, EffectiveTimeout: timeout, Targets: []WaitTarget{{ID: "worker-1", Status: WaitTargetFailed}}}},
		{name: "unknown status", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1"}, EffectiveTimeout: timeout, Targets: []WaitTarget{{ID: "worker-1", Status: "paused"}}}, wantErr: true},
		{name: "unknown target", event: WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1"}, EffectiveTimeout: timeout, Targets: []WaitTarget{{ID: "worker-2", Status: WaitTargetRunning}}}, wantErr: true},
		{name: "nonpositive effective timeout", event: WaitLifecycleEvent{Phase: WaitPhaseStarted, TargetIDs: []string{"worker-1"}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReconcileWaitLifecycle(tt.event)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ReconcileWaitLifecycle() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err == nil {
				if got.Source != "structured_wait" {
					t.Fatalf("source = %q", got.Source)
				}
				if tt.check != nil {
					tt.check(t, got)
				}
			}
		})
	}
}

func TestDegradedWaitLifecycleNamesFallbackAuthority(t *testing.T) {
	t.Parallel()
	got, err := DegradedWaitLifecycle([]string{"worker-2", "worker-1"}, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if got.Authority != WaitAuthorityDegraded || got.Source != "process_liveness" {
		t.Fatalf("fallback authority = %+v", got)
	}
}

func TestWaitLifecycleStateStatusJSONIsPrivacySafe(t *testing.T) {
	t.Parallel()
	got, err := ReconcileWaitLifecycle(WaitLifecycleEvent{Phase: WaitPhaseCompleted, TargetIDs: []string{"worker-1"}, RequestedTimeout: time.Minute, EffectiveTimeout: 30 * time.Second, TimedOut: true, Targets: []WaitTarget{{ID: "worker-1", Status: WaitTargetRunning}}})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"authority", "source", "target_ids", "requested_timeout", "effective_timeout", "timed_out", "targets"} {
		if _, ok := fields[key]; !ok {
			t.Fatalf("status JSON missing %q: %s", key, data)
		}
	}
	for _, forbidden := range []string{"prompt", "path", "command"} {
		if _, ok := fields[forbidden]; ok {
			t.Fatalf("status JSON leaked %q: %s", forbidden, data)
		}
	}
}
