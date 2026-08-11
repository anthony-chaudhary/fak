package sessiondiag

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func TestLifecyclePlanReconcilesCrashParentDeathMissingReceiptAndStaleEdge(t *testing.T) {
	now := time.Date(2026, 8, 11, 22, 0, 0, 0, time.UTC)
	old := now.Add(-time.Hour)
	in := InventoryInput{
		Window:     30 * time.Minute,
		StaleAfter: 10 * time.Minute,
		Turns: []TurnEvidence{
			{ThreadID: "failed-parent", Status: "failed", CompletedAt: old},
			{ThreadID: "dead-child", Status: "failed", CompletedAt: old},
		},
		WriterLocks: []WriterLockEvidence{{ThreadID: "failed-parent", ModifiedAt: old}},
		Threads: append([]ThreadEvidence{
			{ThreadID: "failed-parent", UpdatedAt: old},
			{ThreadID: "dead-child", UpdatedAt: old},
		}, ThreadEvidence{ThreadID: "51000001-0000-4000-8000-000000000001", UpdatedAt: old}),
		GuardReceipts: []GuardReceiptEvidence{{ThreadID: "51000001-0000-4000-8000-000000000001", RecordedAt: now.Add(-time.Minute)}},
		SpawnEdges:    []SpawnEdgeEvidence{{ParentThreadID: "failed-parent", ChildThreadID: "dead-child", Status: "open"}},
		Registrations: []sessionregistry.Record{
			{Schema: sessionregistry.Schema, RegistrationID: "crashed-child", RootRegistrationID: "crashed-child", AttemptID: "a", LaunchKind: "subagent", State: sessionregistry.StateActive, CreatedAt: old, StartedAt: old, Identity: sessionregistry.Identity{Runtime: "codex", PID: 99, ProcessStartedAt: old}},
			{Schema: sessionregistry.Schema, RegistrationID: "missing-start-receipt", RootRegistrationID: "missing-start-receipt", AttemptID: "b", LaunchKind: "guarded_tui", State: sessionregistry.StateRegistered, CreatedAt: old, Identity: sessionregistry.Identity{Runtime: "claude"}},
		},
	}
	report := ReconcileInventory(in, now)
	want := map[string]bool{
		"guard_receipt:51000001-0000-4000-8000-000000000001:retain_receipt": false,
		"registration:crashed-child:append_lost":                            false,
		"registration:missing-start-receipt:append_unknown":                 false,
		"spawn_edge:failed-parent->dead-child:terminalize_unknown":          false,
		"writer_lock:failed-parent:remove":                                  false,
	}
	for _, action := range report.CleanupActions {
		key := action.Artifact + ":" + action.Identity + ":" + action.Action
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, found := range want {
		if !found {
			t.Errorf("missing lifecycle action %s; got %+v", key, report.CleanupActions)
		}
	}
	if !report.ReadOnly {
		t.Fatal("inventory reconciliation must remain dry-run")
	}
}

func TestLifecyclePlanKeepsLivePIDStartRegistrationAndIsIdempotent(t *testing.T) {
	now := time.Date(2026, 8, 11, 22, 0, 0, 0, time.UTC)
	started := now.Add(-time.Hour)
	in := InventoryInput{StaleAfter: time.Minute,
		Processes:     []ProcessEvidence{{PID: 50, Name: "codex.exe", StartedAt: started}},
		Registrations: []sessionregistry.Record{{RegistrationID: "live", RootRegistrationID: "live", AttemptID: "a", LaunchKind: "guarded_tui", State: sessionregistry.StateActive, CreatedAt: started, HeartbeatAt: started, Identity: sessionregistry.Identity{Runtime: "codex", PID: 50, ProcessStartedAt: started}}},
	}
	first, second := ReconcileInventory(in, now), ReconcileInventory(in, now)
	if len(first.RegistrationReconciliation) != 0 || len(first.CleanupActions) != 0 {
		t.Fatalf("live identity proposed cleanup: %+v", first)
	}
	if len(second.CleanupActions) != len(first.CleanupActions) {
		t.Fatalf("non-idempotent plans: %+v / %+v", first.CleanupActions, second.CleanupActions)
	}
}
