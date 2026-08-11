package processforest

import (
	"reflect"
	"testing"
	"time"
)

func TestMixedDepthForestSurvivesProcessExitAndReparent(t *testing.T) {
	r := NewRegistry()
	now := time.Date(2026, 8, 11, 23, 0, 0, 0, time.UTC)
	members := []Member{
		{ForestID: "forest-1", MemberID: "root", RootAuthority: "guard:root", AdapterKind: "fak-harness", Generation: 1, Observation: ProcessObservation{HostID: "host-a", PID: 10, StartedAt: now}, CreatedAt: now},
		{ForestID: "forest-1", MemberID: "codex", ParentMemberID: "root", RootAuthority: "guard:root", AdapterKind: "codex", Generation: 1, Observation: ProcessObservation{HostID: "host-a", PID: 11, StartedAt: now.Add(time.Second)}, CreatedAt: now},
		{ForestID: "forest-1", MemberID: "helper", ParentMemberID: "codex", RootAuthority: "guard:root", AdapterKind: "helper", Generation: 1, Observation: ProcessObservation{HostID: "host-a", PID: 12, StartedAt: now.Add(2 * time.Second)}, CreatedAt: now},
		{ForestID: "forest-1", MemberID: "claude", ParentMemberID: "root", RootAuthority: "guard:root", AdapterKind: "claude", Generation: 1, Observation: ProcessObservation{HostID: "host-a", PID: 13, StartedAt: now.Add(3 * time.Second)}, CreatedAt: now},
	}
	for _, m := range members {
		if err := r.Register(m); err != nil {
			t.Fatalf("register %s: %v", m.MemberID, err)
		}
	}
	if err := r.Terminalize("forest-1", "codex", 2, StateCompleted, "wrapper exited", now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := r.Adopt("forest-1", "helper", "root", "guard:root", 2, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	got, err := r.Snapshot("forest-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Members[2].MemberID != "helper" || got.Members[2].ParentMemberID != "root" {
		t.Fatalf("logical ancestry lost: %+v", got.Members)
	}
	again, _ := r.Snapshot("forest-1")
	if !reflect.DeepEqual(got, again) {
		t.Fatal("snapshot is not deterministic")
	}
}

func TestRegistryRejectsCyclesStaleGenerationDuplicateOwnershipAndPIDReuse(t *testing.T) {
	r := NewRegistry()
	now := time.Now().UTC()
	obs := ProcessObservation{HostID: "host", PID: 7, StartedAt: now}
	root := Member{ForestID: "f", MemberID: "root", RootAuthority: "auth", AdapterKind: "custom", Generation: 1, Observation: obs, CreatedAt: now}
	if err := r.Register(root); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(root); err != nil {
		t.Fatalf("idempotent replay: %v", err)
	}
	if err := r.Register(Member{ForestID: "f", MemberID: "other", RootAuthority: "auth", AdapterKind: "helper", Generation: 1, Observation: obs}); err == nil {
		t.Fatal("duplicate live ownership accepted")
	}
	if err := r.Register(Member{ForestID: "f", MemberID: "child", ParentMemberID: "root", RootAuthority: "auth", AdapterKind: "codex", Generation: 1}); err != nil {
		t.Fatal(err)
	}
	if err := r.Reparent("f", "root", "child", 2, now); err == nil {
		t.Fatal("cycle accepted")
	}
	if err := r.Reparent("f", "child", "root", 1, now); err == nil {
		t.Fatal("stale generation accepted")
	}
	if err := r.Adopt("f", "child", "root", "forged", 2, now); err == nil {
		t.Fatal("forged adoption accepted")
	}
	// Same PID with a different start is a distinct process identity, not an alias.
	if err := r.Register(Member{ForestID: "f", MemberID: "reused", ParentMemberID: "root", RootAuthority: "auth", AdapterKind: "helper", Generation: 1, Observation: ProcessObservation{HostID: "host", PID: 7, StartedAt: now.Add(time.Hour)}}); err != nil {
		t.Fatalf("pid reuse incorrectly aliased: %v", err)
	}
}

func TestTerminalizeIsIdempotent(t *testing.T) {
	r := NewRegistry()
	now := time.Now().UTC()
	m := Member{ForestID: "f", MemberID: "root", RootAuthority: "a", AdapterKind: "fak", Generation: 1, CreatedAt: now}
	if err := r.Register(m); err != nil {
		t.Fatal(err)
	}
	if err := r.Terminalize("f", "root", 2, StateLost, "parent death", now); err != nil {
		t.Fatal(err)
	}
	if err := r.Terminalize("f", "root", 2, StateLost, "parent death", now); err != nil {
		t.Fatalf("terminal replay: %v", err)
	}
}
