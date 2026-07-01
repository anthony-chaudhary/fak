package main

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestServeTasksSnapshotInertByDefault(t *testing.T) {
	t.Setenv("FAK_TASKMGR", "")
	snap, ok := serveTasksSnapshot()
	if ok || snap != nil {
		t.Fatalf("default: serveTasksSnapshot() = (%v, %v), want (nil, false)", snap, ok)
	}
}

func TestServeTasksSnapshotEnabledReturnsValidSnapshot(t *testing.T) {
	t.Setenv("FAK_TASKMGR", "1")
	snap, ok := serveTasksSnapshot()
	if !ok {
		t.Fatalf("enabled: ok = false, want true")
	}
	ts, isSnap := snap.(taskmgr.Snapshot)
	if !isSnap {
		t.Fatalf("snapshot type = %T, want taskmgr.Snapshot", snap)
	}
	if ts.Schema != taskmgr.SchemaSnapshot {
		t.Fatalf("schema = %q, want %q", ts.Schema, taskmgr.SchemaSnapshot)
	}
	if err := taskmgr.ValidateSnapshot(ts); err != nil {
		t.Fatalf("served snapshot failed validation: %v", err)
	}
}

func TestTaskManagerEnabledParsing(t *testing.T) {
	for _, on := range []string{"1", "true", "TRUE", "yes", "on"} {
		t.Setenv("FAK_TASKMGR", on)
		if !taskManagerEnabled() {
			t.Fatalf("FAK_TASKMGR=%q should enable", on)
		}
	}
	for _, off := range []string{"", "0", "false", "no", "off"} {
		t.Setenv("FAK_TASKMGR", off)
		if taskManagerEnabled() {
			t.Fatalf("FAK_TASKMGR=%q should stay disabled", off)
		}
	}
}

// TestTasksProviderInstalledByInit proves the init() wiring ran: the gateway holds
// a provider, so a `fak serve` process exposes the (gated) endpoint.
func TestTasksProviderInstalledByInit(t *testing.T) {
	if !gateway.TasksSnapshotProviderInstalled() {
		t.Fatalf("init did not install the gateway tasks provider")
	}
}

func TestServeTasksSnapshotWitnessFilterProviderShape(t *testing.T) {
	t.Setenv("FAK_TASKMGR", "1")
	snap, ok := serveTasksSnapshot()
	if !ok {
		t.Fatalf("serveTasksSnapshot disabled, want enabled provider")
	}
	b, err := json.Marshal(snap)
	if err != nil {
		t.Fatalf("marshal provider snapshot: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(b, &body); err != nil {
		t.Fatalf("provider snapshot not json object: %v", err)
	}
	if body["schema"] != taskmgr.SchemaSnapshot {
		t.Fatalf("schema = %v, want %s", body["schema"], taskmgr.SchemaSnapshot)
	}
	if _, ok := body["tasks"].([]any); !ok {
		t.Fatalf("provider snapshot missing JSON tasks array: %s", b)
	}
}
