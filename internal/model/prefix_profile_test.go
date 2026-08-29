package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestPrefixProfileIsOptInAndMachineReadable(t *testing.T) {
	oldPath := prefixProfile.path
	t.Cleanup(func() { SetPrefixProfilePath(oldPath) })
	SetPrefixProfilePath("")
	emitPrefixProfile(prefixProfileStart(), "device_clone", "complete", nil, nil)

	SetPrefixProfilePath(filepath.Join(t.TempDir(), "prefix.jsonl"))
	start := prefixProfileStart()
	emitPrefixProfile(start, "device_clone", "complete", nil, nil)
	data, err := os.ReadFile(prefixProfile.path)
	if err != nil {
		t.Fatal(err)
	}
	var event PrefixProfileEvent
	if err := json.Unmarshal(data[:len(data)-1], &event); err != nil {
		t.Fatal(err)
	}
	if event.Schema != "fak.prefix-profile/1" || event.Operation != "device_clone" || event.DurationNS < 0 {
		t.Fatalf("event=%+v", event)
	}
}

func TestPrefixProfileUsesExplicitConfigNotLegacyEnvironment(t *testing.T) {
	oldPath := prefixProfile.path
	t.Cleanup(func() { SetPrefixProfilePath(oldPath) })
	SetPrefixProfilePath("")
	t.Setenv("FAK_PREFIX_PROFILE", filepath.Join(t.TempDir(), "legacy.jsonl"))
	if started := prefixProfileStart(); !started.IsZero() {
		t.Fatalf("legacy environment unexpectedly enabled prefix profiling: %v", started)
	}
}
