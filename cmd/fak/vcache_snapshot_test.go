package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func TestWriteConfiguredVCacheSnapshotUsesEnvPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd-vcache-turns.jsonl")
	t.Setenv(vcachesnapshot.EnvPath, path)

	got, ok, err := writeConfiguredVCacheSnapshot([]vcacheobserve.Turn{{
		Family:            "context",
		ContextEvents:     1,
		ContextShedTokens: 1200,
	}})
	if err != nil {
		t.Fatalf("writeConfiguredVCacheSnapshot() error = %v", err)
	}
	if !ok {
		t.Fatal("writeConfiguredVCacheSnapshot disabled with a file override")
	}
	if got != path {
		t.Fatalf("writeConfiguredVCacheSnapshot path = %q, want %q", got, path)
	}
	turns, readOK, err := vcachesnapshot.Read(path)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	if !readOK || len(turns) != 1 || turns[0].ContextEvents != 1 || turns[0].ContextShedTokens != 1200 {
		t.Fatalf("persisted turns = %+v ok=%v, want one context row", turns, readOK)
	}
}

func TestWriteConfiguredVCacheSnapshotOffSkips(t *testing.T) {
	t.Setenv(vcachesnapshot.EnvPath, "off")

	got, ok, err := writeConfiguredVCacheSnapshot([]vcacheobserve.Turn{{
		Family:    "provider",
		CacheRead: 55,
	}})
	if err != nil {
		t.Fatalf("writeConfiguredVCacheSnapshot(off) error = %v", err)
	}
	if ok || got != "" {
		t.Fatalf("writeConfiguredVCacheSnapshot(off) = path %q ok %v, want disabled", got, ok)
	}
}

func TestWriteConfiguredVCacheSnapshotEmptySkips(t *testing.T) {
	got, ok, err := writeConfiguredVCacheSnapshot(nil)
	if err != nil {
		t.Fatalf("writeConfiguredVCacheSnapshot(empty) error = %v", err)
	}
	if ok || got != "" {
		t.Fatalf("writeConfiguredVCacheSnapshot(empty) = path %q ok %v, want skipped", got, ok)
	}
}

func TestWriteExplicitVCacheSnapshotRequiresEnvPath(t *testing.T) {
	t.Setenv(vcachesnapshot.EnvPath, "")

	got, ok, err := writeExplicitVCacheSnapshot([]vcacheobserve.Turn{{
		Family:    "provider",
		CacheRead: 55,
	}})
	if err != nil {
		t.Fatalf("writeExplicitVCacheSnapshot(no env) error = %v", err)
	}
	if ok || got != "" {
		t.Fatalf("writeExplicitVCacheSnapshot(no env) = path %q ok %v, want skipped", got, ok)
	}
}

func TestWriteExplicitVCacheSnapshotUsesEnvPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replay-vcache-turns.jsonl")
	t.Setenv(vcachesnapshot.EnvPath, path)

	got, ok, err := writeExplicitVCacheSnapshot([]vcacheobserve.Turn{{
		Family:            "context",
		ContextEvents:     1,
		ContextShedTokens: 700,
	}})
	if err != nil {
		t.Fatalf("writeExplicitVCacheSnapshot() error = %v", err)
	}
	if !ok || got != path {
		t.Fatalf("writeExplicitVCacheSnapshot() = path %q ok %v, want %q true", got, ok, path)
	}
}
