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

// The shared guard/serve shutdown tail (persistCacheValueObservations -> serve.go:396,
// guard_child.go:1413) persists this session's observed window through
// writeConfiguredVCacheSnapshot with NO cache flag. This asserts the default emission is
// BOUNDED to the live window and stays replayable via the ordinary Read path — the #1524
// done condition ("a finished session leaves a bounded, replayable cache window without
// extra flags"). A long session must not persist an unbounded per-session log.
func TestWriteConfiguredVCacheSnapshotBoundsLiveWindow(t *testing.T) {
	// No override flags: only the destination path is redirected so the test stays hermetic.
	// The bound itself is the DEFAULT, proving "without extra flags".
	t.Setenv(vcachesnapshot.EnvWindow, "")
	path := filepath.Join(t.TempDir(), "cmd-bounded-vcache-turns.jsonl")
	t.Setenv(vcachesnapshot.EnvPath, path)

	over := vcachesnapshot.DefaultWindowTurns + 25
	turns := make([]vcacheobserve.Turn, over)
	for i := range turns {
		turns[i] = vcacheobserve.Turn{Family: "provider", UnixMillis: int64(i + 1), CacheRead: int64(i)}
	}
	got, ok, err := writeConfiguredVCacheSnapshot(turns)
	if err != nil || !ok {
		t.Fatalf("writeConfiguredVCacheSnapshot() ok=%v err=%v, want a written window", ok, err)
	}
	if got != path {
		t.Fatalf("snapshot path = %q, want %q", got, path)
	}

	// Replayable via the same reader `fak vcache score` uses, and bounded to the tail.
	replay, readOK, err := vcachesnapshot.Read(path)
	if err != nil || !readOK {
		t.Fatalf("replay Read() ok=%v err=%v, want a replayable window", readOK, err)
	}
	if len(replay) != vcachesnapshot.DefaultWindowTurns {
		t.Fatalf("default session persisted %d turns, want the bounded %d", len(replay), vcachesnapshot.DefaultWindowTurns)
	}
	if replay[len(replay)-1].UnixMillis != int64(over) {
		t.Fatalf("bounded window dropped the newest turn: last unix_millis=%d, want %d", replay[len(replay)-1].UnixMillis, over)
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
