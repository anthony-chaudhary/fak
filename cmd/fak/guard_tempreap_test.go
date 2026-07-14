package main

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// TestGuardTempDirOwner pins the parser that gates every reap: only a
// fak-guard-<hook>-<pid>-<random> name with a known hook and a positive pid is
// claimable. Legacy no-PID dirs, reset seed dirs, unknown hooks, and unrelated
// entries must all return ok=false so the reaper leaves them alone.
func TestGuardTempDirOwner(t *testing.T) {
	cases := []struct {
		base     string
		wantHook string
		wantPID  int
		wantOK   bool
	}{
		{"fak-guard-mcp-12345-3980821147", "mcp", 12345, true},
		{"fak-guard-handoff-1-9", "handoff", 1, true},
		{"fak-guard-precompact-777-0", "precompact", 777, true},
		{"fak-guard-pi-42-abcd", "pi", 42, true},
		{"fak-guard-sessionstart-5-x", "sessionstart", 5, true},
		{"fak-guard-stophook-6-y", "stophook", 6, true},
		{"fak-guard-toolproc-8-z", "toolproc", 8, true},
		// reset seeds are forensic evidence, never reaped here.
		{"fak-guard-reset-abc123", "", 0, false},
		{"fak-guard-reset-99-abc", "", 0, false},
		// legacy no-PID hook dir (pre-#3299): only hook + random, no owner to prove dead.
		{"fak-guard-mcp-3980821147", "", 0, false},
		// pid segment not a positive integer.
		{"fak-guard-mcp-abc-xyz", "", 0, false},
		{"fak-guard-mcp-0-9", "", 0, false},
		{"fak-guard-mcp--5-9", "", 0, false},
		// unknown hook token.
		{"fak-guard-unknown-123-456", "", 0, false},
		// not a guard temp dir at all.
		{"unrelated-tmp-dir", "", 0, false},
		{"fak-selfupdate-build-123", "", 0, false},
	}
	for _, c := range cases {
		hook, pid, ok := guardTempDirOwner(c.base)
		if ok != c.wantOK || hook != c.wantHook || pid != c.wantPID {
			t.Errorf("guardTempDirOwner(%q) = (%q, %d, %v), want (%q, %d, %v)",
				c.base, hook, pid, ok, c.wantHook, c.wantPID, c.wantOK)
		}
	}
}

// mkTempDirEntry creates dir root/name (a directory) for the reaper tests.
func mkTempDirEntry(t *testing.T, root, name string) string {
	t.Helper()
	p := filepath.Join(root, name)
	if err := os.Mkdir(p, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", name, err)
	}
	return p
}

// TestGuardReapStaleTempDirs is the core safety proof: only a dead-owner,
// non-self, known-hook dir is removed; a live peer's dir, this process's own
// dir, a legacy no-PID dir, a reset seed dir, an unrelated dir, and a matching
// regular file are all left in place.
func TestGuardReapStaleTempDirs(t *testing.T) {
	root := t.TempDir()
	const selfPID = 1000
	const liveOtherPID = 7777
	alive := func(pid int) bool { return pid == liveOtherPID }

	dead := mkTempDirEntry(t, root, "fak-guard-mcp-4242-1")        // owner dead -> reap
	self := mkTempDirEntry(t, root, "fak-guard-precompact-1000-2") // pid==selfPID -> keep
	live := mkTempDirEntry(t, root, "fak-guard-stophook-7777-3")   // owner alive -> keep
	legacy := mkTempDirEntry(t, root, "fak-guard-mcp-legacy")      // no pid -> keep
	seed := mkTempDirEntry(t, root, "fak-guard-reset-seedfile")    // forensic -> keep
	other := mkTempDirEntry(t, root, "some-other-tmp")             // unrelated -> keep
	deadPI := mkTempDirEntry(t, root, "fak-guard-pi-4243-4")       // owner dead -> reap
	// a regular file that matches the name shape must be ignored (dirs only).
	filePath := filepath.Join(root, "fak-guard-mcp-4244-5")
	if err := os.WriteFile(filePath, []byte("x"), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	reaped := guardReapStaleTempDirs(root, selfPID, alive)
	sort.Strings(reaped)
	want := []string{dead, deadPI}
	sort.Strings(want)
	if len(reaped) != len(want) {
		t.Fatalf("reaped = %v, want %v", reaped, want)
	}
	for i := range want {
		if reaped[i] != want[i] {
			t.Fatalf("reaped = %v, want %v", reaped, want)
		}
	}
	// The reaped dirs are gone; every kept path still exists.
	for _, gone := range []string{dead, deadPI} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Errorf("expected %s reaped, stat err = %v", gone, err)
		}
	}
	for _, kept := range []string{self, live, legacy, seed, other, filePath} {
		if _, err := os.Stat(kept); err != nil {
			t.Errorf("expected %s kept, stat err = %v", kept, err)
		}
	}
}

// TestGuardReapStaleTempDirsMissingRoot proves a missing temp root is a valid
// empty sweep, not a panic or error.
func TestGuardReapStaleTempDirsMissingRoot(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	if got := guardReapStaleTempDirs(missing, 1, func(int) bool { return false }); got != nil {
		t.Fatalf("reap on missing root = %v, want nil", got)
	}
}

// TestGuardMeasureTempDirs checks the observability tally: total count/bytes and
// the dead-owner reclaimable subset, excluding this process's own dir.
func TestGuardMeasureTempDirs(t *testing.T) {
	root := t.TempDir()
	const selfPID = 1000
	alive := func(int) bool { return false } // everything except self reads as dead

	deadDir := mkTempDirEntry(t, root, "fak-guard-mcp-4242-1")
	if err := os.WriteFile(filepath.Join(deadDir, "settings.json"), []byte("0123456789"), 0o600); err != nil {
		t.Fatalf("write dead file: %v", err)
	}
	selfDir := mkTempDirEntry(t, root, "fak-guard-pi-1000-2")
	if err := os.WriteFile(filepath.Join(selfDir, "ext.ts"), []byte("abc"), 0o600); err != nil {
		t.Fatalf("write self file: %v", err)
	}
	mkTempDirEntry(t, root, "fak-guard-reset-ignored") // not counted

	fp := guardMeasureTempDirs(root, selfPID, alive)
	if fp.Count != 2 {
		t.Errorf("Count = %d, want 2", fp.Count)
	}
	if fp.Bytes != 13 {
		t.Errorf("Bytes = %d, want 13", fp.Bytes)
	}
	if fp.DeadCount != 1 {
		t.Errorf("DeadCount = %d, want 1 (self excluded)", fp.DeadCount)
	}
	if fp.DeadBytes != 10 {
		t.Errorf("DeadBytes = %d, want 10", fp.DeadBytes)
	}
}

// TestGuardSessionTempDir proves the creation seam produces a name the reaper's
// parser claims for THIS process — closing the loop between creator and reaper.
func TestGuardSessionTempDir(t *testing.T) {
	dir, err := guardSessionTempDir("mcp")
	if err != nil {
		t.Fatalf("guardSessionTempDir: %v", err)
	}
	defer os.RemoveAll(dir)
	hook, pid, ok := guardTempDirOwner(filepath.Base(dir))
	if !ok || hook != "mcp" || pid != os.Getpid() {
		t.Fatalf("owner(%q) = (%q, %d, %v), want (mcp, %d, true)", filepath.Base(dir), hook, pid, ok, os.Getpid())
	}
}
