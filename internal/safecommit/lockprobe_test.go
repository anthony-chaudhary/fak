package safecommit

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestProbeLockClassifies(t *testing.T) {
	dir := t.TempDir()

	// Absent file: not present, not stale.
	if p := ProbeLock(filepath.Join(dir, "absent.lock")); p.Exists || p.Stale {
		t.Fatalf("absent: got %+v, want Exists=false Stale=false", p)
	}

	// Live holder (this process): present, alive, not stale.
	live := filepath.Join(dir, "live.lock")
	if err := os.WriteFile(live, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if p := ProbeLock(live); !p.Exists || !p.Alive || p.Stale {
		t.Fatalf("live: got %+v, want Exists=true Alive=true Stale=false", p)
	}

	// Dead holder: present, not alive, stale.
	dead := filepath.Join(dir, "dead.lock")
	if err := os.WriteFile(dead, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if p := ProbeLock(dead); !p.Exists || p.Alive || !p.Stale {
		t.Fatalf("dead: got %+v, want Exists=true Alive=false Stale=true", p)
	}

	// Garbage body: present but unattributable => not stale.
	garbage := filepath.Join(dir, "garbage.lock")
	if err := os.WriteFile(garbage, []byte("not-a-pid\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if p := ProbeLock(garbage); !p.Exists || p.HolderPID != 0 || p.Stale {
		t.Fatalf("garbage: got %+v, want Exists=true HolderPID=0 Stale=false", p)
	}
}

func TestReapStaleLockReturnValue(t *testing.T) {
	dir := t.TempDir()

	// Stale => reaped, file gone.
	dead := filepath.Join(dir, "dead.lock")
	if err := os.WriteFile(dead, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !ReapStaleLock(dead) {
		t.Fatal("stale lock: ReapStaleLock returned false, want true")
	}
	if _, err := os.Stat(dead); !os.IsNotExist(err) {
		t.Fatalf("stale lock not removed: %v", err)
	}

	// Live => not reaped, file kept.
	live := filepath.Join(dir, "live.lock")
	if err := os.WriteFile(live, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if ReapStaleLock(live) {
		t.Fatal("live lock: ReapStaleLock returned true, want false (must not reap a live holder)")
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("live lock wrongly removed: %v", err)
	}

	// Absent => not reaped, no panic.
	if ReapStaleLock(filepath.Join(dir, "absent.lock")) {
		t.Fatal("absent lock: ReapStaleLock returned true, want false")
	}
}

func TestProcessAliveExported(t *testing.T) {
	if !ProcessAlive(os.Getpid()) {
		t.Fatal("ProcessAlive(self) = false, want true")
	}
	if ProcessAlive(deadPID(t)) {
		t.Fatal("ProcessAlive(dead) = true, want false")
	}
	if ProcessAlive(0) || ProcessAlive(-1) {
		t.Fatal("ProcessAlive(<=0) = true, want false")
	}
}
