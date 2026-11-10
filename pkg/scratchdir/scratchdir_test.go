package scratchdir

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func makeTree(t *testing.T, root string) {
	t.Helper()
	deep := filepath.Join(root, "a", "b", "c")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{
		filepath.Join(root, "top.txt"),
		filepath.Join(root, "a", "mid.txt"),
		filepath.Join(deep, "leaf.txt"),
	} {
		if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ro := filepath.Join(root, "a", "b", "readonly.txt")
	if err := os.WriteFile(ro, []byte("ro"), 0o444); err != nil {
		t.Fatal(err)
	}
}

func assertGone(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("%s still present (stat err=%v)", path, err)
	}
}

func assertPresent(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("%s missing: %v", path, err)
	}
}

func age(t *testing.T, path string, d time.Duration) {
	t.Helper()
	old := time.Now().Add(-d)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
}

func mkdir(t *testing.T, path string) string {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func entryFor(t *testing.T, rep Report, path string) Entry {
	t.Helper()
	for _, e := range rep.Entries {
		if e.Path == path {
			return e
		}
	}
	t.Fatalf("sweep report has no entry for %s", path)
	return Entry{}
}

// fak-test:runtime fast est=20ms lane=default
func TestCreateRecordsThisProcessAsHolder(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "nested", "parent")
	dir, err := Create(parent, "x-")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(dir) != parent {
		t.Fatalf("Create dir %s not under parent %s", dir, parent)
	}
	b, err := os.ReadFile(filepath.Join(dir, HolderFile))
	if err != nil {
		t.Fatal(err)
	}
	pid, _, ok := parseHolder(b)
	if !ok || pid != os.Getpid() {
		t.Fatalf("holder pid = %d ok=%v, want %d", pid, ok, os.Getpid())
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRemoveAllRemovesNestedTreeAndToleratesMissing(t *testing.T) {
	root := filepath.Join(t.TempDir(), "tree")
	makeTree(t, root)
	if err := RemoveAll(root); err != nil {
		t.Fatalf("RemoveAll: %v", err)
	}
	assertGone(t, root)
	if err := RemoveAll(root); err != nil {
		t.Fatalf("RemoveAll on missing path: %v", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRemoveBottomUpRemovesNestedTree(t *testing.T) {
	root := filepath.Join(t.TempDir(), "tree")
	makeTree(t, root)
	if err := removeBottomUp(root); err != nil {
		t.Fatalf("removeBottomUp: %v", err)
	}
	assertGone(t, root)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepNeverRemovesDirHeldByLiveProcess(t *testing.T) {
	parent := t.TempDir()
	held, err := Create(parent, "s-")
	if err != nil {
		t.Fatal(err)
	}
	age(t, held, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, MinAge: 0, Apply: true})
	if e := entryFor(t, rep, held); e.Verdict != VerdictHeld || e.Removed {
		t.Fatalf("held dir verdict=%s removed=%v, want held/false", e.Verdict, e.Removed)
	}
	if rep.Held != 1 || rep.Removed != 0 {
		t.Fatalf("Held=%d Removed=%d, want 1/0", rep.Held, rep.Removed)
	}
	assertPresent(t, held)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepRemovesStaleKeepsYoung(t *testing.T) {
	parent := t.TempDir()
	stale := mkdir(t, filepath.Join(parent, "s-stale"))
	makeTree(t, stale)
	age(t, stale, 48*time.Hour)
	young := mkdir(t, filepath.Join(parent, "s-young"))
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: true})
	if e := entryFor(t, rep, stale); e.Verdict != VerdictStale || !e.Removed {
		t.Fatalf("stale dir verdict=%s removed=%v", e.Verdict, e.Removed)
	}
	if e := entryFor(t, rep, young); e.Verdict != VerdictYoung || e.Removed {
		t.Fatalf("young dir verdict=%s removed=%v", e.Verdict, e.Removed)
	}
	if rep.Stale != 1 || rep.Young != 1 || rep.Removed != 1 || rep.Failed != 0 {
		t.Fatalf("counts stale=%d young=%d removed=%d failed=%d", rep.Stale, rep.Young, rep.Removed, rep.Failed)
	}
	assertGone(t, stale)
	assertPresent(t, young)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepRemovesOldDirOfDeadHolder(t *testing.T) {
	prev := holderAlive
	holderAlive = func(int, int64) bool { return false }
	t.Cleanup(func() { holderAlive = prev })
	parent := t.TempDir()
	dir, err := Create(parent, "s-")
	if err != nil {
		t.Fatal(err)
	}
	age(t, dir, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: true})
	if e := entryFor(t, rep, dir); e.Verdict != VerdictStale || !e.Removed {
		t.Fatalf("dead-holder dir verdict=%s removed=%v", e.Verdict, e.Removed)
	}
	assertGone(t, dir)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepDryRunMeasuresButKeeps(t *testing.T) {
	parent := t.TempDir()
	stale := mkdir(t, filepath.Join(parent, "s-stale"))
	makeTree(t, stale)
	age(t, stale, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: false, Measure: true})
	if rep.Applied || rep.Stale != 1 || rep.Removed != 0 || rep.StaleBytes <= 0 {
		t.Fatalf("dry run applied=%v stale=%d removed=%d staleBytes=%d", rep.Applied, rep.Stale, rep.Removed, rep.StaleBytes)
	}
	if e := entryFor(t, rep, stale); e.Removed || e.Bytes <= 0 {
		t.Fatalf("dry-run entry removed=%v bytes=%d", e.Removed, e.Bytes)
	}
	assertPresent(t, stale)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepPrefixFiltersOtherDirs(t *testing.T) {
	parent := t.TempDir()
	match := mkdir(t, filepath.Join(parent, "want-1"))
	other := mkdir(t, filepath.Join(parent, "other-1"))
	age(t, match, 48*time.Hour)
	age(t, other, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, Prefix: "want-", MinAge: time.Hour, Apply: true})
	if len(rep.Entries) != 1 || rep.Entries[0].Path != match || rep.Removed != 1 {
		t.Fatalf("entries=%d removed=%d, want only %s swept", len(rep.Entries), rep.Removed, match)
	}
	assertGone(t, match)
	assertPresent(t, other)
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepUnparseableHolderOnRecentDirIsHeld(t *testing.T) {
	parent := t.TempDir()
	dir := mkdir(t, filepath.Join(parent, "s-torn"))
	if err := os.WriteFile(filepath.Join(dir, HolderFile), []byte("pid=notanumber"), 0o644); err != nil {
		t.Fatal(err)
	}
	age(t, dir, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: true})
	if e := entryFor(t, rep, dir); e.Verdict != VerdictHeld || e.Removed {
		t.Fatalf("torn-holder dir verdict=%s removed=%v, want held", e.Verdict, e.Removed)
	}
	assertPresent(t, dir)
}

// fak-test:runtime fast est=10ms lane=default
func TestParseHolderRoundTrip(t *testing.T) {
	pid, _, ok := parseHolder(holderRecord(4242))
	if !ok || pid != 4242 {
		t.Fatalf("parseHolder(holderRecord(4242)) = %d,%v", pid, ok)
	}
	if _, _, ok := parseHolder([]byte("start=" + strconv.Itoa(1))); ok {
		t.Fatalf("record without pid parsed ok")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestSweepRechecksHolderAfterBeforeRemove(t *testing.T) {
	parent := t.TempDir()
	dir := mkdir(t, filepath.Join(parent, "s-stale"))
	makeTree(t, dir)
	age(t, dir, 48*time.Hour)
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: true, Measure: true,
		BeforeRemove: func(path string) {
			if err := os.WriteFile(filepath.Join(path, HolderFile), holderRecord(os.Getpid()), 0o644); err != nil {
				t.Fatal(err)
			}
		},
	})
	if rep.Removed != 0 || rep.Stale != 0 || rep.Held != 1 || rep.StaleBytes != 0 {
		t.Fatalf("removed=%d stale=%d held=%d staleBytes=%d", rep.Removed, rep.Stale, rep.Held, rep.StaleBytes)
	}
	if e := entryFor(t, rep, dir); e.Verdict != VerdictHeld || e.Removed {
		t.Fatalf("verdict=%s removed=%v", e.Verdict, e.Removed)
	}
	assertPresent(t, dir)
}

// fak-test:runtime fast est=20ms lane=default
func TestIsHeldFailsClosedForUnreadableRecord(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, HolderFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if !IsHeld(dir, 30*24*time.Hour) {
		t.Fatal("unreadable holder must retain directory")
	}
}
