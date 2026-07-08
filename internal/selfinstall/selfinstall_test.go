package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// scriptRunner fails the FIRST command whose joined argv contains failOn; everything else
// succeeds. It records the commands it ran so a test can assert the ladder stopped early.
type scriptRunner struct {
	failOn string
	ran    [][]string
}

func (s *scriptRunner) run(_ context.Context, _, name string, args ...string) (string, bool) {
	joined := name + " " + strings.Join(args, " ")
	s.ran = append(s.ran, append([]string{name}, args...))
	if s.failOn != "" && strings.Contains(joined, s.failOn) {
		return "boom: " + s.failOn, false
	}
	return "ok", true
}

func TestInstallHappyPathSwapsOnAllGreen(t *testing.T) {
	r := &scriptRunner{}
	swapped := ""
	swap := func(src, dst string) error { swapped = dst; return nil }

	res := Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	if !res.Installed || res.Stage != StageSwap {
		t.Fatalf("got %+v, want Installed at swap", res)
	}
	if swapped != "/bin/fak" {
		t.Fatalf("swap target = %q, want /bin/fak", swapped)
	}
	// Ladder must have run build, vet, smoke (the tmp binary), then swapped.
	if len(r.ran) != 3 {
		t.Fatalf("ran %d commands, want 3 (build/vet/smoke); got %v", len(r.ran), r.ran)
	}
}

func TestInstallStopsAtFailingGateAndDoesNotSwap(t *testing.T) {
	for _, c := range []struct {
		failOn string
		stage  Stage
		nRan   int // commands attempted before the stop
	}{
		{"build", StageBuild, 1},
		{"vet", StageVet, 2},
		{"version", StageSmoke, 3}, // the smoke command is `<tmp> version`
	} {
		t.Run(string(c.stage), func(t *testing.T) {
			r := &scriptRunner{failOn: c.failOn}
			swapCalled := false
			swap := func(src, dst string) error { swapCalled = true; return nil }

			res := Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
			if res.Installed {
				t.Fatalf("%s gate failed but Install reported Installed", c.stage)
			}
			if res.Stage != c.stage {
				t.Fatalf("stage = %v, want %v", res.Stage, c.stage)
			}
			if swapCalled {
				t.Fatalf("%s gate failed but swap was still called — a non-green binary must NEVER be installed", c.stage)
			}
			if len(r.ran) != c.nRan {
				t.Fatalf("ran %d commands, want %d (ladder should stop at the failing gate)", len(r.ran), c.nRan)
			}
		})
	}
}

func TestInstallReportsSwapFailure(t *testing.T) {
	r := &scriptRunner{} // all gates pass
	swap := func(src, dst string) error { return errSwap }

	res := Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	if res.Installed || res.Stage != StageSwap {
		t.Fatalf("got %+v, want not-installed at swap stage", res)
	}
	if !strings.Contains(res.Detail, "swap-fail") {
		t.Fatalf("detail = %q, want the swap error surfaced", res.Detail)
	}
}

func TestBuildTmpDefaultsToTargetSibling(t *testing.T) {
	var builtTo string
	r := &recordTmp{}
	swap := func(src, dst string) error { return nil }
	_ = Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	builtTo = r.buildOut
	if builtTo != "/bin/fak.new" {
		t.Fatalf("build -o = %q, want /bin/fak.new (sibling default)", builtTo)
	}
}

// recordTmp captures the `-o <path>` the build stage used.
type recordTmp struct{ buildOut string }

func (r *recordTmp) run(_ context.Context, _, name string, args ...string) (string, bool) {
	if name == "go" && len(args) >= 3 && args[0] == "build" && args[1] == "-o" {
		r.buildOut = args[2]
	}
	return "ok", true
}

func TestPrepareOriginAddsAndCleansWorktree(t *testing.T) {
	r := &scriptRunner{}
	dir, cleanup, err := PrepareOrigin(context.Background(), r.run, "/repo", "origin/main", "/repo/.wt")
	if err != nil {
		t.Fatalf("PrepareOrigin err: %v", err)
	}
	if dir != "/repo/.wt" {
		t.Fatalf("dir = %q, want /repo/.wt", dir)
	}
	// It should have fetched then added a detached worktree.
	sawFetch, sawAdd := false, false
	for _, c := range r.ran {
		j := strings.Join(c, " ")
		if strings.Contains(j, "git fetch origin") {
			sawFetch = true
		}
		if strings.Contains(j, "worktree add --detach /repo/.wt origin/main") {
			sawAdd = true
		}
	}
	if !sawFetch || !sawAdd {
		t.Fatalf("prepare did not fetch+add detached worktree; ran %v", r.ran)
	}
	// Cleanup must remove + prune the worktree.
	cleanup()
	sawRemove, sawPrune := false, false
	for _, c := range r.ran {
		j := strings.Join(c, " ")
		if strings.Contains(j, "worktree remove --force /repo/.wt") {
			sawRemove = true
		}
		if strings.Contains(j, "worktree prune") {
			sawPrune = true
		}
	}
	if !sawRemove || !sawPrune {
		t.Fatalf("cleanup did not remove+prune; ran %v", r.ran)
	}
}

func TestPrepareOriginReportsAddFailure(t *testing.T) {
	r := &scriptRunner{failOn: "worktree add"}
	_, _, err := PrepareOrigin(context.Background(), r.run, "/repo", "origin/main", "/repo/.wt")
	if err == nil {
		t.Fatal("PrepareOrigin should return an error when worktree add fails")
	}
}

func TestSingleFlightSecondCallIsBusy(t *testing.T) {
	dir := t.TempDir()
	rel1, err := TrySingleFlight(dir)
	if err != nil {
		t.Fatalf("first TrySingleFlight: %v", err)
	}
	// While the first holds the lock, a second must report ErrBusy, not block or steal.
	if _, err := TrySingleFlight(dir); err != ErrBusy {
		t.Fatalf("second TrySingleFlight while held: got %v, want ErrBusy", err)
	}
	rel1()
	// After release, a new acquire succeeds.
	rel2, err := TrySingleFlight(dir)
	if err != nil {
		t.Fatalf("TrySingleFlight after release: %v", err)
	}
	rel2()
}

func TestWindowsSwapAsidePathUsesPlainOldWhenFree(t *testing.T) {
	got := windowsSwapAsidePath(`C:\work\fak\fak.exe`, 42, func(string) bool { return false })
	if got != `C:\work\fak\fak.exe.old` {
		t.Fatalf("aside path = %q, want plain .old", got)
	}
}

func TestWindowsSwapAsidePathSkipsHeldOldName(t *testing.T) {
	existing := map[string]bool{
		`C:\work\fak\fak.exe.old`:      true,
		`C:\work\fak\fak.exe.old.42.0`: true,
	}
	got := windowsSwapAsidePath(`C:\work\fak\fak.exe`, 42, func(path string) bool {
		return existing[path]
	})
	if got != `C:\work\fak\fak.exe.old.42.1` {
		t.Fatalf("aside path = %q, want first free unique old name", got)
	}
}

func TestPidFromAsideParsesOnlyProducerNames(t *testing.T) {
	const dstBase = "fak.exe"
	// A name windowsSwapAsidePath produces round-trips to its pid.
	if pid, ok := pidFromAside(dstBase, "fak.exe.old.15472.0"); !ok || pid != 15472 {
		t.Fatalf("pidFromAside(.old.15472.0) = (%d,%v), want (15472,true)", pid, ok)
	}
	// Everything that is NOT a "<base>.old.<pid>.<i>" name must be rejected, so the reaper
	// never touches the live binary, its plain .old, or a hand-made backup.
	for _, bad := range []string{
		"fak.exe",                 // the live binary
		"fak.exe.old",             // the plain aside (may still be mapped / conventional)
		"fak.exe.old-20260703",    // manual dated backup
		"fak.exe.old-499587c9",    // manual sha backup
		"fak.exe.old.42.overflow", // the overflow sentinel is not a numeric index
		"fak.exe.old.-1.0",        // non-positive pid
		"fak.exe.old.abc.0",       // non-numeric pid
		"fak.exe.old.42",          // missing index segment
		"other.exe.old.42.0",      // a different binary's aside
		"fak.exe.new",             // the in-flight build candidate
	} {
		if pid, ok := pidFromAside(dstBase, bad); ok {
			t.Fatalf("pidFromAside(%q) = (%d,true), want not-ok", bad, pid)
		}
	}
}

func TestReapStaleAsidesRemovesOnlyDeadOwnerAsides(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak.exe")
	const selfPID = 1000
	alive := func(pid int) bool { return pid == 2000 } // only pid 2000 still building

	// Lay down: the live binary, its plain aside, two hand-made backups, and asides owned by
	// a dead pid (reap), a live pid (keep), and our own pid (keep).
	files := []string{
		"fak.exe",
		"fak.exe.old",
		"fak.exe.old-20260703",
		"fak.exe.old.3000.0", // dead owner -> REAP
		"fak.exe.old.3000.1", // dead owner -> REAP
		"fak.exe.old.2000.0", // live owner -> keep
		"fak.exe.old.1000.0", // our own pid -> keep
	}
	for _, name := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatalf("seed %s: %v", name, err)
		}
	}

	reaped := ReapStaleAsides(target, selfPID, alive)

	want := map[string]bool{
		filepath.Join(dir, "fak.exe.old.3000.0"): true,
		filepath.Join(dir, "fak.exe.old.3000.1"): true,
	}
	if len(reaped) != len(want) {
		t.Fatalf("reaped %v, want exactly the two dead-owner asides", reaped)
	}
	for _, p := range reaped {
		if !want[p] {
			t.Fatalf("reaped unexpected path %q", p)
		}
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("reaped path %q still on disk (stat err %v)", p, err)
		}
	}
	// Everything not owned by a dead pid must survive.
	for _, keep := range []string{"fak.exe", "fak.exe.old", "fak.exe.old-20260703", "fak.exe.old.2000.0", "fak.exe.old.1000.0"} {
		if _, err := os.Stat(filepath.Join(dir, keep)); err != nil {
			t.Fatalf("expected %s to survive, stat err %v", keep, err)
		}
	}
}

func TestReapStaleAsidesNoopWhenDirMissing(t *testing.T) {
	target := filepath.Join(t.TempDir(), "gone", "fak.exe")
	if got := ReapStaleAsides(target, 1, func(int) bool { return false }); got != nil {
		t.Fatalf("reaped %v, want nil when install dir is unreadable", got)
	}
}

var errSwap = swapErr("swap-fail")

type swapErr string

func (e swapErr) Error() string { return string(e) }
