package selfinstall

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
	if s.failOn != "" && strings.Contains(joined, s.failOn) && !(s.failOn == "version" && name == "go") {
		if name == "git" && len(args) >= 4 && args[0] == "worktree" && args[1] == "add" {
			// Model git leaving a partial directory before reporting add failure.
			_ = os.MkdirAll(args[3], 0o755)
		}
		return "boom: " + s.failOn, false
	}
	if name == "git" && len(args) >= 4 && args[0] == "worktree" && args[1] == "add" {
		_ = os.MkdirAll(args[3], 0o755)
	}
	if name == "git" && len(args) >= 4 && args[0] == "worktree" && args[1] == "remove" {
		_ = os.RemoveAll(args[3])
	}
	if name == "git" && len(args) == 2 && args[0] == "status" && args[1] == "--porcelain" {
		return "", true
	}
	if name == "git" && len(args) == 2 && args[0] == "rev-parse" && args[1] == "HEAD" {
		return "0123456789abcdef0123456789abcdef01234567\n", true
	}
	// The smoke stage runs `<tmp> version`; a real freshly-built candidate reports a VCS
	// stamp, which the fail-closed provenance gate (#3350) now requires before the swap.
	if len(args) > 0 && args[0] == "version" {
		return "fak version 9.9.9\nbuild: 1a2b3c4d5e6f a stamped candidate", true
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
	if len(r.ran) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST — closed fixture/contract cardinality
		t.Fatalf("ran %d commands, want 5 (cleanliness/revision/build/vet/smoke); got %v", len(r.ran), r.ran)
	}
}

func TestInstallStopsAtFailingGateAndDoesNotSwap(t *testing.T) {
	for _, c := range []struct {
		failOn string
		stage  Stage
		nRan   int // commands attempted before the stop
	}{
		{"build", StageBuild, 3},
		{"vet", StageVet, 4},
		{"version", StageSmoke, 5}, // the smoke command is `<tmp> version`
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
	// Find the `-o <path>` pair wherever it sits — the build now leads with -buildvcs=true
	// (and may carry -ldflags), so -o is no longer at a fixed index.
	if name == "git" && len(args) == 2 && args[0] == "status" && args[1] == "--porcelain" {
		return "", true
	}
	if name == "git" && len(args) == 2 && args[0] == "rev-parse" && args[1] == "HEAD" {
		return "0123456789abcdef0123456789abcdef01234567\n", true
	}
	if name == "go" && len(args) > 0 && args[0] == "build" {
		for i := 0; i+1 < len(args); i++ {
			if args[i] == "-o" {
				r.buildOut = args[i+1]
				break
			}
		}
	}
	if len(args) > 0 && args[0] == "version" {
		return "fak version 9.9.9\nbuild: 1a2b3c4d5e6f", true
	}
	return "ok", true
}

// recordBuild captures the args of the `go build` invocation so a test can assert what
// ldflags (if any) the build stage passed.
type recordBuild struct{ buildArgs []string }

func (r *recordBuild) run(_ context.Context, _, name string, args ...string) (string, bool) {
	if name == "git" && len(args) == 2 && args[0] == "status" && args[1] == "--porcelain" {
		return "", true
	}
	if name == "git" && len(args) == 2 && args[0] == "rev-parse" && args[1] == "HEAD" {
		return "0123456789abcdef0123456789abcdef01234567\n", true
	}
	if name == "go" && len(args) > 0 && args[0] == "build" {
		r.buildArgs = append([]string{}, args...)
	}
	if len(args) > 0 && args[0] == "version" {
		return "fak version 9.9.9\nbuild: 1a2b3c4d5e6f", true
	}
	return "ok", true
}

func TestInstallBakesVersionLdflagsWhenVersionFilePresent(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "VERSION"), []byte("9.9.9\n"), 0o644); err != nil {
		t.Fatalf("seed VERSION: %v", err)
	}
	r := &recordBuild{}
	swap := func(src, dst string) error { return nil }

	res := Install(context.Background(), r.run, swap, Options{RepoRoot: repo, Target: filepath.Join(repo, "fak")})
	if !res.Installed {
		t.Fatalf("got %+v, want Installed", res)
	}
	joined := strings.Join(r.buildArgs, " ")
	const want = "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=9.9.9"
	if !strings.Contains(joined, "-ldflags") || !strings.Contains(joined, want) {
		t.Fatalf("build args = %v, want -ldflags carrying %q (so the installed binary's version does not depend on a guard's cwd)", r.buildArgs, want)
	}
	// The package + output target must still be present after the ldflags.
	if !strings.Contains(joined, "-o") || !strings.HasSuffix(joined, "./cmd/fak") {
		t.Fatalf("build args = %v, want a well-formed `-o <tmp> ./cmd/fak` tail", r.buildArgs)
	}
}

func TestInstallStillBakesCommitWhenNoVersionFile(t *testing.T) {
	repo := t.TempDir() // deliberately no VERSION file
	r := &recordBuild{}
	swap := func(src, dst string) error { return nil }

	Install(context.Background(), r.run, swap, Options{RepoRoot: repo, Target: filepath.Join(repo, "fak")})
	joined := strings.Join(r.buildArgs, " ")
	const want = "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildCommit=0123456789abcdef0123456789abcdef01234567"
	if !strings.Contains(joined, want) {
		t.Fatalf("build args = %v, want injected commit provenance %q even without VERSION", r.buildArgs, want)
	}
}

func TestPrepareOriginAddsAndCleansWorktree(t *testing.T) {
	r := &scriptRunner{}
	repo := t.TempDir()
	wt := filepath.Join(t.TempDir(), "wt")
	dir, cleanup, err := PrepareOrigin(context.Background(), r.run, repo, "origin/main", wt)
	if err != nil {
		t.Fatalf("PrepareOrigin err: %v", err)
	}
	if dir != wt {
		t.Fatalf("dir = %q, want %q", dir, wt)
	}
	// It should have fetched then added a detached worktree.
	sawFetch, sawAdd := false, false
	for _, c := range r.ran {
		j := strings.Join(c, " ")
		if strings.Contains(j, "git fetch origin") {
			sawFetch = true
		}
		if strings.Contains(j, "worktree add --detach "+wt+" origin/main") {
			sawAdd = true
		}
	}
	if !sawFetch || !sawAdd {
		t.Fatalf("prepare did not fetch+add detached worktree; ran %v", r.ran)
	}
	stamp, present, err := readBuildOwnerStamp(wt)
	if err != nil || !present {
		t.Fatalf("owner stamp missing/unreadable: present=%v err=%v", present, err)
	}
	if stamp.PID != os.Getpid() || stamp.LeaseID == "" || stamp.CreatedAt.IsZero() {
		t.Fatalf("owner stamp = %+v, want current pid + lease + created_at", stamp)
	}
	// Cleanup must remove + prune the worktree.
	cleanup()
	cleanup() // idempotent: explicit-before-exit plus deferred cleanup is safe.
	sawRemove, sawPrune := false, false
	removeCount, pruneCount := 0, 0
	for _, c := range r.ran {
		j := strings.Join(c, " ")
		if strings.Contains(j, "worktree remove --force "+wt) {
			sawRemove = true
			removeCount++
		}
		if strings.Contains(j, "worktree prune") {
			sawPrune = true
			pruneCount++
		}
	}
	if !sawRemove || !sawPrune {
		t.Fatalf("cleanup did not remove+prune; ran %v", r.ran)
	}
	if removeCount != 1 || pruneCount != 1 {
		t.Fatalf("idempotent cleanup ran remove=%d prune=%d, want 1 each: %v", removeCount, pruneCount, r.ran)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("cleanup left worktree directory: %v", err)
	}
	if _, err := os.Stat(BuildOwnerStampPath(wt)); !os.IsNotExist(err) {
		t.Fatalf("cleanup left owner stamp: %v", err)
	}
}

func TestPrepareOriginReportsAddFailure(t *testing.T) {
	r := &scriptRunner{failOn: "worktree add"}
	repo := t.TempDir()
	wt := filepath.Join(t.TempDir(), "wt")
	_, _, err := PrepareOrigin(context.Background(), r.run, repo, "origin/main", wt)
	if err == nil {
		t.Fatal("PrepareOrigin should return an error when worktree add fails")
	}
	if _, statErr := os.Stat(wt); !os.IsNotExist(statErr) {
		t.Fatalf("partial add failure leaked %q: %v", wt, statErr)
	}
	if _, statErr := os.Stat(BuildOwnerStampPath(wt)); !os.IsNotExist(statErr) {
		t.Fatalf("partial add failure leaked owner stamp: %v", statErr)
	}
	sawRemove, sawPrune := false, false
	for _, c := range r.ran {
		j := strings.Join(c, " ")
		sawRemove = sawRemove || strings.Contains(j, "worktree remove --force "+wt)
		sawPrune = sawPrune || strings.Contains(j, "worktree prune")
	}
	if !sawRemove || !sawPrune {
		t.Fatalf("partial add failure did not run source cleanup: %v", r.ran)
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

func TestMeasureAsidesCountsOnlyAsidesAndTagsDeadOwners(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak.exe")
	const selfPID = 1000
	alive := func(pid int) bool { return pid == 2000 } // only 2000 still live

	// Sizes chosen so the byte tallies are unambiguous.
	seed := map[string]int{
		"fak.exe":              50, // live binary — not an aside
		"fak.exe.old":          40, // plain aside — not the .old.<pid>.<i> shape
		"fak.exe.old-20260703": 30, // manual backup — not counted
		"fak.exe.old.3000.0":   10, // dead owner -> counted + reclaimable
		"fak.exe.old.4000.0":   10, // dead owner -> counted + reclaimable
		"fak.exe.old.2000.0":   10, // live owner -> counted, NOT reclaimable
		"fak.exe.old.1000.0":   10, // our own pid -> counted, NOT reclaimable
	}
	for name, n := range seed {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, n), 0o644); err != nil {
			t.Fatalf("seed %s: %v", name, err)
		}
	}

	fp := MeasureAsides(target, selfPID, alive)

	if fp.Count != 4 {
		t.Fatalf("Count = %d, want 4 (the four .old.<pid>.<i> files)", fp.Count)
	}
	if fp.Bytes != 40 {
		t.Fatalf("Bytes = %d, want 40 (4x10, only the asides)", fp.Bytes)
	}
	if fp.DeadCount != 2 {
		t.Fatalf("DeadCount = %d, want 2 (pids 3000 and 4000)", fp.DeadCount)
	}
	if fp.DeadBytes != 20 {
		t.Fatalf("DeadBytes = %d, want 20 (2x10 dead-owner asides)", fp.DeadBytes)
	}
}

func TestMeasureAsidesEmptyWhenDirMissing(t *testing.T) {
	target := filepath.Join(t.TempDir(), "gone", "fak.exe")
	fp := MeasureAsides(target, 1, func(int) bool { return false })
	if fp.Count != 0 || fp.Bytes != 0 || fp.DeadCount != 0 {
		t.Fatalf("footprint = %+v, want zero when install dir is unreadable", fp)
	}
}

var errSwap = swapErr("swap-fail")

type swapErr string

func (e swapErr) Error() string { return string(e) }

// stampRunner models a build ladder whose freshly-built candidate reports versionOut when
// run as `<tmp> version`; every other command (build/vet) succeeds with "ok". It lets a test
// drive the smoke gate against a stamped vs unstamped candidate deterministically.
type stampRunner struct {
	versionOut string
	ran        [][]string
}

func (s *stampRunner) run(_ context.Context, _, name string, args ...string) (string, bool) {
	s.ran = append(s.ran, append([]string{name}, args...))
	if len(args) > 0 && args[0] == "version" {
		return s.versionOut, true
	}
	return "ok", true
}

// TestInstallBuildForcesVCSStamp pins that the build stage forces -buildvcs=true (#3350): a
// self-update build must FAIL rather than silently emit an unstamped binary. Under Go's
// default -buildvcs=auto the detached-worktree build can drop the VCS stamp entirely, so the
// installed guard cannot attest which commit it is — the exact provenance blind spot G2 of
// epic #2218. Red before the fix (no -buildvcs), green after.
func TestInstallBuildForcesVCSStamp(t *testing.T) {
	r := &recordBuild{}
	swap := func(src, dst string) error { return nil }
	Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	joined := strings.Join(r.buildArgs, " ")
	if !strings.Contains(joined, "-buildvcs=true") {
		t.Fatalf("build args = %v, want -buildvcs=true so the build FAILS instead of shipping an unstamped binary (#3350)", r.buildArgs)
	}
	// -buildvcs=true must not displace the well-formed `-o <tmp> ./cmd/fak` tail.
	if !strings.Contains(joined, "-o") || !strings.HasSuffix(joined, "./cmd/fak") {
		t.Fatalf("build args = %v, want a well-formed `-o <tmp> ./cmd/fak` tail", r.buildArgs)
	}
}

// TestInstallSmokeRejectsUnstampedCandidate pins the fail-CLOSED provenance gate (#3350): a
// candidate that builds, vets, and runs but reports NO VCS stamp must NOT be swapped over the
// running fleet. Before the fix the smoke stage checked only the exit status, so an unstamped
// binary (still exits 0 on `version`) passed and swapped in — indistinguishable from a good
// one, blinding every downstream freshness/skew check. Red before, green after.
func TestInstallSmokeRejectsUnstampedCandidate(t *testing.T) {
	r := &stampRunner{versionOut: "fak version 9.9.9\nbuild: (no VCS stamp — built without module/VCS provenance; cannot confirm the commit)"}
	swapCalled := false
	swap := func(src, dst string) error { swapCalled = true; return nil }

	res := Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	if res.Installed {
		t.Fatalf("an UNSTAMPED candidate must NOT be installed; got %+v", res)
	}
	if res.Stage != StageSmoke {
		t.Fatalf("stage = %v, want StageSmoke — the gate must fail closed on a missing VCS stamp", res.Stage)
	}
	if swapCalled {
		t.Fatal("swap was called for an unstamped candidate — the provenance gate is fail-OPEN")
	}
}

// TestInstallSmokeAcceptsStampedCandidate is the positive twin of the reject test: a
// candidate that builds, vets, runs, AND reports a real VCS stamp passes the smoke gate and
// swaps in. It guards the fail-closed gate against over-rejecting a genuinely-stamped binary.
func TestInstallSmokeAcceptsStampedCandidate(t *testing.T) {
	r := &stampRunner{versionOut: "fak version 9.9.9\nbuild: 499587c9deadbeef0011 (committed)"}
	swapped := ""
	swap := func(src, dst string) error { swapped = dst; return nil }

	res := Install(context.Background(), r.run, swap, Options{RepoRoot: "/repo", Target: "/bin/fak"})
	if !res.Installed || res.Stage != StageSwap {
		t.Fatalf("a STAMPED candidate that passes every gate must install; got %+v", res)
	}
	if swapped != "/bin/fak" {
		t.Fatalf("swap target = %q, want /bin/fak", swapped)
	}
}

// TestVersionOutputStamped pins the provenance parse the smoke gate reads: a real revision is
// stamped; the "(no VCS stamp …)" and "module vX" sentinels, a bare build line, and output
// with no build line at all are NOT — the gate must fail closed on each.
func TestVersionOutputStamped(t *testing.T) {
	for _, c := range []struct {
		name string
		out  string
		want bool
	}{
		{"real rev", "fak version 9.9.9\nbuild: 499587c9deadbeef", true},
		{"real rev dirty", "build: 499587c9deadbeef +uncommitted", true},
		{"no vcs stamp", "build: (no VCS stamp — built without module/VCS provenance; cannot confirm the commit)", false},
		{"module version", "build: module v0.1.1", false},
		{"bare build line", "build:", false},
		{"no build line", "fak version 9.9.9\ngo: go1.26", false},
		{"empty", "", false},
	} {
		if got := versionOutputStamped(c.out); got != c.want {
			t.Errorf("%s: versionOutputStamped(%q) = %v, want %v", c.name, c.out, got, c.want)
		}
	}
}

func TestInstallVerifiedCopyCopiesExactBytesAndSwaps(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "verified-fak")
	target := filepath.Join(dir, "hot-copy")
	want := []byte("one gated artifact for every hot copy")
	if err := os.WriteFile(source, want, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}

	var candidate string
	res := InstallVerifiedCopy(func(tmp, dst string) error {
		candidate = tmp
		if dst != target {
			t.Fatalf("swap target = %q, want %q", dst, target)
		}
		got, err := os.ReadFile(tmp)
		if err != nil {
			return err
		}
		if string(got) != string(want) {
			t.Fatalf("candidate bytes = %q, want %q", got, want)
		}
		return os.Rename(tmp, dst)
	}, source, target)
	if !res.Installed || res.Stage != StageSwap {
		t.Fatalf("result = %+v, want installed swap", res)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("installed bytes = %q, want %q", got, want)
	}
	if _, err := os.Stat(candidate); !os.IsNotExist(err) {
		t.Fatalf("candidate still exists after swap: %v", err)
	}
}

func TestInstallVerifiedCopyLeavesTargetOnSwapFailure(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "verified-fak")
	target := filepath.Join(dir, "hot-copy")
	if err := os.WriteFile(source, []byte("verified"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("stale but usable"), 0o755); err != nil {
		t.Fatal(err)
	}

	res := InstallVerifiedCopy(func(_, _ string) error { return os.ErrPermission }, source, target)
	if res.Installed || res.Stage != StageSwap {
		t.Fatalf("result = %+v, want failed swap", res)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "stale but usable" {
		t.Fatalf("target changed after failed swap: %q", got)
	}
}

const (
	cacheTestCommitA = "0123456789abcdef0123456789abcdef01234567"
	cacheTestCommitB = "89abcdef0123456789abcdef0123456789abcdef"
)

// candidateCacheRunner models the entire expensive gate with a virtual clock. The latency
// table never changes between cold and warm runs, so timing assertions measure eliminated
// work rather than scheduler noise. Smoke identities may be queued to model a digest-valid
// cache entry whose executable provenance is nevertheless wrong.
type candidateCacheRunner struct {
	commit        string
	goEnv         string
	artifact      []byte
	builds        int
	vets          int
	smokes        int
	elapsed       time.Duration
	smokeCommits  []string
	lastBuildPath string
}

func newCandidateCacheRunner() *candidateCacheRunner {
	return &candidateCacheRunner{
		commit: cacheTestCommitA,
		goEnv: `go1.26.0
/toolchain/go1.26
auto
windows
amd64
windows
amd64
`,
		artifact: []byte("exact candidate bytes"),
	}
}

func (r *candidateCacheRunner) run(_ context.Context, _ string, name string, args ...string) (string, bool) {
	switch {
	case name == "git" && len(args) == 2 && args[0] == "status":
		r.elapsed += 25 * time.Millisecond
		return "", true
	case name == "git" && len(args) == 2 && args[0] == "rev-parse":
		r.elapsed += 25 * time.Millisecond
		return r.commit + "\n", true
	case name == "go" && len(args) > 0 && args[0] == "env":
		r.elapsed += 50 * time.Millisecond
		return r.goEnv, true
	case name == "go" && len(args) > 0 && args[0] == "build":
		r.elapsed += 5 * time.Second
		r.builds++
		for i := 0; i+1 < len(args); i++ {
			if args[i] != "-o" {
				continue
			}
			r.lastBuildPath = args[i+1]
			if err := os.WriteFile(args[i+1], r.artifact, 0o755); err != nil {
				return err.Error(), false
			}
			return "", true
		}
		return "build has no output path", false
	case name == "go" && len(args) > 0 && args[0] == "vet":
		r.elapsed += 5 * time.Second
		r.vets++
		return "", true
	case len(args) == 2 && args[0] == "version" && args[1] == "--json":
		r.elapsed += 100 * time.Millisecond
		r.smokes++
		commit := r.commit
		if len(r.smokeCommits) > 0 {
			commit = r.smokeCommits[0]
			r.smokeCommits = r.smokeCommits[1:]
		}
		out, err := json.Marshal(candidateVersionIdentity{Commit: commit, Stamped: true})
		if err != nil {
			return err.Error(), false
		}
		return string(out), true
	default:
		return "unexpected command", false
	}
}

// runCompleteCacheTransaction follows the production shape: Install captures the freshly
// gated (or cache-restored) candidate, then RunTransaction stages, snapshots, and activates
// that one candidate across every selected stale target.
func runCompleteCacheTransaction(t *testing.T, r *candidateCacheRunner, opts Options, targets ...string) Result {
	t.Helper()
	var candidate string
	res := Install(context.Background(), r.run, func(src, _ string) error {
		candidate = src
		return nil
	}, opts)
	if !res.Installed || candidate == "" {
		return res
	}
	defer os.Remove(candidate)

	copies := make([]Copy, 0, len(targets))
	for _, target := range targets {
		copies = append(copies, Copy{Source: candidate, Target: target})
	}
	transaction := RunTransaction(copies, func(src, dst string) error {
		r.elapsed += 25 * time.Millisecond
		return OSSwap(src, dst)
	})
	if got, ok := transaction.(Updated); !ok || got.Changed != len(targets) {
		t.Fatalf("transaction = %#v, want Updated across %d targets", transaction, len(targets))
	}
	return res
}

func seedCacheTransactionTargets(t *testing.T, dir string, n int) []string {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.test/candidate-cache\n\ngo 1.26\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cmd", "fak", "main.go"), []byte("package main\nfunc main() {}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	targets := make([]string, 0, n)
	for i := 0; i < n; i++ {
		target := filepath.Join(dir, "stale-"+string(rune('a'+i)))
		if err := os.WriteFile(target, []byte("stale"), 0o755); err != nil {
			t.Fatal(err)
		}
		targets = append(targets, target)
	}
	return targets
}

func TestInstallVerifiedCandidateCacheCompleteTransactionCutsSameEnvelopeFivefold(t *testing.T) {
	dir := t.TempDir()
	r := newCandidateCacheRunner()
	targets := seedCacheTransactionTargets(t, dir, 3)
	opts := Options{
		RepoRoot:       dir,
		Target:         targets[0],
		BuildTmp:       filepath.Join(dir, "candidate"),
		CacheDir:       filepath.Join(dir, "cache"),
		ExpectedCommit: cacheTestCommitA,
	}

	before := r.elapsed
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed {
		t.Fatalf("cold transaction: %+v", got)
	}
	cold := r.elapsed - before
	before = r.elapsed
	got := runCompleteCacheTransaction(t, r, opts, targets...)
	warm := r.elapsed - before
	if !got.Installed || !strings.Contains(got.Detail, "build-input verified candidate cache") {
		t.Fatalf("warm transaction: %+v", got)
	}
	if r.builds != 1 || r.vets != 1 || r.smokes != 2 {
		t.Fatalf("builds=%d vets=%d smokes=%d, want one cold build/vet and smoke on both transactions", r.builds, r.vets, r.smokes)
	}
	ratio := float64(cold) / float64(warm)
	t.Logf("complete stale-update transaction: cold=%s warm=%s speedup=%.2fx", cold, warm, ratio)
	if ratio < 5 {
		t.Fatalf("complete-transaction same-envelope speedup = %.2fx, want >=5x (cold=%s warm=%s)", ratio, cold, warm)
	}
}

func TestInstallCorruptCandidateCacheFallsBackAndAtomicallyRefreshes(t *testing.T) {
	dir := t.TempDir()
	r := newCandidateCacheRunner()
	targets := seedCacheTransactionTargets(t, dir, 1)
	cacheDir := filepath.Join(dir, "cache")
	opts := Options{RepoRoot: dir, Target: targets[0], BuildTmp: filepath.Join(dir, "candidate"), CacheDir: cacheDir, ExpectedCommit: cacheTestCommitA}
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed {
		t.Fatalf("seed transaction: %+v", got)
	}

	manifestData, err := os.ReadFile(candidateCachePaths(cacheDir))
	if err != nil {
		t.Fatal(err)
	}
	var manifest candidateCacheManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatal(err)
	}
	artifact := candidateArtifactPath(cacheDir, manifest.ArtifactDigest)
	if err := os.WriteFile(artifact, []byte("corrupt"), 0o755); err != nil {
		t.Fatal(err)
	}

	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed || strings.Contains(got.Detail, "from build-input") {
		t.Fatalf("corrupt-cache fallback transaction: %+v", got)
	}
	if r.builds != 2 || r.vets != 2 {
		t.Fatalf("builds=%d vets=%d, corrupt cache must rerun the complete build+vet gate", r.builds, r.vets)
	}
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !strings.Contains(got.Detail, "build-input verified candidate cache") {
		t.Fatalf("transaction after atomic refresh: %+v", got)
	}
	if r.builds != 2 || r.vets != 2 {
		t.Fatalf("builds=%d vets=%d after refreshed hit, want no third rebuild", r.builds, r.vets)
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.Contains(entry.Name(), ".tmp-") {
			t.Fatalf("atomic refresh leaked temporary cache entry %q", entry.Name())
		}
	}
}

func TestInstallCandidateCacheIdentityBindsRuntimeInputsAndReusesAcrossCommits(t *testing.T) {
	dir := t.TempDir()
	r := newCandidateCacheRunner()
	targets := seedCacheTransactionTargets(t, dir, 1)
	opts := Options{RepoRoot: dir, Target: targets[0], BuildTmp: filepath.Join(dir, "candidate"), CacheDir: filepath.Join(dir, "cache"), ExpectedCommit: cacheTestCommitA}

	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed || got.Reused {
		t.Fatalf("cold transaction: %+v", got)
	}
	if r.builds != 1 || r.vets != 1 {
		t.Fatalf("cold builds/vets = %d/%d, want 1/1", r.builds, r.vets)
	}

	// A different selected source commit with the same executable graph reuses the
	// already-verified artifact while preserving both provenance identities.
	r.commit = cacheTestCommitB
	r.smokeCommits = []string{cacheTestCommitA}
	opts.ExpectedCommit = cacheTestCommitB
	got := runCompleteCacheTransaction(t, r, opts, targets...)
	if !got.Installed || !got.Reused || got.SourceCommit != cacheTestCommitB || got.ArtifactSourceCommit != cacheTestCommitA {
		t.Fatalf("cross-commit reuse provenance: %+v", got)
	}
	if r.builds != 1 || r.vets != 1 {
		t.Fatalf("cross-commit reuse builds/vets = %d/%d, want 1/1", r.builds, r.vets)
	}
	if got.BuildInputDigest == "" || got.ArtifactDigest == "" || got.ArtifactSize == 0 || got.BuildEnvelope["GOVERSION"] == "" {
		t.Fatalf("cross-commit reuse omitted identities: %+v", got)
	}

	if err := os.WriteFile(filepath.Join(dir, "cmd", "fak", "main.go"), []byte("package main\nfunc main() { println(\"changed\") }\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got = runCompleteCacheTransaction(t, r, opts, targets...)
	if !got.Installed || got.Reused || r.builds != 2 || r.vets != 2 {
		t.Fatalf("runtime source change did not rebuild: %+v builds/vets=%d/%d", got, r.builds, r.vets)
	}

	// VERSION participates through the linker envelope, so it invalidates reuse.
	if err := os.WriteFile(filepath.Join(dir, "VERSION"), []byte("9.9.2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got = runCompleteCacheTransaction(t, r, opts, targets...)
	if !got.Installed || got.Reused {
		t.Fatalf("VERSION change transaction: %+v", got)
	}
	if r.builds != 3 || r.vets != 3 {
		t.Fatalf("VERSION change builds/vets = %d/%d, want 3/3", r.builds, r.vets)
	}
}

func TestInstallCandidateCacheToolchainChangeRebuilds(t *testing.T) {
	dir := t.TempDir()
	r := newCandidateCacheRunner()
	targets := seedCacheTransactionTargets(t, dir, 1)
	opts := Options{RepoRoot: dir, Target: targets[0], BuildTmp: filepath.Join(dir, "candidate"), CacheDir: filepath.Join(dir, "cache"), ExpectedCommit: cacheTestCommitA}
	original := runBuildInputCommand
	t.Cleanup(func() { runBuildInputCommand = original })
	toolchain := "go1.26.7"
	runBuildInputCommand = func(ctx context.Context, workDir string, env []string, args ...string) ([]byte, error) {
		out, err := original(ctx, workDir, env, args...)
		if err != nil || len(args) == 0 || args[0] != "env" {
			return out, err
		}
		var values map[string]string
		if err := json.Unmarshal(out, &values); err != nil {
			return nil, err
		}
		values["GOVERSION"] = toolchain
		return json.Marshal(values)
	}
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed || got.Reused {
		t.Fatalf("cold toolchain transaction: %+v", got)
	}
	toolchain = "go1.27.0"
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed || got.Reused {
		t.Fatalf("toolchain change reused stale artifact: %+v", got)
	}
	if r.builds != 2 || r.vets != 2 {
		t.Fatalf("toolchain change builds/vets = %d/%d, want 2/2", r.builds, r.vets)
	}
}

func TestInstallCandidateCacheProvenanceFailureFallsBackToFullGate(t *testing.T) {
	dir := t.TempDir()
	r := newCandidateCacheRunner()
	targets := seedCacheTransactionTargets(t, dir, 1)
	opts := Options{RepoRoot: dir, Target: targets[0], BuildTmp: filepath.Join(dir, "candidate"), CacheDir: filepath.Join(dir, "cache"), ExpectedCommit: cacheTestCommitA}
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed {
		t.Fatalf("seed transaction: %+v", got)
	}

	// The first identity belongs to the restored cache candidate and must be rejected. The
	// second belongs to the newly built candidate and permits activation only after build+vet.
	r.smokeCommits = []string{cacheTestCommitB, cacheTestCommitA}
	if got := runCompleteCacheTransaction(t, r, opts, targets...); !got.Installed || strings.Contains(got.Detail, "from build-input") {
		t.Fatalf("cached provenance fallback: %+v", got)
	}
	if r.builds != 2 || r.vets != 2 || r.smokes != 3 {
		t.Fatalf("builds=%d vets=%d smokes=%d, want cached smoke followed by full build/vet/fresh smoke", r.builds, r.vets, r.smokes)
	}
}

func TestInstallRejectsMalformedExplicitExpectedCommit(t *testing.T) {
	r := newCandidateCacheRunner()
	swapCalled := false
	res := Install(context.Background(), r.run, func(_, _ string) error {
		swapCalled = true
		return nil
	}, Options{RepoRoot: t.TempDir(), Target: filepath.Join(t.TempDir(), "fak"), ExpectedCommit: "01234567"})
	if res.Installed || res.Stage != StageSmoke || swapCalled {
		t.Fatalf("malformed exact commit must fail closed before build/swap: result=%+v swap=%v", res, swapCalled)
	}
	if r.builds != 0 || r.vets != 0 || r.smokes != 0 {
		t.Fatalf("malformed exact commit ran build/vet/smoke = %d/%d/%d", r.builds, r.vets, r.smokes)
	}
}

func TestInstallRejectsSourceThatDoesNotMatchExpectedCommit(t *testing.T) {
	r := newCandidateCacheRunner()
	res := Install(context.Background(), r.run, func(_, _ string) error {
		t.Fatal("swap called for the wrong source commit")
		return nil
	}, Options{RepoRoot: t.TempDir(), Target: filepath.Join(t.TempDir(), "fak"), CacheDir: t.TempDir(), ExpectedCommit: cacheTestCommitB})
	if res.Installed || res.Stage != StageSmoke || !strings.Contains(res.Detail, cacheTestCommitA) {
		t.Fatalf("source/selection mismatch must fail closed: %+v", res)
	}
	if r.builds != 0 || r.vets != 0 || r.smokes != 0 {
		t.Fatalf("source/selection mismatch ran build/vet/smoke = %d/%d/%d", r.builds, r.vets, r.smokes)
	}
}

func TestCandidateCacheDirUsesCloneSharedGitCommonDir(t *testing.T) {
	commonDir := filepath.Join(t.TempDir(), "primary.git")
	want := filepath.Join(commonDir, "fak", "self-update-cache")
	if got := CandidateCacheDir(commonDir); got != want {
		t.Fatalf("CandidateCacheDir(%q) = %q, want clone-shared %q", commonDir, got, want)
	}
	for _, unresolved := range []string{"", "  \t\r\n"} {
		if got := CandidateCacheDir(unresolved); got != "" {
			t.Fatalf("CandidateCacheDir(%q) = %q, want cache disabled rather than a relative worktree path", unresolved, got)
		}
	}
}
