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
