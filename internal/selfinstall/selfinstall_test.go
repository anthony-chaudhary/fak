package selfinstall

import (
	"context"
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

var errSwap = swapErr("swap-fail")

type swapErr string

func (e swapErr) Error() string { return string(e) }
