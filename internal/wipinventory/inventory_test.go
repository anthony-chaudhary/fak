package wipinventory

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeRunner struct {
	root   string
	worker string
	mu     sync.Mutex
	calls  []string
}

func (f *fakeRunner) Run(dir string, args ...string) ([]byte, error) {
	f.mu.Lock()
	f.calls = append(f.calls, filepath.ToSlash(dir)+" git "+strings.Join(args, " "))
	f.mu.Unlock()
	key := args[0]
	if len(args) > 1 {
		key += " " + args[1]
	}
	switch key {
	case "rev-parse HEAD":
		return []byte("abc123\n"), nil
	case "rev-parse --git-path":
		return []byte(filepath.Join(f.root, ".git", "info", "exclude") + "\n"), nil
	case "status --porcelain=v1":
		if samePath(dir, f.worker) {
			return []byte(" M worker.go\x00?? new-worker.go\x00"), nil
		}
		return []byte(" M tracked.go\x00?? new.go\x00"), nil
	case "ls-files --others":
		return []byte("cache/a\x00cache/b\x00"), nil
	case "worktree list":
		return []byte("worktree " + filepath.ToSlash(f.root) + "\nHEAD abc123\nbranch refs/heads/main\n\nworktree " + filepath.ToSlash(f.worker) + "\nHEAD def456\ndetached\n"), nil
	case "for-each-ref --format=%(refname)%00%(objectname)%00%(creatordate:unix)":
		return []byte("refs/fak/wip/guard\x00feed00\x001700000000\n"), nil
	case "diff-tree --root":
		return []byte("M\ttracked.go\nA\tnew.go\n"), nil
	case "config --get", "config --bool":
		return nil, errors.New("unset")
	case "ls-files -v":
		return []byte("H tracked.go\nS hidden.go\n"), nil
	default:
		return nil, errors.New("unexpected: " + key)
	}
}

func TestCollectSeparatesEveryPopulationWithoutMutation(t *testing.T) {
	root := t.TempDir()
	workerRoot := filepath.Join(root, "workers")
	worker := filepath.Join(workerRoot, workerMarker+"-live")
	stale := filepath.Join(workerRoot, workerMarker+"-stale")
	for _, path := range []string{worker, stale, filepath.Join(root, ".git", "info")} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("cache/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".git", "info", "exclude"), []byte("*.tmp\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	before := treeState(t, root)
	f := &fakeRunner{root: root, worker: worker}
	rep := Collect(root, time.Unix(1700000100, 0), f, Options{WorkerRoot: workerRoot})
	after := treeState(t, root)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("read-only inventory mutated files\nbefore=%v\nafter=%v", before, after)
	}
	if rep.Schema != Schema || !rep.Main.Tracked.Known || rep.Main.Tracked.Count != 1 || rep.Main.Untracked.Count != 1 || rep.Ignored.Count != 2 {
		t.Fatalf("bad main populations: %#v", rep)
	}
	if len(rep.Worktrees) != 1 || rep.Worktrees[0].Tracked.Count != 1 || rep.Worktrees[0].Untracked.Count != 1 {
		t.Fatalf("bad workers: %#v", rep.Worktrees)
	}
	if len(rep.StaleWorkers) != 1 || rep.StaleWorkers[0].Kind != "unregistered-directory" {
		t.Fatalf("bad stale residue: %#v", rep.StaleWorkers)
	}
	if !rep.CheckpointsKnown || len(rep.Checkpoints) != 1 || rep.Checkpoints[0].Changed != 2 || rep.Checkpoints[0].Added != 1 {
		t.Fatalf("bad checkpoints: %#v", rep.Checkpoints)
	}
	if rep.IgnoreInputs.HiddenIndex != 1 || rep.IgnoreInputs.GitignoreHash == "" || rep.IgnoreInputs.ExcludeHash == "" {
		t.Fatalf("bad visibility: %#v", rep.IgnoreInputs)
	}
	for _, call := range f.calls {
		for _, mutator := range []string{" update-ref ", " add ", " commit ", " worktree remove", " clean ", " reset "} {
			if strings.Contains(call, mutator) {
				t.Fatalf("mutating git call: %s", call)
			}
		}
	}
}

func TestCollectMarksStatFailureUnknown(t *testing.T) {
	root := t.TempDir()
	workerRoot := filepath.Join(root, "workers")
	f := &fakeRunner{root: root, worker: filepath.Join(workerRoot, workerMarker+"-live")}
	rep := Collect(root, time.Unix(1700000100, 0), f, Options{WorkerRoot: workerRoot})
	if rep.Main.Untracked.Known || rep.Main.Untracked.Error == "" {
		t.Fatalf("stat failure was not provenance-visible: %#v", rep.Main.Untracked)
	}
}

func TestCollectIsDeterministicExceptObservationTime(t *testing.T) {
	root := t.TempDir()
	workerRoot := filepath.Join(root, "workers")
	worker := filepath.Join(workerRoot, workerMarker+"-live")
	if err := os.MkdirAll(worker, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1700000100, 0)
	a := Collect(root, now, &fakeRunner{root: root, worker: worker}, Options{WorkerRoot: workerRoot})
	b := Collect(root, now, &fakeRunner{root: root, worker: worker}, Options{WorkerRoot: workerRoot})
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("reports differ\na=%#v\nb=%#v", a, b)
	}
}

func TestCollectMarksFailedPopulationUnknown(t *testing.T) {
	rep := Collect(t.TempDir(), time.Unix(1, 0), alwaysFailRunner{}, Options{WorkerRoot: t.TempDir()})
	if rep.Main.Tracked.Known || rep.Ignored.Known || rep.CheckpointsKnown || len(rep.Errors) == 0 {
		t.Fatalf("failures silently became zero: %#v", rep)
	}
}

func TestCollectProbesLargeWorktreeFleetInBoundedParallel(t *testing.T) {
	const worktreeCount = 205
	const probeDelay = 10 * time.Millisecond

	root := t.TempDir()
	workerRoot := filepath.Join(t.TempDir(), "workers")
	if err := os.MkdirAll(workerRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	paths := make([]string, worktreeCount)
	for i := range paths {
		paths[i] = filepath.Join(workerRoot, fmt.Sprintf("%s-%03d", workerMarker, i))
		if err := os.Mkdir(paths[i], 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(paths[i], "untracked.txt"), []byte("age witness\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	failingPath := paths[worktreeCount/2]
	runner := newScaleRunner(root, paths, failingPath, probeDelay)

	started := time.Now()
	rep := Collect(root, time.Unix(1700000100, 0), runner, Options{WorkerRoot: workerRoot})
	elapsed := time.Since(started)

	if got := len(rep.Worktrees); got != worktreeCount {
		t.Fatalf("worktrees=%d, want %d", got, worktreeCount)
	}
	if maxActive := runner.maximumActive(); maxActive != checkoutWorkerLimit {
		t.Fatalf("maximum parallel checkout probes=%d, want fixed ceiling %d", maxActive, checkoutWorkerLimit)
	}
	serialDuration := time.Duration(worktreeCount) * probeDelay
	if elapsed >= serialDuration/2 {
		t.Fatalf("elapsed=%s, want real parallel speedup over serial duration %s", elapsed, serialDuration)
	}
	for i := 1; i < len(rep.Worktrees); i++ {
		if rep.Worktrees[i-1].Path >= rep.Worktrees[i].Path {
			t.Fatalf("worktrees not stably sorted at %d: %q >= %q", i, rep.Worktrees[i-1].Path, rep.Worktrees[i].Path)
		}
	}

	failingSlash := filepath.ToSlash(failingPath)
	foundFailure := false
	for _, checkout := range rep.Worktrees {
		if checkout.Path == failingSlash {
			foundFailure = true
			if checkout.Tracked.Known || checkout.Untracked.Known || !strings.Contains(checkout.Tracked.Error, filepath.Base(failingPath)) {
				t.Fatalf("failure not attached to source checkout: %#v", checkout)
			}
			continue
		}
		if !checkout.Tracked.Known || !checkout.Untracked.Known || checkout.Untracked.Count != 1 || checkout.Untracked.OldestPath != "untracked.txt" {
			t.Fatalf("successful checkout missing status/age evidence at %q: %#v", checkout.Path, checkout)
		}
	}
	if !foundFailure {
		t.Fatalf("missing failing checkout %q", failingSlash)
	}
	wantReportError := "checkout " + failingSlash + ": status failed for " + filepath.Base(failingPath)
	if !containsString(rep.Errors, wantReportError) {
		t.Fatalf("errors=%q, want source-attributed error %q", rep.Errors, wantReportError)
	}
}

type scaleRunner struct {
	root        string
	worktreeOut []byte
	failingPath string
	delay       time.Duration
	mu          sync.Mutex
	active      int
	maxActive   int
}

func newScaleRunner(root string, paths []string, failingPath string, delay time.Duration) *scaleRunner {
	var out strings.Builder
	fmt.Fprintf(&out, "worktree %s\nHEAD main\nbranch refs/heads/main\n", filepath.ToSlash(root))
	for i := len(paths) - 1; i >= 0; i-- {
		fmt.Fprintf(&out, "\nworktree %s\nHEAD %03d\ndetached\n", filepath.ToSlash(paths[i]), i)
	}
	return &scaleRunner{root: root, worktreeOut: []byte(out.String()), failingPath: failingPath, delay: delay}
}

func (r *scaleRunner) Run(dir string, args ...string) ([]byte, error) {
	key := args[0]
	if len(args) > 1 {
		key += " " + args[1]
	}
	switch key {
	case "rev-parse HEAD":
		return []byte("abc123\n"), nil
	case "rev-parse --git-path":
		return []byte(filepath.Join(r.root, ".git", "info", "exclude") + "\n"), nil
	case "status --porcelain=v1":
		r.mu.Lock()
		r.active++
		if r.active > r.maxActive {
			r.maxActive = r.active
		}
		r.mu.Unlock()
		time.Sleep(r.delay)
		r.mu.Lock()
		r.active--
		r.mu.Unlock()
		if samePath(dir, r.failingPath) {
			return nil, fmt.Errorf("status failed for %s", filepath.Base(dir))
		}
		return []byte(" M tracked.go\x00?? untracked.txt\x00"), nil
	case "ls-files --others", "for-each-ref --format=%(refname)%00%(objectname)%00%(creatordate:unix)", "ls-files -v":
		return nil, nil
	case "worktree list":
		return r.worktreeOut, nil
	case "config --get", "config --bool":
		return nil, errors.New("unset")
	default:
		return nil, errors.New("unexpected: " + key)
	}
}

func (r *scaleRunner) maximumActive() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.maxActive
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

type alwaysFailRunner struct{}

func (alwaysFailRunner) Run(string, ...string) ([]byte, error) { return nil, errors.New("boom") }

func treeState(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		info, _ := d.Info()
		out = append(out, filepath.ToSlash(rel)+":"+info.Mode().String()+fmt.Sprintf(":%d", info.Size()))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
