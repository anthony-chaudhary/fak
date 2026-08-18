package wipinventory

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeRunner struct {
	root   string
	worker string
	calls  []string
}

func (f *fakeRunner) Run(dir string, args ...string) ([]byte, error) {
	f.calls = append(f.calls, filepath.ToSlash(dir)+" git "+strings.Join(args, " "))
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
