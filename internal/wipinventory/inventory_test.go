package wipinventory

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestCollectFocusScopesCompletenessToTheFocus(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(t.TempDir(), workerMarker+"-target")
	other := filepath.Join(t.TempDir(), workerMarker+"-other")
	for _, dir := range []string{target, other} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "untracked.txt"), []byte("preserved"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runner := &focusOrderRunner{scaleRunner: newScaleRunner(root, []string{target, other}, other, 0)}
	rep := Collect(root, time.Now(), runner, Options{WorkerRoot: t.TempDir(), Focus: target})
	if rep.Focus == nil || !rep.Focus.Tracked.Known || !rep.Focus.Untracked.Known || rep.Focus.Tracked.Count != 1 || rep.Focus.Untracked.Count != 1 {
		t.Fatalf("target capture lost: %+v", rep.Focus)
	}
	if len(runner.calls) == 0 || runner.calls[0] != target+" status" {
		t.Fatalf("target was not observed first: %v", runner.calls)
	}
	count := 0
	for _, call := range runner.calls {
		if call == target+" status" {
			count++
		}
	}
	if count != 1 || !strings.Contains(strings.Join(rep.Errors, "\n"), filepath.Base(other)) {
		t.Fatalf("focus should be captured once with unrelated error retained: count=%d errors=%v", count, rep.Errors)
	}
}

type focusOrderRunner struct {
	*scaleRunner
	mu    sync.Mutex
	calls []string
}

func (r *focusOrderRunner) Run(dir string, args ...string) ([]byte, error) {
	r.mu.Lock()
	r.calls = append(r.calls, dir+" "+args[0])
	r.mu.Unlock()
	return r.scaleRunner.Run(dir, args...)
}

// fak-test:runtime medium est=2s lane=default
func TestDeadlineRunnerClassifiesExitStatusNotInheritedPipes(t *testing.T) {
	for _, tc := range []struct {
		mode      string
		want      string
		wantError string
	}{
		{"dirty", " M owned.go\x00", ""},
		{"delayed", " M owned.go\x00", ""},
		{"truncated", "", "context deadline exceeded"},
		{"failure", "", "status unavailable"},
		{"stall", "", "context deadline exceeded"},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			budget := 3 * time.Second
			if tc.mode == "stall" || tc.mode == "truncated" {
				budget = 100 * time.Millisecond
			}
			ctx, cancel := context.WithTimeout(context.Background(), budget)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDeadlineRunnerProcessHelper$")
			cmd.Env = append(os.Environ(), "FAK_INVENTORY_PROCESS_HELPER="+tc.mode)
			out, err := runDeadlineCommand(ctx, cmd)
			if string(out) != tc.want {
				t.Fatalf("output=%q want=%q err=%v", out, tc.want, err)
			}
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("successful dirty status treated as failure: %v", err)
				}
				status, captureErr := checkout(t.TempDir(), "", "", time.Now(), deadlineFixtureRunner{ctx: ctx, mode: tc.mode})
				if captureErr != nil || !status.Tracked.Known || !status.Untracked.Known || status.Tracked.Count != 1 || len(status.Tracked.Samples) != 1 || status.Tracked.Samples[0] != "owned.go" {
					t.Fatalf("successful dirty status not recorded as known paths: %+v err=%v", status, captureErr)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) || strings.Contains(err.Error(), "stdout-is-not-an-error") {
					t.Fatalf("failure must name cause/stderr, never stdout: %v", err)
				}
				if tc.mode == "truncated" {
					captureCtx, captureCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
					defer captureCancel()
					status, captureErr := checkout(t.TempDir(), "", "", time.Now(), deadlineFixtureRunner{ctx: captureCtx, mode: tc.mode})
					if captureErr == nil || status.Tracked.Known || status.Untracked.Known {
						t.Fatalf("truncated target status admitted: %+v err=%v", status, captureErr)
					}
				}
			}
		})
	}
}

type deadlineFixtureRunner struct {
	ctx  context.Context
	mode string
}

func (r deadlineFixtureRunner) Run(dir string, _ ...string) ([]byte, error) {
	cmd := exec.CommandContext(r.ctx, os.Args[0], "-test.run=^TestDeadlineRunnerProcessHelper$")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "FAK_INVENTORY_PROCESS_HELPER="+r.mode)
	return runDeadlineCommand(r.ctx, cmd)
}

func TestDeadlineRunnerProcessHelper(t *testing.T) {
	switch os.Getenv("FAK_INVENTORY_PROCESS_HELPER") {
	case "dirty", "delayed", "truncated":
		child := exec.Command(os.Args[0], "-test.run=^TestDeadlineRunnerProcessHelper$")
		childMode := "pipe-holder"
		if os.Getenv("FAK_INVENTORY_PROCESS_HELPER") != "dirty" {
			childMode = "late-writer"
		}
		child.Env = append(os.Environ(), "FAK_INVENTORY_PROCESS_HELPER="+childMode)
		child.Stdout, child.Stderr = os.Stdout, os.Stderr
		if err := child.Start(); err != nil {
			os.Exit(3)
		}
		if childMode == "pipe-holder" {
			fmt.Fprint(os.Stdout, " M owned.go\x00")
		} else {
			fmt.Fprint(os.Stdout, " M own")
		}
		os.Exit(0)
	case "pipe-holder":
		time.Sleep(400 * time.Millisecond)
		os.Exit(0)
	case "late-writer":
		time.Sleep(300 * time.Millisecond)
		fmt.Fprint(os.Stdout, "ed.go\x00")
		os.Exit(0)
	case "failure":
		fmt.Fprint(os.Stdout, "stdout-is-not-an-error")
		fmt.Fprint(os.Stderr, "status unavailable")
		os.Exit(9)
	case "stall":
		time.Sleep(3 * time.Second)
		os.Exit(0)
	}
}

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

type batchTestRunner struct {
	calls []string
}

func (b *batchTestRunner) Run(dir string, args ...string) ([]byte, error) {
	b.calls = append(b.calls, strings.Join(args, " "))
	if strings.HasPrefix(args[0], "for-each-ref") {
		return []byte("refs/fak/wip/c1\x00sha1\x001700000000\nrefs/fak/wip/c2\x00sha2\x001700000010\n"), nil
	}
	return nil, errors.New("unexpected")
}

func (b *batchTestRunner) RunWithStdin(dir string, in []byte, args ...string) ([]byte, error) {
	b.calls = append(b.calls, strings.Join(args, " ")+" (stdin: "+strings.TrimSpace(string(in))+")")
	if args[0] == "diff-tree" {
		return []byte("sha1\nM\tfile1.go\nA\tfile2.go\nsha2\nM\tfile3.go\n"), nil
	}
	return nil, errors.New("unexpected")
}

func TestCheckpointsBatchesMultipleRefs(t *testing.T) {
	r := &batchTestRunner{}
	rep := &Report{}
	cps, ok := checkpoints("/repo", r, rep)
	if !ok || len(cps) != 2 {
		t.Fatalf("expected 2 checkpoints, ok=%v, got: %#v", ok, cps)
	}
	if cps[0].Changed != 2 || cps[0].Added != 1 || len(cps[0].allPaths) != 2 {
		t.Fatalf("c1 population mismatch: %#v", cps[0])
	}
	if cps[1].Changed != 1 || cps[1].Added != 0 || len(cps[1].allPaths) != 1 {
		t.Fatalf("c2 population mismatch: %#v", cps[1])
	}
	diffTreeCalls := 0
	for _, c := range r.calls {
		if strings.HasPrefix(c, "diff-tree") {
			diffTreeCalls++
		}
	}
	if diffTreeCalls != 1 {
		t.Fatalf("expected 1 batched diff-tree call, got %d: %v", diffTreeCalls, r.calls)
	}
}

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
