package workerworktree

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fixedLandProbe struct{ sample landResourceSample }

func (p fixedLandProbe) stop() landResourceSample { return p.sample }

func TestLandProgressFakeClockOrdersEventsAndReconcilesCost(t *testing.T) {
	now := time.Unix(100, 0)
	var events []LandProgressEvent
	cfg := landConfig{
		progressSink: func(event LandProgressEvent) { events = append(events, event) },
		progressNow:  func() time.Time { return now },
		progressProbe: func() landResourceProbe {
			return fixedLandProbe{sample: landResourceSample{
				cpu: 1500 * time.Millisecond, cpuObserved: true,
				peakRSSBytes: 64 << 20, peakRSSObserved: true,
			}}
		},
	}
	recorder := newLandRecorder(cfg)
	recorder.setScan(2, 4096)
	admission := recorder.begin("admission", 0)
	now = now.Add(2 * time.Second)
	admission.complete("ok")
	apply := recorder.begin("apply_index", 1)
	now = now.Add(3 * time.Second)
	apply.complete("ok")
	receipt := recorder.finish(true)

	want := []struct{ phase, state string }{
		{"admission", "started"}, {"admission", "completed"},
		{"apply_index", "started"}, {"apply_index", "completed"},
	}
	if len(events) != len(want) {
		t.Fatalf("events=%+v, want %d boundaries", events, len(want))
	}
	for i := range want {
		if events[i].Phase != want[i].phase || events[i].State != want[i].state {
			t.Fatalf("event[%d]=%+v, want phase=%s state=%s", i, events[i], want[i].phase, want[i].state)
		}
	}
	if receipt.WallMS != 5000 || receipt.AccountedPhaseMS != 5000 || receipt.UnattributedWallMS != 0 {
		t.Fatalf("unreconciled receipt: %+v", receipt)
	}
	if receipt.SlowestPhase != "apply_index" || receipt.SlowestPhaseMS != 3000 {
		t.Fatalf("slowest phase=%q/%d, want apply_index/3000", receipt.SlowestPhase, receipt.SlowestPhaseMS)
	}
	if receipt.CPUms != 1500 || !receipt.CPUObserved || receipt.PeakRSSBytes != 64<<20 || !receipt.PeakRSSObserved {
		t.Fatalf("resource receipt=%+v", receipt)
	}
	if receipt.ScannedFiles != 2 || receipt.ScannedBytes != 4096 || receipt.CacheReuse != "none" {
		t.Fatalf("scan/cache receipt=%+v", receipt)
	}
}

func TestLiveTwoFileLandEmitsEarlyBoundedProgressAndTerminalCost(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=fak test", "GIT_AUTHOR_EMAIL=fak@example.test",
			"GIT_COMMITTER_NAME=fak test", "GIT_COMMITTER_EMAIL=fak@example.test")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git -C %s %s: %v: %s", dir, strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git(root, "init", "-q", "-b", "main")
	git(root, "config", "user.name", "fak test")
	git(root, "config", "user.email", "fak@example.test")
	for _, name := range []string{"a.txt", "b.txt", "peer-owned.txt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("base\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git(root, "add", "a.txt", "b.txt", "peer-owned.txt")
	git(root, "commit", "-qm", "base")
	base := git(root, "rev-parse", "HEAD")
	prepared := Prepare(root, "workerworktree", "9460-live", base, filepath.Join(t.TempDir(), "workers"), nil)
	if !prepared.OK {
		t.Fatalf("prepare: %+v", prepared)
	}
	t.Cleanup(func() { _ = ForceReap(root, prepared.Path, nil) })
	for _, name := range []string{"a.txt", "b.txt"} {
		if err := os.WriteFile(filepath.Join(prepared.Path, name), []byte("changed\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A peer-owned dirty trunk path must remain out of progress and the landed commit.
	if err := os.WriteFile(filepath.Join(root, "peer-owned.txt"), []byte("peer dirty\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	msg := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(msg, []byte("test(workerworktree): progress fixture (fak workerworktree)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	var events []LandProgressEvent
	res := Land(root, prepared.Path, base, msg, []string{"a.txt", "b.txt"}, nil, nil,
		WithLandProgress(func(event LandProgressEvent) { events = append(events, event) }))
	if !res.OK || !res.Committed {
		t.Fatalf("live land: %+v", res)
	}
	if len(events) == 0 || events[0].State != "started" || events[0].Phase != "admission" {
		t.Fatalf("missing immediate admission progress: %+v", events)
	}
	if latency := time.Duration(events[0].ElapsedMS) * time.Millisecond; latency >= 10*time.Second || time.Since(started) < latency {
		t.Fatalf("first event latency=%s, want <10s and within run wall", latency)
	}
	for _, phase := range []string{"admission", "validation_materializer", "prospective_validation", "apply_index", "commit", "recovery_ref_publication", "cas"} {
		if !hasCompletedPhase(events, phase) {
			t.Fatalf("phase %q missing completed boundary: %+v", phase, events)
		}
	}
	if res.LandCost == nil || res.LandCost.ScannedFiles != 2 || res.LandCost.ScannedBytes == 0 {
		t.Fatalf("terminal scan receipt=%+v", res.LandCost)
	}
	if got := res.LandCost.AccountedPhaseMS + res.LandCost.UnattributedWallMS; got != res.LandCost.WallMS {
		t.Fatalf("phase total=%d + unattributed=%d, wall=%d", res.LandCost.AccountedPhaseMS, res.LandCost.UnattributedWallMS, res.LandCost.WallMS)
	}
	encoded, err := json.Marshal(struct {
		Events []LandProgressEvent `json:"events"`
		Cost   *LandCostReceipt    `json:"cost"`
	}{events, res.LandCost})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "peer-owned.txt") {
		t.Fatalf("progress leaked peer path: %s", encoded)
	}
	if got := git(root, "show", "--format=", "--name-only", "HEAD"); strings.Contains(got, "peer-owned.txt") {
		t.Fatalf("land swept peer path: %q", got)
	}
	if body, err := os.ReadFile(filepath.Join(root, "peer-owned.txt")); err != nil || string(body) != "peer dirty\n" {
		t.Fatalf("peer working-tree bytes changed: body=%q err=%v", body, err)
	}
}

func hasCompletedPhase(events []LandProgressEvent, name string) bool {
	for _, event := range events {
		if event.Phase == name && event.State == "completed" {
			return true
		}
	}
	return false
}

func TestWholeTreeAnalysesEmitIndividualDeterministicPhases(t *testing.T) {
	now := time.Unix(200, 0)
	var events []LandProgressEvent
	recorder := newLandRecorder(landConfig{
		progressSink: func(event LandProgressEvent) { events = append(events, event) },
		progressNow:  func() time.Time { return now },
		progressProbe: func() landResourceProbe {
			return fixedLandProbe{}
		},
	})
	oldRead := readDisambiguation
	t.Cleanup(func() { readDisambiguation = oldRead })
	call := 0
	readDisambiguation = func(repo, tree string) DisambiguationWitness {
		call++
		now = now.Add(time.Duration(call) * time.Second)
		return DisambiguationWitness{
			Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true,
			Coverage: 100, ScannedFiles: 10 * call, ScannedBytes: int64(100 * call),
		}
	}
	_, ok := verifyAppliedDisambiguationProgress("/repo", "/worker", "tree", recorder, 2)
	if !ok {
		t.Fatal("valid synthetic witnesses refused")
	}
	receipt := recorder.finish(true)
	want := []string{"whole_tree_analysis_before", "whole_tree_analysis_worktree", "whole_tree_analysis_post_apply"}
	if len(receipt.Phases) != len(want) {
		t.Fatalf("phases=%+v", receipt.Phases)
	}
	for i, name := range want {
		if receipt.Phases[i].Phase != name || receipt.Phases[i].Attempt != 2 || receipt.Phases[i].ElapsedMS != int64((i+1)*1000) {
			t.Fatalf("phase[%d]=%+v, want %s attempt=2 duration=%d", i, receipt.Phases[i], name, (i+1)*1000)
		}
		if !hasCompletedPhase(events, name) {
			t.Fatalf("missing completed event for %s: %+v", name, events)
		}
	}
	if receipt.ScannedFiles != 60 || receipt.ScannedBytes != 600 || receipt.ScanScope != "exact_land_patch_plus_whole_tree_analyses" {
		t.Fatalf("whole-tree scan accounting=%+v", receipt)
	}
	if receipt.AccountedPhaseMS+receipt.UnattributedWallMS != receipt.WallMS {
		t.Fatalf("unreconciled receipt=%+v", receipt)
	}
}
