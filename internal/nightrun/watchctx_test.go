// watchctx_test.go — the #2384 witness set: (1) the descriptor round-trips all
// fields through its durable file; (2) the artifact path is persisted BEFORE launch
// (the descriptor on disk holds absArtifact(opts, t) while the nightrun ledger row is
// still absent); (3) on restart, a live pid reports re-attach (no duplicate launch)
// and a dead or reused pid reports the job gone.
package nightrun

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestWatchContext_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	want := WatchContext{
		WatcherID:          "watcher-abc123",
		JobPID:             4242,
		JobStart:           "boot-2026-07-07T10:00:00Z",
		ArtifactPath:       filepath.Join(dir, "artifacts", "box", "20260707T100000Z-glm-decode.log"),
		LastProgress:       "step 1500/9000 loss=0.42",
		LastProgressUnix:   1783764000,
		ExpectedDoneByUnix: 1783800000,
	}

	path, err := WriteWatchContext(dir, want)
	if err != nil {
		t.Fatalf("WriteWatchContext: %v", err)
	}
	if wantPath := WatchContextPath(dir, want.WatcherID); path != wantPath {
		t.Fatalf("written path = %q, want %q", path, wantPath)
	}

	got, err := ReadWatchContext(dir, want.WatcherID)
	if err != nil {
		t.Fatalf("ReadWatchContext: %v", err)
	}
	if got != want {
		t.Fatalf("round-trip mismatch:\n got  %+v\n want %+v", got, want)
	}

	// Advance rewrites the progress fields and survives a re-write + re-read.
	adv := got.Advance("step 3000/9000 loss=0.31", 1783767600)
	if _, err := WriteWatchContext(dir, adv); err != nil {
		t.Fatalf("WriteWatchContext after Advance: %v", err)
	}
	got2, err := ReadWatchContext(dir, want.WatcherID)
	if err != nil {
		t.Fatalf("ReadWatchContext after Advance: %v", err)
	}
	if got2.LastProgress != "step 3000/9000 loss=0.31" || got2.LastProgressUnix != 1783767600 {
		t.Fatalf("Advance did not persist: %+v", got2)
	}
}

// The pre-launch fixture: given only RunOptions + Task — no process launched, no
// executor run — the descriptor on disk already holds absArtifact(opts, t), while
// the nightrun ledger file does not exist yet. This is the pointer a mid-task
// watcher crash needs and the post-hoc ledger row cannot provide.
func TestWatchContext_ArtifactPersistedBeforeLaunch(t *testing.T) {
	root := t.TempDir()
	opts := RunOptions{
		Root:        root,
		ArtifactDir: filepath.Join(root, "experiments", "nightrun"),
		LedgerPath:  filepath.Join(root, "docs", "nightrun", "collected.jsonl"),
		Now:         time.Date(2026, 7, 7, 10, 0, 0, 0, time.UTC),
		Caps:        Capabilities{Box: "testbox"},
	}
	task := Task{ID: "glm-decode-bench"}

	wc := NewWatchContext("watcher-abc123", 0, "", opts, task)
	if _, err := WriteWatchContext(root, wc); err != nil {
		t.Fatalf("WriteWatchContext: %v", err)
	}

	got, err := ReadWatchContext(root, "watcher-abc123")
	if err != nil {
		t.Fatalf("ReadWatchContext: %v", err)
	}
	if want := absArtifact(opts, task); got.ArtifactPath != want {
		t.Fatalf("descriptor artifact_path = %q, want absArtifact = %q", got.ArtifactPath, want)
	}
	if _, err := os.Stat(opts.LedgerPath); !os.IsNotExist(err) {
		t.Fatalf("ledger row must still be absent pre-launch; stat(%q) err = %v", opts.LedgerPath, err)
	}
}

// The restart cases: seed a descriptor, re-read it as a restarted watcher would,
// and run the re-attach probe through the #2277 seam.
func TestWatchContext_RestartReattachAndGone(t *testing.T) {
	dir := t.TempDir()
	seed := WatchContext{WatcherID: "watcher-abc123", JobPID: 4242, JobStart: "boot-77"}
	if _, err := WriteWatchContext(dir, seed); err != nil {
		t.Fatalf("WriteWatchContext: %v", err)
	}
	wc, err := ReadWatchContext(dir, "watcher-abc123")
	if err != nil {
		t.Fatalf("ReadWatchContext: %v", err)
	}

	alive := func(pid int) (string, bool) {
		if pid == 4242 {
			return "boot-77", true
		}
		return "", false
	}
	if got := wc.Reattach(alive); got != ReattachLive {
		t.Fatalf("live pid: verdict = %q, want %q (re-attach, no duplicate launch)", got, ReattachLive)
	}

	dead := func(int) (string, bool) { return "", false }
	if got := wc.Reattach(dead); got != ReattachGone {
		t.Fatalf("dead pid: verdict = %q, want %q", got, ReattachGone)
	}

	// pid reuse: the pid is held, but by a DIFFERENT process (start identity
	// mismatch) — the recorded job is gone, not alive.
	reused := func(pid int) (string, bool) {
		if pid == 4242 {
			return "boot-99", true
		}
		return "", false
	}
	if got := wc.Reattach(reused); got != ReattachGone {
		t.Fatalf("reused pid: verdict = %q, want %q", got, ReattachGone)
	}

	// No prober / no pid: no verdict — never a false "gone" that triggers a relaunch.
	if got := wc.Reattach(nil); got != ReattachUnknown {
		t.Fatalf("nil prober: verdict = %q, want %q", got, ReattachUnknown)
	}
	if got := (WatchContext{WatcherID: "w", JobPID: 0}).Reattach(alive); got != ReattachUnknown {
		t.Fatalf("no pid: verdict = %q, want %q", got, ReattachUnknown)
	}
}
