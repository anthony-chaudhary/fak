package session

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// memScratchJournal is an in-memory ScratchJournal — the ledger the witnesses read
// back. It records every event in order so a test can assert the lifecycle journaled
// exactly what #2420 requires (a minted path, a GC reclaim with bytes/files, a fork,
// a checkpoint digest, a restore/skip).
type memScratchJournal struct{ events []ScratchEvent }

func (m *memScratchJournal) Record(e ScratchEvent) error {
	m.events = append(m.events, e)
	return nil
}

func (m *memScratchJournal) kinds() []string {
	out := make([]string, 0, len(m.events))
	for _, e := range m.events {
		out = append(out, e.Kind)
	}
	return out
}

func (m *memScratchJournal) last(kind string) (ScratchEvent, bool) {
	for i := len(m.events) - 1; i >= 0; i-- {
		if m.events[i].Kind == kind {
			return m.events[i], true
		}
	}
	return ScratchEvent{}, false
}

// TestScratchpadLeaseLifecycle: session start mints the dir and the ledger records
// it; session end GC journals bytes reclaimed. (#2420 witness 1.)
func TestScratchpadLeaseLifecycle(t *testing.T) {
	base := t.TempDir()
	j := &memScratchJournal{}
	now := time.Unix(1_700_000_000, 0).UTC()

	lease, err := MintScratch(base, "trace-birth", j, now)
	if err != nil {
		t.Fatalf("MintScratch: %v", err)
	}
	if lease.Dir == "" {
		t.Fatal("mint returned an empty scratch dir")
	}
	if fi, err := os.Stat(lease.Dir); err != nil || !fi.IsDir() {
		t.Fatalf("minted scratch dir is not a directory: err=%v", err)
	}
	// The ledger records the mint with the real path (the "recorded on the ledger"
	// half of the witness).
	minted, ok := j.last(EvScratchMinted)
	if !ok {
		t.Fatal("mint was not journaled")
	}
	if minted.Dir != lease.Dir || minted.TraceID != "trace-birth" {
		t.Fatalf("minted event mismatch: got %+v", minted)
	}

	// Write a known number of bytes across two files so the GC event's reclaim
	// figures are exact, not approximate.
	payloadA := []byte("hello scratch")            // 13 bytes
	payloadB := []byte("second file, more bytes!") // 24 bytes
	if err := os.WriteFile(filepath.Join(lease.Dir, "a.txt"), payloadA, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(lease.Dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(lease.Dir, "sub", "b.txt"), payloadB, 0o644); err != nil {
		t.Fatal(err)
	}
	wantBytes := int64(len(payloadA) + len(payloadB))

	ev, err := lease.GC(j, now.Add(time.Minute))
	if err != nil {
		t.Fatalf("GC: %v", err)
	}
	if ev.Kind != EvScratchGC {
		t.Fatalf("GC event kind = %q, want %q", ev.Kind, EvScratchGC)
	}
	if ev.BytesReclaimed != wantBytes {
		t.Fatalf("BytesReclaimed = %d, want %d", ev.BytesReclaimed, wantBytes)
	}
	if ev.FilesDropped != 2 {
		t.Fatalf("FilesDropped = %d, want 2", ev.FilesDropped)
	}
	if _, err := os.Stat(lease.Dir); !os.IsNotExist(err) {
		t.Fatalf("scratch dir survived GC: stat err = %v", err)
	}
	gcEv, ok := j.last(EvScratchGC)
	if !ok || gcEv.BytesReclaimed != wantBytes {
		t.Fatalf("GC was not journaled with the reclaim: %+v ok=%v", gcEv, ok)
	}

	// GC is idempotent: a second reclaim of the now-gone tree reports zero and errors not.
	ev2, err := lease.GC(j, now.Add(2*time.Minute))
	if err != nil {
		t.Fatalf("second GC errored: %v", err)
	}
	if ev2.BytesReclaimed != 0 || ev2.FilesDropped != 0 {
		t.Fatalf("second GC should reclaim nothing, got %+v", ev2)
	}
}

// TestForkScratchIsolation: two forks write the same filename without collision.
// (#2420 witness 2.)
func TestForkScratchIsolation(t *testing.T) {
	base := t.TempDir()
	j := &memScratchJournal{}
	now := time.Unix(1_700_000_100, 0).UTC()

	parent, err := MintScratch(base, "trace-parent", j, now)
	if err != nil {
		t.Fatalf("MintScratch: %v", err)
	}
	// Seed a shared file so we also prove the fork inherits parent state at fork time.
	if err := os.WriteFile(filepath.Join(parent.Dir, "seed.txt"), []byte("seed"), 0o644); err != nil {
		t.Fatal(err)
	}

	forkA, err := parent.Fork(base, "trace-fork-a", j, now)
	if err != nil {
		t.Fatalf("Fork A: %v", err)
	}
	forkB, err := parent.Fork(base, "trace-fork-b", j, now)
	if err != nil {
		t.Fatalf("Fork B: %v", err)
	}
	if forkA.Dir == forkB.Dir {
		t.Fatal("two forks share the same scratch dir — no isolation")
	}
	// Each fork inherited the seed (copy-on-write started from parent state).
	for _, f := range []ScratchLease{forkA, forkB} {
		if b, err := os.ReadFile(filepath.Join(f.Dir, "seed.txt")); err != nil || string(b) != "seed" {
			t.Fatalf("fork %s did not inherit seed: %q err=%v", f.TraceID, b, err)
		}
	}

	// Both forks write the SAME filename with DIFFERENT content — no collision.
	if err := os.WriteFile(filepath.Join(forkA.Dir, "same.txt"), []byte("from A"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(forkB.Dir, "same.txt"), []byte("from B"), 0o644); err != nil {
		t.Fatal(err)
	}
	gotA, err := os.ReadFile(filepath.Join(forkA.Dir, "same.txt"))
	if err != nil {
		t.Fatal(err)
	}
	gotB, err := os.ReadFile(filepath.Join(forkB.Dir, "same.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotA) != "from A" || string(gotB) != "from B" {
		t.Fatalf("fork writes collided: A=%q B=%q", gotA, gotB)
	}
	// A write into one fork must not appear in the other (copy-on-write semantics).
	if _, err := os.Stat(filepath.Join(parent.Dir, "same.txt")); !os.IsNotExist(err) {
		t.Fatalf("fork write leaked back into parent: err=%v", err)
	}
}

// TestCheckpointScratchAxis: checkpoint records the scratch digest; restore-with-
// scratch reproduces it, restore-without leaves current scratch untouched.
// (#2420 witness 3.)
func TestCheckpointScratchAxis(t *testing.T) {
	base := t.TempDir()
	archiveBase := t.TempDir()
	j := &memScratchJournal{}
	now := time.Unix(1_700_000_200, 0).UTC()

	lease, err := MintScratch(base, "trace-ckpt", j, now)
	if err != nil {
		t.Fatalf("MintScratch: %v", err)
	}
	if err := os.WriteFile(filepath.Join(lease.Dir, "state.txt"), []byte("checkpoint me"), 0o644); err != nil {
		t.Fatal(err)
	}

	cp, err := lease.Checkpoint(archiveBase, j, now)
	if err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}
	if cp.Digest == "" {
		t.Fatal("checkpoint recorded no digest")
	}
	// The digest is the checkable identity of the tree at checkpoint time.
	liveDigest, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if cp.Digest != liveDigest {
		t.Fatalf("checkpoint digest %q != live digest %q", cp.Digest, liveDigest)
	}
	ckEv, ok := j.last(EvScratchCheckpoint)
	if !ok || ckEv.Digest != cp.Digest {
		t.Fatalf("checkpoint was not journaled with the digest: %+v ok=%v", ckEv, ok)
	}
	// Checkpoint is not a GC: the live tree survives.
	if _, err := os.Stat(lease.Dir); err != nil {
		t.Fatalf("checkpoint removed the live scratch tree: %v", err)
	}

	// Mutate scratch AFTER the checkpoint, so a restore has something to reverse.
	if err := os.WriteFile(filepath.Join(lease.Dir, "state.txt"), []byte("mutated after checkpoint"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(lease.Dir, "extra.txt"), []byte("added after checkpoint"), 0o644); err != nil {
		t.Fatal(err)
	}
	mutatedDigest, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if mutatedDigest == cp.Digest {
		t.Fatal("mutation did not change the digest — test cannot distinguish restore")
	}

	// RESTORE-WITHOUT-SCRATCH: current scratch is left untouched.
	if err := cp.Restore(lease, false, j, now.Add(time.Minute)); err != nil {
		t.Fatalf("restore-without: %v", err)
	}
	afterSkip, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if afterSkip != mutatedDigest {
		t.Fatalf("restore-without-scratch changed the tree: %q != %q", afterSkip, mutatedDigest)
	}
	if _, ok := j.last(EvScratchRestoreSkipped); !ok {
		t.Fatal("restore-without-scratch was not journaled as a skip")
	}

	// RESTORE-WITH-SCRATCH: reproduces the checkpoint tree exactly.
	if err := cp.Restore(lease, true, j, now.Add(2*time.Minute)); err != nil {
		t.Fatalf("restore-with: %v", err)
	}
	afterRestore, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if afterRestore != cp.Digest {
		t.Fatalf("restore-with-scratch did not reproduce the checkpoint digest: %q != %q", afterRestore, cp.Digest)
	}
	// The post-checkpoint file is gone; the checkpointed content is back.
	if _, err := os.Stat(filepath.Join(lease.Dir, "extra.txt")); !os.IsNotExist(err) {
		t.Fatalf("restore-with-scratch left a post-checkpoint file: err=%v", err)
	}
	if b, err := os.ReadFile(filepath.Join(lease.Dir, "state.txt")); err != nil || string(b) != "checkpoint me" {
		t.Fatalf("restore-with-scratch did not reproduce content: %q err=%v", b, err)
	}
	if ev, ok := j.last(EvScratchRestored); !ok || ev.Digest != cp.Digest {
		t.Fatalf("restore-with-scratch was not journaled with the digest: %+v ok=%v", ev, ok)
	}
}

// TestScratchZeroCheckpointRestoreNoop: a zero checkpoint (no scratch axis) restores
// as a no-op in either mode — the "third OPTIONAL axis" contract.
func TestScratchZeroCheckpointRestoreNoop(t *testing.T) {
	base := t.TempDir()
	lease, err := MintScratch(base, "trace-zero", nil, time.Time{})
	if err != nil {
		t.Fatalf("MintScratch: %v", err)
	}
	if err := os.WriteFile(filepath.Join(lease.Dir, "keep.txt"), []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	before, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	var zero ScratchCheckpoint
	if !zero.IsZero() {
		t.Fatal("zero checkpoint should report IsZero")
	}
	for _, include := range []bool{false, true} {
		if err := zero.Restore(lease, include, nil, time.Time{}); err != nil {
			t.Fatalf("zero-checkpoint restore(include=%v): %v", include, err)
		}
	}
	after, err := lease.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("zero-checkpoint restore mutated the tree: %q != %q", before, after)
	}
}
