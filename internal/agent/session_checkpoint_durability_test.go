package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func durabilityCheckpoint(id string, turn int, msgs ...string) SessionCheckpoint {
	cp := SessionCheckpoint{
		SessionID: id,
		CWD:       "/work",
		Task:      "book flight",
		Model:     "test-model",
		Provider:  "test-provider",
		BaseURL:   "http://localhost:8080/v1",
		Turn:      turn,
		CreatedAt: time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC),
		Status:    "active",
	}
	for _, m := range msgs {
		cp.Messages = append(cp.Messages, Message{Role: RoleUser, Content: m})
	}
	return cp
}

// checkpointDirEntries lists every file in dir, so a leaked temp file is visible.
func checkpointDirEntries(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

func TestSessionCheckpointRoundTripIsVersioned(t *testing.T) {
	dir := t.TempDir()
	cp := durabilityCheckpoint("sess-rt", 3, "hello", "world")
	cp.Messages = append(cp.Messages, Message{Role: RoleAssistant, Content: "ok"})
	if err := SaveSessionCheckpoint(dir, cp); err != nil {
		t.Fatalf("save: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "sess-rt.json"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var onDisk struct {
		Version *int `json:"version"`
	}
	if err := json.Unmarshal(raw, &onDisk); err != nil {
		t.Fatalf("decode on-disk checkpoint: %v", err)
	}
	if onDisk.Version == nil || *onDisk.Version != SessionCheckpointVersion {
		t.Fatalf("on-disk version = %v, want %d", onDisk.Version, SessionCheckpointVersion)
	}

	got, err := LoadSessionCheckpoint("sess-rt", dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Version != SessionCheckpointVersion || got.SessionID != cp.SessionID || got.CWD != cp.CWD ||
		got.Task != cp.Task || got.Model != cp.Model || got.Provider != cp.Provider ||
		got.BaseURL != cp.BaseURL || got.Turn != cp.Turn || got.Status != cp.Status ||
		!got.CreatedAt.Equal(cp.CreatedAt) || got.UpdatedAt.IsZero() {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", *got, cp)
	}
	wantMsgs, _ := json.Marshal(cp.Messages)
	gotMsgs, _ := json.Marshal(got.Messages)
	if !bytes.Equal(gotMsgs, wantMsgs) {
		t.Fatalf("messages round trip:\n got %s\nwant %s", gotMsgs, wantMsgs)
	}
	if names := checkpointDirEntries(t, dir); len(names) != 1 || names[0] != "sess-rt.json" {
		t.Fatalf("dir after save = %v, want only sess-rt.json (no leaked temp file)", names)
	}

	// A second save replaces the first in place under the same name.
	cp2 := durabilityCheckpoint("sess-rt", 4, "hello", "world", "again")
	if err := SaveSessionCheckpoint(dir, cp2); err != nil {
		t.Fatalf("second save: %v", err)
	}
	got2, err := LoadSessionCheckpoint("sess-rt", dir)
	if err != nil {
		t.Fatalf("second load: %v", err)
	}
	if got2.Turn != 4 || len(got2.Messages) != 3 {
		t.Fatalf("second load = turn %d, %d messages; want turn 4, 3 messages", got2.Turn, len(got2.Messages))
	}
	if names := checkpointDirEntries(t, dir); len(names) != 1 {
		t.Fatalf("dir after second save = %v, want one file", names)
	}
}

// TestSessionCheckpointFailedWriteKeepsPrevious is the crash-safety witness: a write
// that fails partway (a torn write), a rename that fails, and a crash that strands a
// torn temp file must all leave the previous checkpoint byte-identical and loadable.
// The historical os.WriteFile truncated the target first, so a torn write destroyed the
// previous good checkpoint too.
func TestSessionCheckpointFailedWriteKeepsPrevious(t *testing.T) {
	setup := func(t *testing.T) (dir, target string, prev []byte) {
		t.Helper()
		dir = t.TempDir()
		if err := SaveSessionCheckpoint(dir, durabilityCheckpoint("sess-crash", 1, "turn one")); err != nil {
			t.Fatalf("save previous: %v", err)
		}
		target = filepath.Join(dir, "sess-crash.json")
		prev, err := os.ReadFile(target)
		if err != nil {
			t.Fatalf("read previous: %v", err)
		}
		return dir, target, prev
	}
	assertPreviousIntact := func(t *testing.T, dir, target string, prev []byte) {
		t.Helper()
		now, err := os.ReadFile(target)
		if err != nil {
			t.Fatalf("previous checkpoint gone: %v", err)
		}
		if !bytes.Equal(now, prev) {
			t.Fatalf("previous checkpoint bytes changed:\n got %s\nwant %s", now, prev)
		}
		got, err := LoadSessionCheckpoint("sess-crash", dir)
		if err != nil {
			t.Fatalf("previous checkpoint not loadable: %v", err)
		}
		if got.Turn != 1 || len(got.Messages) != 1 || got.Messages[0].Content != "turn one" {
			t.Fatalf("loaded %+v, want the previous turn-1 checkpoint", *got)
		}
	}
	next := durabilityCheckpoint("sess-crash", 2, "turn one", "turn two")

	t.Run("torn write", func(t *testing.T) {
		dir, target, prev := setup(t)
		errDiskFull := errors.New("injected: no space left on device")
		orig := checkpointWriteTemp
		t.Cleanup(func() { checkpointWriteTemp = orig })
		checkpointWriteTemp = func(f *os.File, data []byte) error {
			if _, err := f.Write(data[:len(data)/2]); err != nil {
				return err
			}
			return errDiskFull
		}

		err := SaveSessionCheckpoint(dir, next)
		if !errors.Is(err, errDiskFull) {
			t.Fatalf("save error = %v, want the injected write failure", err)
		}
		assertPreviousIntact(t, dir, target, prev)
		if names := checkpointDirEntries(t, dir); len(names) != 1 {
			t.Fatalf("dir after failed save = %v, want only the previous checkpoint", names)
		}
	})

	t.Run("rename fails", func(t *testing.T) {
		dir, target, prev := setup(t)
		errRename := errors.New("injected: rename failed")
		orig := checkpointRename
		t.Cleanup(func() { checkpointRename = orig })
		checkpointRename = func(string, string) error { return errRename }

		err := SaveSessionCheckpoint(dir, next)
		if !errors.Is(err, errRename) {
			t.Fatalf("save error = %v, want the injected rename failure", err)
		}
		assertPreviousIntact(t, dir, target, prev)
		if names := checkpointDirEntries(t, dir); len(names) != 1 {
			t.Fatalf("dir after failed save = %v, want only the previous checkpoint", names)
		}
	})

	t.Run("crash strands torn temp", func(t *testing.T) {
		dir, target, prev := setup(t)
		// A process killed between the temp write and the rename never runs its cleanup,
		// so it leaves a torn temp next to the target. The target must be unaffected.
		full, _ := json.Marshal(next)
		stranded := filepath.Join(dir, ".sess-crash.json.tmp-123456")
		if err := os.WriteFile(stranded, full[:len(full)/2], 0o600); err != nil {
			t.Fatalf("strand temp: %v", err)
		}
		assertPreviousIntact(t, dir, target, prev)
		if _, err := LoadSessionCheckpoint(".sess-crash.json.tmp-123456", dir); err == nil {
			t.Fatal("a stranded temp must never load as a checkpoint")
		}

		// The next save still lands.
		if err := SaveSessionCheckpoint(dir, next); err != nil {
			t.Fatalf("save after crash: %v", err)
		}
		got, err := LoadSessionCheckpoint("sess-crash", dir)
		if err != nil || got.Turn != 2 {
			t.Fatalf("load after recovery save = %+v, %v; want turn 2", got, err)
		}
	})
}

func TestLoadSessionCheckpointRefusesUnknownVersion(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name+".json"), []byte(body), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	write("future", `{"version": 2, "session_id": "future", "turn": 5, "messages": []}`)
	// A future layout that also changed a field's shape must be refused by version, not
	// by whichever decode error it would trip first.
	write("future-shape", `{"version": 99, "session_id": "future-shape", "messages": {"segments": []}}`)
	write("negative", `{"version": -1, "session_id": "negative"}`)
	for _, id := range []string{"future", "future-shape", "negative"} {
		cp, err := LoadSessionCheckpoint(id, dir)
		if !errors.Is(err, ErrUnsupportedSessionCheckpointVersion) {
			t.Fatalf("load %s = %+v, %v; want ErrUnsupportedSessionCheckpointVersion", id, cp, err)
		}
	}

	// A checkpoint written before the version field existed still resumes.
	write("legacy", `{"session_id": "legacy", "task": "t", "turn": 2, "messages": [{"role": "user", "content": "hi"}], "status": "active"}`)
	cp, err := LoadSessionCheckpoint("legacy", dir)
	if err != nil {
		t.Fatalf("load legacy: %v", err)
	}
	if cp.Version != SessionCheckpointVersion || cp.Turn != 2 || len(cp.Messages) != 1 {
		t.Fatalf("legacy load = %+v, want version %d, turn 2, 1 message", *cp, SessionCheckpointVersion)
	}
}

func TestRunArmSurfacesCheckpointSaveError(t *testing.T) {
	// The checkpoint "directory" is a regular file, so every save fails. The turn must
	// still complete, and the failure must be on the witness instead of dropped.
	blocker := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}
	m, err := RunArm(context.Background(), NewMockPlanner("mock-deterministic"), DefaultTask, true, 10, nil,
		WithSessionCheckpoint("sess-unsaved", blocker),
	)
	if err != nil {
		t.Fatalf("RunArm must not fail on a checkpoint save error: %v", err)
	}
	if !m.TaskCompleted {
		t.Fatalf("task should still complete: %+v", m)
	}
	if m.CheckpointSaveErrors == 0 || m.CheckpointSaveError == "" {
		t.Fatalf("checkpoint save failure not surfaced: errors=%d last=%q", m.CheckpointSaveErrors, m.CheckpointSaveError)
	}
	if m.CheckpointSaveErrors != m.Turns {
		t.Fatalf("CheckpointSaveErrors = %d, want one per turn (%d)", m.CheckpointSaveErrors, m.Turns)
	}
}

func TestRunArmRefusesToOverwriteNewerCheckpoint(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "sess-newer.json")
	newer := []byte(`{"version": 2, "session_id": "sess-newer", "turn": 7}`)
	if err := os.WriteFile(target, newer, 0o600); err != nil {
		t.Fatalf("write newer checkpoint: %v", err)
	}
	m, err := RunArm(context.Background(), NewMockPlanner("mock-deterministic"), DefaultTask, true, 10, nil,
		WithSessionCheckpoint("sess-newer", dir),
	)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.CheckpointSaveErrors == 0 || !strings.Contains(m.CheckpointSaveError, "unsupported version") {
		t.Fatalf("refusal not surfaced: errors=%d last=%q", m.CheckpointSaveErrors, m.CheckpointSaveError)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read newer checkpoint: %v", err)
	}
	if !bytes.Equal(got, newer) {
		t.Fatalf("a newer-version checkpoint was overwritten:\n got %s\nwant %s", got, newer)
	}
}
