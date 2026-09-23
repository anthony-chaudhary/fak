package agentqueue

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"syscall"
	"testing"
)

func TestStoreLoadRetriesTransientWindowsSharingFailure(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows sharing retry")
	}
	path := filepath.Join(t.TempDir(), "queue.json")
	want := persistedFixture()
	if err := (Store{Path: path}).Save(want); err != nil {
		t.Fatal(err)
	}
	original := agentqueueReadSnapshotFile
	calls := 0
	agentqueueReadSnapshotFile = func(gotPath string) ([]byte, error) {
		calls++
		if calls == 1 {
			return nil, &os.PathError{Op: "open", Path: gotPath, Err: syscall.Errno(32)}
		}
		return original(gotPath)
	}
	t.Cleanup(func() { agentqueueReadSnapshotFile = original })

	got, err := (Store{Path: path}).Load()
	if err != nil {
		t.Fatalf("Load after transient sharing violation: %v", err)
	}
	if calls != 2 {
		t.Fatalf("reader calls=%d, want one retry", calls)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("snapshot=%+v, want %+v", got, want)
	}
}

func TestStoreLoadBoundsPersistentWindowsSharingFailure(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows sharing retry")
	}
	original := agentqueueReadSnapshotFile
	calls := 0
	agentqueueReadSnapshotFile = func(path string) ([]byte, error) {
		calls++
		return nil, &os.PathError{Op: "open", Path: path, Err: syscall.Errno(33)}
	}
	t.Cleanup(func() { agentqueueReadSnapshotFile = original })

	_, err := (Store{Path: filepath.Join(t.TempDir(), "queue.json")}).Load()
	if err == nil || !errors.Is(err, syscall.Errno(33)) {
		t.Fatalf("Load error=%v, want persistent lock violation", err)
	}
	if calls != 12 {
		t.Fatalf("reader calls=%d, want bounded 12 attempts", calls)
	}
}

func TestStoreLoadDoesNotRetryMissingFile(t *testing.T) {
	original := agentqueueReadSnapshotFile
	calls := 0
	agentqueueReadSnapshotFile = func(path string) ([]byte, error) {
		calls++
		return nil, &os.PathError{Op: "open", Path: path, Err: os.ErrNotExist}
	}
	t.Cleanup(func() { agentqueueReadSnapshotFile = original })

	_, err := (Store{Path: filepath.Join(t.TempDir(), "missing.json")}).Load()
	if err == nil || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Load error=%v, want missing file", err)
	}
	if calls != 1 {
		t.Fatalf("reader calls=%d, missing file must not retry", calls)
	}
}

func TestStoreSaveRetriesTransientWindowsSharingFailure(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows sharing retry")
	}
	path := filepath.Join(t.TempDir(), "queue.json")
	store := Store{Path: path}
	want := persistedFixture()

	original := agentqueueRenameSnapshotFile
	calls := 0
	agentqueueRenameSnapshotFile = func(from, to string) error {
		calls++
		if calls == 1 {
			return &os.PathError{Op: "rename", Path: from, Err: syscall.Errno(5)}
		}
		return original(from, to)
	}
	t.Cleanup(func() { agentqueueRenameSnapshotFile = original })

	if err := store.Save(want); err != nil {
		t.Fatalf("Save after transient access denied: %v", err)
	}
	if calls != 2 {
		t.Fatalf("rename calls=%d, want one retry", calls)
	}
	got, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("snapshot=%+v, want %+v", got, want)
	}
}

func TestStoreSaveBoundsPersistentWindowsSharingFailureWithoutCorruption(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows sharing retry")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "queue.json")
	store := Store{Path: path}
	originalSnapshot := persistedFixture()
	if err := store.Save(originalSnapshot); err != nil {
		t.Fatal(err)
	}

	originalRename := agentqueueRenameSnapshotFile
	calls := 0
	agentqueueRenameSnapshotFile = func(from, _ string) error {
		calls++
		return &os.PathError{Op: "rename", Path: from, Err: syscall.Errno(5)}
	}
	t.Cleanup(func() { agentqueueRenameSnapshotFile = originalRename })

	replacement := persistedFixture()
	replacement.Generation = "generation-replacement"
	err := store.Save(replacement)
	if err == nil || !errors.Is(err, syscall.Errno(5)) {
		t.Fatalf("Save error=%v, want persistent access denied", err)
	}
	if calls != 12 {
		t.Fatalf("rename calls=%d, want bounded 12 attempts", calls)
	}
	got, err := store.Load()
	if err != nil {
		t.Fatalf("Load committed snapshot after failed replacement: %v", err)
	}
	if !reflect.DeepEqual(got, originalSnapshot) {
		t.Fatalf("committed snapshot corrupted: got %+v, want %+v", got, originalSnapshot)
	}
	temps, err := filepath.Glob(filepath.Join(dir, ".agentqueue-*.tmp"))
	if err != nil {
		t.Fatal(err)
	}
	if len(temps) != 0 {
		t.Fatalf("failed Save left snapshot temps: %v", temps)
	}
}
