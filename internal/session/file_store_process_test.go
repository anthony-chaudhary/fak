package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"
)

const fileStoreHelperEnv = "FAK_SESSION_FILE_STORE_HELPER"

func TestFileStoreHelperProcess(t *testing.T) {
	if os.Getenv(fileStoreHelperEnv) == "" {
		return
	}
	path := os.Getenv("FAK_SESSION_FILE_STORE_PATH")
	id := os.Getenv("FAK_SESSION_FILE_STORE_ID")
	rev, err := strconv.ParseUint(os.Getenv("FAK_SESSION_FILE_STORE_REV"), 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	if ready := os.Getenv("FAK_SESSION_FILE_STORE_READY"); ready != "" {
		if err := os.WriteFile(ready, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		start := os.Getenv("FAK_SESSION_FILE_STORE_START")
		deadline := time.Now().Add(10 * time.Second)
		for {
			if _, err := os.Stat(start); err == nil {
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("timed out waiting for process start barrier")
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	if boundary := os.Getenv("FAK_SESSION_FILE_STORE_CRASH_AT"); boundary != "" {
		fileStoreFaultHook = func(stage string) {
			if stage == boundary {
				os.Exit(86)
			}
		}
	}
	if hold := os.Getenv("FAK_SESSION_FILE_STORE_HOLD_LOCK"); hold != "" {
		unlock, err := lockDescriptorFile(path)
		if err != nil {
			t.Fatal(err)
		}
		defer unlock()
		if err := os.WriteFile(hold, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		for {
			time.Sleep(time.Hour)
		}
	}
	if err := NewFileStore(path).Put(Descriptor{ID: id, Rev: rev}); err != nil {
		t.Fatal(err)
	}
}

func TestFileStorePreservesConcurrentCrossProcessUpdates(t *testing.T) {
	const workers = 12
	dir := t.TempDir()
	path := filepath.Join(dir, "sessions.json")
	start := filepath.Join(dir, "start")
	cmds := make([]*exec.Cmd, workers)
	outputs := make([]bytes.Buffer, workers)
	for i := range cmds {
		ready := filepath.Join(dir, fmt.Sprintf("ready-%02d", i))
		cmd := exec.Command(os.Args[0], "-test.run=^TestFileStoreHelperProcess$")
		cmd.Env = append(os.Environ(),
			fileStoreHelperEnv+"=1",
			"FAK_SESSION_FILE_STORE_PATH="+path,
			fmt.Sprintf("FAK_SESSION_FILE_STORE_ID=process-%02d", i),
			fmt.Sprintf("FAK_SESSION_FILE_STORE_REV=%d", i+1),
			"FAK_SESSION_FILE_STORE_READY="+ready,
			"FAK_SESSION_FILE_STORE_START="+start,
		)
		cmd.Stdout = &outputs[i]
		cmd.Stderr = &outputs[i]
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cmds[i] = cmd
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		matches, err := filepath.Glob(filepath.Join(dir, "ready-*"))
		if err != nil {
			t.Fatal(err)
		}
		if len(matches) == workers {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("only %d/%d helpers reached barrier", len(matches), workers)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := os.WriteFile(start, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	for i, cmd := range cmds {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("helper %d: %v\n%s", i, err, outputs[i].String())
		}
	}
	got, err := NewFileStore(path).List()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != workers {
		t.Fatalf("List() returned %d rows, want %d: %+v", len(got), workers, got)
	}
	for i, d := range got {
		wantID := fmt.Sprintf("process-%02d", i)
		if d.ID != wantID || d.Rev != uint64(i+1) {
			t.Fatalf("row %d = {%q rev=%d}, want {%q rev=%d}", i, d.ID, d.Rev, wantID, i+1)
		}
	}
}

func readFileRetrySharing(path string, timeout time.Duration) ([]byte, error) {
	deadline := time.Now().Add(timeout)
	for {
		b, err := os.ReadFile(path)
		if err == nil || time.Now().After(deadline) {
			return b, err
		}
		time.Sleep(time.Millisecond)
	}
}

func TestFileStoreReadersObserveCompleteDocuments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sessions.json")
	store := NewFileStore(path)
	if err := store.Put(Descriptor{ID: "seed"}); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 1)
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			b, err := readFileRetrySharing(path, time.Second)
			if err != nil {
				select {
				case errs <- err:
				default:
				}
				return
			}
			var doc descriptorFile
			if err := json.Unmarshal(b, &doc); err != nil {
				select {
				case errs <- fmt.Errorf("partial descriptor document %q: %w", b, err):
				default:
				}
				return
			}
			if doc.Version != descriptorFileVersion {
				select {
				case errs <- fmt.Errorf("unexpected version %q", doc.Version):
				default:
				}
				return
			}
		}
	}()
	for i := 0; i < 100; i++ {
		if err := store.Put(Descriptor{ID: fmt.Sprintf("row-%03d", i), Rev: uint64(i)}); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()
	select {
	case err := <-errs:
		t.Fatal(err)
	default:
	}
}

func TestFileStoreSameIDLastSuccessfulWriteWins(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sessions.json")
	first := NewFileStore(path)
	second := NewFileStore(path)
	if err := first.Put(Descriptor{ID: "same", Rev: 1}); err != nil {
		t.Fatal(err)
	}
	if err := second.Put(Descriptor{ID: "same", Rev: 2}); err != nil {
		t.Fatal(err)
	}
	got, err := first.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Rev != 2 {
		t.Fatalf("same-ID row = %+v, want last successful write rev 2", got)
	}
}

func runFileStoreHelper(t *testing.T, env ...string) ([]byte, error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestFileStoreHelperProcess$")
	cmd.Env = append(os.Environ(), append([]string{fileStoreHelperEnv + "=1"}, env...)...)
	return cmd.CombinedOutput()
}

func readDescriptorDocument(t *testing.T, path string) descriptorFile {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc descriptorFile
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("descriptor document %q is invalid: %v", b, err)
	}
	if doc.Version != descriptorFileVersion {
		t.Fatalf("descriptor version = %q, want %q", doc.Version, descriptorFileVersion)
	}
	return doc
}

func TestFileStoreCrashBoundariesLeaveValidOldOrNewDocument(t *testing.T) {
	for _, boundary := range []string{"encode", "flush", "close", "replace", "directory-sync"} {
		t.Run(boundary, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "sessions.json")
			if err := NewFileStore(path).Put(Descriptor{ID: "old", Rev: 1}); err != nil {
				t.Fatal(err)
			}
			out, err := runFileStoreHelper(t,
				"FAK_SESSION_FILE_STORE_PATH="+path,
				"FAK_SESSION_FILE_STORE_ID=new",
				"FAK_SESSION_FILE_STORE_REV=2",
				"FAK_SESSION_FILE_STORE_CRASH_AT="+boundary,
			)
			if err == nil {
				t.Fatalf("helper did not crash at %s\n%s", boundary, out)
			}
			doc := readDescriptorDocument(t, path)
			ids := make(map[string]bool, len(doc.Descriptors))
			for _, d := range doc.Descriptors {
				ids[d.ID] = true
			}
			if !ids["old"] {
				t.Fatalf("crash at %s lost old row: %+v", boundary, doc.Descriptors)
			}
			if boundary == "encode" || boundary == "flush" || boundary == "close" {
				if ids["new"] {
					t.Fatalf("pre-publication crash at %s exposed new row: %+v", boundary, doc.Descriptors)
				}
			}
			if boundary == "replace" || boundary == "directory-sync" {
				if !ids["new"] {
					t.Fatalf("post-publication crash at %s omitted new row: %+v", boundary, doc.Descriptors)
				}
			}

			// The next writer owns the abandoned kernel lock and removes stale temps.
			if err := NewFileStore(path).Put(Descriptor{ID: "recovered", Rev: 3}); err != nil {
				t.Fatalf("recovery write after %s crash: %v", boundary, err)
			}
			matches, err := filepath.Glob(filepath.Join(dir, ".session-descriptors-*.tmp"))
			if err != nil {
				t.Fatal(err)
			}
			if len(matches) != 0 {
				t.Fatalf("stale temp artifacts after recovery: %v", matches)
			}
		})
	}
}

func TestFileStoreAbandonedProcessLockRecoversPromptly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sessions.json")
	held := filepath.Join(dir, "held")
	cmd := exec.Command(os.Args[0], "-test.run=^TestFileStoreHelperProcess$")
	cmd.Env = append(os.Environ(),
		fileStoreHelperEnv+"=1",
		"FAK_SESSION_FILE_STORE_PATH="+path,
		"FAK_SESSION_FILE_STORE_ID=holder",
		"FAK_SESSION_FILE_STORE_REV=1",
		"FAK_SESSION_FILE_STORE_HOLD_LOCK="+held,
	)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, err := os.Stat(held); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("helper did not acquire descriptor lock")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Fatal(err)
	}
	_, _ = cmd.Process.Wait()
	started := time.Now()
	if err := NewFileStore(path).Put(Descriptor{ID: "successor", Rev: 2}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("abandoned lock recovery took %s, want <= 1s", elapsed)
	}
}
