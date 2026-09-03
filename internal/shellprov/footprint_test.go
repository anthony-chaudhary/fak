package shellprov

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestFootprintRecorder(t *testing.T) {
	t.Run("SnapshotErrors", func(t *testing.T) {
		recorder := NewFootprintRecorder()

		// Empty directory argument
		if _, err := recorder.Snapshot(""); err == nil {
			t.Fatal("Snapshot(\"\") expected error, got nil")
		}

		// Non-existent directory
		if _, err := recorder.Snapshot(filepath.Join(t.TempDir(), "non-existent")); err == nil {
			t.Fatal("Snapshot(non-existent) expected error, got nil")
		}

		// Target is a file, not a directory
		tempDir := t.TempDir()
		filePath := filepath.Join(tempDir, "file.txt")
		if err := os.WriteFile(filePath, []byte("hello"), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := recorder.Snapshot(filePath); err == nil {
			t.Fatal("Snapshot(file) expected error, got nil")
		}
	})

	t.Run("SnapshotCapturesFilesAndSubdirectories", func(t *testing.T) {
		tempDir := t.TempDir()
		file1 := filepath.Join(tempDir, "root.txt")
		subDir := filepath.Join(tempDir, "sub", "nested")
		if err := os.MkdirAll(subDir, 0755); err != nil {
			t.Fatal(err)
		}
		file2 := filepath.Join(subDir, "nested.txt")

		if err := os.WriteFile(file1, []byte("root file"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file2, []byte("nested file content"), 0644); err != nil {
			t.Fatal(err)
		}

		recorder := NewFootprintRecorder()
		snap, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatalf("Snapshot: %v", err)
		}

		if len(snap.Files) != 2 {
			t.Fatalf("snapshot file count = %d, want 2", len(snap.Files))
		}

		state1, ok1 := snap.Files["root.txt"]
		if !ok1 {
			t.Fatal("missing root.txt in snapshot")
		}
		if state1.Size != int64(len("root file")) {
			t.Fatalf("root.txt size = %d, want %d", state1.Size, len("root file"))
		}
		if state1.Hash == "" {
			t.Fatal("root.txt hash is empty")
		}

		state2, ok2 := snap.Files["sub/nested/nested.txt"]
		if !ok2 {
			t.Fatal("missing sub/nested/nested.txt in snapshot")
		}
		if state2.Size != int64(len("nested file content")) {
			t.Fatalf("nested.txt size = %d, want %d", state2.Size, len("nested file content"))
		}
		if state2.Hash == "" {
			t.Fatal("nested.txt hash is empty")
		}
	})

	t.Run("DiffAddedFiles", func(t *testing.T) {
		tempDir := t.TempDir()
		recorder := NewFootprintRecorder()

		before, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		newFile := filepath.Join(tempDir, "added.txt")
		if err := os.WriteFile(newFile, []byte("added content"), 0644); err != nil {
			t.Fatal(err)
		}

		after, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		fp := recorder.Diff(before, after)
		if len(fp.AddedFiles) != 1 || fp.AddedFiles[0] != "added.txt" {
			t.Fatalf("AddedFiles = %v, want [added.txt]", fp.AddedFiles)
		}
		if len(fp.ModifiedFiles) != 0 {
			t.Fatalf("ModifiedFiles = %v, want empty", fp.ModifiedFiles)
		}
		if len(fp.DeletedFiles) != 0 {
			t.Fatalf("DeletedFiles = %v, want empty", fp.DeletedFiles)
		}
		if fp.TotalMutations != 1 {
			t.Fatalf("TotalMutations = %d, want 1", fp.TotalMutations)
		}
		if fp.Timestamp.IsZero() {
			t.Fatal("Timestamp is zero")
		}
	})

	t.Run("DiffModifiedFiles", func(t *testing.T) {
		tempDir := t.TempDir()
		filePath := filepath.Join(tempDir, "target.txt")
		if err := os.WriteFile(filePath, []byte("initial"), 0644); err != nil {
			t.Fatal(err)
		}

		recorder := NewFootprintRecorder()
		before, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		newContent := []byte("updated content longer")
		if err := os.WriteFile(filePath, newContent, 0644); err != nil {
			t.Fatal(err)
		}

		after, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		fp := recorder.Diff(before, after)
		if len(fp.AddedFiles) != 0 {
			t.Fatalf("AddedFiles = %v, want empty", fp.AddedFiles)
		}
		if len(fp.ModifiedFiles) != 1 || fp.ModifiedFiles[0] != "target.txt" {
			t.Fatalf("ModifiedFiles = %v, want [target.txt]", fp.ModifiedFiles)
		}
		if len(fp.DeletedFiles) != 0 {
			t.Fatalf("DeletedFiles = %v, want empty", fp.DeletedFiles)
		}
		if fp.TotalMutations != 1 {
			t.Fatalf("TotalMutations = %d, want 1", fp.TotalMutations)
		}
		if fp.BytesModified != int64(len(newContent)) {
			t.Fatalf("BytesModified = %d, want %d", fp.BytesModified, len(newContent))
		}
	})

	t.Run("DiffDeletedFiles", func(t *testing.T) {
		tempDir := t.TempDir()
		filePath := filepath.Join(tempDir, "doomed.txt")
		if err := os.WriteFile(filePath, []byte("to be deleted"), 0644); err != nil {
			t.Fatal(err)
		}

		recorder := NewFootprintRecorder()
		before, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		if err := os.Remove(filePath); err != nil {
			t.Fatal(err)
		}

		after, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		fp := recorder.Diff(before, after)
		if len(fp.AddedFiles) != 0 {
			t.Fatalf("AddedFiles = %v, want empty", fp.AddedFiles)
		}
		if len(fp.ModifiedFiles) != 0 {
			t.Fatalf("ModifiedFiles = %v, want empty", fp.ModifiedFiles)
		}
		if len(fp.DeletedFiles) != 1 || fp.DeletedFiles[0] != "doomed.txt" {
			t.Fatalf("DeletedFiles = %v, want [doomed.txt]", fp.DeletedFiles)
		}
		if fp.TotalMutations != 1 {
			t.Fatalf("TotalMutations = %d, want 1", fp.TotalMutations)
		}
	})

	t.Run("DiffMixedMutations", func(t *testing.T) {
		tempDir := t.TempDir()
		delFile := filepath.Join(tempDir, "del.txt")
		modFile := filepath.Join(tempDir, "mod.txt")
		unmodFile := filepath.Join(tempDir, "keep.txt")

		if err := os.WriteFile(delFile, []byte("delete me"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(modFile, []byte("before edit"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(unmodFile, []byte("constant"), 0644); err != nil {
			t.Fatal(err)
		}

		recorder := NewFootprintRecorder()
		before, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		// Perform mutations
		if err := os.Remove(delFile); err != nil {
			t.Fatal(err)
		}
		modContent := []byte("after edit mutated")
		if err := os.WriteFile(modFile, modContent, 0644); err != nil {
			t.Fatal(err)
		}
		addFile := filepath.Join(tempDir, "add.txt")
		if err := os.WriteFile(addFile, []byte("brand new"), 0644); err != nil {
			t.Fatal(err)
		}

		after, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		fp := recorder.Diff(before, after)
		if len(fp.AddedFiles) != 1 || fp.AddedFiles[0] != "add.txt" {
			t.Fatalf("AddedFiles = %v, want [add.txt]", fp.AddedFiles)
		}
		if len(fp.ModifiedFiles) != 1 || fp.ModifiedFiles[0] != "mod.txt" {
			t.Fatalf("ModifiedFiles = %v, want [mod.txt]", fp.ModifiedFiles)
		}
		if len(fp.DeletedFiles) != 1 || fp.DeletedFiles[0] != "del.txt" {
			t.Fatalf("DeletedFiles = %v, want [del.txt]", fp.DeletedFiles)
		}
		if fp.TotalMutations != 3 {
			t.Fatalf("TotalMutations = %d, want 3", fp.TotalMutations)
		}
		if fp.BytesModified != int64(len(modContent)) {
			t.Fatalf("BytesModified = %d, want %d", fp.BytesModified, len(modContent))
		}
	})

	t.Run("DiffNoChanges", func(t *testing.T) {
		tempDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(tempDir, "static.txt"), []byte("same"), 0644); err != nil {
			t.Fatal(err)
		}

		recorder := NewFootprintRecorder()
		before, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}
		after, err := recorder.Snapshot(tempDir)
		if err != nil {
			t.Fatal(err)
		}

		fp := recorder.Diff(before, after)
		if fp.TotalMutations != 0 {
			t.Fatalf("TotalMutations = %d, want 0", fp.TotalMutations)
		}
		if len(fp.AddedFiles) != 0 || len(fp.ModifiedFiles) != 0 || len(fp.DeletedFiles) != 0 {
			t.Fatalf("expected empty mutation slices, got added=%v, mod=%v, del=%v",
				fp.AddedFiles, fp.ModifiedFiles, fp.DeletedFiles)
		}
		if fp.BytesModified != 0 {
			t.Fatalf("BytesModified = %d, want 0", fp.BytesModified)
		}
	})

	t.Run("RecordExecutionSuccess", func(t *testing.T) {
		tempDir := t.TempDir()
		recorder := NewFootprintRecorder()

		sessionID := "test-session-123"
		fp, err := recorder.RecordExecution(tempDir, sessionID, func() error {
			return os.WriteFile(filepath.Join(tempDir, "exec_out.txt"), []byte("output"), 0644)
		})
		if err != nil {
			t.Fatalf("RecordExecution: %v", err)
		}

		if fp.SessionID != sessionID {
			t.Fatalf("SessionID = %q, want %q", fp.SessionID, sessionID)
		}
		if len(fp.AddedFiles) != 1 || fp.AddedFiles[0] != "exec_out.txt" {
			t.Fatalf("AddedFiles = %v, want [exec_out.txt]", fp.AddedFiles)
		}
		if fp.TotalMutations != 1 {
			t.Fatalf("TotalMutations = %d, want 1", fp.TotalMutations)
		}
	})

	t.Run("RecordExecutionPartialError", func(t *testing.T) {
		tempDir := t.TempDir()
		recorder := NewFootprintRecorder()

		sessionID := "test-session-err"
		boomErr := errors.New("boom")
		fp, err := recorder.RecordExecution(tempDir, sessionID, func() error {
			// Creates a file before failing
			_ = os.WriteFile(filepath.Join(tempDir, "partial.txt"), []byte("partial"), 0644)
			return boomErr
		})

		if !errors.Is(err, boomErr) {
			t.Fatalf("RecordExecution err = %v, want boomErr", err)
		}
		if fp == nil {
			t.Fatal("expected non-nil footprint on partial execution error")
		}
		if len(fp.AddedFiles) != 1 || fp.AddedFiles[0] != "partial.txt" {
			t.Fatalf("AddedFiles = %v, want [partial.txt]", fp.AddedFiles)
		}
		if fp.TotalMutations != 1 {
			t.Fatalf("TotalMutations = %d, want 1", fp.TotalMutations)
		}
	})

	t.Run("RecordExecutionInvalidDir", func(t *testing.T) {
		recorder := NewFootprintRecorder()
		ran := false
		_, err := recorder.RecordExecution("", "sess", func() error {
			ran = true
			return nil
		})
		if err == nil {
			t.Fatal("expected error on empty dir")
		}
		if ran {
			t.Fatal("fn should not have been executed on invalid dir")
		}
	})

	t.Run("WriteJSONLRoundTrip", func(t *testing.T) {
		fp := ExecutionFootprint{
			SessionID:      "sess-round-trip",
			AddedFiles:     []string{"a.txt", "b.txt"},
			ModifiedFiles:  []string{"c.txt"},
			DeletedFiles:   []string{"d.txt"},
			TotalMutations: 4,
			BytesModified:  1024,
			Timestamp:      time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC),
		}

		var buf bytes.Buffer
		recorder := NewFootprintRecorder()
		if err := recorder.WriteJSONL(&buf, fp); err != nil {
			t.Fatalf("WriteJSONL: %v", err)
		}

		line := buf.Bytes()
		if len(line) == 0 || line[len(line)-1] != '\n' {
			t.Fatal("WriteJSONL output does not end with newline")
		}

		var decoded ExecutionFootprint
		if err := json.Unmarshal(bytes.TrimSuffix(line, []byte{'\n'}), &decoded); err != nil {
			t.Fatalf("Unmarshal JSON: %v", err)
		}

		if decoded.SessionID != fp.SessionID {
			t.Errorf("SessionID = %q, want %q", decoded.SessionID, fp.SessionID)
		}
		if !reflect.DeepEqual(decoded.AddedFiles, fp.AddedFiles) {
			t.Errorf("AddedFiles = %v, want %v", decoded.AddedFiles, fp.AddedFiles)
		}
		if !reflect.DeepEqual(decoded.ModifiedFiles, fp.ModifiedFiles) {
			t.Errorf("ModifiedFiles = %v, want %v", decoded.ModifiedFiles, fp.ModifiedFiles)
		}
		if !reflect.DeepEqual(decoded.DeletedFiles, fp.DeletedFiles) {
			t.Errorf("DeletedFiles = %v, want %v", decoded.DeletedFiles, fp.DeletedFiles)
		}
		if decoded.TotalMutations != fp.TotalMutations {
			t.Errorf("TotalMutations = %d, want %d", decoded.TotalMutations, fp.TotalMutations)
		}
		if decoded.BytesModified != fp.BytesModified {
			t.Errorf("BytesModified = %d, want %d", decoded.BytesModified, fp.BytesModified)
		}
		if !decoded.Timestamp.Equal(fp.Timestamp) {
			t.Errorf("Timestamp = %v, want %v", decoded.Timestamp, fp.Timestamp)
		}
	})

	t.Run("WriteJSONLNilWriter", func(t *testing.T) {
		recorder := NewFootprintRecorder()
		if err := recorder.WriteJSONL(nil, ExecutionFootprint{}); err == nil {
			t.Fatal("expected error with nil writer")
		}
	})

	t.Run("PackageLevelConvenienceFunctions", func(t *testing.T) {
		tempDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(tempDir, "pkg.txt"), []byte("pkg test"), 0644); err != nil {
			t.Fatal(err)
		}

		snap1, err := Snapshot(tempDir)
		if err != nil {
			t.Fatalf("Snapshot: %v", err)
		}

		diff := Diff(snap1, snap1)
		if diff.TotalMutations != 0 {
			t.Fatalf("Diff TotalMutations = %d, want 0", diff.TotalMutations)
		}

		var buf bytes.Buffer
		if err := WriteJSONL(&buf, diff); err != nil {
			t.Fatalf("WriteJSONL: %v", err)
		}

		fp, err := RecordExecution(tempDir, "pkg-session", func() error {
			return nil
		})
		if err != nil {
			t.Fatalf("RecordExecution: %v", err)
		}
		if fp.SessionID != "pkg-session" {
			t.Fatalf("SessionID = %q, want pkg-session", fp.SessionID)
		}
	})

	t.Run("CleanConceptTokensRule", func(t *testing.T) {
		// Contract requirement:
		// DO NOT use tokens like Context, Ctx, Render, Guard, Gate in new exported symbols.
		forbidden := []string{"context", "ctx", "render", "guard", "gate"}

		checkName := func(name string) {
			lower := strings.ToLower(name)
			for _, token := range forbidden {
				if strings.Contains(lower, token) {
					t.Fatalf("exported symbol %q contains forbidden concept token %q", name, token)
				}
			}
		}

		// Inspect ExecutionFootprint struct fields
		efType := reflect.TypeOf(ExecutionFootprint{})
		checkName(efType.Name())
		for i := 0; i < efType.NumField(); i++ {
			f := efType.Field(i)
			if f.IsExported() {
				checkName(f.Name)
			}
		}

		// Inspect FootprintRecorder methods
		recType := reflect.TypeOf(&FootprintRecorder{})
		checkName(recType.Elem().Name())
		for i := 0; i < recType.NumMethod(); i++ {
			m := recType.Method(i)
			if m.IsExported() {
				checkName(m.Name)
			}
		}

		// Inspect FilesystemSnapshot
		snapType := reflect.TypeOf(FilesystemSnapshot{})
		checkName(snapType.Name())
		for i := 0; i < snapType.NumField(); i++ {
			f := snapType.Field(i)
			if f.IsExported() {
				checkName(f.Name)
			}
		}

		// Inspect FileState
		fsType := reflect.TypeOf(FileState{})
		checkName(fsType.Name())
		for i := 0; i < fsType.NumField(); i++ {
			f := fsType.Field(i)
			if f.IsExported() {
				checkName(f.Name)
			}
		}
	})
}
