package model

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41ActivationTracePublicationUsesOwnedTemporaryFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.json")
	legacyTemporary := path + ".tmp"
	const sentinel = "another writer owns this file"
	if err := os.WriteFile(legacyTemporary, []byte(sentinel), 0o600); err != nil {
		t.Fatal(err)
	}
	want := computetrace.Artifact{Schema: computetrace.Schema, InputTokens: []int{1, 3, 5}}
	if err := writeV41ActivationTrace(path, want); err != nil {
		t.Fatal(err)
	}
	untouched, err := os.ReadFile(legacyTemporary)
	if err != nil || string(untouched) != sentinel {
		t.Fatalf("writer changed another writer's temporary file: %q, %v", untouched, err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	got, readErr := computetrace.Read(f)
	closeErr := f.Close()
	if readErr != nil || closeErr != nil || !slices.Equal(got.InputTokens, want.InputTokens) {
		t.Fatalf("published artifact: %+v, read=%v, close=%v", got, readErr, closeErr)
	}
	remaining, err := filepath.Glob(filepath.Join(dir, ".fak-v41-trace-*"))
	if err != nil || len(remaining) != 0 {
		t.Fatalf("unremoved owned temporary files: %v, %v", remaining, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41ActivationTracePublicationUsesOriginalCreationMode(t *testing.T) {
	dir := t.TempDir()
	// This control asks the kernel for the old writer's fresh-file mode under
	// the same umask and directory defaults. Never change the process umask.
	control := filepath.Join(dir, "original-mode")
	if err := os.WriteFile(control, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	controlInfo, err := os.Stat(control)
	if err != nil {
		t.Fatal(err)
	}
	wantMode := controlInfo.Mode().Perm()
	if wantMode&0o044 == 0 {
		t.Log("current creation policy masks group/other read bits; this run cannot distinguish the old 0600 regression")
	}
	for _, existing := range []bool{false, true} {
		name := "fresh.json"
		if existing {
			name = "replace.json"
		}
		path := filepath.Join(dir, name)
		if existing {
			// The original rename-based writer replaced the destination mode;
			// copying a previous destination's 0600 mode would be a new policy.
			if err := os.WriteFile(path, []byte("previous output"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if err := writeV41ActivationTrace(path, computetrace.Artifact{Schema: computetrace.Schema}); err != nil {
			t.Fatal(err)
		}
		gotInfo, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if gotMode := gotInfo.Mode().Perm(); gotMode != wantMode {
			t.Fatalf("existing=%v: published mode %04o, fresh original-mode control %04o", existing, gotMode, wantMode)
		}
		remaining, err := filepath.Glob(filepath.Join(dir, ".fak-v41-trace-*"))
		if err != nil || len(remaining) != 0 {
			t.Fatalf("unremoved owned temporary files: %v, %v", remaining, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41ActivationTraceTemporaryRejectsNameCollisions(t *testing.T) {
	for _, tc := range []struct {
		name        string
		symlink     bool
		caseVariant bool
	}{
		{name: "file"},
		{name: "case-variant", caseVariant: true},
		{name: "symlink", symlink: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			first := bytes.Repeat([]byte{0}, 16)
			second := bytes.Repeat([]byte{1}, 16)
			third := bytes.Repeat([]byte{2}, 16)
			name := func(suffix []byte) string {
				return filepath.Join(dir, fmt.Sprintf(".fak-v41-trace-%x", suffix))
			}
			path, collision := name(first), name(second)
			if tc.caseVariant {
				path = filepath.Join(dir, fmt.Sprintf(".FAK-V41-TRACE-%x", first))
			}
			const sentinel = "another writer owns this file"
			victim := collision
			if tc.symlink {
				victim = filepath.Join(dir, "symlink-target")
			}
			if err := os.WriteFile(victim, []byte(sentinel), 0o600); err != nil {
				t.Fatal(err)
			}
			if tc.symlink {
				if err := os.Symlink(victim, collision); err != nil {
					t.Skipf("symlink creation unavailable: %v", err)
				}
			}
			entropy := bytes.NewReader(bytes.Join([][]byte{first, second, third}, nil))
			f, err := createV41ActivationTraceTemp(path, entropy)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				_ = f.Close()
				_ = os.Remove(f.Name())
			})
			if f.Name() != name(third) {
				t.Fatalf("temporary file %q, want third noncolliding name %q", f.Name(), name(third))
			}
			if _, err := os.Lstat(path); !os.IsNotExist(err) {
				t.Fatalf("temporary creation touched the unpublished destination: %v", err)
			}
			if got, err := os.ReadFile(victim); err != nil || string(got) != sentinel {
				t.Fatalf("collision modified the existing file: %q, %v", got, err)
			}
			if tc.symlink {
				info, err := os.Lstat(collision)
				if err != nil || info.Mode()&os.ModeSymlink == 0 {
					t.Fatalf("collision removed or replaced the symlink: %v", err)
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41ActivationTracePublicationFailureReleasesOwner(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing", "trace.json")
	withV41ActivationTraceEnv(t, path)
	m := &Model{}
	for _, forwardErr := range []error{nil, errors.New("forward failure")} {
		finish := m.v41ActivationTraceFromEnv([]int{1})
		if finish == nil {
			t.Fatal("env capture did not acquire a recorder")
		}
		got := finish(forwardErr)
		if got == nil || (forwardErr != nil && got != forwardErr) {
			t.Fatalf("error precedence: got %v, forward error %v", got, forwardErr)
		}
		if computetrace.Enabled() {
			t.Fatal("recorder remained enabled after failed publication")
		}
		if !v41ActivationTraceEnvMu.TryLock() {
			t.Fatal("env mutex remained locked after failed publication")
		}
		v41ActivationTraceEnvMu.Unlock()
	}
}
