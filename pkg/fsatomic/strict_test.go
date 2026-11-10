package fsatomic

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=0.1s lane=default
func TestWriteFileAtomicStrictPreservesPeerStoreOnRenameFailure(t *testing.T) {
	old := []byte(`{"schema_version":1,"epoch":7,"leases":[{"id":"peer-lease","holder":"peer","node_id":"peer-node"}]}` + "\n")
	updated := []byte(`{"schema_version":1,"epoch":8,"leases":[{"id":"peer-lease"},{"id":"new-lease"}]}` + "\n")
	errRetry := errors.New("simulated Windows sharing refusal")
	errFinal := errors.New("permanent rename failure")
	for _, scenario := range []struct {
		name    string
		calls   int
		wantErr error
	}{
		{"initial-refusal", 1, errFinal},
		{"second-refusal", 2, errFinal},
		{"retry-exhausted", renameAttempts, errRetry},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "leases.json")
			if err := os.WriteFile(path, old, 0o600); err != nil {
				t.Fatal(err)
			}
			before, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			// Execute the production strict replacement policy with a simulated
			// Windows transient classifier. Only syscalls and sleeps are fake.
			rename := func(from, to string) error {
				calls++
				got, err := os.ReadFile(to)
				if err != nil || !bytes.Equal(got, old) {
					t.Fatal("destination changed or disappeared before replacement")
				}
				tmp, err := os.ReadFile(from)
				if err != nil || !bytes.Equal(tmp, updated) {
					t.Fatal("replacement is not the complete new state")
				}
				if calls < scenario.calls || scenario.calls == renameAttempts {
					return errRetry
				}
				return errFinal
			}
			err = WriteFileAtomicWithRename(path, updated, 0o644, func(from, to string) error {
				return strictRenameWith(from, to, rename, func(err error) bool { return errors.Is(err, errRetry) }, func(time.Duration) {})
			})
			if !errors.Is(err, scenario.wantErr) || calls != scenario.calls {
				t.Fatalf("err=%v calls=%d, want failure after %d attempts", err, calls, scenario.calls)
			}
			got, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(got, old) {
				t.Fatal("failed strict replacement lost peer state")
			}
			after, err := os.Stat(path)
			if err != nil || !os.SameFile(before, after) || before.Mode() != after.Mode() {
				t.Fatal("failed strict replacement changed destination identity or mode")
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 1 || entries[0].Name() != "leases.json" {
				t.Fatal("failed strict replacement leaked temporary files")
			}
		})
	}
	t.Run("public-injected-syscall", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "leases.json")
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		err := WriteFileAtomicStrictWithRename(path, updated, 0o644, func(string, string) error { return errFinal })
		if !errors.Is(err, errFinal) {
			t.Fatalf("strict syscall error=%v", err)
		}
		got, readErr := os.ReadFile(path)
		entries, dirErr := os.ReadDir(dir)
		if readErr != nil || !bytes.Equal(got, old) || dirErr != nil || len(entries) != 1 {
			t.Fatal("injected strict writer did not preserve state and clean temporary")
		}
		if err := WriteFileAtomicStrict(path, updated, 0o644); err != nil {
			t.Fatal(err)
		}
		if got, err := os.ReadFile(path); err != nil || !bytes.Equal(got, updated) {
			t.Fatal("successful strict writer did not replace complete state")
		}
	})
}
