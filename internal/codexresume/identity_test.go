package codexresume

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const (
	testThreadIDOne = "019ff1af-9d63-7452-8109-58172f63c3e9"
	testThreadIDTwo = "019ff1af-9d63-7452-8109-58172f63c3ea"
)

func TestThreadIdentityValidation(t *testing.T) {
	got, err := NewCodexThreadIdentity(testThreadIDOne)
	if err != nil {
		t.Fatal(err)
	}
	if got.Provider != ThreadProviderCodex || got.ID != testThreadIDOne {
		t.Fatalf("identity=%+v", got)
	}
	for _, invalid := range []string{"", "thread-one", "019ff1af-9d63-7452-8109-not-a-uuid", strings.ToUpper(testThreadIDOne)} {
		if _, err := NewCodexThreadIdentity(invalid); err == nil {
			t.Fatalf("NewCodexThreadIdentity(%q) succeeded", invalid)
		}
	}
}

func TestWriterResourceHandleBindsExactThreadAndCanonicalLock(t *testing.T) {
	one, _ := NewCodexThreadIdentity(testThreadIDOne)
	two, _ := NewCodexThreadIdentity(testThreadIDTwo)
	lockPath := filepath.Join(t.TempDir(), "locks", "writer.lock")
	first, err := NewWriterResourceHandle(one, lockPath)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewWriterResourceHandle(two, lockPath)
	if err != nil {
		t.Fatal(err)
	}
	if first.ResourceID == second.ResourceID {
		t.Fatalf("different threads share resource ID %q", first.ResourceID)
	}
	equivalent, err := NewWriterResourceHandle(one, filepath.Join(filepath.Dir(lockPath), ".", filepath.Base(lockPath)))
	if err != nil {
		t.Fatal(err)
	}
	if first.ResourceID != equivalent.ResourceID || first.LockPath != equivalent.LockPath {
		t.Fatalf("canonical equivalents differ: first=%+v equivalent=%+v", first, equivalent)
	}
	if runtime.GOOS == "windows" {
		caseVariant, err := NewWriterResourceHandle(one, strings.ToUpper(lockPath))
		if err != nil {
			t.Fatal(err)
		}
		if first.ResourceID != caseVariant.ResourceID {
			t.Fatalf("Windows case-equivalent paths differ: %q != %q", first.ResourceID, caseVariant.ResourceID)
		}
	}
}

func TestWriterResourceHandleValidationRejectsMismatchAndUnsupportedVersion(t *testing.T) {
	thread, _ := NewCodexThreadIdentity(testThreadIDOne)
	handle, err := NewWriterResourceHandle(thread, filepath.Join(t.TempDir(), "writer.lock"))
	if err != nil {
		t.Fatal(err)
	}
	mismatched := handle
	mismatched.Thread, _ = NewCodexThreadIdentity(testThreadIDTwo)
	if err := mismatched.Validate(); err == nil {
		t.Fatal("resource/thread mismatch accepted")
	}
	unsupported := handle
	unsupported.Version = "writer-resource-v2"
	if err := unsupported.Validate(); err == nil {
		t.Fatal("unsupported handle version accepted")
	}
}
