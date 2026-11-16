package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestReapMalformedLooseRefsReapsAnyUnusableLeaseRef pins the widened scope: the
// "bad object" wedge is a property of the refs/fak/locks/ NAMESPACE, not of the
// session-* kind alone, so a torn contract-/epoch-/bare-id ref is reaped too.
// Every retained class is asserted so the repair can never widen into deleting a
// valid lease or an ordinary ref.
func TestReapMalformedLooseRefsReapsAnyUnusableLeaseRef(t *testing.T) {
	gitDir := t.TempDir()
	locks := filepath.Join(gitDir, "refs", "fak", "locks")
	if err := os.MkdirAll(filepath.Join(locks, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	valid := "0123456789abcdef0123456789abcdef01234567\n"
	files := map[string][]byte{
		// Unusable: not a full object id and not a symref.
		"session-torn":   make([]byte, 41), // the crash-mid-update-ref shape
		"contract-short": []byte("deadbeef\n"),
		"epoch-garbage":  []byte("not-an-object\n"),
		"intent-bare":    []byte("refs/fak/locks/other\n"),
		"plain":          []byte("x\n"),
		// Retained: a valid id, even when the object is missing (object-DB damage).
		"session-good":     []byte(valid),
		"contract-missing": []byte("0123456789abcdef0123456789abcdef01234567\n"),
		// Retained: a symref resolves, so it never wedges a traversal.
		"intent-symref": []byte("ref: refs/fak/locks/other\n"),
		// Retained: Git's sibling lock is another writer's in-flight claim.
		"session-writing.lock": []byte("partial"),
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(locks, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir + "\n", 0, nil
	}}
	got, err := s.ReapMalformedLooseRefs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		refPrefix + "session-torn":   true,
		refPrefix + "contract-short": true,
		refPrefix + "epoch-garbage":  true,
		refPrefix + "intent-bare":    true,
		refPrefix + "plain":          true,
	}
	if len(got) != len(want) {
		t.Fatalf("reaped = %#v, want %d entries", got, len(want))
	}
	for _, ref := range got {
		if !want[ref] {
			t.Fatalf("unexpected reap %q (reaped = %#v)", ref, got)
		}
		delete(want, ref)
	}
	if len(want) != 0 {
		t.Fatalf("missed reap(s) %#v (reaped = %#v)", want, got)
	}
	for _, name := range []string{"session-torn", "contract-short", "epoch-garbage", "intent-bare", "plain"} {
		if _, err := os.Stat(filepath.Join(locks, name)); !os.IsNotExist(err) {
			t.Fatalf("%s still exists: %v", name, err)
		}
	}
	for _, name := range []string{"session-good", "contract-missing", "intent-symref", "session-writing.lock", "nested"} {
		if _, err := os.Stat(filepath.Join(locks, name)); err != nil {
			t.Fatalf("retained ref %s was removed: %v", name, err)
		}
	}
}

// TestReapMalformedLooseRefsNeverTouchesOrdinaryRefs pins the blast radius: a
// torn ref OUTSIDE the lease namespace is not this repair's business, and must
// survive even though it would wedge `--all` just as hard.
func TestReapMalformedLooseRefsNeverTouchesOrdinaryRefs(t *testing.T) {
	gitDir := t.TempDir()
	locks := filepath.Join(gitDir, "refs", "fak", "locks")
	if err := os.MkdirAll(locks, 0o755); err != nil {
		t.Fatal(err)
	}
	// A torn branch ref, beside the lease namespace the repair owns.
	heads := filepath.Join(gitDir, "refs", "heads")
	if err := os.MkdirAll(heads, 0o755); err != nil {
		t.Fatal(err)
	}
	tornHead := filepath.Join(heads, "torn")
	if err := os.WriteFile(tornHead, []byte("not-an-object\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir + "\n", 0, nil
	}}
	got, err := s.ReapMalformedLooseRefs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("reaped = %#v, want none", got)
	}
	if _, err := os.Stat(tornHead); err != nil {
		t.Fatalf("ordinary ref was touched: %v", err)
	}
}

func TestReapMalformedLooseRefsDefersToConcurrentGitWriter(t *testing.T) {
	gitDir := t.TempDir()
	locks := filepath.Join(gitDir, "refs", "fak", "locks")
	if err := os.MkdirAll(locks, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(locks, "session-torn")
	if err := os.WriteFile(path, make([]byte, 41), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+".lock", []byte("writer owns this"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir, 0, nil
	}}
	got, err := s.ReapMalformedLooseRefs(context.Background())
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("concurrent writer's ref was removed: %v", err)
	}
}

func TestReapMalformedLooseRefsMissingDirectoryIsClean(t *testing.T) {
	gitDir := t.TempDir()
	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir, 0, nil
	}}
	got, err := s.ReapMalformedLooseRefs(context.Background())
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
