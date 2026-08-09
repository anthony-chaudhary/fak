package claimcheck

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFingerprintChangesWithUncommittedWorkingTreeBytes(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	path := filepath.Join("internal", "leaf.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package leaf\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	before, err := Fingerprint("abc123", []string{path}, "rules-v1", "judge-v1")
	if err != nil {
		t.Fatalf("Fingerprint(before): %v", err)
	}
	// This edit is intentionally never committed. The fingerprint must describe
	// bytes under review, not merely the unchanged HEAD named above.
	if err := os.WriteFile(path, []byte("package leaf\n\nconst Dirty = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	after, err := Fingerprint("abc123", []string{path}, "rules-v1", "judge-v1")
	if err != nil {
		t.Fatalf("Fingerprint(after): %v", err)
	}
	if before == after {
		t.Fatalf("fingerprint reused across an uncommitted edit: %s", before)
	}
}

func TestFingerprintIncludesUntrackedFileAndIsOrderIndependent(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	for name, content := range map[string]string{"new.txt": "untracked", "tracked.txt": "working tree"} {
		if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	first, err := Fingerprint("abc123", []string{"tracked.txt", "new.txt"}, "rules-v1", "judge-v1")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Fingerprint("abc123", []string{"new.txt", "tracked.txt"}, "rules-v1", "judge-v1")
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("path order changed fingerprint: %s != %s", first, second)
	}
}

func TestFingerprintFailsClosedOnIncompleteOrUnreadableInput(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile("scope.go", []byte("package scope\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name      string
		head      string
		paths     []string
		rules     string
		evaluator string
	}{
		{name: "missing head", paths: []string{"scope.go"}, rules: "rules-v1", evaluator: "judge-v1"},
		{name: "missing paths", head: "abc123", rules: "rules-v1", evaluator: "judge-v1"},
		{name: "missing rules digest", head: "abc123", paths: []string{"scope.go"}, evaluator: "judge-v1"},
		{name: "missing evaluator digest", head: "abc123", paths: []string{"scope.go"}, rules: "rules-v1"},
		{name: "unreadable path", head: "abc123", paths: []string{"missing.go"}, rules: "rules-v1", evaluator: "judge-v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Fingerprint(tt.head, tt.paths, tt.rules, tt.evaluator)
			if err == nil {
				t.Fatalf("Fingerprint() = %q, want fail-closed error", got)
			}
			if got != "" {
				t.Fatalf("Fingerprint() returned reusable key %q with error %v", got, err)
			}
		})
	}
}
