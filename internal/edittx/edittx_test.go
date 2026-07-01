package edittx

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestApplyFailingValidationRollsBackFiveFiles(t *testing.T) {
	root := t.TempDir()
	paths := []string{"a.txt", "b.txt", "c.txt", "d.txt", "e.txt"}
	for i, p := range paths {
		writeFile(t, root, p, string(rune('A'+i))+"\n")
	}
	before := digestTreeForTest(t, root, paths)

	spec := Spec{Edits: fiveEdits("new\n"), Checks: []string{"validate"}}
	res := Apply(context.Background(), Options{
		Root: root,
		Spec: spec,
		Run: func(_ context.Context, root, command string) CheckResult {
			if b, err := os.ReadFile(filepath.Join(root, "c.txt")); err != nil || string(b) != "new\n" {
				t.Fatalf("validation did not see the applied five-file transaction: c.txt=%q err=%v", b, err)
			}
			return CheckResult{Command: command, ExitCode: 1, Output: "file 3 failed validation"}
		},
	})
	if res.OK || res.Applied || !res.RolledBack || res.Reason != ReasonCheckFailed {
		t.Fatalf("Apply failing validation = %+v, want CHECK_FAILED rollback", res)
	}
	after := digestTreeForTest(t, root, paths)
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("rollback did not restore byte-identical tree:\nbefore=%v\nafter=%v", before, after)
	}
}

func TestApplyPassingSetAppliesAllFiveFiles(t *testing.T) {
	root := t.TempDir()
	paths := []string{"a.txt", "b.txt", "c.txt", "d.txt", "e.txt"}
	for _, p := range paths {
		writeFile(t, root, p, "old\n")
	}

	res := Apply(context.Background(), Options{Root: root, Spec: Spec{Edits: fiveEdits("new\n")}})
	if !res.OK || !res.Applied || res.RolledBack || res.Reason != "" {
		t.Fatalf("Apply passing set = %+v, want applied OK", res)
	}
	for _, p := range paths {
		if got := readFile(t, root, p); got != "new\n" {
			t.Fatalf("%s = %q, want new content", p, got)
		}
	}
}

func TestApplyRejectsEscapingPathBeforeTouchingTree(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "keep.txt", "old\n")
	content := "bad\n"
	for _, path := range []string{"../escape.txt", `..\escape.txt`} {
		res := Apply(context.Background(), Options{Root: root, Spec: Spec{Edits: []Edit{{Path: path, Content: &content}}}})
		if res.OK || res.Reason != ReasonInvalidPath {
			t.Fatalf("Apply escaping path %q = %+v, want INVALID_PATH", path, res)
		}
	}
	if got := readFile(t, root, "keep.txt"); got != "old\n" {
		t.Fatalf("unrelated file changed on invalid path: %q", got)
	}
}

func TestApplyRejectsSymlinkParentEscape(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	content := "bad\n"
	res := Apply(context.Background(), Options{Root: root, Spec: Spec{Edits: []Edit{{Path: "link/escape.txt", Content: &content}}}})
	if res.OK || res.Reason != ReasonInvalidPath {
		t.Fatalf("Apply symlink escape = %+v, want INVALID_PATH", res)
	}
	if _, err := os.Stat(filepath.Join(outside, "escape.txt")); !os.IsNotExist(err) {
		t.Fatalf("outside symlink target was written, stat err=%v", err)
	}
}

func fiveEdits(content string) []Edit {
	edits := make([]Edit, 0, 5)
	for _, p := range []string{"a.txt", "b.txt", "c.txt", "d.txt", "e.txt"} {
		c := content
		edits = append(edits, Edit{Path: p, Content: &c})
	}
	return edits
}

func digestTreeForTest(t *testing.T, root string, paths []string) map[string]string {
	t.Helper()
	got, err := DigestTree(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func writeFile(t *testing.T, root, path, content string) {
	t.Helper()
	abs := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, root, path string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(b), "\r\n", "\n")
}
