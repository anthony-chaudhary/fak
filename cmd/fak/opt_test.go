package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeOptDiscoverWorkspace drops a single annotated const into a temp dir so
// DiscoverDir has exactly one well-formed target to harvest.
func writeOptDiscoverWorkspace(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	src := `package knobs
// fak:opttarget name=test-knob metric=test_score dir=higher sweep=1,2,3 measurer=fake
const TestKnob = 1
`
	if err := os.WriteFile(filepath.Join(tmp, "knobs.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return tmp
}

func TestOptDiscover(t *testing.T) {
	tmp := writeOptDiscoverWorkspace(t)

	t.Run("human", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runOpt(&out, &errb, []string{"discover", "--workspace", tmp})
		if code != 0 {
			t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
		}
		if !strings.Contains(out.String(), "test-knob") {
			t.Fatalf("stdout missing target name %q:\n%s", "test-knob", out.String())
		}
	})

	t.Run("check-present", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runOpt(&out, &errb, []string{"discover", "--workspace", tmp, "--check", "test-knob"})
		if code != 0 {
			t.Fatalf("exit = %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
		}
	})

	t.Run("check-missing", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runOpt(&out, &errb, []string{"discover", "--workspace", tmp, "--check", "missing-knob"})
		if code != 1 {
			t.Fatalf("exit = %d, want 1\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
		}
	})
}
