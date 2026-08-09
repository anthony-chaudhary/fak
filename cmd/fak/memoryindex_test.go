package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestMemoryIndexCheckThenWrite(t *testing.T) {
	d := t.TempDir()
	_ = os.WriteFile(filepath.Join(d, "MEMORY.md"), []byte("# Memory\n"), 0644)
	_ = os.WriteFile(filepath.Join(d, "a.md"), []byte("---\nname: a\ndescription: A\nmetadata:\n type: project\n---\n"), 0644)
	var out, err bytes.Buffer
	if c := runMemoryIndex(&out, &err, []string{"--dir", d, "--json"}); c != 1 {
		t.Fatalf("check=%d out=%s err=%s", c, out.String(), err.String())
	}
	if got, e := os.ReadFile(filepath.Join(d, "MEMORY.md")); e != nil || string(got) != "# Memory\n" {
		t.Fatalf("check mode changed index: %q err=%v", got, e)
	}
	out.Reset()
	if c := runMemoryIndex(&out, &err, []string{"--dir", d, "--write"}); c != 0 {
		t.Fatalf("write=%d %s", c, err.String())
	}
	if c := runMemoryIndex(&out, &err, []string{"--dir", d}); c != 0 {
		t.Fatalf("clean check=%d", c)
	}
}
