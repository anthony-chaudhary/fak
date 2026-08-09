package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestIndexGraphReadsHEADNotWorkingTree(t *testing.T) {
	d := t.TempDir()
	run := func(a ...string) {
		c := exec.Command("git", a...)
		c.Dir = d
		if b, e := c.CombinedOutput(); e != nil {
			t.Fatalf("git %v: %v %s", a, e, b)
		}
	}
	run("init", "-q")
	run("config", "user.email", "t@example.com")
	run("config", "user.name", "T")
	os.WriteFile(filepath.Join(d, "README.md"), []byte("[doc](docs/a.md)"), 0644)
	os.Mkdir(filepath.Join(d, "docs"), 0755)
	os.WriteFile(filepath.Join(d, "docs", "a.md"), []byte("head"), 0644)
	run("add", ".")
	run("commit", "-qm", "fixture")
	os.WriteFile(filepath.Join(d, "README.md"), []byte("[bad](missing.md)"), 0644)
	old, _ := os.Getwd()
	os.Chdir(d)
	defer os.Chdir(old)
	var out, er bytes.Buffer
	if code := indexGraphMain(&out, &er, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	if strings.Contains(out.String(), "missing.md") {
		t.Fatalf("used dirty worktree: %s", out.String())
	}
	if !strings.Contains(out.String(), `"rule": "R-LINK"`) {
		t.Fatalf("missing named rule: %s", out.String())
	}
}
