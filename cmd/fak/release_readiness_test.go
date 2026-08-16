package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestReleaseReadinessJSONIsNativeAndGateable(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		_ = os.MkdirAll(filepath.Dir(p), 0755)
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("cmd/fak/main.go", `package main
func f(){switch "" {case "release":;case "release-staleness":}}
`)
	write("AGENTS.md", "## Release\n")
	write("llms.txt", "skills/release\n")
	write("Makefile", "release-staleness:\n")
	write(".claude/skills/release/SKILL.md", "# release\n")
	c := exec.Command("git", "init")
	c.Dir = root
	if out, err := c.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
	var out, errb bytes.Buffer
	rc := runReleaseReadiness(&out, &errb, []string{"--workspace", root, "--json", "--skip-gh"})
	if rc != 0 {
		t.Fatalf("exit=%d stderr=%s", rc, errb.String())
	}
	var p map[string]any
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatalf("json: %v: %s", err, out.String())
	}
	if p["schema"] != "fak-release-readiness/1" {
		t.Fatalf("schema=%v", p["schema"])
	}
	out.Reset()
	errb.Reset()
	if rc := runReleaseReadiness(&out, &errb, []string{"--workspace", root, "--check", "--skip-gh"}); rc != 1 {
		t.Fatalf("check exit=%d, want 1", rc)
	}
}
