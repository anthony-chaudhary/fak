package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func TestArchitectureJSON(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"leaf":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/leaf/leaf.go", "package leaf\n")
	var out, errout bytes.Buffer
	if rc := runArchitecture(&out, &errout, []string{"--workspace", root, "--leaf", "leaf", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	var r archreport.Report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != "fak-architecture/1" || len(r.Leaves) != 1 || r.Leaves[0].DeclaredTierName != "primitive" {
		t.Fatalf("%+v", r)
	}
}
func mustWriteArchitectureFile(t *testing.T, root, path, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestRunArchitectureTextNamesDependents(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runArchitecture(&out, &errOut, []string{"--leaf", "archreport"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "dependents=") {
		t.Fatalf("output does not name dependents: %s", out.String())
	}
}
