package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeBoundaryFile creates root/rel (and parents) with content.
func writeBoundaryFile(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", p, err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
}

// A workspace with no boundary tells exits 0 and reports an honest empty,
// non-null tell list under --json.
func TestBoundaryCleanWorkspace(t *testing.T) {
	root := t.TempDir()
	writeBoundaryFile(t, root, "go.mod", "module example.com/x\n\ngo 1.22\n")
	writeBoundaryFile(t, root, "cmd/demo/main.go", "package main\n\nfunc main() {}\n")
	writeBoundaryFile(t, root, "internal/x/x.go", "package x\n\nfunc F() int { return 1 }\n")

	var out, errOut bytes.Buffer
	rc := runBoundary(&out, &errOut, []string{"--workspace", root, "--json"})
	if rc != 0 {
		t.Fatalf("clean workspace: rc=%d want 0 (stderr=%s)", rc, errOut.String())
	}
	var rep boundaryReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out.String())
	}
	if rep.Schema != boundarySchema {
		t.Errorf("schema=%q want %q", rep.Schema, boundarySchema)
	}
	if !rep.OK || rep.Count != 0 {
		t.Errorf("clean: ok=%v count=%d, want ok=true count=0", rep.OK, rep.Count)
	}
	if rep.Tells == nil {
		t.Error("tells is null; want [] so the JSON view is honest on a clean run")
	}
}

// A hardcoded download URL outside the audited chokepoint is an urllint tell:
// the verb must surface it, attribute it to urllint with the closed-vocabulary
// code, and exit 1.
func TestBoundaryDirtyWorkspaceURLTell(t *testing.T) {
	root := t.TempDir()
	writeBoundaryFile(t, root, "go.mod", "module example.com/x\n\ngo 1.22\n")
	writeBoundaryFile(t, root, "cmd/demo/main.go", "package main\n\nfunc main() {}\n")
	// A pasted model-download URL outside cmd/simpledemo escapes the audited builder.
	writeBoundaryFile(t, root, "internal/x/bad.go",
		"package x\n\nconst U = \"https://huggingface.co/org/repo/resolve/main/model.gguf\"\n")

	var out, errOut bytes.Buffer
	rc := runBoundary(&out, &errOut, []string{"--workspace", root, "--json"})
	if rc != 1 {
		t.Fatalf("dirty workspace: rc=%d want 1 (stderr=%s)", rc, errOut.String())
	}
	var rep boundaryReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out.String())
	}
	if rep.OK || rep.Count < 1 {
		t.Fatalf("dirty: ok=%v count=%d, want ok=false count>=1", rep.OK, rep.Count)
	}
	if rep.ByLinter["urllint"] < 1 {
		t.Errorf("expected an urllint tell, by_linter=%v", rep.ByLinter)
	}
	found := false
	for _, tell := range rep.Tells {
		if tell.Linter == "urllint" && tell.Code == "UNVERIFIED_EXTERNAL_URL" {
			found = true
		}
	}
	if !found {
		t.Errorf("no urllint/UNVERIFIED_EXTERNAL_URL tell in %+v", rep.Tells)
	}
}

// The audited chokepoint (cmd/simpledemo/main.go) is allowlisted: a download URL
// there is NOT a tell — proving the verb carries the same allow-set as the test.
func TestBoundaryAllowlistedChokepoint(t *testing.T) {
	root := t.TempDir()
	writeBoundaryFile(t, root, "go.mod", "module example.com/x\n\ngo 1.22\n")
	writeBoundaryFile(t, root, "cmd/demo/main.go", "package main\n\nfunc main() {}\n")
	writeBoundaryFile(t, root, "internal/x/x.go", "package x\n")
	writeBoundaryFile(t, root, "cmd/simpledemo/main.go",
		"package main\n\nconst U = \"https://huggingface.co/org/repo/resolve/main/model.gguf\"\n\nfunc main() {}\n")

	var out, errOut bytes.Buffer
	rc := runBoundary(&out, &errOut, []string{"--workspace", root})
	if rc != 0 {
		t.Fatalf("allowlisted chokepoint: rc=%d want 0 (stderr=%s, stdout=%s)", rc, errOut.String(), out.String())
	}
}

// A stray positional is a usage error (exit 2), per the cmd/fak convention.
func TestBoundaryRejectsPositional(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runBoundary(&out, &errOut, []string{"extra"}); rc != 2 {
		t.Fatalf("positional: rc=%d want 2", rc)
	}
}
