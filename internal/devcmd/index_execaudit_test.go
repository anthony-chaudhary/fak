package devcmd

// CLI proof for #5648. internal/devindex proves the DERIVATION (which packages are
// executable, tested, reached); this file proves the half that only exists here: the
// mapping from a verdict to an EXIT CODE, which is what lets the audit be used as a
// gate instead of a report.
//
// The load-bearing case is fail-closed. `fak index execaudit` over a tree whose source
// metadata cannot be loaded must exit non-zero and SAY could-not-establish-domain — the
// one bug that would quietly retire the instrument is an audit that folds over an empty
// domain, finds nothing wrong, and prints success.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// writeExecAuditFile writes one fixture file, creating parents.
func writeExecAuditFile(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", rel, err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

// runExecAudit runs the subcommand and returns its exit code plus both streams.
func runExecAudit(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	rc := RunIndex(&out, &errb, append([]string{"execaudit"}, args...))
	return rc, out.String(), errb.String()
}

// newExecAuditRoot returns a fixture root carrying the dos.toml every `fak index`
// subcommand needs to load the catalog before it dispatches. Whether the tree is also
// a Go MODULE is what each test varies.
func newExecAuditRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeExecAuditFile(t, root, "dos.toml", "[lanes.trees]\ndevindex = [\"internal/devindex/**\"]\n")
	return root
}

// TestIndexExecAuditFailsClosedOnUnresolvableDomain: no module, so no domain. The exit
// code must be the failing one and the JSON must carry the refusal rather than an empty
// green document a caller could mistake for "nothing wrong".
func TestIndexExecAuditFailsClosedOnUnresolvableDomain(t *testing.T) {
	bare := newExecAuditRoot(t)
	writeExecAuditFile(t, bare, "notes.txt", "no go module here\n")

	rc, out, _ := runExecAudit(t, "--json", "--root", bare)
	if rc != 1 {
		t.Fatalf("rc = %d, want 1 — an unresolvable domain must not exit green", rc)
	}
	var res devindex.ExecAuditResult
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("execaudit --json is not valid JSON: %v\n%s", err, out)
	}
	if res.Status != devindex.ExecDomainNotEstablished {
		t.Errorf("status = %q, want %q", res.Status, devindex.ExecDomainNotEstablished)
	}
	if res.Established {
		t.Error("Established = true for a tree with no resolvable module")
	}
	if res.Reason == "" {
		t.Error("fail-closed JSON carries no reason")
	}

	// The table mode shares the exit code and names the refusal on stderr.
	rc, _, errs := runExecAudit(t, "--root", bare)
	if rc != 1 {
		t.Errorf("table-mode rc = %d, want 1", rc)
	}
	if !strings.Contains(errs, devindex.ExecDomainNotEstablished) {
		t.Errorf("table-mode stderr does not name the refusal: %q", errs)
	}
}

// TestIndexExecAuditExitCodeTracksTheVerdict: a fully wired executable exits 0, and
// adding a single buildable-but-unwired package flips the same tree to 1 and names it.
func TestIndexExecAuditExitCodeTracksTheVerdict(t *testing.T) {
	root := newExecAuditRoot(t)
	writeExecAuditFile(t, root, "go.mod", "module example.com/cli\n\ngo 1.22\n")
	writeExecAuditFile(t, root, "cmd/wired/main.go", "package main\n\nfunc main() {}\n")
	writeExecAuditFile(t, root, "cmd/wired/main_test.go",
		"package main\n\nimport \"testing\"\n\nfunc TestBuilds(t *testing.T) { _ = t }\n")
	writeExecAuditFile(t, root, "Makefile", "build:\n\tgo build -o bin/ ./cmd/wired\n")

	if rc, out, errs := runExecAudit(t, "--root", root); rc != 0 {
		t.Fatalf("wired+tested tree rc = %d, want 0\nstdout:\n%s\nstderr:\n%s", rc, out, errs)
	}

	// The unwired package is named only by its own source and by a markdown inventory
	// row — the evidence a mention-counting audit would wrongly accept.
	writeExecAuditFile(t, root, "cmd/stranded/main.go",
		"package main\n\n// cmd/stranded is listed in the inventory below.\n\nfunc main() {}\n")
	writeExecAuditFile(t, root, "docs/INVENTORY.md",
		"# Commands\n\n| command | purpose |\n| --- | --- |\n| cmd/stranded | the unwired one |\n")

	rc, out, errs := runExecAudit(t, "--root", root)
	if rc != 1 {
		t.Fatalf("tree with an unwired executable rc = %d, want 1\nstdout:\n%s\nstderr:\n%s", rc, out, errs)
	}
	if !strings.Contains(out, "cmd/stranded") || !strings.Contains(out, string(devindex.ExecStatusOrphan)) {
		t.Errorf("failing readout does not name the orphan:\n%s", out)
	}
}
