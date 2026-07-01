package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/edittx"
)

func TestEditTxCLIRollsBackOnFailingCheck(t *testing.T) {
	root := t.TempDir()
	writeEditTxCLIFile(t, root, "a.txt", "old\n")
	spec := `{"edits":[{"path":"a.txt","content":"new\n"}]}`
	var out, errBuf bytes.Buffer
	code := runEditTx(&out, &errBuf, strings.NewReader(spec), []string{
		"--workspace", root,
		"--spec", "-",
		"--check", "exit 7",
		"--json",
	})
	if code != 1 {
		t.Fatalf("runEditTx failing check exit = %d, want 1; stdout=%s stderr=%s", code, out.String(), errBuf.String())
	}
	var res edittx.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("result JSON: %v\n%s", err, out.String())
	}
	if res.Reason != edittx.ReasonCheckFailed || !res.RolledBack {
		t.Fatalf("result = %+v, want CHECK_FAILED rollback", res)
	}
	if got := readEditTxCLIFile(t, root, "a.txt"); got != "old\n" {
		t.Fatalf("a.txt after rollback = %q, want old", got)
	}
}

func TestEditTxCLIAppliesSpecFile(t *testing.T) {
	root := t.TempDir()
	specPath := filepath.Join(root, "tx.json")
	spec := `{"edits":[{"path":"a.txt","content":"A\n"},{"path":"b.txt","content":"B\n"}]}`
	if err := os.WriteFile(specPath, []byte(spec), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errBuf bytes.Buffer
	code := runEditTx(&out, &errBuf, strings.NewReader(""), []string{
		"--workspace", root,
		"--spec", specPath,
	})
	if code != 0 {
		t.Fatalf("runEditTx apply exit = %d, want 0; stdout=%s stderr=%s", code, out.String(), errBuf.String())
	}
	if got := readEditTxCLIFile(t, root, "a.txt"); got != "A\n" {
		t.Fatalf("a.txt = %q, want A", got)
	}
	if got := readEditTxCLIFile(t, root, "b.txt"); got != "B\n" {
		t.Fatalf("b.txt = %q, want B", got)
	}
	if !strings.Contains(out.String(), "applied 2 edit") {
		t.Fatalf("human output missing apply summary: %s", out.String())
	}
}

func writeEditTxCLIFile(t *testing.T, root, path, content string) {
	t.Helper()
	abs := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readEditTxCLIFile(t *testing.T, root, path string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(b), "\r\n", "\n")
}
