package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCommandAddReadCleanup(t *testing.T) {
	ws := t.TempDir()
	var out, errOut bytes.Buffer
	add := []string{"decisions", "add", "--workspace", ws, "--key", "bench-6349-blank", "--action", "OPEN_ISSUE", "--severity", "P1", "--payload", `{"case":"blank"}`, "--json"}
	if code := run(&out, &errOut, add); code != 0 {
		t.Fatalf("add code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, add); code != 0 {
		t.Fatalf("repeat code=%d stderr=%s", code, errOut.String())
	}
	var repeated struct {
		Created bool `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &repeated); err != nil || repeated.Created {
		t.Fatalf("repeat=%s err=%v", out.String(), err)
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--native=false", "--json"}); code != 0 {
		t.Fatalf("list code=%d stderr=%s", code, errOut.String())
	}
	var rows []map[string]any
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 {
		t.Fatalf("list=%s err=%v", out.String(), err)
	}
	if rows[0]["key"] != "bench-6349-blank" || rows[0]["action"] != "OPEN_ISSUE" || rows[0]["severity"] != "P1" {
		t.Fatalf("wrong row: %+v", rows[0])
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "remove", "--workspace", ws, "--key", "bench-6349-blank", "--json"}); code != 0 {
		t.Fatalf("remove code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--native=false", "--json"}); code != 0 {
		t.Fatal(errOut.String())
	}
	if string(bytes.TrimSpace(out.Bytes())) != "[]" {
		t.Fatalf("residue: %s", out.String())
	}
}

func TestListDelegatesNativeReader(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fixture; exercised under WSL validation")
	}
	ws := t.TempDir()
	bin := t.TempDir()
	script := filepath.Join(bin, "dos")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nprintf '[{\"kind\":\"NATIVE\"}]'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var rows []map[string]any
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 || rows[0]["kind"] != "NATIVE" {
		t.Fatalf("rows=%s err=%v", out.String(), err)
	}
}
