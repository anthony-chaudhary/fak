package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunPrivatePathUsesOpaquePrivateRoot(t *testing.T) {
	private := filepath.Join(t.TempDir(), "private")
	if err := os.MkdirAll(private, 0o755); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runPrivatePath(&out, &errOut, []string{"--root", private, "--create", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Path    string `json:"path"`
		Created bool   `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Created {
		t.Fatal("created=false")
	}
	if strings.Contains(strings.ToLower(got.Path), "dgx") {
		t.Fatalf("hardware identity leaked: %q", got.Path)
	}
	if !strings.HasPrefix(got.Path, filepath.Join(private, "fleet-runs", "codex")) {
		t.Fatalf("path=%q", got.Path)
	}
}

func TestRunPrivatePathRejectsLabels(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runPrivatePath(&out, &errOut, []string{"machine-label"}); code != 2 {
		t.Fatalf("code=%d", code)
	}
	if !strings.Contains(errOut.String(), "usage:") {
		t.Fatalf("stderr=%q", errOut.String())
	}
}
