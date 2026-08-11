package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunTurnTaxVisualCheck(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	data := filepath.Join(root, "tools", "hero_turntax.data.json")
	var stdout, stderr bytes.Buffer
	if code := runTurnTaxVisual(&stdout, &stderr, []string{"--data", data, "--check"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "is up to date") {
		t.Fatalf("stdout=%q", stdout.String())
	}
}

func TestRunTurnTaxVisualRejectsDrift(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(root, "tools", "hero_turntax.data.json"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	data = bytes.Replace(data, []byte(`"out_svg": "visuals/60-hero-turntax-curves.svg"`), []byte(`"out_svg": "stale.svg"`), 1)
	path := filepath.Join(dir, "data.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runTurnTaxVisual(&stdout, &stderr, []string{"--data", path, "--check"}); code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "drift") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}
