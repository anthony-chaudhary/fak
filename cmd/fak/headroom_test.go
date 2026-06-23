package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunHeadroomListShowsPlugins(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"list"}); code != 0 {
		t.Fatalf("list exit=%d", code)
	}
	for _, want := range []string{"noop", "native", "headroom"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("list missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunHeadroomStatus(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d", code)
	}
	if !strings.Contains(out.String(), "selected:") || !strings.Contains(out.String(), "headroom url:") {
		t.Fatalf("status output unexpected:\n%s", out.String())
	}
}

func TestRunHeadroomCompressFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.json")
	pretty := "{\n    \"a\": 1,\n    \"b\": [\n        1,\n        2,\n        3\n    ]\n}\n"
	if err := os.WriteFile(path, []byte(pretty), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "native", path}); code != 0 {
		t.Fatalf("compress exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "compressed: true") || !strings.Contains(s, "json-min") {
		t.Fatalf("compress did not report a JSON saving:\n%s", s)
	}
}

func TestRunHeadroomCompressEmit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.json")
	if err := os.WriteFile(path, []byte("{\n  \"x\": 1\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "native", "--emit", path}); code != 0 {
		t.Fatalf("emit exit=%d", code)
	}
	if out.String() != `{"x":1}` {
		t.Fatalf("emit should write minified bytes, got %q", out.String())
	}
}

func TestRunHeadroomUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"frobnicate"}); code != 2 {
		t.Fatalf("unknown subcommand exit=%d, want 2", code)
	}
}

func TestRunHeadroomCompressUnknownVia(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "nope", path}); code != 2 {
		t.Fatalf("unknown --via exit=%d, want 2", code)
	}
}
