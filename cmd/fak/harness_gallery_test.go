package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessgallery"
)

func TestHarnessGalleryCLI(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessGallery(&out, &errOut, []string{"list", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var items []harnessgallery.Blueprint
	if err := json.Unmarshal(out.Bytes(), &items); err != nil || len(items) != 4 {
		t.Fatalf("items=%d err=%v", len(items), err)
	}
	out.Reset()
	errOut.Reset()
	dir := filepath.Join(t.TempDir(), "pack")
	if code := runHarnessGallery(&out, &errOut, []string{"init", "--id", "coding-workspace", "--dir", dir, "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "harness.pack.json") {
		t.Fatalf("out=%s", out.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runHarnessGallery(&out, &errOut, []string{"selfcheck"}); code != 0 || !strings.Contains(out.String(), "blueprints=4") {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
}
