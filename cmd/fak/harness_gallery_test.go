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
	if code := runHarnessGallery(&out, &errOut, []string{"list"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"Harness starter packs", "Use when:", "Outcome:", "readonly-support", "Next: inspect one pack"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("list missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errOut.Reset()
	if code := runHarnessGallery(&out, &errOut, []string{"show", "--id", "readonly-support"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"Readonly Support Desk (readonly-support)", "For:", "Does not get:", "Ten-minute path:", "Next: fak harness gallery init"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("show missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errOut.Reset()
	if code := runHarnessGallery(&out, &errOut, []string{"show", "--id", "readonly-support", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var shown harnessgallery.Blueprint
	if err := json.Unmarshal(out.Bytes(), &shown); err != nil || shown.ID != "readonly-support" {
		t.Fatalf("shown=%+v err=%v", shown, err)
	}

	out.Reset()
	errOut.Reset()
	dir := filepath.Join(t.TempDir(), "pack")
	if code := runHarnessGallery(&out, &errOut, []string{"init", "--id", "coding-workspace", "--dir", dir}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"Initialized coding-workspace", "Created:", "harness.pack.json", "Next:", "Edit " + filepath.Join(dir, "harness.pack.json")} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("init missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errOut.Reset()
	if code := runHarnessGallery(&out, &errOut, []string{"init", "--id", "coding-workspace", "--dir", dir}); code != 0 || !strings.Contains(out.String(), "Preserved:") {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}

	out.Reset()
	errOut.Reset()
	if code := runHarnessGallery(&out, &errOut, []string{"selfcheck"}); code != 0 || !strings.Contains(out.String(), "blueprints=4") {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
}
