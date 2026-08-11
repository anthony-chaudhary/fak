package main

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAbsolutePathFlagsPreserveCallerPathsAcrossNestedModule(t *testing.T) {
	caller := filepath.Join(string(filepath.Separator), "repo")
	got, err := absolutePathFlags([]string{
		"-config", "tools/videogen/projects/demo/render.json",
		"-timeline=timeline.json",
		"-verify",
	}, caller)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"-config", filepath.Join(caller, "tools/videogen/projects/demo/render.json"),
		"-timeline=" + filepath.Join(caller, "timeline.json"),
		"-verify",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("absolutePathFlags = %#v, want %#v", got, want)
	}
}

func TestAbsolutePathFlagsRefuseMissingValue(t *testing.T) {
	if _, err := absolutePathFlags([]string{"-config"}, "/repo"); err == nil {
		t.Fatal("missing -config value succeeded")
	}
}

func TestProjectNamesArePortableSlugs(t *testing.T) {
	for _, name := range []string{"nix-engine-delivery", "gt-cycle-2", "a"} {
		if !projectName.MatchString(name) {
			t.Errorf("valid name %q rejected", name)
		}
	}
	for _, name := range []string{"", "Upper", "../escape", "has space", "-leading"} {
		if projectName.MatchString(name) {
			t.Errorf("invalid name %q accepted", name)
		}
	}
}

func TestNewProjectCopiesTemplateAndRefusesOverwrite(t *testing.T) {
	root := t.TempDir()
	template := filepath.Join(root, "tools", "videogen", "templates", "terminal")
	if err := os.MkdirAll(template, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(template, "render.json"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := newProject(root, "shared-demo")
	if err != nil {
		t.Fatal(err)
	}
	if got != "tools/videogen/projects/shared-demo" {
		t.Fatalf("newProject path = %q", got)
	}
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(got), "render.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "{}\n" {
		t.Fatalf("copied render.json = %q", raw)
	}
	if _, err := newProject(root, "shared-demo"); err == nil {
		t.Fatal("second newProject overwrote the existing project")
	}
}

func TestRendererModuleHasOneSharedOwner(t *testing.T) {
	root := t.TempDir()
	legacy := filepath.Join(root, "deploy", "nix", "proof-e2e-2026-07-27", "renderer")
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(legacy, "go.mod"), []byte("module old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := rendererModule(root); err == nil {
		t.Fatal("rendererModule accepted the legacy topic-local renderer; the shared tool would have two owners")
	}

	shared := filepath.Join(root, "tools", "videogen", "terminal")
	if err := os.MkdirAll(shared, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(shared, "go.mod"), []byte("module shared\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := rendererModule(root)
	if err != nil {
		t.Fatal(err)
	}
	if got != shared {
		t.Fatalf("rendererModule shared = %q, want %q", got, shared)
	}
}

func TestRunWithoutArgumentsPrintsSharedUsage(t *testing.T) {
	var out bytes.Buffer
	if err := run(nil, nil, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"-new NAME", "-config FILE -verify", "-record-typescript"} {
		if !bytes.Contains(out.Bytes(), []byte(want)) {
			t.Errorf("usage missing %q:\n%s", want, out.String())
		}
	}
}

func TestExtractFFmpegFlag(t *testing.T) {
	base := t.TempDir()
	args, got, err := extractFFmpegFlag([]string{"-config", "render.json", "-ffmpeg", "bin/ffmpeg.exe", "-all"}, base)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(base, "bin", "ffmpeg.exe")
	if got != want {
		t.Fatalf("ffmpeg = %q, want %q", got, want)
	}
	if diff := strings.Join(args, " "); diff != "-config render.json -all" {
		t.Fatalf("remaining args = %q", diff)
	}
}

func TestExtractFFmpegFlagNeedsValue(t *testing.T) {
	if _, _, err := extractFFmpegFlag([]string{"-ffmpeg"}, t.TempDir()); err == nil {
		t.Fatal("expected missing value error")
	}
}
