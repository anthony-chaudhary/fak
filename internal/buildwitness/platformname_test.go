package buildwitness

import (
	"go/build"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFilenamePlatformTags(t *testing.T) {
	cases := map[string][]string{
		"closed_windows.go":        {"windows"},
		"closed_windows_test.go":   {"windows"},
		"x_linux_amd64.go":         {"linux", "amd64"},
		"x_linux_amd64_test.go":    {"linux", "amd64"},
		"x_arm64.go":               {"arm64"},
		"x_darwin.s":               {"darwin"},
		"windows.go":               nil, // the element before the first "_" is ignored
		"linux_test.go":            nil,
		"closed_window_history.go": nil,
		"closed_file_window.go":    nil,
		"windows_helper.go":        nil,
		"x_test.go":                nil,
		"x.go":                     nil,
	}
	for name, want := range cases {
		if got := FilenamePlatformTags(name); !reflect.DeepEqual(got, want) {
			t.Errorf("FilenamePlatformTags(%q) = %v, want %v", name, got, want)
		}
	}
}

// TestPlatformNameListsMatchGoBuild pins knownOS/knownArch to go/build itself: every listed name
// must make go/build exclude a suffixed, tag-less file under a context of some OTHER platform. A
// list entry go/build does not honor would refuse a file Go actually compiles everywhere.
func TestPlatformNameListsMatchGoBuild(t *testing.T) {
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH = "plan9", "386"
	if knownOS["plan9"] {
		ctx.GOOS = "aix"
	}
	ctx.OpenFile = func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader("package p\n")), nil }
	check := func(tag string, other string, isOS bool) {
		c := ctx
		if isOS {
			c.GOOS = other
		} else {
			c.GOARCH = other
		}
		ok, err := c.MatchFile("dir", "x_"+tag+".go")
		if err != nil {
			t.Fatalf("MatchFile x_%s.go: %v", tag, err)
		}
		if ok {
			t.Errorf("go/build compiles x_%s.go under %s/%s; %q is not a filename constraint and must leave the list", tag, c.GOOS, c.GOARCH, tag)
		}
	}
	for os := range knownOS {
		other := "linux"
		if os == "linux" || os == "android" {
			other = "windows"
		}
		check(os, other, true)
	}
	for arch := range knownArch {
		other := "amd64"
		if arch == "amd64" || arch == "amd64p32" || arch == "386" {
			other = "arm64"
		}
		check(arch, other, false)
	}
}

// TestCheckPlatformFilenameRefusesIncidentShape replays f9d4cca4: a word that happens to be a GOOS
// ends the basename and the file has no build line, so the gate must refuse it.
func TestCheckPlatformFilenameRefusesIncidentShape(t *testing.T) {
	src := []byte("package ctxmmu\n\nfunc (c *compactor) fileWindows() {}\n")
	for _, rel := range []string{"internal/ctxmmu/closed_windows.go", "internal/ctxmmu/closed_windows_test.go", "internal/x/foo_windows.go"} {
		v := CheckPlatformFilename(rel, src)
		if v == nil {
			t.Fatalf("%s without a build line was admitted", rel)
		}
		if !reflect.DeepEqual(v.Missing, []string{"windows"}) || v.HasLine {
			t.Errorf("%s: violation = %+v, want Missing=[windows] HasLine=false", rel, v)
		}
		if msg := v.String(); !strings.Contains(msg, "//go:build windows") || !strings.Contains(msg, "rename") {
			t.Errorf("message %q does not name both fixes", msg)
		}
	}
	// A build line that names some other tag does not make the platform restriction explicit.
	v := CheckPlatformFilename("internal/x/foo_linux_amd64.go", []byte("//go:build linux && cgo\n\npackage x\n"))
	if v == nil || !reflect.DeepEqual(v.Missing, []string{"amd64"}) || !v.HasLine {
		t.Errorf("partial build line: violation = %+v, want Missing=[amd64] HasLine=true", v)
	}
}

func TestCheckPlatformFilenameAdmitsExplicitAndUnconstrained(t *testing.T) {
	cases := map[string]string{
		"cmd/fak/model_artifact_identity_windows.go": "//go:build windows\n\npackage main\n",
		"internal/x/foo_windows_test.go":             "// Copyright header.\n\n//go:build windows && !race\n\npackage x\n",
		"internal/x/foo_linux_arm64.go":              "//go:build linux && arm64\n\npackage x\n",
		"internal/x/foo_darwin.go":                   "// +build darwin\n\npackage x\n",
		"internal/x/foo_unix_windows.go":             "//go:build !windows\n\npackage x\n", // names it; satisfiability is the author's call
		"internal/x/closed_window_history.go":        "package x\n",
		"internal/x/windows.go":                      "package x\n",
	}
	for rel, src := range cases {
		if v := CheckPlatformFilename(rel, []byte(src)); v != nil {
			t.Errorf("%s refused: %s", rel, v)
		}
	}
}

func TestCheckPlatformFilenamesHonorsAllowlist(t *testing.T) {
	root := t.TempDir()
	write := func(rel, src string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("a/foo_windows.go", "package a\n")
	write("a/bar_windows.go", "package a\n")
	write("a/ok_windows.go", "//go:build windows\n\npackage a\n")
	write("testdata/fixture_linux.go", "package fixture\n")

	old := PlatformNameAllowlist
	PlatformNameAllowlist = map[string]string{"a/bar_windows.go": "reviewed: test fixture"}
	t.Cleanup(func() { PlatformNameAllowlist = old })

	rels, err := ScanPlatformFilenames(root)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"a/bar_windows.go", "a/foo_windows.go", "a/ok_windows.go"}; !reflect.DeepEqual(rels, want) {
		t.Fatalf("ScanPlatformFilenames = %v, want %v (testdata skipped)", rels, want)
	}
	bad, err := CheckPlatformFilenames(root, append(rels, "a/deleted_windows.go", "a/README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if len(bad) != 1 || bad[0].Path != "a/foo_windows.go" {
		t.Fatalf("CheckPlatformFilenames = %v, want only a/foo_windows.go", bad)
	}
}
