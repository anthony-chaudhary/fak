package main

// wip_fence_test.go — the pure-transform tables for `fak wip fence`/`unfence`
// (#4153) plus the STRONG round-trip witness (modeled on internal/buildwitness):
// in a hermetic temp module, a broken WIP file reds the default build, fencing it
// turns the default build green, and `-tags wip_<slug>` compiles it back in.

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestWipFenceSlug(t *testing.T) {
	cases := []struct{ in, want string }{
		{"new_thing.go", "new_thing"},
		{"Foo_test.go", "foo"},
		{"weird-Name.go", "weird_name"},
		{".go", "wip"},
		{"", "wip"},
		{"cmd/fak/wip_fence.go", "wip_fence"},
		{`C:\work\fak\cmd\fak\dispatch_tick.go`, "dispatch_tick"},
		{"---.go", "wip"},
		{"a..b!!c.go", "a_b_c"},
		{"_lead_trail_.go", "lead_trail"},
	}
	for _, tc := range cases {
		if got := wipFenceSlug(tc.in); got != tc.want {
			t.Errorf("wipFenceSlug(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestFenceText(t *testing.T) {
	src := "package foo\n\nfunc F() {}\n"
	out, changed := fenceText(src, "x")
	if !changed {
		t.Fatalf("fenceText: expected changed=true on an unfenced file")
	}
	want := "//go:build wip_x\n\npackage foo\n\nfunc F() {}\n"
	if out != want {
		t.Fatalf("fenceText header shape:\n got %q\nwant %q", out, want)
	}

	// Idempotent: fencing twice is fencing once.
	out2, changed2 := fenceText(out, "x")
	if changed2 || out2 != out {
		t.Fatalf("fenceText not idempotent: changed=%v\n got %q\nwant %q", changed2, out2, out)
	}

	// A file already carrying ANY //go:build constraint is left unchanged — never stack.
	for _, existing := range []string{
		"//go:build linux\n\npackage foo\n",
		"//go:build wip_other\n\npackage foo\n",
		"// Copyright.\n\n//go:build !windows\n\npackage foo\n",
	} {
		if got, ch := fenceText(existing, "x"); ch || got != existing {
			t.Fatalf("fenceText stacked a constraint onto %q: changed=%v got %q", existing, ch, got)
		}
	}

	// A leading doc comment is NOT a constraint: the fence still goes on top.
	doc := "// Package foo does things.\npackage foo\n"
	got, ch := fenceText(doc, "y")
	if !ch || !strings.HasPrefix(got, "//go:build wip_y\n\n// Package foo does things.") {
		t.Fatalf("fenceText over a doc comment: changed=%v got %q", ch, got)
	}
}

func TestUnfenceText(t *testing.T) {
	// unfence(fence(src)) round-trips back to src EXACTLY.
	for _, src := range []string{
		"package foo\n\nfunc F() {}\n",
		"// Package foo does things.\npackage foo\n",
		"package foo", // no trailing newline
		"",
	} {
		fenced, ch := fenceText(src, "x")
		if !ch {
			t.Fatalf("fenceText(%q): expected changed=true", src)
		}
		out, slug, changed := unfenceText(fenced)
		if !changed || slug != "x" || out != src {
			t.Fatalf("round-trip broke: changed=%v slug=%q\n got %q\nwant %q", changed, slug, out, src)
		}
	}

	// Unfencing an unfenced file is a no-op.
	plain := "package foo\n"
	if out, slug, ch := unfenceText(plain); ch || slug != "" || out != plain {
		t.Fatalf("unfenceText on unfenced file: changed=%v slug=%q got %q", ch, slug, out)
	}

	// A non-wip_ constraint is a real build gate — never removed.
	for _, keep := range []string{
		"//go:build linux\n\npackage foo\n",
		"//go:build wip_x && linux\n\npackage foo\n", // compound: not a pure fence
	} {
		if out, _, ch := unfenceText(keep); ch || out != keep {
			t.Fatalf("unfenceText removed a non-fence constraint from %q: got %q", keep, out)
		}
	}
}

// TestWipFenceBuildRoundTrip is the strong structural witness: in a hermetic temp
// module, an untracked-style WIP file referencing an undefined symbol reds the
// default `go build ./...`; after fenceText the default build is GREEN (peers/CI
// unpoisoned); and `go build -tags wip_demo` compiles the fenced file back IN
// (red again on the undefined symbol — the WIP is still buildable-by-tag).
func TestWipFenceBuildRoundTrip(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH; the build round-trip witness cannot run here")
	}
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if werr := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); werr != nil {
			t.Fatal(werr)
		}
	}
	write("go.mod", "module wipfencetest\n\ngo 1.21\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	wipSrc := "package main\n\nfunc callsMissing() { missingSymbol() }\n"
	write("wip.go", wipSrc)

	nullDev := "/dev/null"
	if runtime.GOOS == "windows" {
		nullDev = "NUL"
	}
	build := func(tags string) (string, error) {
		t.Helper()
		args := []string{"build"}
		if tags != "" {
			args = append(args, "-tags", tags)
		}
		args = append(args, "-o", nullDev, "./...")
		cmd := exec.Command(goBin, args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=")
		out, berr := cmd.CombinedOutput()
		return string(out), berr
	}

	if out, berr := build(""); berr == nil {
		t.Fatalf("expected the unfenced wip.go to red the default build; it compiled:\n%s", out)
	} else if !strings.Contains(out, "missingSymbol") {
		t.Fatalf("default build red for the wrong reason (want undefined missingSymbol):\n%s", out)
	}

	fenced, changed := fenceText(wipSrc, "demo")
	if !changed {
		t.Fatal("fenceText: expected changed=true on the unfenced wip.go")
	}
	write("wip.go", fenced)

	if out, berr := build(""); berr != nil {
		t.Fatalf("default build still red after fencing wip.go:\n%s", out)
	}
	if out, berr := build("wip_demo"); berr == nil {
		t.Fatalf("-tags wip_demo should compile the fenced file back IN (red on missingSymbol); it was green:\n%s", out)
	} else if !strings.Contains(out, "missingSymbol") {
		t.Fatalf("-tags wip_demo red for the wrong reason (want undefined missingSymbol):\n%s", out)
	}
}
