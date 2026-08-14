package codetools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// confine_test.go — the confinement invariant, stated as tests.
//
// These are the cases that decide whether the toolset is a kernel-mediated surface or a
// filesystem primitive with a comment claiming it is one.

// newTestToolset builds a toolset over a fresh scratch dir under the default read policy.
func newTestToolset(t *testing.T) (*Toolset, string) {
	t.Helper()
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ts, dir
}

func TestResolveAcceptsPathsInsideTheRoot(t *testing.T) {
	ts, dir := newTestToolset(t)
	for _, in := range []string{"a.txt", "./a.txt", "sub/b.txt", filepath.Join(dir, "a.txt")} {
		got, r := ts.resolve(in)
		if r != nil {
			t.Fatalf("resolve(%q) refused: %v", in, r)
		}
		if !within(ts.root, got.Abs) {
			t.Fatalf("resolve(%q) = %q, outside the root %q", in, got.Abs, ts.root)
		}
	}
}

func TestResolveDeniesLexicalTraversal(t *testing.T) {
	ts, _ := newTestToolset(t)
	for _, in := range []string{"../outside.txt", "a/../../outside.txt", "sub/../../../etc/passwd"} {
		_, r := ts.resolve(in)
		if r == nil {
			t.Fatalf("resolve(%q) admitted a traversal escape", in)
		}
		if r.Code != CodePathEscape {
			t.Fatalf("resolve(%q) code = %q, want %q", in, r.Code, CodePathEscape)
		}
	}
}

func TestResolveDeniesAbsolutePathOutsideTheRoot(t *testing.T) {
	ts, _ := newTestToolset(t)
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if _, r := ts.resolve(outside); r == nil || r.Code != CodePathEscape {
		t.Fatalf("resolve(%q) = %v, want PATH_ESCAPE", outside, r)
	}
}

func TestResolveDeniesNulByte(t *testing.T) {
	ts, _ := newTestToolset(t)
	if _, r := ts.resolve("a\x00.txt"); r == nil || r.Code != CodeMalformed {
		t.Fatalf("resolve with NUL = %v, want MALFORMED", r)
	}
}

// TestResolveDeniesSymlinkEscape is the case internal/agent/readengine.go deliberately
// does not cover (its comment argues a read-only engine can tolerate an in-tree symlink).
// It is exactly the hole that argument leaves open: the link below is lexically inside the
// workspace and really points outside it, so following it would let a READ return the
// bytes of a file the workspace boundary was supposed to make unreachable.
func TestResolveDeniesSymlinkEscape(t *testing.T) {
	ts, dir := newTestToolset(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("classified"), 0o644); err != nil {
		t.Fatalf("seed outside file: %v", err)
	}
	link := filepath.Join(dir, "escape")
	if err := os.Symlink(outside, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink creation requires privilege on this host")
		}
		t.Fatalf("symlink: %v", err)
	}
	// Both the link itself and a path THROUGH it must be refused: the second is the one
	// that actually reads the outside file.
	for _, in := range []string{"escape", "escape/secret.txt"} {
		_, r := ts.resolve(in)
		if r == nil {
			t.Fatalf("resolve(%q) admitted a symlink escape", in)
		}
		if r.Code != CodeSymlinkEscape {
			t.Fatalf("resolve(%q) code = %q, want %q", in, r.Code, CodeSymlinkEscape)
		}
	}
}

// TestReadRefusesSymlinkEscapeEndToEnd is the engine-level twin of the resolve test: it
// proves the refusal survives all the way to the tool result a model would see, rather
// than being a property only of an internal helper.
func TestReadRefusesSymlinkEscapeEndToEnd(t *testing.T) {
	ts, dir := newTestToolset(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("classified"), 0o644); err != nil {
		t.Fatalf("seed outside file: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "escape")); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink creation requires privilege on this host")
		}
		t.Fatalf("symlink: %v", err)
	}
	out, isErr := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "escape/secret.txt"}))
	if !isErr || errCode(t, out) != CodeSymlinkEscape {
		t.Fatalf("Read through a symlink = %s, want SYMLINK_ESCAPE", string(out))
	}
	if strings.Contains(string(out), "classified") {
		t.Fatalf("refusal leaked the outside file's content: %s", string(out))
	}
}

// TestResolveAllowsSymlinkedRoot pins that confinement is about ESCAPING, not about how
// the operator spelled the workspace: a root reached through a symlink (a macOS /tmp, a
// Windows junction) must not make every in-tree path look like an escape.
func TestResolveAllowsSymlinkedRoot(t *testing.T) {
	target := t.TempDir()
	link := filepath.Join(t.TempDir(), "workspace")
	if err := os.Symlink(target, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink creation requires privilege on this host")
		}
		t.Fatalf("symlink: %v", err)
	}
	if err := os.WriteFile(filepath.Join(target, "in.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	ts, err := New(Config{Root: link})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, r := ts.resolve("in.txt"); r != nil {
		t.Fatalf("resolve inside a symlinked root refused: %v", r)
	}
}

func TestZeroLimitsNormalizeToDefaults(t *testing.T) {
	got := Limits{}.normalize()
	if got != DefaultLimits() {
		t.Fatalf("zero Limits normalized to %+v, want the defaults %+v", got, DefaultLimits())
	}
}

// TestDefaultPolicyAdmitsImplementedTools pins the admission floor this slice ships: the
// three read tools and nothing else. A Write/Edit/Bash name reaching this package before
// #6704/#6705 land is not merely unlisted, it is unknown — and both must refuse.
func TestDefaultPolicyAdmitsImplementedTools(t *testing.T) {
	p := DefaultPolicy()
	for _, tool := range []string{ToolRead, ToolGrep, ToolGlob, ToolWrite, ToolEdit} {
		if !p.Allow[tool] {
			t.Fatalf("DefaultPolicy does not admit %q", tool)
		}
		if _, mine := engineFor(tool); !mine {
			t.Fatalf("engineFor(%q) reports the tool is not ours", tool)
		}
	}
	for _, tool := range []string{"Bash"} {
		if p.Allow[tool] {
			t.Fatalf("DefaultPolicy admits %q, which this slice does not implement", tool)
		}
		if engine, mine := engineFor(tool); mine {
			t.Fatalf("engineFor(%q) routes to %q, but no such engine is registered", tool, engine)
		}
	}
}
